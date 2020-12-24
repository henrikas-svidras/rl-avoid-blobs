# pylint: disable=missing-docstring

import pygame
import pygame.freetype
import numpy as np
import matplotlib.pyplot as plt
import math
from objects import Player,\
                    Enemy

import tensorflow as tf

##### Keras setup
# Configuration parameters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
max_steps_per_episode = 10000

eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

num_inputs = 6
num_actions = 4
num_hidden = 128

inputs = tf.keras.layers.Input(shape=num_inputs)
common = tf.keras.layers.Dense(num_hidden, activation="relu")(inputs)
action = tf.keras.layers.Dense(num_actions, activation='softmax')(common)
critic = tf.keras.layers.Dense(num_actions)(common)
model = tf.keras.Model(inputs=inputs, outputs=[action,critic])
print(model.summary())
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
huber_loss = tf.keras.losses.Huber()


action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0

#####

pygame.init()

# Set up the drawing window
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000
MAX_ENEMY = 4
screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])

# Create a custom event for adding a new enemy
#ADDENEMY = pygame.USEREVENT + 1
#pygame.time.set_timer(ADDENEMY, 250)

generation = 0
time_passed = 0
USER_PERMISSION = True

times_survived = []
figure, ax = plt.subplots(1,1)


def get_enemy_states(enemies, scr_x, scr_y):
    xy_coords = []
    for enemy in enemies:
        x = enemy.rect.centerx
        y = enemy.rect.centery
        xy_coords.append([x, y])
    xy_coords.append([scr_x, scr_y])
    xy_coords.append([0, 0])
    xy_coords = np.array(xy_coords)

    dists = np.sqrt(np.sum(np.square(xy_coords), axis=1))
    return dists

while time_passed<60 and USER_PERMISSION:

    # Run until the user asks to quit
    RUNNING = True
    GAME_OVER = False
    player = Player(bot=True)

    episode_reward = 0
    with tf.GradientTape() as tape:
        player = Player(bot=True)

        # Create groups to hold enemy sprites and all sprites
        # - enemies is used for collision detection and position updates
        # - all_sprites is used for rendering
        enemies = pygame.sprite.Group()
        all_sprites = pygame.sprite.Group()
        all_sprites.add(player)

        myfont = pygame.freetype.SysFont('Comic Sans MS', 30)
        clock = pygame.time.Clock()


        time_passed = 0
        time_after_death = 0
        RUNNING = True
        reward_history = []
        action_prob_history_x = []

        while len(enemies) < MAX_ENEMY:
                print('adding new enemies')
                # Create the new enemy and add it to sprite groups
                new_enemy = Enemy()
                enemies.add(new_enemy)
                all_sprites.add(new_enemy)

        dists = get_enemy_states(enemies, SCREEN_WIDTH, SCREEN_HEIGHT)

        state = tf.convert_to_tensor(np.vstack(dists-np.sqrt(player.rect.centerx**2+player.rect.centery**2)))

        print(f'Starting generation {generation}')
        while RUNNING:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    RUNNING = False
                    USER_PERMISSION = False
                ##########################

            screen.fill((0, 0, 0))

            action_probs, critic_value = model(tf.transpose(state))  
            action = np.random.choice(4, p=np.squeeze(action_probs))

            if not GAME_OVER:
                critic_value_history.append(critic_value[0, 0])
                action_probs_history.append(tf.math.log(action_probs[0, action]))

            action_x = 0
            action_y = 0
            if action == 0:
                action_x = 1
            if action == 1:
                action_x = -1
            if action == 2:
                action_y = 1
            if action == 3:
                action_y = -1

            player.update(None, (action_x, action_y))

            # Draw all sprites
            for entity in all_sprites:
                screen.blit(entity.surf, entity.rect)

            for enemy in enemies:
                enemy.steer(player.rect.centerx, player.rect.centery)
                enemy.drag()
            enemies.update()

            # Check if any enemies have collided with the player
            if player.alive():
                if pygame.sprite.spritecollideany(player, enemies):
                # player.kill()
                    GAME_OVER = True
            if not GAME_OVER:
                time_passed+=clock.get_time()/1000
                myfont.render_to(screen, (SCREEN_WIDTH*0.8, SCREEN_HEIGHT*0.1), f"{time_passed:.2g} s", (220, 0, 0))
                dists = get_enemy_states(enemies, SCREEN_WIDTH, SCREEN_HEIGHT)

                state = tf.convert_to_tensor(np.vstack(dists-np.sqrt(player.rect.centerx**2+player.rect.centery**2)))

                reward = math.exp(time_passed-15)-int(player.hitting_side_wall)-int(player.hitting_top_wall)
                rewards_history.append(reward)
                episode_reward += reward
            else:
                myfont.render_to(screen, (SCREEN_WIDTH*0.8, SCREEN_HEIGHT*0.1), f"{time_passed:.2f} s", (220, 0, 0))
                myfont.render_to(screen, (SCREEN_WIDTH/2, SCREEN_HEIGHT/2), "Big ded", (220, 0, 0))
                if time_after_death > 5:
                    RUNNING = False
                    time_after_death = 0
                else:
                    time_after_death += clock.get_time()/1000
            
            myfont.render_to(screen, (SCREEN_WIDTH*0.1, SCREEN_HEIGHT*0.05), f"Generation {generation}", (220, 0, 0))
            myfont.render_to(screen, (SCREEN_WIDTH*0.1, SCREEN_HEIGHT*0.1), f"running reward: {running_reward:.2f}", (220, 0, 0))
            # Flip the display
            pygame.display.flip()
            # Ensure program maintains a rate of 30 frames per second
            # Setup the clock for a decent framerate
            clock.tick(30)
        ax.plot(generation, time_passed, 'or')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Time survided, s')
        plt.show(block=False)

        # Update running reward to check condition for solving
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        returns = []
        discounted_sum = 0

        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize

        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        # Calculating loss values to update our network
        history = zip(action_probs_history, critic_value_history, returns)

        actor_losses = action_probs_history
        critic_losses = []
        
        for log_prob, value, ret in history:
    
        #     # At this point in history, the critic estimated that we would get a
        #     # total reward = `value` in the future. We took an action with log probability
        #     # of `log_prob` and ended up recieving a total reward = `ret`.
        #     # The actor must be updated so that it predicts an action that leads to
        #     # high rewards (compared to critic's estimate) with high probability.
             diff = ret - value
        #     print(diff)

             actor_losses.append(-log_prob * diff)  # actor loss

        #     # The critic must be updated so that it predicts a better estimate of
        #     # the future rewards.
             critic_losses.append(
        #         huber_loss(value, ret)
                  huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))

             )

        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)

        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Clear the loss and reward history
 
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()

        generation+=1

    # Log details
    episode_count += 1
    if episode_count % 10 == 0:
        template = "running reward: {:.2f} at episode {}"
        print(template.format(running_reward, episode_count))


    # Create groups to hold enemy sprites and all sprites
    # - enemies is used for collision detection and position updates
    # - all_sprites is used for rendering
    enemies = pygame.sprite.Group()
    all_sprites = pygame.sprite.Group()
    all_sprites.add(player)

# Done! Time to quit.
pygame.quit()

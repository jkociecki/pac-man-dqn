#
#
#
# def play_pacman(model_path="models/pacman_dqn_best.pth", fps=30):
#     env = AdvancedPacmanEnv(render=True, fps=fps)
#     initial_state = env.reset()
#     state_matrix_shape = initial_state["matrix"].shape
#     state_feature_size = len(initial_state["features"])
#     config = DQNConfig()
#     agent = AdvancedDQNAgent(
#         state_matrix_shape=state_matrix_shape,
#         state_feature_size=state_feature_size,
#         action_size=4,  # UP, RIGHT, DOWN, LEFT
#         config=config
#     )
#     if not agent.load(model_path):
#         print(f"Failed to load model from {model_path}")
#         env.close()
#         return
#     agent.epsilon = 0.05
#     state = env.reset()
#     done = False
#     total_reward = 0
#
#     while not done:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 env.close()
#                 return
#         action = agent.get_action(state, eval_mode=True)
#         next_state, reward, done, info = env.step(action)
#         state = next_state
#         total_reward += reward
#         print(f"Score: {info['score']}, Lives: {info['lives']}, Reward: {reward:.1f}, Total Reward: {total_reward:.1f}")
#
#     print(f"Game Over! Final Score: {info['score']}")
#     time.sleep(2)
#     env.close()

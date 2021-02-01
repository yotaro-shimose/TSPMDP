from tspmdp.env import TSPMDP
import tensorflow as tf


def raise_at_final(env: TSPMDP, actions: list):
    env.reset()
    answer = False
    for i in range(len(actions)):
        action = actions[i]
        try:
            action = tf.constant([action], dtype=tf.int32)
            env.step(action)
        except tf.errors.InvalidArgumentError:
            if i == len(actions) - 1:
                answer = True
            else:
                break
    return answer


def complete_synario(env: TSPMDP, actions: list):
    env.reset()
    answer = False
    is_terminal = False
    for i in range(len(actions)):
        action = actions[i]
        try:
            action = tf.constant([action], dtype=tf.int32)
            _, _, is_terminal = env.step(action)
        except tf.errors.InvalidArgumentError:
            break
    else:
        if is_terminal:
            answer = True
    return answer


def test_env_mask():
    env = TSPMDP(batch_size=1, n_nodes=10)
    actions = [0]
    raise_at_final(env, actions)
    actions = [3, 4, 4]
    raise_at_final(env, actions)
    actions = [1, 2, 3, 4, 0]
    raise_at_final(env, actions)
    actions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 9]


def test_env_complete():
    env = TSPMDP(batch_size=1, n_nodes=10)
    actions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
    complete_synario(env, actions)


def test_env_synchronization():
    original = TSPMDP(batch_size=1, n_nodes=10)
    original.reset()
    copy = TSPMDP(batch_size=1, n_nodes=10)
    copy.import_states(original.export_states())
    original_sum = 0
    copy_sum = 0
    actions = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i in range(len(actions)):
        action = actions[i]
        action = tf.constant([action], dtype=tf.int32)
        _, original_reward, _ = original.step(action)
        _, copy_reward, _ = copy.step(action)
        original_sum += original_reward
        copy_sum += copy_reward
    assert original_sum == copy_sum


def test_env_reward_on_episode():
    B, N = 128, 100
    original = TSPMDP(batch_size=B, n_nodes=N, reward_on_episode=True)
    original.reset()
    copy = TSPMDP(batch_size=B, n_nodes=N, reward_on_episode=False)
    init_state = original.export_states()
    init_state.pop("rewards")
    copy_sum = 0
    copy.import_states(init_state)
    actions = tf.constant([list(range(1, N))
                           for _ in range(B)], dtype=tf.int32)
    original_reward = None
    for j in range(actions.shape[1]):
        action = actions[:, j]
        _, original_reward, _ = original.step(action)
        _, copy_reward, _ = copy.step(action)
        assert tf.reduce_all(
            original_reward == 0.) or j == actions.shape[1] - 1
        copy_sum += copy_reward
    tf.assert_equal(original_reward, copy_sum)


def test_env_rewards():
    batch_size = 1
    n_nodes = 4
    reward_on_episode = False
    env = TSPMDP(batch_size=batch_size, n_nodes=n_nodes,
                 reward_on_episode=reward_on_episode)
    env.reset()
    state_dict = env.export_states()
    new_coordinates = tf.constant([[[0., 0.],
                                    [0., 1.],
                                    [1., 1.],
                                    [1., 0.]]])
    state_dict["coordinates"] = new_coordinates
    env.import_states(state_dict)
    actions = [1, 2, 3]
    episode_reward = 0
    for action in actions:
        _, reward, _ = env.step(tf.constant([action], dtype=tf.int32))
        episode_reward += reward
    assert episode_reward == -n_nodes

import unittest
import numpy as np
from twenty48stoch import Twenty48stoch

UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3

class TestTwenty48stoch(unittest.TestCase):

    def setUp(self):
        self.env = Twenty48stoch(render_mode='human')

    def test_reset(self):
        # Test if reset function initializes a valid state

        state, _ = self.env.reset()
        self.assertEqual(state.shape, (4 * 4 * 12,))
        self.assertIsNotNone(self.env.state)

    def test_step(self):
        # Test if step function returns valid output

        self.env.reset()
        old_state = np.copy(self.env.state)
        action = 0
        new_state, reward, terminal, _, _ = self.env.step(action)
        self.assertEqual(new_state.shape, (4 * 4 * 12,))
        self.assertNotEqual(np.sum(old_state), np.sum(self.env.state))

    def test_invalid_action(self):
        # Test if step function raises an exception for invalid actions

        self.env.reset()
        invalid_action = 4
        with self.assertRaises(Exception):
            self.env.step(invalid_action)

    def test_move_left(self):
        # Test if the move_left function properly shifts and merges tiles

        row = np.array([1, 0, 1, 0])
        expected = np.array([2, 0, 0, 0])
        self.env._move_left(row)
        self.env._merge_left(row)
        self.env._move_left(row)
        np.testing.assert_array_equal(row, expected)

    def test_move_right(self):
        # Test if the move_right function properly shifts and merges tiles

        row = np.array([1, 0, 1, 0])
        expected = np.array([0, 0, 0, 2])
        self.env._move_right(row)
        self.env._merge_right(row)
        self.env._move_right(row)
        np.testing.assert_array_equal(row, expected)

    def test_move_up(self):
        # Test if the move_up function properly shifts and merges tiles

        self.env.state = np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        expected_state = np.array([
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])

        transposed_state = self.env.state.T
        for row in transposed_state:
            self.env._move_left(row)
            self.env._merge_left(row)
            self.env._move_left(row)

        self.env.state = transposed_state.T
        np.testing.assert_array_equal(self.env.state, expected_state)

    def test_move_down(self):
        # Test if the move_down function properly shifts and merges tiles

        self.env.state = np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        expected_state = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [2, 0, 0, 0]
        ])

        transposed_state = self.env.state.T
        for row in transposed_state:
            self.env._move_right(row)
            self.env._merge_right(row)
            self.env._move_right(row)

        self.env.state = transposed_state.T
        np.testing.assert_array_equal(self.env.state, expected_state)


    def test_terminal(self):
        # Test if terminal condition is True when 2048 is reached
        self.env.state = np.array([[11, 0, 0, 0],
                                   [0, 0, 0, 0],
                                   [0, 0, 0, 0],
                                   [0, 0, 0, 0]])
        self.assertTrue(self.env._is_terminal())

    def test_non_terminal(self):
        # Test if terminal condition is False for a non-terminal state

        self.env.reset()
        self.assertFalse(self.env._is_terminal())

    def test_spawn_tile(self):
        # Test if spawn_tile function spawns a single new tile with value 1 or 2

        self.env.reset()
        self.env.state = np.array([[0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0]])
        self.env.spawn_tile()
        spawned_tiles = np.count_nonzero(self.env.state)
        spawned_value = self.env.state[self.env.state.nonzero()]

        self.assertEqual(spawned_tiles, 1, "Expected only one tile to be spawned")
        self.assertIn(spawned_value, [1, 2], "Expected the spawned tile to be either 1 or 2")
    
    def test_reward_for_merging_tiles(self):
        # Test if rewards are given correctly for merging tiles

        self.env.state = np.array([
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])

        _, reward, _, _, _ = self.env.step(LEFT)
        assert reward == 4, f"Expected reward: 4, got: {reward}"

    def test_reward_for_multiple_merges(self):
        # Test if rewards are given correctly for multiple merges

        self.env.state = np.array([
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])

        _, reward, _, _, _ = self.env.step(LEFT)
        assert reward == 8, f"Expected reward: 8, got: {reward}"

    def test_reward_for_no_change(self):
        # Test if rewards are given correctly for no change in the state

        self.env.state = np.array([
            [2, 3, 4, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])

        _, reward, _, _, _ = self.env.step(LEFT)
        assert reward == -0.1, f"Expected reward: -0.1, got: {reward}"

    def test_reward_for_terminal_state(self):
        # Test if rewards are given correctly for terminal state

        self.env.state = np.array([
            [11, 3, 4, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])

        _, reward, _, _, _ = self.env.step(LEFT)
        assert reward == 0, f"Expected reward: 0, got: {reward}"

    def test_game_over(self):
        # Test if terminal condition is True for a game over state (no moves possible)

        self.env.state = np.array([
            [1, 2, 1, 2],
            [2, 1, 2, 1],
            [1, 2, 1, 2],
            [2, 1, 2, 1]
        ])
        self.assertTrue(self.env._is_terminal())

    def test_not_game_over(self):
        # Test if terminal condition is False when there is a valid move available

        self.env.state = np.array([
            [1, 2, 1, 2],
            [2, 1, 2, 1],
            [1, 2, 1, 2],
            [2, 1, 2, 0]
        ])
        self.assertFalse(self.env._is_terminal())
    
    def test_spawn_tile_after_move(self):
        # Test if a tile is spawned after a move

        self.env.state = np.array([
            [1, 0, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        old_state = np.copy(self.env.state)
        self.env.step(LEFT)
        new_state = self.env.state
        self.assertEqual(np.count_nonzero(old_state), np.count_nonzero(new_state), "Expected one tile to be spawned")

    def test_step_no_action_possible(self):
        # Test if the state remains unchanged when no action is possible

        self.env.state = np.array([
            [1, 2, 1, 2],
            [2, 1, 2, 1],
            [1, 2, 1, 2],
            [2, 1, 2, 1]
        ])
        old_state = np.copy(self.env.state)
        one_hot = self.env.one_hot_encode(old_state)
        old_state = np.array(one_hot, dtype=np.uint8).flatten()
        action = LEFT
        new_state, _, _, _, _ = self.env.step(action)
        np.testing.assert_array_equal(old_state, new_state, "Expected no change in state")

    def test_step_terminal_values(self):
        # Test if step function returns correct terminal values

        self.env.state = np.array([
            [11, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        _, _, terminal, _, _ = self.env.step(LEFT)
        self.assertTrue(terminal, "Expected terminal to be True")

        self.env.reset()
        _, _, terminal, _, _ = self.env.step(LEFT)
        self.assertFalse(terminal, "Expected terminal to be False")

    def test_merge_left_points(self):
        # Test if merge_left function returns correct points

        row = np.array([1, 1, 1, 0])
        expected_points = 4
        points = self.env._merge_left(row)
        self.assertEqual(points, expected_points)

    def test_merge_right_points(self):
        # Test if merge_right function returns correct points

        row = np.array([0, 1, 1, 1])
        expected_points = 4
        points = self.env._merge_right(row)
        self.assertEqual(points, expected_points)

    def test_reset_clears_terminal(self):
        # Test if reset function clears the terminal state

        self.env.state = np.array([
            [11, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])
        self.env.reset()
        self.assertFalse(self.env._is_terminal())

if __name__ == '__main__':
    unittest.main()
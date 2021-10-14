import random
import numpy as np
from itertools import product
import multiprocessing as mp
import os
import cv2
import pybullet as pb
from pybullet_utils import bullet_client
import pybullet_data
from time import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir_out', default='test/', type=str, help="Where experiments should be saved")
parser.add_argument('--seed', default=1, type=int, help="Random seed")
parser.add_argument('--n_balls', default=3, type=int, help="# of balls in the scene")
parser.add_argument('--n_examples', default=10, type=int, help="# of experiments to generate")
args = parser.parse_args()

COLORS = ['red', 'green', 'blue', 'yellow'][:args.n_balls]

W, H = 112, 112  # Image shape
RANGE_POS = 3
RANGE_SPEED = 3
EPSILON = 100  # Threshold for constraints


def check_bayes(alt_ab, alt_cd, ab, cd):
    """ Check the identifiability contraint
    :param alt_ab: list of alternative trajectories from AB
    :param alt_cd: list of alternative trajectories from CD
    :param ab: AB candidate
    :param cd: CD candidate
    :return: True if experiment is identifiable
    """
    for i in range(len(alt_ab)):
        if alt_ab[i] == ab:
            if alt_cd[i] != cd:
                return False
    return True


def check_counterfactual(alt_cd, cd, mass_permutation):
    """ Check the counterfactuality constraint
    :param alt_cd: list of alternative trajectories from CD
    :param cd: CD candidate
    :param mass_permutation: List of every mass permutation
    :return: List of counterfactual objects. Experiment is cf if len()>0
    """
    counterfactual_cubes = []
    for k in range(cd.n_balls):
        alter_cf = cd.confounders.copy()
        alter_cf[k] = 1 if alter_cf[k] == 10 else 10
        alt_trajectory = alt_cd[mass_permutation.index(alter_cf)]
        if alt_trajectory != cd:
            counterfactual_cubes.append(k)
    return counterfactual_cubes


class Generator:
    def __init__(self, dir_out, seed, n_balls, nb_examples):
        """
        Class that oversees the experiment generation
        :param dir_out: Where experiments should be saved
        :param seed: Random seed
        :param n_balls: # of balls in the scene
        :param nb_examples: # of experiments to generate
        """
        self.dir_out = dir_out
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)

        self.mass_permutation = [list(combo) for combo in product([1, 10], repeat=args.n_balls)]

        self.logs_cf = {str(d): 0 for d in self.mass_permutation}  # Usefull to ensure balance in the dataset
        self.n_balls = n_balls

        # LOGS variables
        self.list_time = []
        self.nb_examples = nb_examples

        self.total_trial_counter = 0
        self.ab_trial_counter = 0
        self.cd_trial_counter = 0

    def generate(self):
        """Generate the experiments"""
        nb_ex = 0
        # Choose colors, masses configuration and if we apply a remove do-operation
        do_remove_op, index_cf, colors = self.get_configuration_example()
        t = time()
        while nb_ex < self.nb_examples:

            # Step 1 : find a valid AB
            self.total_trial_counter += 1
            ab = self.find_valid_AB(self.mass_permutation[index_cf])
            ab = self.simulate_one(ab, colors)

            # Step 2 : find a valid CD
            do_op, cf_cubes, cd = self.find_valid_CD(ab, do_remove_op, colors, index_cf)

            if cd is not None:  # If a valid CD has been found
                self.list_time.append(time() - t)
                self.logs_cf[str(self.mass_permutation[index_cf])] += 1  # Update the logs for dataset balance
                nb_ex += 1
                ab, cd = self.simulate_final(ab, cd, colors)  # Simulate AB and CD with rendering
                self.save(ab, cd, colors, do_op, cf_cubes, nb_ex)  # Save the experiment
                t = time()
                # Choose new configuration
                do_remove_op, index_cf, colors = self.get_configuration_example()

    def get_configuration_example(self):
        """Sample a do-operation, colors and masses. Try to ensure balance in the masses distribution"""
        do_remove_op = random.random() < 0.3  # 30% of chance of being a remove operation

        # Search for the masses with the less representation in previous experiments
        cf = min(self.mass_permutation, key=lambda x: self.logs_cf[str(x)])
        index_cf = self.mass_permutation.index(cf)

        # Randomly sample colors
        colors = random.sample(COLORS, args.n_balls)
        return do_remove_op, index_cf, colors

    def find_valid_AB(self, masse):
        """No constraint on A, simply return a random candidate"""
        self.ab_trial_counter += 1
        candidate = Arena(self.n_balls, masse)
        return candidate

    def find_valid_CD(self, ab, do_remove_op, colors, index_cf, ):
        """
        Search for a valid CD trajectory.
        :param ab: AB candidate
        :param do_remove_op: Bool, True if the do-op should be a remove op
        :param colors: Colors list
        :param index_cf: index of the masse configuration in self.mass_configuration
        :return: the do-operation parameters, list of counterfactual objects, CD candidate
        """
        found_cd = False
        n_trials = 0
        while found_cd is False and n_trials < 10:  # Try 10 different do-op, else quit
            self.cd_trial_counter += 1
            if do_remove_op:
                n_trials = 10
                cd = ab.remove_ball()  # Remove a (random) ball
                do_op = {"operation": "remove", "amplitude": 0, "cube": -1}
            else:
                do_op, cd = ab.generate_random_do_operation()  # Generate a random do-op
            if cd != []:  # do-op sampling may failed, in this case, try again
                if do_op['operation'] is not None:
                    # Simulate all alternative traj. from CD
                    alt_cd = self.simulate_all(cd, colors)
                    cd.trajectory = alt_cd[index_cf].trajectory.copy()
                    # Check counterfactuality constraint
                    counterfactual_cubes = check_counterfactual(alt_cd, cd, self.mass_permutation)
                    if len(counterfactual_cubes) > 0:
                        # Simulate all alternative traj. from AB
                        alt_ab = self.simulate_all(ab, colors)
                        if check_bayes(alt_ab, alt_cd, ab, cd):  # Check identifiability constraint
                            found_cd = True
            n_trials += 1
        if found_cd:
            return do_op, counterfactual_cubes, cd
        else:
            return None, None, None

    def simulate_all(self, tower, colors):
        """
        Simulate every outcomes with every mass configuration for a given initial condition
        :param tower: initial condition
        :param colors: list of object colors
        :return: list of outcomes for each mass configuration
        """
        towers = [tower.clone(m) for m in self.mass_permutation]
        childPipes, parentPipes = [], []
        processes = []

        # Simulation are multiprocess, to go faster
        for pr in range(len(towers)):  # Create pipes to get the simulation
            parentPipe, childPipe = mp.Pipe()
            parentPipes.append(parentPipe)
            childPipes.append(childPipe)

        for rank in range(len(towers)):  # Run the processes
            simulator = Simulator(25, 6, 0, W, H)  # Simulate at 25 FPS, for 6 second, no substeps.
            p = mp.Process(target=simulator.run, args=(childPipes[rank], towers[rank], 0, colors, False))
            p.start()
            processes.append(p)

        for rank, p in enumerate(processes):  # Get the simulation
            state, _, _, _ = parentPipes[rank].recv()
            towers[rank].trajectory = state
            p.join()

        return towers

    def simulate_one(self, arena, colors):
        """
        Simulate a single trajectory without rendering
        :param arena: initial condition
        :param colors: list of colors
        :return: outcome
        """
        parentPipe, childPipe = mp.Pipe()
        simulator = Simulator(25, 6, 0, W, H)
        p = mp.Process(target=simulator.run, args=(childPipe, arena, 0, colors, False))
        p.start()
        state, _, _, _ = parentPipe.recv()
        arena.trajectory = state
        p.join()

        return arena

    def simulate_final(self, ab, cd, colors):
        """
        Simulate with rendering
        :param ab: AB candidate
        :param cd: CD candidate
        :param colors: colors list
        :return: simulated trajectories
        """

        childPipes, parentPipes = [], []
        for pr in range(2):  # Create pipes to get the simulation
            parentPipe, childPipe = mp.Pipe()
            parentPipes.append(parentPipe)
            childPipes.append(childPipe)

        simulator = Simulator(25, 6, 0, W, H)
        plane_id = random.randint(0, 3)
        p_ab = mp.Process(target=simulator.run, args=(childPipes[0], ab, plane_id, colors, True))
        p_cd = mp.Process(target=simulator.run, args=(childPipes[1], cd, plane_id, colors, True))

        p_ab.start()
        p_cd.start()

        # Get results for AB and CD
        ab.trajectory, ab.rgb, ab.depth, ab.seg = parentPipes[0].recv()
        cd.trajectory, cd.rgb, cd.depth, cd.seg = parentPipes[1].recv()

        p_ab.join()
        p_cd.join()

        return ab, cd

    def save(self, ab, cd, colors, do_op, cf_cubes, n):
        """
        Save the experiment
        :param ab: AB candidate
        :param cd: CD candidate
        :param colors: colors list
        :param do_op: do-operation parameters
        :param cf_cubes: list of counterfactual cubes
        :param n: index of this experiments
        :return:
        """

        assert ab.confounders == cd.confounders
        assert len(cf_cubes) > 0

        # Create the paths
        out_dir = self.dir_out + str(self.seed) + "_" + str(n) + "/"
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(out_dir + 'ab', exist_ok=True)
        os.makedirs(out_dir + 'cd', exist_ok=True)

        # Add a column of zero if the do-operation is a remove operation
        cd.fill_trajectory(ab.n_balls)

        # Save ground truth trajectory as numpy array + save confounders
        np.save(out_dir + "ab/states.npy", ab.trajectory)
        np.save(out_dir + "cd/states.npy", cd.trajectory)
        np.save(out_dir + "confounders.npy", ab.confounders)

        # Write do-op parameter in a file
        with open(out_dir + "do_op.txt", 'w') as f:
            if do_op["operation"] == "remove":
                f.write(f"Remove the {colors[-1]} cube")
            else:
                f.write(
                    f"Move the {colors[do_op['cube']]} cube of {do_op['amplitude']} in the {do_op['operation']} direction")

        # Write colors in a file
        with open(out_dir + "COLORS.txt", 'w') as f:
            f.write(str(colors))

        # Write list of cf cubes in a file
        with open(out_dir + "cd/counterfactual_balls.txt", 'w') as f:
            f.write("Cubes that strongly depend on their masses\n")
            f.write('\n'.join([f"idx:{i}, colors={colors[i]}" for i in cf_cubes]))

        # SAVE RGB
        writer = cv2.VideoWriter(out_dir + 'ab/rgb.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (W, H))
        for rgb in ab.rgb:
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        writer.release()
        writer = cv2.VideoWriter(out_dir + 'cd/rgb.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (W, H))
        for rgb in cd.rgb:
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        writer.release()

        # SAVE DEPTH
        writer = cv2.VideoWriter(out_dir + 'ab/depth.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (W, H))
        for rgb in ab.depth:
            rgb = np.round(rgb * 255)
            writer.write(cv2.cvtColor(rgb.astype(np.uint8).reshape((W, H, 1)), cv2.COLOR_GRAY2BGR))
        writer.release()
        writer = cv2.VideoWriter(out_dir + 'cd/depth.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (W, H))
        for rgb in cd.depth:
            rgb = np.round(rgb * 255)
            writer.write(cv2.cvtColor(rgb.astype(np.uint8).reshape((W, H, 1)), cv2.COLOR_GRAY2BGR))
        writer.release()

        # SAVE SEGMENTATION
        writer = cv2.VideoWriter(out_dir + 'ab/segmentation.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (W, H))
        for rgb in ab.seg:
            writer.write(cv2.cvtColor(rgb.astype(np.uint8).reshape((W, H, 1)), cv2.COLOR_GRAY2BGR))
        writer.release()
        writer = cv2.VideoWriter(out_dir + 'cd/segmentation.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (W, H))
        for rgb in cd.seg:
            writer.write(cv2.cvtColor(rgb.astype(np.uint8).reshape((W, H, 1)), cv2.COLOR_GRAY2BGR))
        writer.release()

        # Write some logs
        with open("logs_create_dataset_" + str(self.seed) + ".txt", "a") as f:
            f.write(
                f"{n}/{self.nb_examples} in {self.total_trial_counter} trial ({self.ab_trial_counter} on AB, {self.cd_trial_counter} on CD), took {round(self.list_time[-1], 1)} seconds (Average {round(np.mean(self.list_time), 2)})\n")

        self.total_trial_counter = 0
        self.ab_trial_counter = 0
        self.cd_trial_counter = 0


class Arena:
    def __init__(self, n_balls, confounders):
        """Class that model a trajectory"""

        self.start_position = []
        self.start_speed = []
        self.n_balls = n_balls

        self.confounders = confounders
        self.trajectory = None
        self.rgb = None
        self.depth = None
        self.seg = None

        self.init()

    def init(self):
        """Generate random initial condition"""
        for _ in range(self.n_balls):
            no_overlap = False  # Make sure that there is no overlap between balls
            while no_overlap is False:
                no_overlap = True
                cand_pose = RANGE_POS * (2 * np.random.random(2) - 1)  # Generate random position for a ball
                for balls in self.start_position:
                    if np.sqrt(((balls - cand_pose) ** 2).sum()) < 1:
                        no_overlap = False

            self.start_position.append(cand_pose)
            self.start_speed.append(RANGE_SPEED * (2 * np.random.random(2) - 1))  # Random speed

    def compute_valid_movement_range(self, ball_idx):
        """For do-op sampling : return maximal displacement in each direction
        without getting out of the limits"""
        x, y = self.start_position[ball_idx]
        delta_x = np.array([-3.5 + x, 3.5 - x])
        delta_y = np.array([-3.5 + y, 3.5 - y])
        return delta_x, delta_y

    def remove_ball(self):
        """Remove a random ball from the scene"""
        new_arena = Arena(self.n_balls - 1, self.confounders.copy())
        new_arena.start_position = [n.copy() for n in self.start_position[:-1]]
        new_arena.start_speed = [n.copy() for n in self.start_speed[:-1]]
        return new_arena

    def clone(self, cf=None):
        """Clone an arena"""
        if cf is None:
            new_arena = Arena(self.n_balls, self.confounders.copy())
        else:
            new_arena = Arena(self.n_balls, cf)
        new_arena.start_position = [n.copy() for n in self.start_position]
        new_arena.start_speed = [n.copy() for n in self.start_speed]
        return new_arena

    def generate_random_do_operation(self):
        """Return a do-operation candidate"""
        ball_idx = random.randint(0, self.n_balls - 1)  # Select random ball for do-op
        delta_x, delta_y = self.compute_valid_movement_range(ball_idx)

        operation = random.choice(['x', 'y'])  # Select a direction
        delta = delta_x if operation == "x" else delta_y
        delta = delta * 0.9

        # List of possible do-op (bounded in absolute value)
        epsilons = list(np.arange(0.5, delta[1], 0.1)) + list(np.arange(delta[0], -0.5, 0.1))

        if len(epsilons) == 0:  # If there is no possible movement in this direction, quit...
            return {"operation": None, "cube": None, "amplitude": None}, []

        # Sample a displacement amplitude
        amplitude = random.choices(epsilons, k=1)[0]
        cd = self.clone()

        # Apply do-op
        cd.start_position[ball_idx][0 if operation == "x" else 1] += amplitude

        # Check for overlap
        for i in cd.start_position:
            for j in cd.start_position:
                if (i != j).all() and np.sqrt(((i - j) ** 2).sum()) < 1:
                    return {"operation": None, "cube": None, "amplitude": None}, []
        return {"operation": operation, "amplitude": amplitude, "cube": ball_idx}, cd

    def fill_trajectory(self, n_balls):
        """Make sure that states are of good shape"""
        T, K, S = self.trajectory.shape
        if K != n_balls:
            self.trajectory = np.concatenate([self.trajectory, np.zeros((T, 1, S))], axis=1)

    def __eq__(self, other):
        """Check if two trajectories are equal or not"""
        if other == []:
            return False
        error = np.zeros(self.n_balls)

        # Compute MSE on 3D position per object (the measure is independant from the number of object)
        for k in range(other.trajectory.shape[1]):
            error[k] = np.sqrt(((self.trajectory[:, k, :2] - other.trajectory[:, k, :2]) ** 2).sum(-1)).sum(0)
        # If 1 object MSE is above threshold, trajectories are different.
        return (error > EPSILON).sum() == 0


class Simulator:
    def __init__(self, fps, time_duration, num_substeps=1000, W=448, H=448):
        """
        Class that model the physics simulator
        :param fps: frame per second
        :param time_duration: simulation time length
        :param num_substeps: substeps for simulation accuracy
        :param W: Width of image
        :param H: Height of image
        """
        self.fixed_timestep = 1 / fps
        self.nb_steps = time_duration * fps
        self.num_substeps = num_substeps

        self.p = None
        self.W, self.H = W, H

    def run(self, pipe, arena, plane_id, colors, rendering=False):
        """
        Run the simulator
        :param pipe: multiprocess pipe to output the results
        :param arena: initial condition
        :param plane_id: id of the place for the ground
        :param colors: colors list
        :param rendering: activate or not the rendering
        :return: None
        """

        # Initialize the simulator
        self.p = bullet_client.BulletClient(pb.DIRECT)
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.p.setGravity(0, 0, -10)
        self.p.setPhysicsEngineParameter(fixedTimeStep=self.fixed_timestep, numSolverIterations=10000,
                                         solverResidualThreshold=1e-10,
                                         numSubSteps=self.num_substeps)

        # Init the environnement
        list_cube = self._init(arena, colors, plane_id)

        # Logs
        seq_states = np.zeros((self.nb_steps, arena.n_balls, 3 + 4 + 3 + 3))
        list_rgb = []
        list_depth = []
        list_seg = []

        # Simulate
        for t in range(self.nb_steps):
            for i, cube in enumerate(list_cube):
                pos, angle = self.p.getBasePositionAndOrientation(cube)
                vel_pose, vel_angle = self.p.getBaseVelocity(cube)
                seq_states[t, i] = list(pos) + list(angle) + list(vel_pose) + list(vel_angle)

            if rendering:
                img_arr = self.get_rendering()
                rgb = img_arr[2][:, :, :3]
                list_depth.append(img_arr[3])
                list_seg.append(img_arr[4])
                list_rgb.append(rgb)

            self.p.stepSimulation()

        pipe.send((seq_states, list_rgb, list_depth, list_seg))
        pipe.close()

    def _init(self, arena, colors, plane_id):
        """
        Init the scene with corresponding objects
        :param arena: initial condition
        :param colors: colors list
        :param plane_id: index of the ground texture
        :return:
        """

        # Load ground
        pb.loadURDF(f"../data_generation/urdf/plane_{plane_id}/plane.urdf", useMaximalCoordinates=True)

        # Walls
        limit = 4.3
        angle = (np.pi / 2.)
        pb.loadURDF(f"../data_generation/urdf/plane_white/plane.urdf", [0, limit, 0],
                    pb.getQuaternionFromEuler([angle, 0, 0]), useMaximalCoordinates=True)
        pb.loadURDF(f"../data_generation/urdf/plane_white/plane.urdf", [0, -limit, 0],
                    pb.getQuaternionFromEuler([-angle, 0, 0]), useMaximalCoordinates=True)
        pb.loadURDF(f"../data_generation/urdf/plane_white/plane.urdf", [limit, 0, 0],
                    pb.getQuaternionFromEuler([0, -angle, 0]), useMaximalCoordinates=True)
        pb.loadURDF(f"../data_generation/urdf/plane_white/plane.urdf", [-limit, 0, 0],
                    pb.getQuaternionFromEuler([0, angle, 0]), useMaximalCoordinates=True)

        # Add balls
        list_balls = []
        for i in range(arena.n_balls):
            color = colors[i]
            x, y = arena.start_position[i]
            cube = self.p.loadURDF(f"../data_generation/urdf/{color}/ball.urdf",
                                   [x, y, 0.5],
                                   useMaximalCoordinates=True)
            pb.changeDynamics(cube, -1,
                              mass=arena.confounders[i],
                              lateralFriction=0,
                              restitution=1)  # Change physical parameters

            vx, vy = arena.start_speed[i]
            pb.resetBaseVelocity(cube, [vx, vy, 0])  # Change initial speed
            list_balls.append(cube)
        return list_balls

    def get_rendering(self):
        """ Rendering of the environment """
        viewMatrix = pb.computeViewMatrix([0, 0.01, 8], [0, 0, 0], [0, 0, 1])
        projectionMatrix = pb.computeProjectionMatrixFOV(60, self.W / self.H, 4, 20)
        img_arr = pb.getCameraImage(self.W, self.H, viewMatrix, projectionMatrix,
                                    shadow=0,
                                    lightDirection=[1, 1, 1],
                                    renderer=pb.ER_BULLET_HARDWARE_OPENGL)
        return img_arr


if __name__ == '__main__':
    g = Generator(dir_out=args.dir_out, seed=args.seed, n_balls=args.n_balls, nb_examples=args.n_examples)
    g.generate()

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
import matplotlib.pyplot as plt
import argparse

COLORS = ['red', 'green', 'blue', 'yellow']

parser = argparse.ArgumentParser()
parser.add_argument('--dir_out', default='test/', type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--n_examples', default=10, type=int)
args = parser.parse_args()

W, H = 448, 448
RANGE_POS = 3
RANGE_SPEED = 10
EPSILON = 100


def check_bayes(alt_ab, alt_cd, ab, cd):
    for i in range(len(alt_ab)):
        if alt_ab[i] == ab:
            if alt_cd[i] != cd:
                return False
    return True


def check_counterfactual(alt_cd, cd, mass_permutation):
    counterfactual_cubes = []
    for k in range(2):
        alter_cf = cd.confounders.copy()
        alter_cf[k] = 1 if alter_cf[k] == 10 else 10
        alt_trajectory = alt_cd[mass_permutation.index(alter_cf)]
        if alt_trajectory != cd:
            counterfactual_cubes.append(k)
    return counterfactual_cubes


class Generator:
    def __init__(self, dir_out, seed, nb_examples):
        self.dir_out = dir_out
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)

        self.mass_permutation = [list(combo) for combo in product([1, 10], repeat=2)]
        self.logs_cf = {str(d): 0 for d in self.mass_permutation}
        self.logs_moving = {"ball": 0, "cylinder": 0}
        self.logs_cylinder_orientation = {'up': 0, 'down': 0}

        self.list_time = []
        self.nb_examples = nb_examples

        self.total_trial_counter = 0
        self.ab_trial_counter = 0
        self.cd_trial_counter = 0

    def generate(self):
        nb_ex = 0
        cylinder_orientation, moving_object, index_cf, colors = self.get_configuration_example()

        t = time()
        while nb_ex < self.nb_examples:

            self.total_trial_counter += 1
            ab = self.find_valid_AB(self.mass_permutation[index_cf], moving_object, cylinder_orientation)
            ab = self.simulate_one(ab, colors)
            do_op, cf_cubes, cd = self.find_valid_CD(ab, colors, index_cf)

            if cd is not None:
                self.list_time.append(time() - t)
                self.logs_cf[str(self.mass_permutation[index_cf])] += 1
                self.logs_moving[moving_object] += 1
                self.logs_cylinder_orientation[cylinder_orientation] += 1
                cylinder_orientation, moving_object, index_cf, colors = self.get_configuration_example()
                nb_ex += 1

                ab, cd = self.simulate_final(ab, cd, colors)
                self.save(ab, cd, colors, do_op, cf_cubes, nb_ex)
                t = time()

    def get_configuration_example(self):
        cf = min(self.mass_permutation, key=lambda x: self.logs_cf[str(x)])
        moving_object = min(["ball", "cylinder"], key=lambda x: self.logs_moving[x])
        cylinder_orientation = min(["up", "down"], key=lambda x: self.logs_cylinder_orientation[x])
        index_cf = self.mass_permutation.index(cf)
        colors = random.sample(COLORS, 2)
        return cylinder_orientation, moving_object, index_cf, colors

    def find_valid_AB(self, masse, moving_object, cylinder_orientation):
        self.ab_trial_counter += 1
        candidate = Arena(masse, moving_object, cylinder_orientation)
        return candidate

    def find_valid_CD(self, ab, colors, index_cf, ):
        found_cd = False
        n_trials = 0
        while found_cd is False and n_trials < 10:
            self.cd_trial_counter += 1
            do_op, cd = ab.generate_random_do_operation()
            if cd != []:
                if do_op['operation'] is not None:
                    alt_cd = self.simulate_all(cd, colors)
                    cd.trajectory = alt_cd[index_cf].trajectory.copy()
                    counterfactual_cubes = check_counterfactual(alt_cd, cd, self.mass_permutation)
                    if len(counterfactual_cubes) > 0:
                        alt_ab = self.simulate_all(ab, colors)
                        if check_bayes(alt_ab, alt_cd, ab, cd):
                            found_cd = True
                        else:
                            print("Bayes Error non zero")
                    else:
                        print("Not CF")
                else:
                    print("Do op is None")
            n_trials += 1
        if found_cd:
            return do_op, counterfactual_cubes, cd
        else:
            return None, None, None

    def simulate_all(self, tower, colors):
        towers = [tower.clone(m) for m in self.mass_permutation]
        childPipes, parentPipes = [], []
        processes = []

        for pr in range(len(towers)):  # Create pipes to get the simulation
            parentPipe, childPipe = mp.Pipe()
            parentPipes.append(parentPipe)
            childPipes.append(childPipe)

        for rank in range(len(towers)):  # Run the processes$
            simulator = Simulator(25, 3, W=448, H=448)
            p = mp.Process(target=simulator.run, args=(childPipes[rank], towers[rank], 0, colors, False))
            p.start()
            processes.append(p)

        for rank, p in enumerate(processes):  # Get the simulation
            state, _ = parentPipes[rank].recv()
            towers[rank].trajectory = state
            p.join()

        return towers

    def simulate_one(self, arena, colors):
        parentPipe, childPipe = mp.Pipe()
        simulator = Simulator(25, 3, W=448, H=448)
        p = mp.Process(target=simulator.run, args=(childPipe, arena, 0, colors, False))
        p.start()
        state, _ = parentPipe.recv()
        arena.trajectory = state
        p.join()

        return arena

    def simulate_final(self, ab, cd, colors):
        childPipes, parentPipes = [], []
        for pr in range(2):  # Create pipes to get the simulation
            parentPipe, childPipe = mp.Pipe()
            parentPipes.append(parentPipe)
            childPipes.append(childPipe)

        simulator = Simulator(25, 3, W=W, H=H)
        plane_id = random.randint(0, 3)
        p_ab = mp.Process(target=simulator.run, args=(childPipes[0], ab, plane_id, colors, True))
        p_cd = mp.Process(target=simulator.run, args=(childPipes[1], cd, plane_id, colors, True))

        p_ab.start()
        p_cd.start()

        ab.trajectory, ab.rgb = parentPipes[0].recv()
        cd.trajectory, cd.rgb = parentPipes[1].recv()

        p_ab.join()
        p_cd.join()

        return ab, cd

    def save(self, ab, cd, colors, do_op, cf_cubes, n):
        assert ab.confounders == cd.confounders
        assert len(cf_cubes) > 0

        out_dir = self.dir_out + str(self.seed) + "_" + str(n) + "/"
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(out_dir + 'ab', exist_ok=True)
        os.makedirs(out_dir + 'cd', exist_ok=True)

        np.save(out_dir + "ab/states.npy", ab.trajectory)
        np.save(out_dir + "cd/states.npy", cd.trajectory)
        np.save(out_dir + "confounders.npy", ab.confounders)

        with open(out_dir + "do_op.txt", 'w') as f:
            f.write(f"Moving {ab.moving_object}\n")
            if do_op["operation"] == "cylinder_rotation":
                f.write(f"Rotate cylinder from {ab.cylinder_orientation} to {cd.cylinder_orientation}")
            else:
                f.write(
                    f"Move the {do_op['object']} of {do_op['amplitude']} in the {do_op['operation']} direction")

        with open(out_dir + "COLORS.txt", 'w') as f:
            f.write(str(colors))

        writer = cv2.VideoWriter(out_dir + 'ab/rgb.mp4',
                                 cv2.VideoWriter_fourcc(*'mp4v'),
                                 25,
                                 (448, 448)
                                 )
        for rgb in ab.rgb:
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        writer.release()

        writer = cv2.VideoWriter(out_dir + 'cd/rgb.mp4',
                                 cv2.VideoWriter_fourcc(*'mp4v'),
                                 25,
                                 (448, 448)
                                 )
        for rgb in cd.rgb:
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        writer.release()

        with open("logs_collision_" + str(self.seed) + ".txt", "a") as f:
            f.write(
                f"{n}/{self.nb_examples} in {self.total_trial_counter} trial ({self.ab_trial_counter} on AB, {self.cd_trial_counter} on CD), took {round(self.list_time[-1], 1)} seconds (Average {round(np.mean(self.list_time), 2)})\n")

        self.total_trial_counter = 0
        self.ab_trial_counter = 0
        self.cd_trial_counter = 0


class Arena:
    def __init__(self, confounders, moving_object, cylinder_orientation):
        self.start_position_cylinder = []
        self.start_position_ball = []

        self.start_speed_cylinder = []
        self.start_speed_ball = []

        self.cylinder_angle = []

        self.confounders = confounders
        self.trajectory = None
        self.rgb = None

        self.moving_object = moving_object
        self.cylinder_orientation = cylinder_orientation
        self.init()

    def init(self):
        self.cylinder_angle = 360 * np.random.random(3)
        if self.moving_object == "ball":
            self.start_speed_cylinder = np.zeros(3)
            if self.cylinder_orientation == "up":
                self.start_position_cylinder = [0, 0, 1.501]
            else:
                self.start_position_cylinder = [0, 0, 0.501]

            movement_range = list(np.linspace(-RANGE_POS, -1, 10)) + list(np.linspace(1, RANGE_POS, 10))
            x, y = random.choices(movement_range, k=2)
            z = 1 + random.random() * 2
            self.start_position_ball = np.array([x, y, z])
            self.start_speed_ball = -self.start_position_ball / np.linalg.norm(self.start_position_ball) * RANGE_SPEED
        elif self.moving_object == "cylinder":
            self.start_speed_ball = np.zeros(3)
            self.start_position_ball = np.array([0, 0, 0.5])
            movement_range = list(np.linspace(-RANGE_POS, -1, 10)) + list(np.linspace(1, RANGE_POS, 10))
            x, y = random.choices(movement_range, k=2)
            z = 1.5 + random.random() * 2
            self.start_position_cylinder = np.array([x, y, z])
            self.start_speed_cylinder = -self.start_position_cylinder / np.linalg.norm(
                self.start_position_cylinder) * RANGE_SPEED

    def rotate_cylinder(self):
        assert self.moving_object == "ball"
        self.cylinder_orientation = "up" if self.cylinder_orientation == "down" else "down"
        if self.cylinder_orientation == "up":
            self.start_position_cylinder = [0, 0, 1.501]
        else:
            self.start_position_cylinder = [0, 0, 0.501]

    def clone(self, cf=None):
        if cf is None:
            new_arena = Arena(self.confounders.copy(), self.moving_object, self.cylinder_orientation)
        else:
            new_arena = Arena(cf, self.moving_object, self.cylinder_orientation)
        new_arena.start_position_cylinder = self.start_position_cylinder.copy()
        new_arena.start_position_ball = self.start_position_ball.copy()

        new_arena.start_speed_cylinder = self.start_speed_cylinder.copy()
        new_arena.start_speed_ball = self.start_speed_ball.copy()

        new_arena.cylinder_angle = self.cylinder_angle.copy()

        return new_arena

    def generate_random_do_operation(self):
        if self.moving_object == "ball" and random.random() < 0.3:
            cd = self.clone()
            cd.rotate_cylinder()
            return {"operation": "cylinder_rotation", "amplitude": 0, "object": "cylinder"}, cd

        operation = random.choice(['x', 'y', 'z'])
        amplitude = 2 * random.random() - 1
        while abs(amplitude) < 0.1:
            amplitude = 2 * random.random() - 1
        cd = self.clone()
        if self.moving_object == "ball":
            if operation == "x":
                cd.start_position_ball[0] += amplitude
            elif operation == "y":
                cd.start_position_ball[1] += amplitude
            else:
                cd.start_position_ball[2] += amplitude
            start = cd.start_position_ball
            if -1 < start[0] < 1 or -1 < start[1] < 1 or start[2] < 1:
                return {"operation": None, "object": "sphere", "amplitude": None}, []

        elif self.moving_object == "cylinder":
            if operation == "x":
                cd.start_position_cylinder[0] += amplitude
            elif operation == "y":
                cd.start_position_cylinder[1] += amplitude
            else:
                cd.start_position_cylinder[2] += amplitude
            start = cd.start_position_cylinder
            if -1 < start[0] < 1 or -1 < start[1] < 1 or start[2] < 1.5:
                return {"operation": None, "object": "cylinder", "amplitude": None}, []

        return {"operation": operation, "amplitude": amplitude, "object": cd.moving_object}, cd

    def fill_trajectory(self, n_balls):
        T, K, S = self.trajectory.shape
        if K != n_balls:
            self.trajectory = np.concatenate([self.trajectory, np.zeros((T, 1, S))], axis=1)

    def __eq__(self, other):
        if other == []:
            return False
        error = np.zeros(2)

        for k in range(other.trajectory.shape[1]):
            error[k] = np.sqrt(((self.trajectory[:, k, :2] - other.trajectory[:, k, :2]) ** 2).sum(-1)).sum(0)
        return (error > EPSILON).sum() == 0


class Simulator:
    def __init__(self, fps, time_duration, num_substeps=1000, W=448, H=448):
        self.fixed_timestep = 1 / fps
        self.nb_steps = time_duration * fps
        self.num_substeps = num_substeps

        self.p = None
        self.W, self.H = W, H

    def run(self, pipe, arena, plane_id, colors, rendering=False):
        self.p = bullet_client.BulletClient(pb.DIRECT)
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.p.setGravity(0, 0, -10)
        self.p.setPhysicsEngineParameter(fixedTimeStep=self.fixed_timestep, numSolverIterations=10000,
                                         solverResidualThreshold=1e-10,
                                         numSubSteps=self.num_substeps)

        list_cube = self._init(arena, colors, plane_id)
        seq_states = np.zeros((self.nb_steps, 2, 3 + 4 + 3 + 3))
        list_rgb = []

        for t in range(self.nb_steps):
            for i, cube in enumerate(list_cube):
                pos, angle = self.p.getBasePositionAndOrientation(cube)
                vel_pose, vel_angle = self.p.getBaseVelocity(cube)
                seq_states[t, i] = list(pos) + list(angle) + list(vel_pose) + list(vel_angle)

            if rendering:
                img_arr = self.get_rendering()
                rgb = img_arr[2][:, :, :3]
                list_rgb.append(rgb)

            self.p.stepSimulation()
        pipe.send((seq_states, list_rgb))
        pipe.close()

    def _init(self, arena, colors, plane_id):
        pb.loadURDF(f"../data_generation/urdf/plane_{plane_id}/plane.urdf", useMaximalCoordinates=True)

        list_objects = []
        if arena.moving_object == "ball":
            if arena.cylinder_orientation == "up":
                orientation = pb.getQuaternionFromEuler([np.pi / 2, 0, arena.cylinder_angle[2]])
            else:
                orientation = pb.getQuaternionFromEuler([0, 0, arena.cylinder_angle[2]])
        else:
            orientation = pb.getQuaternionFromEuler(arena.cylinder_angle)
        cylinder = self.p.loadURDF(f"../data_generation/urdf/{colors[0]}/cylinder.urdf", arena.start_position_cylinder,
                                   orientation, useMaximalCoordinates=True)
        pb.changeDynamics(cylinder, -1,
                          mass=arena.confounders[0],
                          lateralFriction=0.5,
                          restitution=1)
        vx, vy, vz = arena.start_speed_cylinder
        pb.resetBaseVelocity(cylinder, [vx, vy, vz])

        ball = self.p.loadURDF(f"../data_generation/urdf/{colors[1]}/ball.urdf", arena.start_position_ball,
                               useMaximalCoordinates=True)
        pb.changeDynamics(ball, -1,
                          mass=arena.confounders[1],
                          lateralFriction=0.5,
                          restitution=1)
        vx, vy, vz = arena.start_speed_ball
        pb.resetBaseVelocity(ball, [vx, vy, vz])
        list_objects.append(cylinder)
        list_objects.append(ball)
        return list_objects

    def get_rendering(self):
        """ Rendering of the environment """
        viewMatrix = pb.computeViewMatrix([0, -7, 4.5], [0, 0, 1.5], [0, 0, 1])
        projectionMatrix = pb.computeProjectionMatrixFOV(60, self.W / self.H, 4, 20)
        img_arr = pb.getCameraImage(self.W, self.H, viewMatrix, projectionMatrix,
                                    shadow=0,
                                    lightDirection=[1, 1, 1],
                                    renderer=pb.ER_BULLET_HARDWARE_OPENGL)
        return img_arr


if __name__ == '__main__':
    g = Generator(dir_out=args.dir_out, seed=args.seed, nb_examples=args.n_examples)
    g.generate()

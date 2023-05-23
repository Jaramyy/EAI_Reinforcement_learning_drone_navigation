from omniisaacgymenvs.tasks.base.rl_task import RLTask

from typing import Optional
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView

from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.objects import DynamicSphere

from skrl.utils import omniverse_isaacgym_utils




import carb
import numpy as np
import torch


TASK_CFG = {"test": False,
            "device_id": 0,
            "headless": True,
            "sim_device": "gpu",
            "enable_livestream": False,
            "task": {"name": "iris",
                     "physics_engine": "physx",
                     "env": {
                             "numEnvs": 1024,
                             "envSpacing": 2.5,
                            #  "episodeLength": 100,
                             "enableDebugVis": False,
                             "clipObservations": 5.0,
                             "clipActions": 1.0,
                             "maxEpisodeLength": 1000 #700 1400
                            #  "controlFrequencyInv": 4,
                            #  "actionScale": 2.5,
                            #  "dofVelocityScale": 0.1,
                            #  "controlSpace": "cartesian"
                             },
                     "sim": {
                             "dt": 0.005,  # 1 / 120
                             "use_gpu_pipeline": True,
                             "gravity": [0.0, 0.0, -9.81],
                             "add_ground_plane": True,
                             "use_flatcache": True,
                             "enable_scene_query_support": False,
                             "enable_cameras": False,
                             "disable_contact_processing": False,
                            #  "default_physics_material": {"static_friction": 1.0,
                                                        #  "dynamic_friction": 1.0,
                                                        #  "restitution": 0.0},
                             "physx": {
                                      "worker_thread_count": 6, #4
                                      "solver_type": 1,
                                      "use_gpu": True,
                                      "solver_position_iteration_count": 6,
                                      "solver_velocity_iteration_count": 1,
                                      "contact_offset": 0.02,
                                      "rest_offset": 0.001,
                                      "bounce_threshold_velocity": 0.2,
                                      "friction_offset_threshold": 0.04,
                                      "friction_correlation_distance": 0.025,
                                      "enable_sleeping": True,
                                      "enable_stabilization": True,
                                      "max_depenetration_velocity": 1000.0,
                                      "gpu_max_rigid_contact_count": 524288,
                                      "gpu_max_rigid_patch_count": 33554432,
                                      "gpu_found_lost_pairs_capacity": 524288,
                                      "gpu_found_lost_aggregate_pairs_capacity": 262144,
                                      "gpu_total_aggregate_pairs_capacity": 1048576,
                                      "gpu_max_soft_body_contacts": 1048576,
                                      "gpu_max_particle_contacts": 1048576,
                                      "gpu_heap_capacity": 33554432,
                                      "gpu_temp_buffer_capacity": 16777216,
                                      "gpu_max_num_partitions": 8
                                      },

                            #  "robot": {"override_usd_defaults": False,
                            #            "fixed_base": False,
                            #            "enable_self_collisions": False,
                            #            "enable_gyroscopic_forces": True,
                            #            "solver_position_iteration_count": 4,
                            #            "solver_velocity_iteration_count": 1,
                            #            "sleep_threshold": 0.005,
                            #            "stabilization_threshold": 0.001,
                            #            "density": -1,
                            #            "max_depenetration_velocity": 1000.0,
                            #            "contact_offset": 0.005,
                            #            "rest_offset": 0.0},
                            "iris":{

                                        # -1 to use default values
                                        "override_usd_defaults": False,
                                        "enable_self_collisions": True,
                                        "enable_gyroscopic_forces": True,
                                        # also in stage params
                                        # per-actor
                                        "solver_position_iteration_count": 6,
                                        "solver_velocity_iteration_count": 1,
                                        "sleep_threshold": 0.005,
                                        "stabilization_threshold": 0.001,
                                        # per-body
                                        "density": -1,
                                        "max_depenetration_velocity": 1000.0
                                        },
                            "ball":{
                                # -1 to use default values
                                "override_usd_defaults": False,
                                "make_kinematic": True,
                                "enable_self_collisions": False,
                                "enable_gyroscopic_forces": True,
                                # also in stage params
                                # per-actor
                                "solver_position_iteration_count": 6,
                                "solver_velocity_iteration_count": 1,
                                "sleep_threshold": 0.005,
                                "stabilization_threshold": 0.001,
                                # per-body
                                "density": -1,
                                "max_depenetration_velocity": 1000.0
                                    },

                             "target": {"override_usd_defaults": False,
                                        "fixed_base": True,
                                        "enable_self_collisions": False,
                                        "enable_gyroscopic_forces": True,
                                        "solver_position_iteration_count": 4,
                                        "solver_velocity_iteration_count": 1,
                                        "sleep_threshold": 0.005,
                                        "stabilization_threshold": 0.001,
                                        "density": -1,
                                        "max_depenetration_velocity": 1000.0,
                                        "contact_offset": 0.005,
                                        "rest_offset": 0.0}
                                }
                        }
                }
class iris(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "iris",
        usd_path: Optional[str] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        scale: Optional[np.array] = None
    ) -> None:
        """[summary]
        """
        
        self._usd_path = usd_path
        self._name = name

        if self._usd_path is None:
            assets_root_path = get_assets_root_path()
            if assets_root_path is None:
                carb.log_error("Could not find Isaac Sim assets folder")
            # self._usd_path = assets_root_path + "/Isaac/Robots/Crazyflie/cf2x.usd"
            print("assets_root_path = ",assets_root_path)
            self._usd_path = "/home/jaramy/PegasusSimulator/extensions/pegasus.simulator/pegasus/simulator/assets/Robots/Iris/iris.usd"

        add_reference_to_stage(self._usd_path, prim_path)
        scale = torch.tensor([1, 1, 1])

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=translation,
            orientation=orientation,
            scale=scale
        )

class irisView(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "irisView"
    ) -> None:
        """[summary]
        """

        super().__init__(
            prim_paths_expr=prim_paths_expr,
            name=name,
        )

        self.physics_rotors = [RigidPrimView(prim_paths_expr=f"/World/envs/.*/iris/rotor{i}",
                                             name=f"rotor{i}_prop_view") for i in range(0, 4)]
        
EPS = 1e-6   # small constant to avoid divisions by 0 and log(0)          
class irisTask(RLTask):
    def __init__(
            self,
            name,
            sim_config,
            env,
            offset=None
    ) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg["env"]["maxEpisodeLength"]

        self.dt = self._task_cfg["sim"]["dt"]

        self._num_observations = 22  # original 18 ops + 4 prev_action
        self._num_actions = 4

        self._crazyflie_position = torch.tensor([0, 0, 1.0])
        self._ball_position = torch.tensor([0, 0, 1.0])

        # self._ball_position = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        # self._ball_position[:, 2] = 1

        RLTask.__init__(self, name=name, env=env)

        # parameters for the crazyflie
        self.arm_length = 0.2

        # parameters for the controller
        self.motor_damp_time_up = 0.2
        self.motor_damp_time_down = 0.2 #0.15

        # I use the multiplier 4, since 4*T ~ time for a step response to finish, where
        # T is a time constant of the first-order filter
        self.motor_tau_up = 4 * self.dt / (self.motor_damp_time_up + EPS)
        self.motor_tau_down = 4 * self.dt / (self.motor_damp_time_down + EPS)

        self.thrusts = torch.zeros((self._num_envs, 4, 3), dtype=torch.float32, device=self._device)
        self.thrust_cmds_damp = torch.zeros((self._num_envs, 4), dtype=torch.float32, device=self._device)
        self.thrust_rot_damp = torch.zeros((self._num_envs, 4), dtype=torch.float32, device=self._device)

        # thrust max
        self.mass = 1.5  #1.5
        self.thrust_to_weight = 10.0

        self.motor_assymetry = np.array([1.0, 1.0, 1.0, 1.0])
        # re-normalizing to sum-up to 4
        self.motor_assymetry = self.motor_assymetry * 4. / np.sum(self.motor_assymetry)

        self.grav_z = -1.0 * self._task_cfg["sim"]["gravity"][2]
        thrust_max = self.grav_z * self.mass * self.thrust_to_weight * self.motor_assymetry / 4.0
        self.thrust_max = torch.tensor(thrust_max, device=self._device, dtype=torch.float32)

        self.motor_linearity = 1.0
        self.prop_max_rot = 900  #433.3

        self.target_positions = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        self.target_positions[:, 2] = 1
        self.actions = torch.zeros((self._num_envs, 4), device=self._device, dtype=torch.float32)
        self.prev_actions = torch.zeros((self._num_envs, 4), device=self._device, dtype=torch.float32)

        self.prev_obs_buf = torch.zeros((self._num_envs, 18), device=self._device, dtype=torch.float32)
        
        self.all_indices = torch.arange(self._num_envs, dtype=torch.int32, device=self._device)

        # Extra info
        self.extras = {}

        torch_zeros = lambda: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums = {"rew_pos": torch_zeros(), "rew_orient": torch_zeros(), "rew_effort": torch_zeros(),
                             "rew_spin": torch_zeros(),
                             "raw_dist": torch_zeros(), "raw_orient": torch_zeros(), "raw_effort": torch_zeros(),
                             "raw_spin": torch_zeros()}
        return

    def set_up_scene(self, scene) -> None:
        self.get_iris()
        self.get_target()
        RLTask.set_up_scene(self, scene)
        self._copters = irisView(prim_paths_expr="/World/envs/.*/iris", name="iris_view")
        self._balls = RigidPrimView(prim_paths_expr="/World/envs/.*/ball")
        scene.add(self._copters)
        scene.add(self._balls)
        for i in range(4):
            scene.add(self._copters.physics_rotors[i])
        return
 
    
    def get_iris(self):
        copter = iris(prim_path=self.default_zero_env_path + "/iris", name="iris",translation=self._crazyflie_position)
        self._sim_config.apply_articulation_settings("iris", get_prim_at_path(copter.prim_path),
                                                     self._sim_config.parse_actor_config("iris"))

    def get_target(self):
        radius = 0.1
        color = torch.tensor([0, 1, 0])
        ball = DynamicSphere(
            prim_path=self.default_zero_env_path + "/ball",
            translation=self._ball_position,
            name="target_0",
            radius=radius,
            color=color)
        self._sim_config.apply_articulation_settings("ball", get_prim_at_path(ball.prim_path),
                                                     self._sim_config.parse_actor_config("ball"))
        ball.set_collision_enabled(False)

    def get_observations(self) -> dict:
        self.root_pos, self.root_rot = self._copters.get_world_poses(clone=False)
        self.root_velocities = self._copters.get_velocities(clone=False)

        root_positions = self.root_pos - self._env_pos
        root_quats = self.root_rot

        rot_x = quat_axis(root_quats, 0)
        rot_y = quat_axis(root_quats, 1)
        rot_z = quat_axis(root_quats, 2)

        root_linvels = self.root_velocities[:, :3]
        root_angvels = self.root_velocities[:, 3:]
        
        

        self.obs_buf[..., 0:3] = self.target_positions - root_positions

        self.obs_buf[..., 3:6] = rot_x
        self.obs_buf[..., 6:9] = rot_y
        self.obs_buf[..., 9:12] = rot_z

        self.obs_buf[..., 12:15] = root_linvels
        self.obs_buf[..., 15:18] = root_angvels
        # self.prev_obs_buf[..., 18:36] = self.obs_buf
        # self.obs_buf[..., 18:36] = self.prev_obs_buf
        # self.obs_buf[..., 36:40] = self.prev_actions
        self.obs_buf[..., 18:22] = self.prev_actions

        # print(self.obs_buf + self.prev_actions)
        

        # print(self.obs_buf.size())
        
        observations = {
            self._copters.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def pre_physics_step(self, actions) -> None:
        if not self._env._world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        set_target_ids = (self.progress_buf % 500 == 0).nonzero(as_tuple=False).squeeze(-1)
        if len(set_target_ids) > 0:
            self.set_targets(set_target_ids)

        actions = actions.clone().to(self._device)
        self.actions = actions
        self.prev_actions = self.actions
        # print(self.prev_actions.size())

        # clamp to [-1.0, 1.0]
        thrust_cmds = torch.clamp(actions, min=-1.0, max=1.0)
        # scale to [0.0, 1.0]
        thrust_cmds = (thrust_cmds + 1.0) / 2.0
        # filtering the thruster and adding noise
        motor_tau = self.motor_tau_up * torch.ones((self._num_envs, 4), dtype=torch.float32, device=self._device)
        motor_tau[thrust_cmds < self.thrust_cmds_damp] = self.motor_tau_down
        motor_tau[motor_tau > 1.0] = 1.0

        # Since NN commands thrusts we need to convert to rot vel and back
        thrust_rot = thrust_cmds ** 0.5
        self.thrust_rot_damp = motor_tau * (thrust_rot - self.thrust_rot_damp) + self.thrust_rot_damp
        self.thrust_cmds_damp = self.thrust_rot_damp ** 2

        ## Adding noise
        thrust_noise = 0.01 * torch.randn(4, dtype=torch.float32, device=self._device)
        thrust_noise = thrust_cmds * thrust_noise
        self.thrust_cmds_damp = torch.clamp(self.thrust_cmds_damp + thrust_noise, min=0.0, max=1.0)

        thrusts = self.thrust_max * self.thrust_cmds_damp

        # thrusts given rotation
        root_quats = self.root_rot
        rot_x = quat_axis(root_quats, 0)
        rot_y = quat_axis(root_quats, 1)
        rot_z = quat_axis(root_quats, 2)
        rot_matrix = torch.cat((rot_x, rot_y, rot_z), 1).reshape(-1, 3, 3)

        force_x = torch.zeros(self._num_envs, 4, dtype=torch.float32, device=self._device)
        force_y = torch.zeros(self._num_envs, 4, dtype=torch.float32, device=self._device)
        force_xy = torch.cat((force_x, force_y), 1).reshape(-1, 4, 2)
        
        thrusts = thrusts.reshape(-1, 4, 1)
        thrusts = torch.cat((force_xy, thrusts), 2)

        thrusts_0 = thrusts[:, 0]
        thrusts_0 = thrusts_0[:, :, None]

        thrusts_1 = thrusts[:, 1]
        thrusts_1 = thrusts_1[:, :, None]

        thrusts_2 = thrusts[:, 2]
        thrusts_2 = thrusts_2[:, :, None]

        thrusts_3 = thrusts[:, 3]
        thrusts_3 = thrusts_3[:, :, None]

        mod_thrusts_0 = torch.matmul(rot_matrix, thrusts_0)
        mod_thrusts_1 = torch.matmul(rot_matrix, thrusts_1)
        mod_thrusts_2 = torch.matmul(rot_matrix, thrusts_2)
        mod_thrusts_3 = torch.matmul(rot_matrix, thrusts_3)

        self.thrusts[:, 0] = torch.squeeze(mod_thrusts_0)
        self.thrusts[:, 1] = torch.squeeze(mod_thrusts_1)
        self.thrusts[:, 2] = torch.squeeze(mod_thrusts_2)
        self.thrusts[:, 3] = torch.squeeze(mod_thrusts_3)

        # clear actions for reset envs
        self.thrusts[reset_env_ids] = 0

        # spin spinning rotors
        prop_rot = self.thrust_cmds_damp * self.prop_max_rot
        self.dof_vel[:, 0] = prop_rot[:, 0]
        self.dof_vel[:, 1] = -1.0 * prop_rot[:, 1]
        self.dof_vel[:, 2] = prop_rot[:, 2]
        self.dof_vel[:, 3] = -1.0 * prop_rot[:, 3]

        self._copters.set_joint_velocities(self.dof_vel)

        # apply actions
        for i in range(4):
            self._copters.physics_rotors[i].apply_forces(self.thrusts[:, i], indices=self.all_indices)

    def post_reset(self):
        self.root_pos, self.root_rot = self._copters.get_world_poses()
        self.root_velocities = self._copters.get_velocities()
        self.dof_pos = self._copters.get_joint_positions()
        self.dof_vel = self._copters.get_joint_velocities()

        self.initial_ball_pos, self.initial_ball_rot = self._balls.get_world_poses(clone=False)
        self.initial_root_pos, self.initial_root_rot = self.root_pos.clone(), self.root_rot.clone()

        # control parameters
        self.thrusts = torch.zeros((self._num_envs, 4, 3), dtype=torch.float32, device=self._device)
        self.thrust_cmds_damp = torch.zeros((self._num_envs, 4), dtype=torch.float32, device=self._device)
        self.thrust_rot_damp = torch.zeros((self._num_envs, 4), dtype=torch.float32, device=self._device)

        self.set_targets(self.all_indices)

    # def set_targets(self, env_ids):
    #     num_sets = len(env_ids)
    #     envs_long = env_ids.long()
    #     # set target position randomly with x, y in (0, 0) and z in (2)
    #     self.target_positions[envs_long, 0:2] = torch.zeros((num_sets, 2), device=self._device)
    #     self.target_positions[envs_long, 2] = torch.ones(num_sets, device=self._device) * 2.0

    #     # shift the target up so it visually aligns better
    #     ball_pos = self.target_positions[envs_long] + self._env_pos[envs_long]
    #     ball_pos[:, 2] += 0.0
    #     self._balls.set_world_poses(ball_pos[:, 0:3], self.initial_ball_rot[envs_long].clone(), indices=env_ids)

    # def set_targets(self, env_ids):
    #     num_sets = len(env_ids)
    #     envs_long = env_ids.long()
    #     # set target position randomly with x, y in (-1, 1) and z in (1, 2)
    #     # self.target_positions[envs_long, 0:2] = torch.rand((num_sets, 2), device=self._device) * 2 - 1.95
    #     self.target_positions[envs_long, 0:2] = torch.rand((num_sets, 2), device=self._device) * 1.0 -  0.5
    #     self.target_positions[envs_long, 2] = torch.rand(num_sets, device=self._device) + 0.03

    #     # shift the target up so it visually aligns better
    #     ball_pos = self.target_positions[envs_long] + self._env_pos[envs_long]
    #     ball_pos[:, 2] += 0.4
    #     self._balls.set_world_poses(ball_pos[:, 0:3], self.initial_ball_rot[envs_long].clone(), indices=env_ids)
    def set_targets(self, env_ids):
        num_sets = len(env_ids)
        envs_long = env_ids.long()
        # set target position randomly with x, y in (-1, 1) and z in (1, 2)
        # self.target_positions[envs_long, 0:2] = torch.rand((num_sets, 2), device=self._device) * 2 - 1
        # self.target_positions[envs_long, 2] = torch.rand(num_sets, device=self._device) + 1

        self.target_positions[envs_long, 0:2] = torch.rand((num_sets, 2), device=self._device)*0.2-0.1
        self.target_positions[envs_long, 2] = torch.ones(num_sets, device=self._device) * 2.0

        # shift the target up so it visually aligns better
        ball_pos = self.target_positions[envs_long] + self._env_pos[envs_long]
        ball_pos[:, 2] += 0.0
        self._balls.set_world_poses(ball_pos[:, 0:3], self.initial_ball_rot[envs_long].clone(), indices=env_ids)

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        self.dof_pos[env_ids, :] = torch_rand_float(-0.0, 0.0, (num_resets, self._copters.num_dof), device=self._device)
        self.dof_vel[env_ids, :] = 0

        root_pos = self.initial_root_pos.clone()
        root_pos[env_ids, 0] += torch_rand_float(-0.0, 0.0, (num_resets, 1), device=self._device).view(-1)
        root_pos[env_ids, 1] += torch_rand_float(-0.0, 0.0, (num_resets, 1), device=self._device).view(-1)
        root_pos[env_ids, 2] += torch_rand_float(-0.0, 0.0, (num_resets, 1), device=self._device).view(-1)
        root_velocities = self.root_velocities.clone()
        root_velocities[env_ids] = 0

        # apply resets
        self._copters.set_joint_positions(self.dof_pos[env_ids], indices=env_ids)
        self._copters.set_joint_velocities(self.dof_vel[env_ids], indices=env_ids)

        self._copters.set_world_poses(root_pos[env_ids], self.initial_root_rot[env_ids].clone(), indices=env_ids)
        self._copters.set_velocities(root_velocities[env_ids], indices=env_ids)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        self.thrust_cmds_damp[env_ids] = 0
        self.thrust_rot_damp[env_ids] = 0


        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"][key] = torch.mean(
                self.episode_sums[key][env_ids]) / self._max_episode_length
            self.episode_sums[key][env_ids] = 0.

    def calculate_metrics(self) -> None:
        root_positions = self.root_pos - self._env_pos
        root_quats = self.root_rot
        root_angvels = self.root_velocities[:, 3:]

        # pos reward
        target_dist = torch.sqrt(torch.square(self.target_positions - root_positions).sum(-1))
        pos_reward = 1.0 / (2.0 + target_dist + target_dist)
        self.target_dist = target_dist
        self.root_positions = root_positions

        # orient reward
        ups = quat_axis(root_quats, 2)
        self.orient_z = ups[..., 2]
        up_reward = torch.clamp(ups[..., 2], min=0.0, max=1.0)
        # ups = quat_axis(root_quats, 2)
        
        # tiltage = torch.abs(1 - ups[..., 2])
        # up_reward = 1.0 / (1.0 + 30 * tiltage * tiltage)
  

        # effort reward
        effort = torch.square(self.actions).sum(-1)
        effort_reward = 0.05 * torch.exp(-0.5 * effort)

        # spin reward
        # spin = torch.square(root_angvels).sum(-1)
        # spin_reward = 0.01 * torch.exp(-1.0 * spin)
        
        spinnage = torch.abs(root_angvels[..., 2])
        spinnage_reward = 1.0 / (1.0 + 10 * spinnage * spinnage)

        # combined reward
        self.rew_buf[:] = pos_reward + pos_reward * (up_reward + spinnage_reward) - effort_reward


        # log episode reward sums
        self.episode_sums["rew_pos"] += pos_reward
        self.episode_sums["rew_orient"] += up_reward
        self.episode_sums["rew_effort"] += effort_reward
        self.episode_sums["rew_spin"] += spinnage_reward#spin_reward

        # log raw info
        self.episode_sums["raw_dist"] += target_dist
        self.episode_sums["raw_orient"] += ups[..., 2]
        self.episode_sums["raw_effort"] += effort
        self.episode_sums["raw_spin"] += spinnage #spin

    def is_done(self) -> None:
        # resets due to misbehavior
        ones = torch.ones_like(self.reset_buf)
        die = torch.zeros_like(self.reset_buf)
        die = torch.where(self.target_dist > 20.0, ones, die)

        # z >= 0.5 & z <= 5.0 & up > 0
        die = torch.where(self.root_positions[..., 2] < 0.5, ones, die)
        die = torch.where(self.root_positions[..., 2] > 10.0, ones, die)
        die = torch.where(self.orient_z < 0.0, ones, die)

        # resets due to episode length
        self.reset_buf[:] = torch.where(self.progress_buf >= self._max_episode_length - 1, ones, die)

    # def is_done(self) -> None:
    #     # resets due to misbehavior
    #     ones = torch.ones_like(self.reset_buf)
    #     die = torch.zeros_like(self.reset_buf)
    #     die = torch.where(self.target_dist > 20.0, ones, die)
    #     die = torch.where(self.root_positions[..., 2] < 0.5, ones, die)

    #     # resets due to episode length
    #     self.reset_buf[:] = torch.where(self.progress_buf >= self._max_episode_length - 1, ones, die)
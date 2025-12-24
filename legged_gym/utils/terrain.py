# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
from numpy.random import choice
from scipy import interpolate
import random
from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.utils import trimesh

class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:
        self.trimeshes = []
        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:    
            self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
    
    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
        
    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.vertical_scale,
                              horizontal_scale=self.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
    
    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.length_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        slope = difficulty * 0.4
        step_height = 0.05 + 0.18 * difficulty
        high_step_height = 0.2 + 0.3 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty==0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        if choice < self.proportions[0]:
            if choice < self.proportions[0]/ 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < self.proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            if choice<self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
        elif choice < self.proportions[5]:
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
        elif choice < self.proportions[6]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        elif choice < self.proportions[7]:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        elif choice < self.proportions[8]: # 需确保 cfg.terrain_proportions 至少有9个元素
            backfill_mounds_terrain(terrain, difficulty, platform_size=0.)
        elif choice < self.proportions[9]: # 需确保 cfg.terrain_proportions 至少有10个元素
            agri_ridge_terrain(terrain, difficulty, platform_size=0.)
        elif choice < self.proportions[11]:
            if choice<self.proportions[10]:
                high_step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=high_step_height, platform_size=3.)
            self.add_roughness(terrain, difficulty)
        elif choice < self.proportions[12]:
            crawling_tunnel_terrain(terrain, difficulty, platform_size=2.)
        elif choice < self.proportions[13]:
            suspended_grate_terrain(terrain, difficulty, platform_size=2.)
        elif choice < self.proportions[14]:
            floating_stepping_blocks_terrain(terrain, difficulty, platform_size=2.)
        elif choice < self.proportions[15]:
            choppy_waves_terrain(terrain, difficulty, platform_size=2.)
        elif choice < self.proportions[16]:
            diagonal_trenches_terrain(terrain, difficulty, platform_size=2.)
        elif choice < self.proportions[18]:
            if choice<self.proportions[17]:
                step_height *= -1
            print("Using hollow pyramid stairs terrain")
            hollow_pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=2.)
        else:
            mud_slope_terrain(terrain, slope=slope, platform_size=0., noise_scale=0.05)
        
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

        if hasattr(terrain, 'trimeshes') and terrain.trimeshes:
            # print(f">>> DEBUG: Found {len(terrain.trimeshes)} meshes from sub-terrain ({row}, {col}). Collecting them.")
            
            origin_x_m = i * self.env_length
            origin_y_m = j * self.env_width
            
            for vertices, triangles in terrain.trimeshes:
                world_vertices = vertices.copy()
                world_vertices[:, 0] += origin_x_m
                world_vertices[:, 1] += origin_y_m
                self.trimeshes.append((world_vertices, triangles))


    def add_roughness(self, terrain, difficulty=1):
        cfgheight = [0.02, 0.06]
        max_height = (cfgheight[1] - cfgheight[0]) * difficulty + cfgheight[0]
        height = random.uniform(cfgheight[0], max_height)
        terrain_utils.random_uniform_terrain(terrain, min_height=-height, max_height=height, step=0.005, downsampled_scale=0.075)

def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth


def mud_slope_terrain(terrain, slope, platform_size=1.0, noise_scale=0.1):
    """
    生成泥石流陡坡：线性坡度 + 随机噪声
    :param terrain: 地形对象
    :param slope: 坡度 (例如 0.4 表示很陡)
    :param platform_size: 起始平台长度 (米)
    :param noise_scale: 噪声幅度 (米)，例如 0.1 表示地面会有 ±10cm 的起伏
    """
    # 1. 获取网格尺寸
    # 在 legged_gym 中，width 通常对应 x 轴 (前进方向)，length 对应 y 轴
    num_rows = int(terrain.length)
    num_cols = int(terrain.width)

    # 2. 生成基础坡度 (沿着 x 轴上升)
    # 创建 0 到 max_x 的线性数组
    x = np.linspace(0, num_rows * terrain.horizontal_scale, num_rows)
    # 扩展成 2D 矩阵 (rows, cols)
    height_field = np.tile(x, (num_cols, 1)).T * slope

    # 3. 添加随机噪声 (泥石流的核心)
    # 生成一个和地图一样大的随机数矩阵
    noise = np.random.uniform(-noise_scale, noise_scale, size=height_field.shape)
    height_field += noise

    # 4. 确保起始平台是平的 (否则机器人一出生就翻车)
    platform_len = int(platform_size / terrain.horizontal_scale)
    # 把前几米的高度强制设为 0
    height_field[:platform_len, :] = 0

    # 5. 写入 terrain.height_field_raw
    # 注意：Isaac Gym 底层是 int16，需要除以垂直缩放比例
    terrain.height_field_raw[:, :] = (height_field / terrain.vertical_scale).astype(np.int16)
    
    return terrain

# ==============================================================================
#  Helper Function (为了代码简洁，不需要复制，直接嵌入逻辑即可，但逻辑如下)
# ==============================================================================
# 逻辑：找到网格中心 -> 计算平台像素半径 -> 将该矩形区域高度置零

# ==============================================================================
#  1. 施工区域回填土丘 (Backfill Mounds) - Center Platform
# ==============================================================================
def backfill_mounds_terrain(terrain, difficulty, platform_size=2.0):
    num_rows = int(terrain.length)
    num_cols = int(terrain.width)
    
    # 物理坐标网格
    x = np.linspace(0, num_rows * terrain.horizontal_scale, num_rows)
    y = np.linspace(0, num_cols * terrain.horizontal_scale, num_cols)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    
    height_field = np.zeros((num_rows, num_cols), dtype=np.float32)
    
    min_h = 0.05 + difficulty * 0.05
    max_h = 0.15 + difficulty * 0.15
    min_r = 0.5
    max_r = 1.2 + difficulty * 0.5
    num_mounds = int(5 + difficulty * 10) 

    # 计算中心点坐标 (米)
    center_x_m = num_rows * terrain.horizontal_scale / 2
    center_y_m = num_cols * terrain.horizontal_scale / 2
    
    # 为了防止土丘生成切到平台边缘，我们在生成时避开中心区域
    # 平台半径 + 最大土丘半径
    safe_radius = (platform_size / 2.0) + max_r

    mounds_created = 0
    attempts = 0
    # 简单的防止死循环
    while mounds_created < num_mounds and attempts < num_mounds * 5:
        attempts += 1
        
        # 全图随机生成中心
        c_x = np.random.uniform(max_r, (num_rows * terrain.horizontal_scale) - max_r)
        c_y = np.random.uniform(max_r, (num_cols * terrain.horizontal_scale) - max_r)
        
        # 检查是否太靠近中心平台
        dist_to_center = np.sqrt((c_x - center_x_m)**2 + (c_y - center_y_m)**2)
        if dist_to_center < safe_radius:
            continue

        h = np.random.uniform(min_h, max_h)
        r = np.random.uniform(min_r, max_r)
        
        dist = np.sqrt((xx - c_x)**2 + (yy - c_y)**2)
        mound = h * (1 - dist / r)
        mound = np.maximum(0, mound)
        height_field = np.maximum(height_field, mound)
        mounds_created += 1

    # 【强制清理中心平台】确保绝对平整
    mid_r = num_rows // 2
    mid_c = num_cols // 2
    half_p_pixels = int(platform_size / terrain.horizontal_scale / 2)
    
    r1 = max(0, mid_r - half_p_pixels)
    r2 = min(num_rows, mid_r + half_p_pixels)
    c1 = max(0, mid_c - half_p_pixels)
    c2 = min(num_cols, mid_c + half_p_pixels)
    
    height_field[r1:r2, c1:c2] = 0

    terrain.height_field_raw[:, :] = (height_field / terrain.vertical_scale).astype(np.int16)


# ==============================================================================
#  2. 农田田埂 (Agri Ridge) - Center Platform
# ==============================================================================
def agri_ridge_terrain(terrain, difficulty, platform_size=2.0):
    num_rows = int(terrain.length)
    num_cols = int(terrain.width)
    height_field = np.zeros((num_rows, num_cols), dtype=np.float32)
    
    ridge_top_width = 0.3
    ridge_height = 0.08 + difficulty * 0.08
    ridge_base_width = 1.0 + (1.0 - difficulty) * 0.5 
    gap_size = 1.5
    
    x_coords = np.linspace(0, num_rows * terrain.horizontal_scale, num_rows)
    profile_1d = np.zeros(num_rows)
    
    # 从头开始生成，不考虑平台，最后挖空
    current_x = 0.5 # 稍微留点边距
    
    while current_x < (num_rows * terrain.horizontal_scale) - ridge_base_width:
        x_start = current_x
        x_top_start = x_start + (ridge_base_width - ridge_top_width) / 2
        x_top_end = x_top_start + ridge_top_width
        x_end = x_start + ridge_base_width
        
        mask_up = (x_coords >= x_start) & (x_coords < x_top_start)
        if np.any(mask_up):
            slope = ridge_height / (x_top_start - x_start)
            profile_1d[mask_up] = slope * (x_coords[mask_up] - x_start)
            
        mask_top = (x_coords >= x_top_start) & (x_coords <= x_top_end)
        profile_1d[mask_top] = ridge_height
        
        mask_down = (x_coords > x_top_end) & (x_coords <= x_end)
        if np.any(mask_down):
            slope = ridge_height / (x_end - x_top_end)
            profile_1d[mask_down] = ridge_height - slope * (x_coords[mask_down] - x_top_end)
            
        current_x += gap_size
        
    height_field = np.tile(profile_1d.reshape(-1, 1), (1, num_cols))
    
    # 【强制清理中心平台】
    mid_r = num_rows // 2
    mid_c = num_cols // 2
    half_p_pixels = int(platform_size / terrain.horizontal_scale / 2)
    
    r1 = max(0, mid_r - half_p_pixels)
    r2 = min(num_rows, mid_r + half_p_pixels)
    c1 = max(0, mid_c - half_p_pixels)
    c2 = min(num_cols, mid_c + half_p_pixels)
    
    height_field[r1:r2, c1:c2] = 0
    
    terrain.height_field_raw[:, :] = (height_field / terrain.vertical_scale).astype(np.int16)


# ==============================================================================
#  3. 废墟瓦砾 (Rubble Ruins) - Center Platform
# ==============================================================================
def rubble_ruins_terrain(terrain, difficulty, platform_size=2.0):
    num_rows = int(terrain.length)
    num_cols = int(terrain.width)
    height_field = np.zeros((num_rows, num_cols), dtype=np.float32)

    num_blocks = int(50 + difficulty * 100)
    min_h = 0.05
    max_h = 0.15 + difficulty * 0.25
    
    min_size = int(0.2 / terrain.horizontal_scale)
    max_size = int(0.6 / terrain.horizontal_scale)

    for _ in range(num_blocks):
        r = np.random.randint(1, num_rows - 5)
        c = np.random.randint(1, num_cols - 1)
        w = np.random.randint(min_size, max_size)
        l = np.random.randint(min_size, max_size)
        h = np.random.uniform(min_h, max_h)
        
        r_end = min(r + w, num_rows)
        c_end = min(c + l, num_cols)
        height_field[r:r_end, c:c_end] = h

    # 【强制清理中心平台】
    mid_r = num_rows // 2
    mid_c = num_cols // 2
    half_p_pixels = int(platform_size / terrain.horizontal_scale / 2)
    
    r1 = max(0, mid_r - half_p_pixels)
    r2 = min(num_rows, mid_r + half_p_pixels)
    c1 = max(0, mid_c - half_p_pixels)
    c2 = min(num_cols, mid_c + half_p_pixels)
    
    height_field[r1:r2, c1:c2] = 0

    terrain.height_field_raw[:, :] = (height_field / terrain.vertical_scale).astype(np.int16)


# ==============================================================================
#  4. 独木桥 (Balance Beam) - Center Platform
# ==============================================================================
def balance_beam_terrain(terrain, difficulty, platform_size=2.0):
    num_rows = int(terrain.length)
    num_cols = int(terrain.width)
    
    pit_depth = 2.0 
    # 整个地图默认是深坑
    height_field = np.full((num_rows, num_cols), -pit_depth, dtype=np.float32)
    
    num_beams = 2 if difficulty > 0.5 else 3
    beam_width_m = 0.4 - difficulty * 0.2 
    beam_width_px = int(beam_width_m / terrain.horizontal_scale)
    
    centers = np.linspace(num_cols * 0.2, num_cols * 0.8, num_beams).astype(int)
    
    for c in centers:
        y_start = max(0, c - beam_width_px // 2)
        y_end = min(num_cols, c + beam_width_px // 2)
        height_field[:, y_start:y_end] = 0
        
        # 仅在非平台区域加噪声，避免出生点不平
        noise = np.random.uniform(-0.02, 0.02, size=(num_rows, y_end-y_start))
        height_field[:, y_start:y_end] += noise
        
    # 【强制清理中心平台】 - 这里实际上是“填平”深坑，连接独木桥
    mid_r = num_rows // 2
    mid_c = num_cols // 2
    half_p_pixels = int(platform_size / terrain.horizontal_scale / 2)
    
    r1 = max(0, mid_r - half_p_pixels)
    r2 = min(num_rows, mid_r + half_p_pixels)
    c1 = max(0, mid_c - half_p_pixels)
    c2 = min(num_cols, mid_c + half_p_pixels)
    
    height_field[r1:r2, c1:c2] = 0
    
    terrain.height_field_raw[:, :] = (height_field / terrain.vertical_scale).astype(np.int16)


# ==============================================================================
#  5. 梅花桩 (Staggered Pillars) - Center Platform
# ==============================================================================
def staggered_pillars_terrain(terrain, difficulty, platform_size=2.0):
    num_rows = int(terrain.length)
    num_cols = int(terrain.width)
    height_field = np.zeros((num_rows, num_cols), dtype=np.float32)
    
    pillar_height = 0.1 + difficulty * 0.2 
    pillar_size = 0.4 
    gap_size = 0.2 + difficulty * 0.2
    
    size_px = int(pillar_size / terrain.horizontal_scale)
    gap_px = int(gap_size / terrain.horizontal_scale)
    total_unit = size_px + gap_px
    
    for i in range(0, num_rows, total_unit):
        for j in range(0, num_cols, total_unit):
            offset = 0 if (i // total_unit) % 2 == 0 else total_unit // 2
            
            c_start = j + offset
            c_end = min(c_start + size_px, num_cols)
            r_end = min(i + size_px, num_rows)
            
            if c_start < num_cols:
                height_field[i:r_end, c_start:c_end] = pillar_height
                tilt = np.random.uniform(-0.02, 0.02)
                height_field[i:r_end, c_start:c_end] += tilt

    # 【强制清理中心平台】
    mid_r = num_rows // 2
    mid_c = num_cols // 2
    half_p_pixels = int(platform_size / terrain.horizontal_scale / 2)
    
    r1 = max(0, mid_r - half_p_pixels)
    r2 = min(num_rows, mid_r + half_p_pixels)
    c1 = max(0, mid_c - half_p_pixels)
    c2 = min(num_cols, mid_c + half_p_pixels)
    
    height_field[r1:r2, c1:c2] = 0
    
    terrain.height_field_raw[:, :] = (height_field / terrain.vertical_scale).astype(np.int16)


# ==============================================================================
#  6. 交叉波浪 (Choppy Waves) - Center Platform
# ==============================================================================
def choppy_waves_terrain(terrain, difficulty, platform_size=2.0):
    num_rows = int(terrain.length)
    num_cols = int(terrain.width)
    
    x = np.linspace(0, num_rows * terrain.horizontal_scale, num_rows)
    y = np.linspace(0, num_cols * terrain.horizontal_scale, num_cols)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    
    amplitude = 0.1 + difficulty * 0.15 
    freq_x = 1.0 + difficulty * 1.0
    freq_y = 1.0 + difficulty * 1.0
    
    height_field = amplitude * np.sin(freq_x * xx) * np.cos(freq_y * yy)
    height_field += amplitude
    
    # 【强制清理中心平台】
    mid_r = num_rows // 2
    mid_c = num_cols // 2
    half_p_pixels = int(platform_size / terrain.horizontal_scale / 2)
    
    r1 = max(0, mid_r - half_p_pixels)
    r2 = min(num_rows, mid_r + half_p_pixels)
    c1 = max(0, mid_c - half_p_pixels)
    c2 = min(num_cols, mid_c + half_p_pixels)
    
    height_field[r1:r2, c1:c2] = 0
    
    terrain.height_field_raw[:, :] = (height_field / terrain.vertical_scale).astype(np.int16)


# ==============================================================================
#  7. 斜向沟壑 (Diagonal Trenches) - Center Platform
# ==============================================================================
def diagonal_trenches_terrain(terrain, difficulty, platform_size=2.0):
    num_rows = int(terrain.length)
    num_cols = int(terrain.width)
    height_field = np.zeros((num_rows, num_cols), dtype=np.float32)
    
    trench_width = 0.4 + difficulty * 0.3 
    trench_depth = 0.5 + difficulty * 0.5 
    land_width = 1.0 - difficulty * 0.2
    
    x = np.linspace(0, num_rows * terrain.horizontal_scale, num_rows)
    y = np.linspace(0, num_cols * terrain.horizontal_scale, num_cols)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    
    period = trench_width + land_width
    pos = (xx + yy) % period
    
    mask_trench = pos < trench_width
    height_field[mask_trench] = -trench_depth
    
    # 【强制清理中心平台】
    mid_r = num_rows // 2
    mid_c = num_cols // 2
    half_p_pixels = int(platform_size / terrain.horizontal_scale / 2)
    
    r1 = max(0, mid_r - half_p_pixels)
    r2 = min(num_rows, mid_r + half_p_pixels)
    c1 = max(0, mid_c - half_p_pixels)
    c2 = min(num_cols, mid_c + half_p_pixels)
    
    height_field[r1:r2, c1:c2] = 0
    
    terrain.height_field_raw[:, :] = (height_field / terrain.vertical_scale).astype(np.int16)


def hollow_pyramid_stairs_terrain(terrain, step_width, step_height, platform_size=1.):
    """
    生成镂空的金字塔楼梯 (Hollow Pyramid Stairs)。
    
    特征：
    - 视觉上是金字塔形状的楼梯。
    - 物理上是"悬浮"的板子，板子下方是深渊。
    - 地形 height_field_raw 被挖空，完全依赖 trimesh 进行碰撞检测。
    
    Parameters:
        terrain (terrain): 地形对象
        step_width (float): 台阶宽度 [meters]
        step_height (float): 台阶高度 [meters]
        platform_size (float): 顶部平台大小 [meters]
    """
    if not hasattr(terrain, 'trimeshes'):
        terrain.trimeshes = []
    # 1. 参数转换
    step_width_px = int(step_width / terrain.horizontal_scale)
    # 注意：trimesh 生成需要真实的米单位，所以这里保留 step_height 为 float
    step_height_m = step_height 
    platform_size_px = int(platform_size / terrain.horizontal_scale)
    
    # 2. 初始化边界
    start_x = 0
    stop_x = terrain.width
    start_y = 0
    stop_y = terrain.length
    
    current_height_m = 0.0
    step_thickness = 0.05  # 台阶厚度，与您提供的 hollow_stairs 保持一致

    
    if not hasattr(terrain, 'trimeshes'):
        terrain.trimeshes = []

    # 4. 循环生成层级 (金字塔逻辑)
    while (stop_x - start_x) > platform_size_px and (stop_y - start_y) > platform_size_px:
        # 收缩边界 (向内进阶)
        start_x += step_width_px
        stop_x -= step_width_px
        start_y += step_width_px
        stop_y -= step_width_px
        
        # 提升高度
        current_height_m += step_height_m
        
        # === 生成当前层级的"环形"网格 ===
        # 我们需要用4个长方体拼成一个正方形的环，代表这一层的台阶
        
        # 计算当前环的尺寸 (米)
        # 注意：这里的 start_x 等已经是收缩后的内圈起点，
        # 但这一层的台阶实际上占据了 [start_x, start_x + step_width] 的位置？
        # 对比实心金字塔逻辑：它是先把区域填高。
        # 所以对于镂空，我们在这个区域生成板子。
        
        # 尺寸定义：
        # 环的外径由当前 start/stop 决定 (因为是填满 start_x 到 stop_x)
        # 环的厚度是 step_width
        
        # 为了避免重叠，我们将环拆分为：
        # 1. 底部横条 (Bottom X-strip)
        # 2. 顶部横条 (Top X-strip)
        # 3. 左侧竖条 (Left Y-strip, 减去上下高度)
        # 4. 右侧竖条 (Right Y-strip, 减去上下高度)
        
        # 转换坐标到米
        inner_w_px = stop_x - start_x
        inner_l_px = stop_y - start_y
        
        # 实际上，上面的逻辑是填充了整个内部。
        # 如果我们要表现"台阶"，我们需要生成的是位于当前边界 *边缘* 的一圈。
        # 在实心金字塔代码中：terrain[start_x:stop_x] = height
        # 这意味着整个内部平台都升高了。
        # 如果是镂空金字塔，为了让它像楼梯，每一层应该是一个中空的环，或者是实心的平台？
        # 通常"Hollow Stairs"意味着每一级是悬浮的，中间不要填满，直到最后。
        # 但金字塔结构意味着下一级会覆盖上一级的中间。
        # 所以我们只需要生成当前层级 *暴露* 出来的部分（即外圈），或者生成整个平面让下一层盖在上面。
        # 为了节约资源并体现"镂空"（侧面看有缝隙），我们生成一个"环"。
        
        # 环的参数：
        # 外边界: start_x, stop_x, start_y, stop_y
        # 内边界 (下一级台阶的位置): start_x + step_width, ... (但在这一层循环里我们不知道下一层是否还会生成)
        # 既然是模仿实心逻辑，这一层其实就是一个大的矩形板。
        # 为了做成"空心楼梯"（悬空板），我们直接生成一个填满 start_x 到 stop_x 的大薄板即可？
        # 不，那样就看不出"镂空"了（如果从下往上看），而且会重叠。
        # 建议：生成一个"环"，宽度为 step_width。
        
        # --- 计算环的几何参数 ---
        # 环的 4 个边
        
        # 1. 下边 (South): 覆盖整个 X 宽度
        # Center X = (start_x + stop_x) / 2
        # Center Y = start_y + step_width/2  <-- 稍微存疑，实心代码是填充 start_y 到 stop_y
        # 我们假设这层台阶是覆盖整个当前矩形的。为了做成"阶梯"，其实只需要生成最外圈的 step_width 宽度的边框。
        # 因为内圈会被下一层更高的台阶覆盖（或者悬空）。
        # 如果内圈悬空，那就是"天井"式楼梯。
        # 这里假设生成标准的金字塔台阶，只是台阶本身是薄板。
        
        # 所以我们生成一个 环 (Ring)。
        
        # 下边 (Bottom Strip)
        # 范围: x[start_x, stop_x], y[start_y, start_y + step_width_px]
        size_bottom = (
            (stop_x - start_x) * terrain.horizontal_scale, 
            step_width_px * terrain.horizontal_scale, 
            step_thickness
        )
        pos_bottom = (
            (start_x + stop_x) / 2 * terrain.horizontal_scale,
            (start_y + step_width_px / 2) * terrain.horizontal_scale,
            current_height_m - step_thickness / 2
        )
        
        # 上边 (Top Strip)
        # 范围: x[start_x, stop_x], y[stop_y - step_width_px, stop_y]
        size_top = size_bottom # 尺寸一样
        pos_top = (
            (start_x + stop_x) / 2 * terrain.horizontal_scale,
            (stop_y - step_width_px / 2) * terrain.horizontal_scale,
            current_height_m - step_thickness / 2
        )
        
        # 左边 (Left Strip)
        # 范围: x[start_x, start_x + step_width_px], y[start_y + step_width, stop_y - step_width] (避免重叠)
        size_left = (
            step_width_px * terrain.horizontal_scale,
            (stop_y - start_y - 2 * step_width_px) * terrain.horizontal_scale,
            step_thickness
        )
        pos_left = (
            (start_x + step_width_px / 2) * terrain.horizontal_scale,
            (start_y + stop_y) / 2 * terrain.horizontal_scale,
            current_height_m - step_thickness / 2
        )
        
        # 右边 (Right Strip)
        # 范围: x[stop_x - step_width_px, stop_x], y...
        size_right = size_left
        pos_right = (
            (stop_x - step_width_px / 2) * terrain.horizontal_scale,
            (start_y + stop_y) / 2 * terrain.horizontal_scale,
            current_height_m - step_thickness / 2
        )
        
        # 生成4个部分的 Mesh
        mesh_bottom = trimesh.box_trimesh(size_bottom, pos_bottom)
        mesh_top = trimesh.box_trimesh(size_top, pos_top)
        mesh_left = trimesh.box_trimesh(size_left, pos_left)
        mesh_right = trimesh.box_trimesh(size_right, pos_right)
        
        # 合并这4个部分为一个 Mesh (当前层的环)
        ring_mesh = trimesh.combine_trimeshes(mesh_bottom, mesh_top)
        ring_mesh = trimesh.combine_trimeshes(ring_mesh, mesh_left)
        ring_mesh = trimesh.combine_trimeshes(ring_mesh, mesh_right)
        
        # 添加到地形
        terrain.trimeshes.append(ring_mesh)

    # 5. 生成顶部平台 (Top Platform)
    # 循环结束后，中间剩余的区域 [start_x, stop_x] x [start_y, stop_y] 是顶部平台
    platform_size_x = (stop_x - start_x) * terrain.horizontal_scale
    platform_size_y = (stop_y - start_y) * terrain.horizontal_scale
    
    # 平台高度再升一级? 或者保持最后一级的高度? 
    # 实心逻辑中，循环里先 height+=step_height，所以当前 current_height_m 已经是这一层的了。
    # 顶部平台应该再高一级，或者填满最后一级。
    # 通常金字塔顶部是平的，高度比最后一级台阶高，或者就是最后一级填实。
    # 我们让它再高一级作为最终目标点。
    current_height_m += step_height_m
    
    platform_size_vec = (platform_size_x, platform_size_y, step_thickness)
    platform_pos = (
        (start_x + stop_x) / 2 * terrain.horizontal_scale,
        (start_y + stop_y) / 2 * terrain.horizontal_scale,
        current_height_m - step_thickness / 2
    )

    # 3. 将地形高度场挖成深坑 (模拟 hollow 效果)
    # 设置为一个很大的负值，确保机器人踩空时会掉落，而不是踩在不可见的地板上
    if step_height > 0:
        terrain.height_field_raw[:, :] = 0
        terrain.height_field_raw[start_x:stop_x,
                                  start_y:stop_y] = int(current_height_m / terrain.vertical_scale)
    else:
        terrain.height_field_raw[:, :] = current_height_m / terrain.vertical_scale   # 大负值，模拟深坑

    platform_mesh = trimesh.box_trimesh(platform_size_vec, platform_pos)
    terrain.trimeshes.append(platform_mesh)
    
    return terrain

def set_center_spawn_height(terrain, platform_size, height_m):
    """
    强制设置地图中心区域的 height_field_raw，确保机器人出生在正确的高度。
    """
    num_rows = int(terrain.length)
    num_cols = int(terrain.width)
    mid_r = num_rows // 2
    mid_c = num_cols // 2
    
    # 计算平台半径 (像素)
    half_p_pixels = int(platform_size / terrain.horizontal_scale / 2)
    # 确保至少有 1 个像素
    half_p_pixels = max(1, half_p_pixels)
    
    r1 = max(0, mid_r - half_p_pixels)
    r2 = min(num_rows, mid_r + half_p_pixels)
    c1 = max(0, mid_c - half_p_pixels)
    c2 = min(num_cols, mid_c + half_p_pixels)
    
    # 将高度转换为 int16 (height / vertical_scale)
    height_int16 = int(height_m / terrain.vertical_scale)
    
    terrain.height_field_raw[r1:r2, c1:c2] = height_int16


# ==============================================================================
#  2. 匍匐隧道 (Crawling Tunnel) - 修正版
# ==============================================================================
def crawling_tunnel_terrain(terrain, difficulty, platform_size=2.0):
    terrain.height_field_raw[:, :] = -1000 # 挖空
    if not hasattr(terrain, 'trimeshes'): terrain.trimeshes = []
    
    num_rows = int(terrain.length)
    num_cols = int(terrain.width)
    
    path_width_m = 1.5 + (1.0 - difficulty) * 0.5
    ceiling_height = 1.2 - difficulty * 0.85 
    thickness = 0.1
    
    # 1. 地板 Mesh (贯穿全图)
    floor_size = (num_rows * terrain.horizontal_scale, path_width_m, thickness)
    floor_pos = ((num_rows * terrain.horizontal_scale) / 2, (num_cols * terrain.horizontal_scale) / 2, -thickness / 2)
    terrain.trimeshes.append(trimesh.box_trimesh(floor_size, floor_pos))
    
    # 2. 顶棚 Mesh (避开中心出生点)
    # 既然平台在中心，我们需要把顶棚分成两段，或者把出生点留出来
    # 这里我们选择“两头有顶棚，中间留空”或者“只有一边有顶棚”
    # 为了简化，我们让顶棚覆盖除了中心平台以外的区域
    
    center_m = (num_rows * terrain.horizontal_scale) / 2
    safe_radius = platform_size / 2.0 + 0.5 # 多留点空间
    
    # 左侧顶棚 (0 到 center - safe)
    len_1 = max(0, center_m - safe_radius)
    if len_1 > 0:
        c1_size = (len_1, path_width_m + 0.5, thickness)
        c1_pos = (len_1 / 2, (num_cols * terrain.horizontal_scale) / 2, ceiling_height + thickness / 2)
        terrain.trimeshes.append(trimesh.box_trimesh(c1_size, c1_pos))
        
    # 右侧顶棚 (center + safe 到 end)
    len_2 = (num_rows * terrain.horizontal_scale) - (center_m + safe_radius)
    if len_2 > 0:
        c2_size = (len_2, path_width_m + 0.5, thickness)
        c2_pos = (center_m + safe_radius + len_2 / 2, (num_cols * terrain.horizontal_scale) / 2, ceiling_height + thickness / 2)
        terrain.trimeshes.append(trimesh.box_trimesh(c2_size, c2_pos))

    # 【关键修正】：设置中心出生点高度为 0
    set_center_spawn_height(terrain, platform_size, 0.0)

# ==============================================================================
#  3. 悬空格栅 (Suspended Grate) - 修正版
# ==============================================================================
def suspended_grate_terrain(terrain, difficulty, platform_size=2.0):
    terrain.height_field_raw[:, :] = -1000
    if not hasattr(terrain, 'trimeshes'): terrain.trimeshes = []
    
    num_rows = int(terrain.length)
    num_cols = int(terrain.width)
    
    path_width_m = 2.0
    thickness = 0.1
    beam_width = 0.2 - difficulty * 0.1 
    gap_width = 0.05 + difficulty * 0.1
    
    total_len_m = num_rows * terrain.horizontal_scale
    center_x_m = total_len_m / 2
    
    # 1. 生成中心安全平台 Mesh
    # 必须在中心放一个实心板子，否则出生时脚会卡住
    plat_size = (platform_size, path_width_m, thickness)
    plat_pos = (center_x_m, (num_cols * terrain.horizontal_scale) / 2, -thickness / 2)
    terrain.trimeshes.append(trimesh.box_trimesh(plat_size, plat_pos))
    
    # 2. 生成两侧的格栅
    # 向左生成
    current_x = center_x_m - platform_size / 2 - gap_width
    while current_x > 0:
        if current_x - beam_width < 0: break
        beam_size = (beam_width, path_width_m, thickness)
        beam_pos = (current_x - beam_width / 2, (num_cols * terrain.horizontal_scale) / 2, -thickness / 2)
        terrain.trimeshes.append(trimesh.box_trimesh(beam_size, beam_pos))
        current_x -= (beam_width + gap_width)
        
    # 向右生成
    current_x = center_x_m + platform_size / 2 + gap_width
    while current_x < total_len_m:
        if current_x + beam_width > total_len_m: break
        beam_size = (beam_width, path_width_m, thickness)
        beam_pos = (current_x + beam_width / 2, (num_cols * terrain.horizontal_scale) / 2, -thickness / 2)
        terrain.trimeshes.append(trimesh.box_trimesh(beam_size, beam_pos))
        current_x += (beam_width + gap_width)
        
    # 【关键修正】：设置中心出生点高度为 0
    set_center_spawn_height(terrain, platform_size, 0.0)

# ==============================================================================
#  4. 悬浮方块 (Floating Blocks) - 修正版
# ==============================================================================
def floating_stepping_blocks_terrain(terrain, difficulty, platform_size=2.0):
    terrain.height_field_raw[:, :] = -1000
    if not hasattr(terrain, 'trimeshes'): terrain.trimeshes = []
    
    num_rows = int(terrain.length)
    num_cols = int(terrain.width)
    
    block_size = 0.5 - difficulty * 0.25
    gap_range = 0.1 + difficulty * 0.2
    height_noise = 0.1 * difficulty
    thickness = 0.2
    
    center_x_m = num_rows * terrain.horizontal_scale / 2
    center_y_m = num_cols * terrain.horizontal_scale / 2
    
    # 1. 生成中心安全平台 Mesh
    plat_size = (platform_size, platform_size, thickness)
    plat_pos = (center_x_m, center_y_m, -thickness / 2)
    terrain.trimeshes.append(trimesh.box_trimesh(plat_size, plat_pos))
    
    # 2. 填充四周的方块
    # 为了避免和中心平台重叠，我们遍历全图，如果方块在中心平台范围内则跳过
    start_x = 0
    end_x = num_rows * terrain.horizontal_scale
    y_min = center_y_m - 2.0 # 宽度4米
    y_max = center_y_m + 2.0
    
    safe_radius = platform_size / 2.0 + 0.2
    
    current_x = start_x
    while current_x < end_x:
        current_y = y_min
        while current_y < y_max:
            # 计算方块中心到地图中心的距离
            cx = current_x + block_size/2
            cy = current_y + block_size/2
            dist = np.sqrt((cx - center_x_m)**2 + (cy - center_y_m)**2)
            
            # 如果在安全区外，则生成方块
            if dist > safe_radius:
                if np.random.uniform() > 0.2:
                    h_offset = np.random.uniform(-height_noise, height_noise)
                    b_size = (block_size, block_size, thickness)
                    b_pos = (cx, cy, h_offset - thickness/2)
                    terrain.trimeshes.append(trimesh.box_trimesh(b_size, b_pos))
            
            current_y += block_size + gap_range
        current_x += block_size + gap_range

    # 【关键修正】：设置中心出生点高度为 0
    set_center_spawn_height(terrain, platform_size, 0.0)
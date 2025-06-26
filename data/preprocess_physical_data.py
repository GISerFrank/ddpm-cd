import os
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
import geopandas as gpd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings

# 忽略 rasterio 和 geopandas 可能产生的运行时警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==============================================================================
# --- 1. 核心配置区域 (请根据您的项目结构修改这里) ---
# ==============================================================================
CONFIG = {
    # 指向您第一阶段生成的数据集目录
    "dataset_root": "./GVLM_CD_split_dataset_geotiff",
    
    # 原始物理数据的根目录
    "physical_data_source_root": "./physical_data_sources",

    # 预处理后数据的输出根目录
    "output_root": "./preprocessed_physical_data",

    # 要处理的数据集划分
    "splits_to_process": ["train", "val", "test"],

    # 定义物理数据源
    # - `type`: 'raster' 或 'vector'
    # - `path_template`: 文件路径模板，{site_name} 会被自动替换
    # - `attribute_column`: (仅用于vector) Shapefile中要栅格化的属性列名
    # - `resampling`: (仅用于raster) 'bilinear', 'nearest', 'cubic' 等
    "physical_data_layers": {
        "dem": {
            "type": "raster",
            "path_template": "{site_name}/DEM.tif",
            "resampling": Resampling.bilinear,
        },
        "slope": {
            "type": "derived", # 特殊类型，从DEM计算
            "source": "dem", # 依赖于DEM
        },
        "lithology": {
            "type": "vector",
            "path_template": "{site_name}/Lithology.shp",
            "attribute_column": "LITHO_ID", # 假设Shapefile里有这一列代表岩性ID
        },
        "soil": {
            "type": "vector",
            "path_template": "{site_name}/Soil.shp",
            "attribute_column": "SOIL_ID", # 假设Shapefile里有这一列代表土壤ID
        }
    },
    
    # 定义最终输出文件中各个通道的顺序
    "output_channel_order": ["dem", "slope", "lithology", "soil"]
}

# ==============================================================================
# --- 2. 辅助函数和虚拟数据生成器 ---
# ==============================================================================

def load_patch_metadata(list_dir, splits):
    """从 train/val/test.txt 加载所有需要处理的图像块元数据"""
    all_patches = []
    for split in splits:
        list_path = Path(list_dir) / f"{split}.txt"
        if not list_path.exists():
            print(f"警告: 列表文件未找到，跳过: {list_path}")
            continue
        with open(list_path, 'r') as f:
            for line in f:
                # 假设行格式为: "site/pre... site/post... site/mask..."
                parts = line.strip().split()
                if not parts: continue
                # 从第一个路径中解析信息
                full_path = Path(parts[0])
                patch_name = full_path.name
                site_name = full_path.parts[-4]
                
                all_patches.append({
                    "split": split,
                    "site_name": site_name,
                    "patch_name_base": Path(patch_name).stem,
                })
    return all_patches

def create_dummy_physical_data(source_root, sites_list):
    """创建一个虚拟的DEM (raster) 和 岩性 (vector) 数据集用于演示"""
    print("\n--- 正在创建虚拟物理数据源 (用于演示) ---")
    source_root = Path(source_root)
    for site_name in list(set(d['site_name'] for d in sites_list))[:2]: # 只为前两个地点创建
        site_path = source_root / site_name
        site_path.mkdir(parents=True, exist_ok=True)
        
        # 1. 创建虚拟 DEM GeoTIFF
        dem_path = site_path / "DEM.tif"
        if not dem_path.exists():
            print(f"  正在创建虚拟DEM: {dem_path}")
            h, w = 2000, 2000
            transform = rasterio.transform.from_origin(-120, 40, 0.01, 0.01)
            profile = {'driver': 'GTiff', 'height': h, 'width': w, 'count': 1,
                       'dtype': 'float32', 'crs': 'EPSG:4326', 'transform': transform}
            # 创建一个有梯度的表面
            x = np.linspace(500, 1500, w)
            y = np.linspace(200, 1200, h)
            xx, yy = np.meshgrid(x, y)
            dummy_dem_data = (xx + yy).astype(np.float32)
            with rasterio.open(dem_path, 'w', **profile) as dst:
                dst.write(dummy_dem_data, 1)

        # 2. 创建虚拟岩性 Shapefile
        litho_path = site_path / "Lithology.shp"
        if not litho_path.exists():
            from shapely.geometry import Polygon
            print(f"  正在创建虚拟岩性Shapefile: {litho_path}")
            # 创建几个多边形区域代表不同的岩性
            poly1 = Polygon([(0, 0), (0, 10), (10, 10), (10, 0)])
            poly2 = Polygon([(10, 10), (10, 20), (20, 20), (20, 10)])
            gdf = gpd.GeoDataFrame(
                {'LITHO_ID': [1, 2]}, # 对应CONFIG中的attribute_column
                geometry=[poly1, poly2],
                crs="EPSG:4326"
            )
            # 将坐标变换到与DEM相同的范围
            gdf.geometry = gdf.geometry.translate(xoff=-119.5, yoff=40.5)
            gdf.to_file(litho_path, driver='ESRI Shapefile')
            
        # 3. 创建虚拟土壤 Shapefile (为完整性)
        soil_path = site_path / "Soil.shp"
        if not soil_path.exists():
            print(f"  正在创建虚拟土壤Shapefile: {soil_path}")
            poly = Polygon([(0, 0), (0, 20), (20, 20), (20, 0)])
            gdf = gpd.GeoDataFrame({'SOIL_ID': [10]}, geometry=[poly], crs="EPSG:4326")
            gdf.geometry = gdf.geometry.translate(xoff=-119.5, yoff=40.5)
            gdf.to_file(soil_path)
            
    print("--- 虚拟数据创建完毕 ---\n")

def get_target_geometry(reference_patch_path):
    """从参考的光学影像小块中获取精确的地理对齐信息"""
    with rasterio.open(reference_patch_path) as ref_patch:
        return {
            "transform": ref_patch.transform,
            "crs": ref_patch.crs,
            "shape": (ref_patch.height, ref_patch.width)
        }

def calculate_slope(dem_array, transform):
    """从DEM数组计算坡度（单位：度）"""
    # 从transform获取像素大小（分辨率）
    pixel_size_x, pixel_size_y = transform[0], -transform[4]
    
    gy, gx = np.gradient(dem_array, pixel_size_y, pixel_size_x)
    slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
    return np.degrees(slope_rad).astype(np.float32)


# ==============================================================================
# --- 3. 主执行函数 ---
# ==============================================================================

def main():
    """主执行函数"""
    print("--- 开始物理数据预处理 ---")
    
    # 步骤 0: 加载并验证配置
    dataset_root = Path(CONFIG["dataset_root"])
    source_root = Path(CONFIG["physical_data_source_root"])
    output_root = Path(CONFIG["output_root"])
    
    list_dir = dataset_root / "list"
    all_patches = load_patch_metadata(list_dir, CONFIG["splits_to_process"])
    
    if not all_patches:
        print("错误: 未找到任何图像块元数据。请检查 'dataset_root' 和 'list' 文件夹路径是否正确。")
        # 尝试创建虚拟数据以进行演示
        dummy_sites = [{'site_name': s} for s in ['Site_A', 'Site_B']]
        create_dummy_physical_data(source_root, dummy_sites)
        print("\n请注意：由于未找到元数据，仅创建了演示用的物理数据。")
        print("请将您的数据放入正确位置后重新运行。")
        return
        
    create_dummy_physical_data(source_root, all_patches) # 检查并创建虚拟数据

    # 使用缓存来避免重复打开大的源文件
    source_file_cache = {}

    print(f"总共找到 {len(all_patches)} 个图像块需要处理。")
    
    # 步骤 1: 遍历所有图像块
    for patch_info in tqdm(all_patches, desc="正在处理物理数据切片"):
        try:
            split = patch_info["split"]
            site_name = patch_info["site_name"]
            patch_base = patch_info["patch_name_base"]
            
            # 创建输出目录
            output_dir = output_root / split / site_name
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{patch_base}.npy"
            
            # 如果文件已存在，则跳过
            if output_path.exists():
                continue

            # 步骤 2: 获取此图像块的“黄金标准”地理信息
            reference_tiff_path = dataset_root / split / site_name / "pre_event" / "tiff" / f"{patch_base}.tif"
            if not reference_tiff_path.exists():
                print(f"警告: 参考TIF文件未找到，跳过: {reference_tiff_path}")
                continue
            target_geom = get_target_geometry(reference_tiff_path)

            processed_layers = {}

            # 步骤 3: 按顺序处理每个物理数据层
            for layer_name, layer_config in CONFIG["physical_data_layers"].items():
                
                # --- A. 处理栅格数据源 ---
                if layer_config["type"] == "raster":
                    source_path = source_root / layer_config["path_template"].format(site_name=site_name)
                    if source_path not in source_file_cache:
                        if not source_path.exists(): raise FileNotFoundError
                        source_file_cache[source_path] = rasterio.open(source_path)
                    
                    source_raster = source_file_cache[source_path]
                    
                    # 创建目标数组
                    aligned_patch = np.zeros(target_geom["shape"], dtype=np.float32)
                    reproject(
                        source=rasterio.band(source_raster, 1),
                        destination=aligned_patch,
                        src_transform=source_raster.transform,
                        src_crs=source_raster.crs,
                        dst_transform=target_geom["transform"],
                        dst_crs=target_geom["crs"],
                        resampling=layer_config["resampling"]
                    )
                    processed_layers[layer_name] = aligned_patch

                # --- B. 处理矢量数据源 ---
                elif layer_config["type"] == "vector":
                    source_path = source_root / layer_config["path_template"].format(site_name=site_name)
                    if source_path not in source_file_cache:
                         if not source_path.exists(): raise FileNotFoundError
                         source_file_cache[source_path] = gpd.read_file(source_path)

                    source_gdf = source_file_cache[source_path]
                    
                    # 确保矢量数据的CRS与目标一致
                    reprojected_gdf = source_gdf.to_crs(target_geom["crs"])
                    
                    # 准备栅格化的形状
                    shapes = [(geom, value) for geom, value in zip(
                        reprojected_gdf.geometry, 
                        reprojected_gdf[layer_config["attribute_column"]]
                    )]
                    
                    aligned_patch = rasterize(
                        shapes=shapes,
                        out_shape=target_geom["shape"],
                        transform=target_geom["transform"],
                        fill=0, # 未覆盖区域的默认值
                        dtype=np.uint8 # 通常类别数据是整数
                    )
                    processed_layers[layer_name] = aligned_patch

                # --- C. 处理衍生数据源 (如坡度) ---
                elif layer_config["type"] == "derived":
                    source_layer_name = layer_config["source"]
                    if source_layer_name not in processed_layers:
                        print(f"错误: 衍生层 '{layer_name}' 的源 '{source_layer_name}' 尚未处理。请检查CONFIG顺序。")
                        continue
                    
                    if layer_name == "slope":
                        dem_patch = processed_layers[source_layer_name]
                        slope_patch = calculate_slope(dem_patch, target_geom["transform"])
                        processed_layers[layer_name] = slope_patch

            # 步骤 4: 堆叠并保存
            final_stack_list = []
            for name in CONFIG["output_channel_order"]:
                if name in processed_layers:
                    final_stack_list.append(processed_layers[name])
                else:
                    # 如果某个层处理失败或未配置，用0填充
                    final_stack_list.append(np.zeros(target_geom["shape"], dtype=np.float32))

            final_stack = np.stack(final_stack_list, axis=0)
            np.save(output_path, final_stack)
        
        except FileNotFoundError:
            print(f"警告: 找不到 {patch_info['site_name']} 的源物理数据文件，跳过该图像块。")
        except Exception as e:
            print(f"处理 {patch_info['patch_name_base']} 时发生未知错误: {e}")

    # 清理缓存
    for f in source_file_cache.values():
        f.close()
        
    print("\n--- 所有物理数据预处理完成！---")
    print(f"预处理后的数据已保存至: {output_root}")


if __name__ == "__main__":
    main()
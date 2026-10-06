"""把ecg图片目录切分成 N 份软链接子目录,用于 SLURM array + 单卡多进程并行"""
import os
import math
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="原始图片目录")
    parser.add_argument("--dst", required=True, help="切分后存放shard的目录")
    parser.add_argument("--n_gpus", type=int, default=6, help="SLURM array里用几张卡(数组长度)")
    parser.add_argument("--procs_per_gpu", type=int, default=3, help="每张卡内并发跑几个进程")
    parser.add_argument("--extensions", nargs="+", default=[".png", ".jpg", ".jpeg", ".JPG"])
    args = parser.parse_args()

    total_shards = args.n_gpus * args.procs_per_gpu

    files = []
    for root, _, fs in os.walk(args.src):
        for f in fs:
            if f.endswith(tuple(args.extensions)):
                files.append(os.path.join(root, f))
    files.sort()

    n = len(files)
    shard_size = math.ceil(n / total_shards)
    print(f"共 {n} 张图,切成 {total_shards} 份(={args.n_gpus}卡 x {args.procs_per_gpu}进程/卡),每份约 {shard_size} 张")

    for i in range(total_shards):
        shard_dir = os.path.join(args.dst, f"shard_{i}")
        os.makedirs(shard_dir, exist_ok=True)
        shard_files = files[i * shard_size : (i + 1) * shard_size]
        for f in shard_files:
            link_path = os.path.join(shard_dir, os.path.basename(f))
            if not os.path.exists(link_path):
                os.symlink(f, link_path)
        print(f"shard_{i}: {len(shard_files)} 张 -> {shard_dir}")

if __name__ == "__main__":
    main()
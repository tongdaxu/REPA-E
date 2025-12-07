from calculate_fid import calculate_fid_given_paths

fid_num = 50000
sample_folder_dir = '/workspace/cogview_dev/xutd/xu/REPA-E/samples/sit-xl-dinov2-b-enc8-repae-sdvae-flow-6r-1.5-1500k_0400000_cfg1.0-0.0-1.0'
fid_reference_file = '/workspace/cogview_dev/xutd/xu/LightningDiT/VIRTUAL_imagenet256_labeled.npz'
fid = calculate_fid_given_paths(
    [fid_reference_file, sample_folder_dir],
    batch_size=50,
    dims=2048,
    device='cuda',
    num_workers=8,
    sp_len = fid_num
)
print('fid=',fid)


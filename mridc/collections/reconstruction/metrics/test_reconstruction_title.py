import os
from evaluation_metrics import ssim, nrmse, nmse, haarpsi, vsi
import h5py
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from Sense_algorithm import sense


def construct_gt(kspaces, sensitivity_maps, shift=True, norm=None, method="SENSE", one_coil=False):
    # contruct image per coil
    coils, x, y = kspaces.shape
    if shift:
        recon = np.fft.ifftshift(kspaces, axes=(-2, -1))
    recon = np.fft.ifft2(recon, axes=(-2, -1), norm=norm)
    if shift:
        recon = np.fft.fftshift(recon, axes=(-2, -1))
    # combine coils with RSS
    if one_coil:
        return recon
    elif method == "RSS":
        return np.sqrt((np.abs(recon) ** 2).sum(0))
    elif method == "SENSE":
        return np.abs(sense(recon, sensitivity_maps))


def show_mag_phase_real_img_images(ground_truth, recon):
    # Plot magnitude, angle, real and imaginary component
    fig, axes = plt.subplots(2, 2)
    axes[0, 0].imshow(np.abs(ground_truth), cmap='gray')
    axes[0, 0].set_title("Ground truth")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(np.abs(recon), cmap='gray')
    axes[0, 1].set_title("Reconstruction")
    axes[0, 1].axis('off')

    axes[1, 0].imshow(np.angle(ground_truth), cmap='gray')
    axes[1, 0].set_title("Phase ground truth")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(np.angle(recon), cmap='gray')
    axes[1, 1].set_title("Phase Reconstruction")
    axes[1, 1].axis('off')

    # fig.suptitle("Coil 2", fontsize=30)
    plt.show()
    return


def crop_image(image, cropsize):
    x, y = image.shape
    w = int(np.trunc((x - cropsize[0]) / 2))
    h = int(np.trunc((y - cropsize[1]) / 2))
    return image[w:x-w, h:y-h]


def qualitatively_evaluate_recons(datapath, reconpath):
    for root, _, files in os.walk(datapath):
        for file in files:
            print(file)
            gt_data = h5py.File(datapath + file)
            recon_data = h5py.File(reconpath + file)
            print(recon_data)
            # show one slice rec and gt of subject
            slice = 5
            plt.subplot(1, 2, 1)
            plt.imshow(gt_data['reconstruction_rss'][slice][0], cmap='gray')
            plt.colorbar()
            plt.title("Reconstruction")
            plt.subplot(1, 2, 2)
            plt.imshow(np.abs(recon_data['reconstruction'][slice][0]), cmap='gray')
            plt.colorbar()
            plt.title("Ground truth")
            plt.show()
    return


def quantitively_evaluate_recons(datapath, reconpath, dataset="FastMRI"):
    # loop through files in testmap
    ssim_scores = []
    nrmse_scores = []
    psnr_scores = []
    for root, _, files in os.walk(reconpath):
        for file in files:
            # load in the data
            print(file)
            gt_data = h5py.File(datapath + file)
            recon_data = h5py.File(reconpath + file)
            print(gt_data.keys())
            print(recon_data.keys())
            # get the recon and ground truth image
            rec = np.squeeze(np.abs(recon_data['reconstruction']))
            kspaces = gt_data['kspace']
            sensitivity_maps = gt_data['sensitivity_map']
            num_slices = rec.shape[0]
            subject_ssim = []
            subject_nmse = []

            # calculate nrmse and ssim per slice
            for slice in range(num_slices):
                _, x, y = rec.shape
                recon_slice = rec[slice]
                recon_slice = recon_slice / np.mean(recon_slice)
                gt_slice = construct_gt(kspaces[slice], sensitivity_maps[slice], shift=True)
                gt_slice = gt_slice / np.mean(gt_slice)
                if dataset == "FastMRI":
                    gt_slice = crop_image(gt_slice, [320, 320])
                gt_slice = np.expand_dims(gt_slice, 0)
                recon_slice = np.expand_dims(recon_slice, 0)

                # Calculate quantitative scores of reconstruction
                sim_score = ssim(gt_slice, recon_slice)
                error_score = nmse(gt_slice[0], recon_slice[0])

                subject_ssim.append(sim_score)
                subject_nmse.append(error_score)

                print("SSIM", sim_score)
                plt.subplot(1, 2, 1)
                gt = np.abs(gt_data['reconstruction_rss'][slice][0])
                plt.imshow(np.rot90(gt, k=-1), cmap='gray')
                plt.axis('off')
                # Add file name to title
                file_name = os.path.splitext(file)[0]
                plt.title(f"Ground Truth\nFile: {file_name}")

                plt.subplot(1, 2, 2)
                recon = np.squeeze(np.abs(recon_data['reconstruction'][slice]))
                plt.imshow(np.rot90(recon, k=-1), cmap='gray')
                plt.axis('off')
                # Add file name to title
                plt.title(f"Reconstruction SSIM: {np.round(sim_score, 2)}\nFile: {file_name}")

                plt.show()



            # print scores of subject
            print("SSIM mean subject:", np.round(np.mean(subject_ssim), 3))
            print("NMSE mean subject:", np.round(np.mean(subject_nmse), 4))
            print("PSNR mean subject:", np.round(np.mean(subject_psnr), 1))

            ssim_scores += subject_ssim
            nrmse_scores += subject_nmse
            psnr_scores += subject_psnr

    print("SSIM mean all subjects:", np.mean(ssim_scores))
    print("NMSE mean all subjects:", np.mean(nrmse_scores))
    print("PSNR mean all subjects:", np.mean(psnr_scores))
    return ssim_scores



def plot_recons_next_to_each_other(path1, path2):
    for root, _, files in os.walk(path1):
        for file in files:
            slice = 22
            data1 = np.abs(h5py.File(path1 + file)['reconstruction'][slice][0])
            data2 = np.abs(h5py.File(path2 + file)['reconstruction'][slice][0])

            # Plot magnitude, angle, real and imaginary component
            fig, axes = plt.subplots(2, 1)
            axes[0, 0].imshow(data1, cmap='gray')
            axes[0, 0].set_title("30")
            axes[0, 0].axis('off')

            axes[1, 0].imshow(data2, cmap='gray')
            axes[1, 0].set_title("10")
            axes[1, 0].axis('off')

            fig.suptitle("Recons", fontsize=30)
            plt.show()
    return


if __name__ == "__main__":
    # path1 = '/home/dmvandenberg/scratch/CIRIM_models_FastMRI/1066_slices_2x/default/TestFastMRI/reconstructions/'
    # path2 = '/home/dmvandenberg/scratch/CIRIM_models_FastMRI/run3_10subjects_2x/default/Test_fastMRI/reconstructions/'
    # path3 = '/home/dmvandenberg/scratch/CIRIM_models_FastMRI/718_slices_2x/default/TestFastMRI/reconstructions/'
    # path4 = '/scratch/dmvandenberg/FastMRI data/2coil_data/multicoil_test/'
    # plot_recons_next_to_each_other(path1, path2, path3, path4)

    # Path to ground truths
    data_path = '/scratch/helaajati/val/'
    # Path to reconstructions
    # recon_path = '/scratch/dmvandenberg/CIRIM_models_Esaote/Eerste_experiment/Spine test/default/2023-11-26_20-34-31/reconstructions/'
    recon_path = "/scratch/helaajati/Saved_models1/default/2023-12-07_12-16-57/reconstructions/"
    #get SSIM / NRMSE / PSNR
    ssim_list = quantitively_evaluate_recons(data_path, recon_path, dataset='Esaote')




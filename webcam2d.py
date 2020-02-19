import time

import skimage.transform
from PIL import Image
import cv2
import tensorflow as tf
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork
from utils.general import detect_keypoints, trafo_coords, plot_hand, plot_hand_3d



def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

    im_width, im_height = (cap.get(3), cap.get(4))

    # cv2.namedWindow('Hand Detection', cv2.WINDOW_NORMAL)

    image_tf = tf.placeholder(tf.float32, shape=(1, 240, 320, 3))
    hand_side_tf = tf.constant([[1.0, 0.0]])  # left hand (true for all samples provided)
    evaluation = tf.placeholder_with_default(True, shape=())

    # build network
    net = ColorHandPose3DNetwork()
    keypoints_scoremap, _, scale_crop, center = net.inference2d(image_tf)

    # Start TF
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    NUM_CORES = 6  # Choose how many cores to use.
    sess = tf.Session(
        config=tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES,
                              intra_op_parallelism_threads=NUM_CORES))

    # initialize network
    net.init(sess, weight_files=['./weights/handsegnet-rhd.pickle',
                                 './weights/posenet-rhd-stb.pickle'], exclude_var_list=['PosePrior', 'ViewpointNet'])

    while True:
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        ret, frame = cap.read()
        image_np = cv2.flip(frame, 1)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        # cv2.imshow('Single-Hand Detection',
        #            cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

        # if cv2.waitKey(25) & 0xFF == ord('a'):
        if True:  # lol kek
            print('webcam2d')
            image_raw = image_np  # scipy.misc.imread(image_np)
            image_raw = scipy.misc.imresize(image_raw, (240, 320))
            image_v = np.expand_dims((image_raw.astype('float') / 255.0) - 0.5, 0)

            # im = Image.fromarray(image_raw)
            # im.save(f'results/testwebcam.png')

            s_time = time.time()
            keypoints_scoremap_v, \
            scale_crop_v, center_v = sess.run([keypoints_scoremap, scale_crop, center], feed_dict={image_tf: image_v})
            print(f'TIME: {time.time()-s_time}')

            keypoints_scoremap_v = np.squeeze(keypoints_scoremap_v)
            # kp_uv21_gt = np.squeeze(kp_uv21_gt)
            # kp_vis = np.squeeze(kp_vis)

            # detect keypoints
            coord_hw_pred_crop = detect_keypoints(np.squeeze(keypoints_scoremap_v))
            coord_hw_pred = trafo_coords(coord_hw_pred_crop, center_v, scale_crop_v, 256)
            coord_uv_pred = np.stack([coord_hw_pred[:, 1], coord_hw_pred[:, 0]], 1)

            # visualize
            fig = plt.figure(1)
            ax1 = fig.add_subplot(221)
            # ax2 = fig.add_subplot(222)
            # ax3 = fig.add_subplot(223)
            # ax4 = fig.add_subplot(224, projection='3d')
            ax1.imshow(image_raw)
            plot_hand(coord_hw_pred, ax1)
            # ax2.imshow(image_crop_v)
            # plot_hand(coord_hw_pred_crop, ax2)
            # ax3.imshow(np.argmax(hand_scoremap_v, 2))
            # plot_hand_3d(keypoint_coord3d_v, ax4)

            # ax4.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view
            # ax4.set_xlim([-3, 3])
            # ax4.set_ylim([-3, 1])
            # ax4.set_zlim([-3, 3])
            plt.savefig('results/plot2d.png')

            # plt.show()
            # time.sleep(10)
            fig.canvas.draw()

            # convert canvas to image
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                                sep='')
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            # img is rgb, convert to opencv's default bgr
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # display image with opencv or any operation you like
            cv2.imshow("plot", img)

            plt.clf()

            # cv2.destroyAllWindows()
            # break

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()

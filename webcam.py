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
    hand_scoremap_tf, image_crop_tf, scale_tf, center_tf, \
    keypoints_scoremap_tf, keypoint_coord3d_tf = net.inference(image_tf, hand_side_tf, evaluation)

    # Start TF
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    NUM_CORES = 6  # Choose how many cores to use.
    sess = tf.Session(
        config=tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES,
                              intra_op_parallelism_threads=NUM_CORES))

    # initialize network
    net.init(sess)

    while True:
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        ret, frame = cap.read()
        image_np = cv2.flip(frame, 1)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        # cv2.imshow('Single-Hand Detection',
        #            cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

        # if cv2.waitKey(25) & 0xFF == ord('a'):
        if True:  # lol kek
            print('screenshot')
            image_raw = image_np  # scipy.misc.imread(image_np)
            image_raw = scipy.misc.imresize(image_raw, (240, 320))
            image_v = np.expand_dims((image_raw.astype('float') / 255.0) - 0.5, 0)

            im = Image.fromarray(image_raw)
            # im.save(f'results/testwebcam.png')

            s_time = time.time()
            hand_scoremap_v, image_crop_v, scale_v, center_v, \
            keypoints_scoremap_v, keypoint_coord3d_v = sess.run([hand_scoremap_tf, image_crop_tf, scale_tf, center_tf,
                                                                 keypoints_scoremap_tf, keypoint_coord3d_tf],
                                                                feed_dict={image_tf: image_v})
            print(f'time: {time.time()-s_time}')

            hand_scoremap_v = np.squeeze(hand_scoremap_v)
            image_crop_v = np.squeeze(image_crop_v)
            keypoints_scoremap_v = np.squeeze(keypoints_scoremap_v)
            keypoint_coord3d_v = np.squeeze(keypoint_coord3d_v)


            # post processing
            image_crop_v = ((image_crop_v + 0.5) * 255).astype('uint8')
            coord_hw_crop = detect_keypoints(np.squeeze(keypoints_scoremap_v))
            coord_hw = trafo_coords(coord_hw_crop, center_v, scale_v, 256)

            # # visualize
            # fig = plt.figure(1)
            # ax1 = fig.add_subplot(221)
            # ax2 = fig.add_subplot(222)
            # ax3 = fig.add_subplot(223)
            # # ax4 = fig.add_subplot(224, projection='3d')
            # ax1.imshow(image_raw)
            # plot_hand(coord_hw, ax1)
            # ax2.imshow(image_crop_v)
            # plot_hand(coord_hw_crop, ax2)
            # ax3.imshow(np.argmax(hand_scoremap_v, 2))
            # # plot_hand_3d(keypoint_coord3d_v, ax4)
            #
            # # ax4.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view
            # # ax4.set_xlim([-3, 3])
            # # ax4.set_ylim([-3, 1])
            # # ax4.set_zlim([-3, 3])
            # plt.savefig('results/plot.png')
            # # plt.show()
            #
            # # redraw the canvas
            # fig.canvas.draw()
            #
            # # convert canvas to image
            # img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
            #                     sep='')
            # img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            #
            # # img is rgb, convert to opencv's default bgr
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            #
            # # display image with opencv or any operation you like
            # cv2.imshow("plot", img)
            opencv_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            cv2.imshow("plot", opencv_image)

            # plt.clf()

            # time.sleep(10)

            # cv2.destroyAllWindows()
            # break

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()

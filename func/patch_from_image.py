import cv2
import numpy as np

def patch_from_image(im, wh, step):
    def sliding_window(image, stepSize, windowSize):
        for y in xrange(0, image.shape[0], stepSize):
            for x in xrange(0, image.shape[1], stepSize):
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
    def pyramid(image, scale=1.5, minSize=(50, 50)):
        yield image

    im = cv2.resize(im, (1000, np.int32(np.ceil(1000. / im.shape[1] * im.shape[0]))))
    """ crop """
    _, im_bw = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    im_bw_dilate = cv2.dilate(im_bw, np.ones((3, 3), np.uint8), iterations=1)
    erosion = cv2.erode(im_bw_dilate, np.ones((3, 3), np.uint8), iterations=1)
    image_back = erosion
    image_back = 255 - image_back
    if np.sum(image_back) < 10000:
        return []

    mean_0 = np.nonzero(np.mean(image_back, axis=0))
    mean_1 = np.nonzero(np.mean(image_back, axis=1))
    x0, y0, x1, y1 = mean_0[0][0], mean_1[0][0], mean_0[0][-1], mean_1[0][-1]
    crop_image = im[y0 : y1, x0 : x1 + 1]
    if crop_image.shape[1] < 300:
        crop_image = np.hstack((crop_image, crop_image, crop_image))

    _, im_bw = cv2.threshold(crop_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    im_bw_dilate = cv2.dilate(im_bw, np.ones((3, 3), np.uint8), iterations=1)
    erosion = cv2.erode(im_bw_dilate, np.ones((3, 3), np.uint8), iterations=1)
    image_back = erosion
    image_back = 255 - image_back
    if np.sum(image_back) < 10000:
        return []

    def myfunc(stepSize, winW, winH, imts):
        for resized in pyramid(crop_image, scale=1.25):
            for (x, y, window) in sliding_window(resized, stepSize=stepSize, windowSize=(winW, winH)):
                if window.shape[0] != winH or window.shape[1] != winW:
                    continue
                crop_output = resized[y : y + winH, x : x + winW]
                crop_output = cv2.resize(crop_output, (224, 224))

                _, im_bw = cv2.threshold(crop_output, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                im_bw_dilate = cv2.dilate(im_bw, np.ones((3, 3), np.uint8), iterations=1)
                erosion = cv2.erode(im_bw_dilate, np.ones((3, 3), np.uint8), iterations=1)
                image_back = erosion
                image_back = 255 - image_back

                _, crop_output = cv2.threshold(crop_output, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                if np.sum(image_back) < 10000:
                    continue
                imts.append(np.atleast_3d(crop_output))
        return imts
    imts = []
    imts = myfunc(step, wh, wh, imts)
    imts = np.float32(imts)
    return imts

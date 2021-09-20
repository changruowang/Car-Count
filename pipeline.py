'''
Author: your name
Date: 2021-09-19 13:08:20
LastEditTime: 2021-09-20 11:23:05
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: \car_conut\pipeline.py
'''
import os
import logging
import csv

import numpy as np
import cv2

import utils


DIVIDER_COLOUR = (255, 255, 0)
BOUNDING_BOX_COLOUR = (255, 0, 0)
CENTROID_COLOUR = (0, 0, 255)
CAR_COLOURS = [(0, 0, 255)]
EXIT_COLOR = (66, 183, 42)


class PipelineRunner(object):
    '''
        Very simple pipline.
        Just run passed processors in order with passing context from one to 
        another.
        You can also set log level for processors.
    '''

    def __init__(self, pipeline=None, log_level=logging.DEBUG):
        self.pipeline = pipeline or []
        self.context = {}
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(log_level)
        self.log_level = log_level
        self.set_log_level()

    def set_context(self, data):
        self.context = data

    def add(self, processor):
        if not isinstance(processor, PipelineProcessor):
            raise Exception(
                'Processor should be an isinstance of PipelineProcessor.')
        processor.log.setLevel(self.log_level)
        self.pipeline.append(processor)

    def remove(self, name):
        for i, p in enumerate(self.pipeline):
            if p.__class__.__name__ == name:
                del self.pipeline[i]
                return True
        return False

    def set_log_level(self):
        for p in self.pipeline:
            p.log.setLevel(self.log_level)

    def run(self):
        for p in self.pipeline:
            self.context = p(self.context)

        self.log.debug("Frame #%d processed.", self.context['frame_number'])

        return self.context


class PipelineProcessor(object):
    '''
        Base class for processors.
    '''

    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)


class CascadeDetection(PipelineProcessor):
    def __init__ (self,save_image=False,image_dir='images'):
        super(CascadeDetection, self).__init__()
        self.save_image = save_image
        self.image_dir = image_dir
        self.major = cv2.__version__.split('.')[0]

    def car_detect_demo (self,image):
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) 
        car_detector = cv2.CascadeClassifier(r"./cascade05022.xml")  
        cars = car_detector.detectMultiScale(gray,1.1,2)   

        matches = []
        for pos in cars:
            x, y, w, h = (pos[0], pos[1], pos[2], pos[3])
            centroid = utils.get_centroid(x, y, w, h)
            matches.append(((x, y, w, h), centroid))
 
        return matches
   
    def __call__(self,context):
        frame = context['frame'].copy()
        frame_number = context['frame_number']
        cap = self.car_detect_demo(frame)
        
        context['objects'] = cap

        return context

class ContourDetection(PipelineProcessor):
    '''
        Detecting moving objects.
        Purpose of this processor is to subtrac background, get moving objects
        and detect them with a cv2.findContours method, and then filter off-by
        width and height. 
        bg_subtractor - background subtractor isinstance.
        min_contour_width - min bounding rectangle width.
        min_contour_height - min bounding rectangle height.
        save_image - if True will save detected objects mask to file.
        image_dir - where to save images(must exist).        
    '''

    def __init__(self, bg_subtractor, min_contour_width=35, min_contour_height=35, save_image=False, image_dir='images'):
        super(ContourDetection, self).__init__()

        self.bg_subtractor = bg_subtractor
        self.min_contour_width = min_contour_width
        self.min_contour_height = min_contour_height
        self.save_image = save_image
        self.image_dir = image_dir
        self.major = cv2.__version__.split('.')[0]

    def filter_mask(self, img, a=None):
        '''
            This filters are hand-picked just based on visual tests
        '''

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

        # Fill any small holes
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        # Remove noise
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

        # Dilate to merge adjacent blobs
        dilation = cv2.dilate(opening, kernel, iterations=2)

        return dilation

    def detect_vehicles(self, fg_mask, context):

        matches = []

        # finding external contours
        if self.major == '3':
            _, contours, _ = cv2.findContours(
                fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        else:
            contours, _ = cv2.findContours(
                fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

        for (i, contour) in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(contour)
            contour_valid = (w >= self.min_contour_width) and (
                h >= self.min_contour_height)

            if not contour_valid:
                continue

            centroid = utils.get_centroid(x, y, w, h)

            matches.append(((x, y, w, h), centroid))

        return matches

    def __call__(self, context):
        frame = context['frame'].copy()
        frame_number = context['frame_number']

        fg_mask = self.bg_subtractor.apply(frame, None, 0.001)
        # just thresholding values
        fg_mask[fg_mask < 240] = 0
        fg_mask = self.filter_mask(fg_mask, frame_number)

        if self.save_image:
            utils.save_frame(fg_mask, self.image_dir +
                             "/mask_%04d.png" % frame_number, flip=False)
        # else:
        #     cv2.imshow("fg mask", fg_mask)
        #     cv2.imshow("fg mask", fg_mask)
        context['objects'] = self.detect_vehicles(fg_mask, context)
        context['fg_mask'] = fg_mask

        return context


class VehicleCounter(PipelineProcessor):
    '''
        Counting vehicles that entered in exit zone.
        Purpose of this class based on detected object and local cache create
        objects pathes and count that entered in exit zone defined by exit masks.
        exit_masks - list of the exit masks.
        path_size - max number of points in a path.
        max_dst - max distance between two points.
        1. 轨迹维护的思路：就是最近临，将当前帧所有检测的点 和 历史轨迹去比较，将距离最近 且 满足距离阈值的点和轨迹匹配
            距离的计算方法，根据前两次的历史位置预测当前的位置，然后和检测的位置计算距离
        2. 计数的方法：遍历所有轨迹，取最近两次的位置，上一次位置在检测区外，这一次的位置在检测区内，就计数。

    '''

    def __init__(self, exit_masks=[], path_size=10, max_dst=30, x_weight=1.0, y_weight=1.0):
        super(VehicleCounter, self).__init__()

        self.exit_masks = exit_masks

        self.vehicle_count = 0
        self.path_size = path_size
        self.pathes = []
        self.max_dst = max_dst
        self.x_weight = x_weight
        self.y_weight = y_weight

    def check_exit(self, point):
        for exit_mask in self.exit_masks:
            try:
                if exit_mask[point[1]][point[0]] == 255:
                    return True
            except:
                return True
        return False

    def __call__(self, context):
        objects = context['objects']
        context['exit_masks'] = self.exit_masks
        context['pathes'] = self.pathes
        context['vehicle_count'] = self.vehicle_count
        if not objects:
            return context

        points = np.array(objects)[:, 0:2]
        points = points.tolist()
        # 系统会维护一个轨迹列表 self.pathes 是一个三维列表  [ [[(x,y,w,h),(cx,cy)],[x,y],[x,y]],  [[],[],[]] ... ]  保存N条轨迹，每个轨迹又是由N个点组成
        # 初始 一个轨迹就一个点
        if not self.pathes:
            for match in points:
                self.pathes.append([match])
        # 系统里已经存在轨迹后
        else:
            # link new points with old pathes based on minimum distance between
            # points
            new_pathes = []
            # 首先遍历 每个轨迹 path_i
            for path in self.pathes:
                _min = 999999
                _match = None

                # 然后取当前帧检测得到的所有车辆的位置p，找距离轨迹 path_i 最近的 点p
                for p in points:
                    if len(path) == 1: 
                        # 如果历史轨迹中只有一个点，直接计算和上一个点的距离 d
                        d = utils.distance(p[0], path[-1][0])
                    else:
                        # 如果历史轨迹中由两个以上的点，根据历史前两次的位置预测当前帧应该存在的位置 并 和检测的p计算距离
                        # distance from predicted next point to current
                        xn = 2 * path[-1][0][0] - path[-2][0][0]
                        yn = 2 * path[-1][0][1] - path[-2][0][1]
                        d = utils.distance(
                            p[0], (xn, yn),
                            x_weight=self.x_weight,
                            y_weight=self.y_weight
                        )
                    # 如果 最近的距离 d 小于阈值 就将p 添加到轨迹 path_i 中
                    if d < _min:
                        _min = d
                        _match = p
                # 如果 点 p 已经添加到某条轨迹中 就要把该点删除，如果点没有添加到轨迹中，那就以他为起点重新开始一条轨迹
                if _match and _min <= self.max_dst:
                    points.remove(_match)
                    path.append(_match)
                    new_pathes.append(path)

                # 如果历史轨迹 没有个 当前检测的任何一个点关联上，说明此时出现了中断，但是不舍弃这个轨迹，后续检测还可以和他关联
                if _match is None:
                    new_pathes.append(path)

            self.pathes = new_pathes

            # 这几行代码就是 剩下的 没和轨迹关联上的点添加 作为新的轨迹起始。但是处在检测区域的点不会被加入新轨迹。
            if len(points):
                for p in points:
                    # do not add points that already should be counted
                    if self.check_exit(p[1]):
                        continue
                    self.pathes.append([p])

        # 只保存前 N 个历史轨迹
        for i, _ in enumerate(self.pathes):
            self.pathes[i] = self.pathes[i][self.path_size * -1:]

        # count vehicles and drop counted pathes:
        new_pathes = []
        for i, path in enumerate(self.pathes):
            d = path[-2:]

            if (
                # 上一次位置在检测区外，这一次的位置在检测区内，就计数
                len(d) >= 2 and
                # prev point not in exit zone
                not self.check_exit(d[0][1]) and
                # current point in exit zone
                self.check_exit(d[1][1]) and
                # path len is bigger then min
                self.path_size <= len(path)
            ):
                self.vehicle_count += 1
            else:
                # prevent linking with path that already in exit zone
                add = True
                for p in path:
                    if self.check_exit(p[1]):
                        add = False
                        break
                if add:
                    new_pathes.append(path)

        self.pathes = new_pathes

        context['pathes'] = self.pathes
        context['objects'] = objects
        context['vehicle_count'] = self.vehicle_count

        self.log.debug('#VEHICLES FOUND: %s' % self.vehicle_count)

        return context


class CsvWriter(PipelineProcessor):

    def __init__(self, path, name, start_time=0, fps=15):
        super(CsvWriter, self).__init__()

        self.fp = open(os.path.join(path, name), 'w')
        self.writer = csv.DictWriter(self.fp, fieldnames=['time', 'vehicles'])
        self.writer.writeheader()
        self.start_time = start_time
        self.fps = fps
        self.path = path
        self.name = name
        self.prev = None

    def __call__(self, context):
        frame_number = context['frame_number']
        count = _count = context['vehicle_count']

        if self.prev:
            _count = count - self.prev

        time = ((self.start_time + int(frame_number / self.fps)) * 100 
                + int(100.0 / self.fps) * (frame_number % self.fps))
        self.writer.writerow({'time': time, 'vehicles': _count})
        self.prev = count

        return context


class Visualizer(PipelineProcessor):

    def __init__(self, save_image=True, image_dir='images'):
        super(Visualizer, self).__init__()

        self.save_image = save_image
        self.image_dir = image_dir

    def check_exit(self, point, exit_masks=[]):
        for exit_mask in exit_masks:
            if exit_mask[point[1]][point[0]] == 255:
                return True
        return False

    def draw_pathes(self, img, pathes):
        if not img.any():
            return

        for i, path in enumerate(pathes):
            path = np.array(path)[:, 1].tolist()
            for point in path:
                cv2.circle(img, point, 2, CAR_COLOURS[0], -1)
                cv2.polylines(img, [np.int32(path)], False, CAR_COLOURS[0], 1)

        return img

    def draw_boxes(self, img, pathes, exit_masks=[]):
        for (i, match) in enumerate(pathes):

            contour, centroid = match[-1][:2]
            if self.check_exit(centroid, exit_masks):
                continue

            x, y, w, h = contour

            cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1),
                          BOUNDING_BOX_COLOUR, 1)
            cv2.circle(img, centroid, 2, CENTROID_COLOUR, -1)

        return img

    def draw_ui(self, img, vehicle_count, exit_masks=[]):

        # this just add green mask with opacity to the image
        for exit_mask in exit_masks:
            _img = np.zeros(img.shape, img.dtype)
            _img[:, :] = EXIT_COLOR
            mask = cv2.bitwise_and(_img, _img, mask=exit_mask)
            cv2.addWeighted(mask, 1, img, 1, 0, img)

        # drawing top block with counts
        cv2.rectangle(img, (0, 0), (img.shape[1], 50), (0, 0, 0), cv2.FILLED)
        cv2.putText(img, ("Vehicles passed: {total} ".format(total=vehicle_count)), (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        return img

    def __call__(self, context):
        frame = context['frame'].copy()
        frame_number = context['frame_number']
        pathes = context['pathes']
        exit_masks = context['exit_masks']
        vehicle_count = context['vehicle_count']

        frame = self.draw_ui(frame, vehicle_count, exit_masks)
        frame = self.draw_pathes(frame, pathes)
        frame = self.draw_boxes(frame, pathes, exit_masks)

        utils.save_frame(frame, self.image_dir +
                         "/processed_%04d.png" % frame_number)

        context['frame_draw'] = frame 
        return context



# class VehicleCounter(PipelineProcessor):
#     '''
#         Counting vehicles that entered in exit zone.
#         Purpose of this class based on detected object and local cache create
#         objects pathes and count that entered in exit zone defined by exit masks.
#         exit_masks - list of the exit masks.
#         path_size - max number of points in a path.
#         max_dst - max distance between two points.
#     '''

#     def __init__(self, exit_masks=[], path_size=10, max_dst=30, x_weight=1.0, y_weight=1.0):
#         super(VehicleCounter, self).__init__()

#         self.exit_masks = exit_masks

#         self.vehicle_count = 0
#         self.path_size = path_size
#         self.pathes = []
#         self.max_dst = max_dst
#         self.x_weight = x_weight
#         self.y_weight = y_weight

#     def check_exit(self, point):
#         for exit_mask in self.exit_masks:
#             try:
#                 if exit_mask[point[1]][point[0]] == 255:
#                     return True
#             except:
#                 return True
#         return False

#     def __call__(self, context):
#         objects = context['objects']
#         context['exit_masks'] = self.exit_masks
#         context['pathes'] = self.pathes
#         context['vehicle_count'] = self.vehicle_count
#         if not objects:
#             return context

#         points = np.array(objects)[:, 0:2]
#         points = points.tolist()

#         # add new points if pathes is empty
#         if not self.pathes:
#             for match in points:
#                 self.pathes.append([match])

#         else:
#             # link new points with old pathes based on minimum distance between
#             # points
#             new_pathes = []

#             for path in self.pathes:
#                 _min = 999999
#                 _match = None
#                 for p in points:
#                     if len(path) == 1:
#                         # distance from last point to current
#                         d = utils.distance(p[0], path[-1][0])
#                     else:
#                         # based on 2 prev points predict next point and calculate
#                         # distance from predicted next point to current
#                         xn = 2 * path[-1][0][0] - path[-2][0][0]
#                         yn = 2 * path[-1][0][1] - path[-2][0][1]
#                         d = utils.distance(
#                             p[0], (xn, yn),
#                             x_weight=self.x_weight,
#                             y_weight=self.y_weight
#                         )

#                     if d < _min:
#                         _min = d
#                         _match = p

#                 if _match and _min <= self.max_dst:
#                     points.remove(_match)
#                     path.append(_match)
#                     new_pathes.append(path)

#                 # do not drop path if current frame has no matches
#                 if _match is None:
#                     new_pathes.append(path)

#             self.pathes = new_pathes

#             # add new pathes
#             if len(points):
#                 for p in points:
#                     # do not add points that already should be counted
#                     if self.check_exit(p[1]):
#                         continue
#                     self.pathes.append([p])

#         # save only last N points in path
#         for i, _ in enumerate(self.pathes):
#             self.pathes[i] = self.pathes[i][self.path_size * -1:]

#         # count vehicles and drop counted pathes:
#         new_pathes = []
#         for i, path in enumerate(self.pathes):
#             d = path[-2:]

#             if (
#                 # need at list two points to count
#                 len(d) >= 2 and
#                 # prev point not in exit zone
#                 not self.check_exit(d[0][1]) and
#                 # current point in exit zone
#                 self.check_exit(d[1][1]) and
#                 # path len is bigger then min
#                 self.path_size <= len(path)
#             ):
#                 self.vehicle_count += 1
#             else:
#                 # prevent linking with path that already in exit zone
#                 add = True
#                 for p in path:
#                     if self.check_exit(p[1]):
#                         add = False
#                         break
#                 if add:
#                     new_pathes.append(path)

#         self.pathes = new_pathes

#         context['pathes'] = self.pathes
#         context['objects'] = objects
#         context['vehicle_count'] = self.vehicle_count

#         self.log.debug('#VEHICLES FOUND: %s' % self.vehicle_count)

#         return context
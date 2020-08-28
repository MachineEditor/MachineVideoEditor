import numpy as np
import cv2

class IEPolysPoints:
    def __init__(self, IEPolys_parent, type):
        self.parent = IEPolys_parent
        self.type = type
        self.points = np.empty( (0,2), dtype=np.int32 )
        self.n_max = self.n = 0

    def add(self,x,y):
        self.points = np.append(self.points[0:self.n], [ (x,y) ], axis=0)
        self.n_max = self.n = self.n + 1
        self.parent.dirty = True

    def n_dec(self):
        self.n = max(0, self.n-1)
        self.parent.dirty = True
        return self.n

    def n_inc(self):
        self.n = min(len(self.points), self.n+1)
        self.parent.dirty = True
        return self.n

    def n_clip(self):
        self.points = self.points[0:self.n]
        self.n_max = self.n

    def cur_point(self):
        return self.points[self.n-1]

    def points_to_n(self):
        return self.points[0:self.n]

    def set_points(self, points):
        self.points = np.array(points)
        self.n_max = self.n = len(points)
        self.parent.dirty = True

class IEPolys:
    def __init__(self):
        self.list = []
        self.n_max = self.n = 0
        self.dirty = True

    def add(self, type):
        self.list = self.list[0:self.n]
        self.list.append ( IEPolysPoints(self, type) )
        self.n_max = self.n = self.n + 1
        self.dirty = True

    def n_dec(self):
        self.n = max(0, self.n-1)
        self.dirty = True
        return self.n
        
    def n_inc(self):
        self.n = min(len(self.list), self.n+1)
        self.dirty = True
        return self.n

    def n_list(self):
        return self.list[self.n-1]

    def n_clip(self):
        self.list = self.list[0:self.n]
        self.n_max = self.n
        if self.n > 0:
            self.list[-1].n_clip()

    def __iter__(self):
        for n in range(self.n):
            yield self.list[n]

    def switch_dirty(self):
        d = self.dirty
        self.dirty = False
        return d

    def overlay_mask(self, mask):
        h,w,c = mask.shape
        white = (1,)*c
        black = (0,)*c
        for n in range(self.n):
            poly = self.list[n]
            if poly.n > 0:
                cv2.fillPoly(mask, [poly.points_to_n()], white if poly.type == 1 else black )

    def dump(self):
        result = []
        for n in range(self.n):
            l = self.list[n]
            result += [ (l.type, l.points_to_n().tolist() ) ]
        return result

    @staticmethod
    def load(ie_polys=None):
        obj = IEPolys()
        if ie_polys is not None:
            for (type, points) in ie_polys:
                obj.add(type)
                obj.n_list().set_points(points)
        return obj
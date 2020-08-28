import numpy as np
import cv2
from enum import IntEnum


class SegIEPolyType(IntEnum):
    EXCLUDE = 0
    INCLUDE = 1
    


class SegIEPoly():
    def __init__(self, type=None, pts=None, **kwargs):
        self.type = type
        
        if pts is None:
            pts = np.empty( (0,2), dtype=np.float32 )
        else:
            pts = np.float32(pts)
        self.pts = pts
        self.n_max = self.n = len(pts)

    def dump(self):
        return {'type': int(self.type),
                'pts' : self.get_pts(),
               }
    
    def identical(self, b):
        if self.n != b.n:
            return False            
        return (self.pts[0:self.n] == b.pts[0:b.n]).all()        
        
    def get_type(self):
        return self.type

    def add_pt(self, x, y):
        self.pts = np.append(self.pts[0:self.n], [ ( float(x), float(y) ) ], axis=0).astype(np.float32)
        self.n_max = self.n = self.n + 1

    def undo(self):
        self.n = max(0, self.n-1)
        return self.n

    def redo(self):
        self.n = min(len(self.pts), self.n+1)
        return self.n

    def redo_clip(self):
        self.pts = self.pts[0:self.n]
        self.n_max = self.n

    def insert_pt(self, n, pt):
        if n < 0 or n > self.n:
            raise ValueError("insert_pt out of range")
        self.pts = np.concatenate( (self.pts[0:n], pt[None,...].astype(np.float32), self.pts[n:]), axis=0)
        self.n_max = self.n = self.n+1
        
    def remove_pt(self, n):
        if n < 0 or n >= self.n:
            raise ValueError("remove_pt out of range")
        self.pts = np.concatenate( (self.pts[0:n], self.pts[n+1:]), axis=0)
        self.n_max = self.n = self.n-1
        
    def get_last_point(self):
        return self.pts[self.n-1].copy()

    def get_pts(self):
        return self.pts[0:self.n].copy()
        
    def get_pts_count(self):
        return self.n

    def set_point(self, id, pt):
        self.pts[id] = pt
        
    def set_points(self, pts):
        self.pts = np.array(pts)
        self.n_max = self.n = len(pts)
        
    
        

class SegIEPolys():
    def __init__(self):
        self.polys = []

    def identical(self, b):
        polys_len = len(self.polys)
        o_polys_len = len(b.polys)
        if polys_len != o_polys_len:
            return False
        
        return all ([ a_poly.identical(b_poly) for a_poly, b_poly in zip(self.polys, b.polys) ])
        
    def add_poly(self, ie_poly_type):       
        poly = SegIEPoly(ie_poly_type)
        self.polys.append (poly)
        return poly

    def remove_poly(self, poly):
        if poly in self.polys:
            self.polys.remove(poly)

    def has_polys(self):
        return len(self.polys) != 0
        
    def get_poly(self, id):
        return self.polys[id]

    def get_polys(self):
        return self.polys
        
    def get_pts_count(self):
        return sum([poly.get_pts_count() for poly in self.polys])
        
    def sort(self):
        poly_by_type = { SegIEPolyType.EXCLUDE : [], SegIEPolyType.INCLUDE : [] }

        for poly in self.polys:
            poly_by_type[poly.type].append(poly)
            
        self.polys = poly_by_type[SegIEPolyType.INCLUDE] + poly_by_type[SegIEPolyType.EXCLUDE]

    def __iter__(self):
        for poly in self.polys:
            yield poly

    def overlay_mask(self, mask):
        h,w,c = mask.shape
        white = (1,)*c
        black = (0,)*c
        for poly in self.polys:
            pts = poly.get_pts().astype(np.int32)
            if len(pts) != 0:
                cv2.fillPoly(mask, [pts], white if poly.type == SegIEPolyType.INCLUDE else black )

    def dump(self):
        return {'polys' : [ poly.dump() for poly in self.polys ] }

    @staticmethod
    def load(data=None):
        ie_polys = SegIEPolys()
        if data is not None:
            if isinstance(data, list):
                # Backward comp
                ie_polys.polys = [ SegIEPoly(type=type, pts=pts) for (type, pts) in data ]
            elif isinstance(data, dict):                
                ie_polys.polys = [ SegIEPoly(**poly_cfg) for poly_cfg in data['polys'] ]
                
        ie_polys.sort()   
                
        return ie_polys
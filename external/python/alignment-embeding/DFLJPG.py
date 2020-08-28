import pickle
import struct

import cv2
import numpy as np

import FaceType
import SegIEPolys

import zlib

class DFLJPG(object):
    def __init__(self, filename):
        self.filename = filename
        self.data = b""
        self.length = 0
        self.chunks = []
        self.dfl_dict = None
        self.shape = (0,0,0)

    @staticmethod
    def load_raw(filename, loader_func=None):
        try:
            if loader_func is not None:
                data = loader_func(filename)
            else:
                with open(filename, "rb") as f:
                    data = f.read()
        except:
            raise FileNotFoundError(filename)

        try:
            inst = DFLJPG(filename)
            inst.data = data
            inst.length = len(data)
            inst_length = inst.length
            chunks = []
            data_counter = 0
            while data_counter < inst_length:
                chunk_m_l, chunk_m_h = struct.unpack ("BB", data[data_counter:data_counter+2])
                data_counter += 2

                if chunk_m_l != 0xFF:
                    raise ValueError(f"No Valid JPG info in {filename}")

                chunk_name = None
                chunk_size = None
                chunk_data = None
                chunk_ex_data = None
                is_unk_chunk = False

                if chunk_m_h & 0xF0 == 0xD0:
                    n = chunk_m_h & 0x0F

                    if n >= 0 and n <= 7:
                        chunk_name = "RST%d" % (n)
                        chunk_size = 0
                    elif n == 0x8:
                        chunk_name = "SOI"
                        chunk_size = 0
                        if len(chunks) != 0:
                            raise Exception("")
                    elif n == 0x9:
                        chunk_name = "EOI"
                        chunk_size = 0
                    elif n == 0xA:
                        chunk_name = "SOS"
                    elif n == 0xB:
                        chunk_name = "DQT"
                    elif n == 0xD:
                        chunk_name = "DRI"
                        chunk_size = 2
                    else:
                        is_unk_chunk = True
                elif chunk_m_h & 0xF0 == 0xC0:
                    n = chunk_m_h & 0x0F
                    if n == 0:
                        chunk_name = "SOF0"
                    elif n == 2:
                        chunk_name = "SOF2"
                    elif n == 4:
                        chunk_name = "DHT"
                    else:
                        is_unk_chunk = True
                elif chunk_m_h & 0xF0 == 0xE0:
                    n = chunk_m_h & 0x0F
                    chunk_name = "APP%d" % (n)
                else:
                    is_unk_chunk = True

                if chunk_size == None: #variable size
                    chunk_size, = struct.unpack (">H", data[data_counter:data_counter+2])
                    chunk_size -= 2
                    data_counter += 2

                if chunk_size > 0:
                    chunk_data = data[data_counter:data_counter+chunk_size]
                    data_counter += chunk_size

                if chunk_name == "SOS":
                    c = data_counter
                    while c < inst_length and (data[c] != 0xFF or data[c+1] != 0xD9):
                        c += 1

                    chunk_ex_data = data[data_counter:c]
                    data_counter = c

                chunks.append ({'name' : chunk_name,
                                'm_h' : chunk_m_h,
                                'data' : chunk_data,
                                'ex_data' : chunk_ex_data,
                                })
            inst.chunks = chunks

            return inst
        except Exception as e:
            raise Exception (f"Corrupted JPG file {filename} {e}")

    @staticmethod
    def load(filename, loader_func=None):
        try:
            inst = DFLJPG.load_raw (filename, loader_func=loader_func)
            inst.dfl_dict = {}

            for chunk in inst.chunks:
                if chunk['name'] == 'APP0':
                    d, c = chunk['data'], 0
                    c, id, _ = struct_unpack (d, c, "=4sB")

                    if id == b"JFIF":
                        c, ver_major, ver_minor, units, Xdensity, Ydensity, Xthumbnail, Ythumbnail = struct_unpack (d, c, "=BBBHHBB")
                        #if units == 0:
                        #    inst.shape = (Ydensity, Xdensity, 3)
                    else:
                        raise Exception("Unknown jpeg ID: %s" % (id) )
                elif chunk['name'] == 'SOF0' or chunk['name'] == 'SOF2':
                    d, c = chunk['data'], 0
                    c, precision, height, width = struct_unpack (d, c, ">BHH")
                    inst.shape = (height, width, 3)

                elif chunk['name'] == 'APP15':
                    if type(chunk['data']) == bytes:
                        inst.dfl_dict = pickle.loads(chunk['data'])

            return inst
        except Exception as e:
            print (e)
            return None

    def has_data(self):
        return len(self.dfl_dict.keys()) != 0

    def save(self):
        try:
            with open(self.filename, "wb") as f:
                f.write ( self.dump() )
        except:
            raise Exception( f'cannot save {self.filename}' )

    def dump(self):
        data = b""

        dict_data = self.dfl_dict
        
        # Remove None keys
        for key in list(dict_data.keys()):
            if dict_data[key] is None:                
                dict_data.pop(key)

        for chunk in self.chunks:
            if chunk['name'] == 'APP15':
                self.chunks.remove(chunk)
                break

        last_app_chunk = 0
        for i, chunk in enumerate (self.chunks):
            if chunk['m_h'] & 0xF0 == 0xE0:
                last_app_chunk = i

        dflchunk = {'name' : 'APP15',
                    'm_h' : 0xEF,
                    'data' : pickle.dumps(dict_data),
                    'ex_data' : None,
                    }
        self.chunks.insert (last_app_chunk+1, dflchunk)


        for chunk in self.chunks:
            data += struct.pack ("BB", 0xFF, chunk['m_h'] )
            chunk_data = chunk['data']
            if chunk_data is not None:
                data += struct.pack (">H", len(chunk_data)+2 )
                data += chunk_data

            chunk_ex_data = chunk['ex_data']
            if chunk_ex_data is not None:
                data += chunk_ex_data

        return data

    def get_shape(self):
        return self.shape

    def get_height(self):
        for chunk in self.chunks:
            if type(chunk) == IHDR:
                return chunk.height
        return 0

    def get_dict(self):
        return self.dfl_dict

    def set_dict (self, dict_data=None):
        self.dfl_dict = dict_data

    def get_face_type(self):            return self.dfl_dict.get('face_type', FaceType.FaceType.toString (FaceType.FaceType.FULL) )
    def set_face_type(self, face_type): self.dfl_dict['face_type'] = face_type

    def get_landmarks(self):            return np.array ( self.dfl_dict['landmarks'] )
    def set_landmarks(self, landmarks): self.dfl_dict['landmarks'] = landmarks

    def get_eyebrows_expand_mod(self):                      return self.dfl_dict.get ('eyebrows_expand_mod', 1.0)
    def set_eyebrows_expand_mod(self, eyebrows_expand_mod): self.dfl_dict['eyebrows_expand_mod'] = eyebrows_expand_mod

    def get_source_filename(self):                  return self.dfl_dict.get ('source_filename', None)
    def set_source_filename(self, source_filename): self.dfl_dict['source_filename'] = source_filename

    def get_source_rect(self):              return self.dfl_dict.get ('source_rect', None)
    def set_source_rect(self, source_rect): self.dfl_dict['source_rect'] = source_rect

    def get_source_landmarks(self):                     return np.array ( self.dfl_dict.get('source_landmarks', None) )
    def set_source_landmarks(self, source_landmarks):   self.dfl_dict['source_landmarks'] = source_landmarks

    def get_image_to_face_mat(self):
        mat = self.dfl_dict.get ('image_to_face_mat', None)
        if mat is not None:
            return np.array (mat)
        return None
    def set_image_to_face_mat(self, image_to_face_mat):   self.dfl_dict['image_to_face_mat'] = image_to_face_mat

    def get_seg_ie_polys(self): 
        d = self.dfl_dict.get('seg_ie_polys',None)
        if d is not None:
            d = SegIEPolys.SegIEPolys.load(d)
           
        return d
        
    def set_seg_ie_polys(self, seg_ie_polys):
        # if seg_ie_polys is not None:        
        #     if not isinstance(seg_ie_polys, SegIEPolys.SegIEPolys):
        #         raise ValueError('seg_ie_polys should be instance of SegIEPolys')
            
        #     if seg_ie_polys.has_polys():
        #         seg_ie_polys = seg_ie_polys.dump()
        #     else:
        #         seg_ie_polys = None

        # build expected dict
        # print (seg_ie_polys)
        # if seg_ie_polys is None or len(seg_ie_polys) == 0:
        #     return

        if seg_ie_polys is None:
            return

        if hasattr(seg_ie_polys, '__len__') and len(seg_ie_polys) == 0:
            return

        if isinstance(seg_ie_polys, SegIEPolys.SegIEPolys):
            if seg_ie_polys.has_polys():
                seg_ie_polys = seg_ie_polys.dump()
            return

        seg_ser = {'polys': [{'type': int(poly[0]), 'pts': [[float(point[0]), float(point[1])]  for point in poly[1]]} for poly in seg_ie_polys] }
        
        self.dfl_dict['seg_ie_polys'] = seg_ser

    def get_xseg_mask(self): 
        mask_buf = self.dfl_dict.get('xseg_mask',None)
        if mask_buf is None:
            return None

        #return mask_buf   
        img = cv2.imdecode(mask_buf, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 2:
            img = img[...,None]
        # don't have a imdecode/imencode in js
        img = img.astype(np.float32) / 255.0
        # print (img.shape)
        return zlib.compress(img)
        # return img.astype(np.float32) / 255.0
        
        
    def set_xseg_mask(self, mask_a):
        if mask_a is None:
            self.dfl_dict['xseg_mask'] = None
            return
            
        ret, buf = cv2.imencode( '.png', np.clip( mask_a*255, 0, 255 ).astype(np.uint8) )
        if not ret:
            raise Exception("unable to generate PNG data for set_xseg_mask")
        
        self.dfl_dict['xseg_mask'] = buf

    @staticmethod
    def embed_data(filename, face_type=None,
                             landmarks=None,
                             seg_ie_polys=None,
                             source_filename=None,
                             source_rect=None,
                             source_landmarks=None,
                             image_to_face_mat=None,
                             xseg_mask=None,
                             eyebrows_expand_mod=None,
                             **kwargs
                   ):

        inst = DFLJPG.load (filename)
        inst.set_landmarks(landmarks)
        inst.set_eyebrows_expand_mod(eyebrows_expand_mod)
        inst.set_source_filename(source_filename)
        inst.set_source_rect(source_rect)
        inst.set_source_landmarks(source_landmarks)
        inst.set_image_to_face_mat(image_to_face_mat)
        inst.set_face_type(face_type)
        inst.set_seg_ie_polys(seg_ie_polys)
        if xseg_mask is not None:
            inst.set_xseg_mask(xseg_mask)
        # inst.setDFLDictData ({
        #                         'face_type': face_type,
        #                         'landmarks': landmarks,
        #                         'ie_polys' : ie_polys.dump() if ie_polys is not None else None,
        #                         'source_filename': source_filename,
        #                         'source_rect': source_rect,
        #                         'source_landmarks': source_landmarks,
        #                         'image_to_face_mat': image_to_face_mat,
        #                         'fanseg_mask' : fanseg_mask,
        #                         'pitch_yaw_roll' : pitch_yaw_roll,
        #                         'eyebrows_expand_mod' : eyebrows_expand_mod
        #                      })

        inst.save()

    def embed_and_set(self, filename, face_type=None,
                                landmarks=None,
                                seg_ie_polys=None,
                                source_filename=None,
                                source_rect=None,
                                source_landmarks=None,
                                image_to_face_mat=None,
                                xseg_mask=None,
                                eyebrows_expand_mod=None,
                                **kwargs
                    ):
        if face_type is None: face_type = self.get_face_type()
        if landmarks is None: landmarks = self.get_landmarks()
        if seg_ie_polys is None: seg_ie_polys = self.get_seg_ie_polys()
        if source_filename is None: source_filename = self.get_source_filename()
        if source_rect is None: source_rect = self.get_source_rect()
        if source_landmarks is None: source_landmarks = self.get_source_landmarks()
        if image_to_face_mat is None: image_to_face_mat = self.get_image_to_face_mat()
        # if xseg_mask is None: xseg_mask = self.get_xseg_mask() do not init like that since I return a zip archive
        if eyebrows_expand_mod is None: eyebrows_expand_mod = self.get_eyebrows_expand_mod()

        DFLJPG.embed_data (filename, face_type=face_type,
                                     landmarks=landmarks,
                                     seg_ie_polys=seg_ie_polys,
                                     source_filename=source_filename,
                                     source_rect=source_rect,
                                     source_landmarks=source_landmarks,
                                     image_to_face_mat=image_to_face_mat,
                                     xseg_mask=xseg_mask,
                                     eyebrows_expand_mod=eyebrows_expand_mod)

def struct_unpack(data, counter, fmt):
    fmt_size = struct.calcsize(fmt)
    return (counter+fmt_size,) + struct.unpack (fmt, data[counter:counter+fmt_size])




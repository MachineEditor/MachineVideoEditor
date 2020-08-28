import pickle
import string
import struct
import zlib

import cv2
import numpy as np

import FaceType
from IEPolys import IEPolys

PNG_HEADER = b"\x89PNG\r\n\x1a\n"

class Chunk(object):
    def __init__(self, name=None, data=None):
        self.length = 0
        self.crc = 0
        self.name = name if name else "noNe"
        self.data = data if data else b""

    @classmethod
    def load(cls, data):
        """Load a chunk including header and footer"""
        inst = cls()
        if len(data) < 12:
            msg = "Chunk-data too small"
            raise ValueError(msg)

        # chunk header & data
        (inst.length, raw_name) = struct.unpack("!I4s", data[0:8])
        inst.data = data[8:-4]
        inst.verify_length()
        inst.name = raw_name.decode("ascii")
        inst.verify_name()

        # chunk crc
        inst.crc = struct.unpack("!I", data[8+inst.length:8+inst.length+4])[0]
        inst.verify_crc()

        return inst

    def dump(self, auto_crc=True, auto_length=True):
        """Return the chunk including header and footer"""
        if auto_length: self.update_length()
        if auto_crc: self.update_crc()
        self.verify_name()
        return struct.pack("!I", self.length) + self.get_raw_name() + self.data + struct.pack("!I", self.crc)

    def verify_length(self):
        if len(self.data) != self.length:
            msg = "Data length ({}) does not match length in chunk header ({})".format(len(self.data), self.length)
            raise ValueError(msg)
        return True

    def verify_name(self):
        for c in self.name:
            if c not in string.ascii_letters:
                msg = "Invalid character in chunk name: {}".format(repr(self.name))
                raise ValueError(msg)
            return True

    def verify_crc(self):
        calculated_crc = self.get_crc()
        if self.crc != calculated_crc:
            msg = "CRC mismatch: {:08X} (header), {:08X} (calculated)".format(self.crc, calculated_crc)
            raise ValueError(msg)
        return True

    def update_length(self):
        self.length = len(self.data)

    def update_crc(self):
        self.crc = self.get_crc()

    def get_crc(self):
        return zlib.crc32(self.get_raw_name() + self.data)

    def get_raw_name(self):
        return self.name if isinstance(self.name, bytes) else self.name.encode("ascii")

    # name helper methods

    def ancillary(self, set=None):
        """Set and get ancillary=True/critical=False bit"""
        if set is True:
            self.name[0] = self.name[0].lower()
        elif set is False:
            self.name[0] = self.name[0].upper()
        return self.name[0].islower()

    def private(self, set=None):
        """Set and get private=True/public=False bit"""
        if set is True:
            self.name[1] = self.name[1].lower()
        elif set is False:
            self.name[1] = self.name[1].upper()
        return self.name[1].islower()

    def reserved(self, set=None):
        """Set and get reserved_valid=True/invalid=False bit"""
        if set is True:
            self.name[2] = self.name[2].upper()
        elif set is False:
            self.name[2] = self.name[2].lower()
        return self.name[2].isupper()

    def safe_to_copy(self, set=None):
        """Set and get save_to_copy=True/unsafe=False bit"""
        if set is True:
            self.name[3] = self.name[3].lower()
        elif set is False:
            self.name[3] = self.name[3].upper()
        return self.name[3].islower()

    def __str__(self):
        return "<Chunk '{name}' length={length} crc={crc:08X}>".format(**self.__dict__)

class IHDR(Chunk):
	"""IHDR Chunk
	width, height, bit_depth, color_type, compression_method,
	filter_method, interlace_method contain the data extracted
	from the chunk. Modify those and use and build() to recreate
	the chunk. Valid values for bit_depth depend on the color_type
	and can be looked up in color_types or in the PNG specification

	See:
	http://www.libpng.org/pub/png/spec/1.2/PNG-Chunks.html#C.IHDR
	"""
	# color types with name & allowed bit depths
	COLOR_TYPE_GRAY  = 0
	COLOR_TYPE_RGB   = 2
	COLOR_TYPE_PLTE  = 3
	COLOR_TYPE_GRAYA = 4
	COLOR_TYPE_RGBA  = 6
	color_types = {
		COLOR_TYPE_GRAY:	("Grayscale", (1,2,4,8,16)),
		COLOR_TYPE_RGB:		("RGB", (8,16)),
		COLOR_TYPE_PLTE:	("Palette", (1,2,4,8)),
		COLOR_TYPE_GRAYA:	("Greyscale+Alpha", (8,16)),
		COLOR_TYPE_RGBA:	("RGBA", (8,16)),
	}

	def __init__(self, width=0, height=0, bit_depth=8, color_type=2, \
	             compression_method=0, filter_method=0, interlace_method=0):
		self.width = width
		self.height = height
		self.bit_depth = bit_depth
		self.color_type = color_type
		self.compression_method = compression_method
		self.filter_method = filter_method
		self.interlace_method = interlace_method
		super().__init__("IHDR")

	@classmethod
	def load(cls, data):
		inst = super().load(data)
		fields = struct.unpack("!IIBBBBB", inst.data)
		inst.width = fields[0]
		inst.height = fields[1]
		inst.bit_depth = fields[2] # per channel
		inst.color_type = fields[3] # see specs
		inst.compression_method = fields[4] # always 0(=deflate/inflate)
		inst.filter_method = fields[5] # always 0(=adaptive filtering with 5 methods)
		inst.interlace_method = fields[6] # 0(=no interlace) or 1(=Adam7 interlace)
		return inst

	def dump(self):
		self.data = struct.pack("!IIBBBBB", \
			self.width, self.height, self.bit_depth, self.color_type, \
			self.compression_method, self.filter_method, self.interlace_method)
		return super().dump()

	def __str__(self):
		return "<Chunk:IHDR geometry={width}x{height} bit_depth={bit_depth} color_type={}>" \
			.format(self.color_types[self.color_type][0], **self.__dict__)

class IEND(Chunk):
    def __init__(self):
        super().__init__("IEND")

    def dump(self):
        if len(self.data) != 0:
            msg = "IEND has data which is not allowed"
            raise ValueError(msg)
        if self.length != 0:
            msg = "IEND data lenght is not 0 which is not allowed"
            raise ValueError(msg)
        return super().dump()

    def __str__(self):
        return "<Chunk:IEND>".format(**self.__dict__)

class DFLChunk(Chunk):
    def __init__(self, dict_data=None):
        super().__init__("fcWp")
        self.dict_data = dict_data

    def setDictData(self, dict_data):
        self.dict_data = dict_data

    def getDictData(self):
        return self.dict_data

    @classmethod
    def load(cls, data):
        inst = super().load(data)
        inst.dict_data = pickle.loads( inst.data )
        return inst

    def dump(self):
        self.data = pickle.dumps (self.dict_data)
        return super().dump()

chunk_map = {
    b"IHDR": IHDR,
    b"fcWp": DFLChunk,
    b"IEND": IEND
}

class DFLPNG(object):
    def __init__(self):
        self.data = b""
        self.length = 0
        self.chunks = []
        self.dfl_dict = None

    @staticmethod
    def load_raw(filename):
        try:
            with open(filename, "rb") as f:
                data = f.read()
        except:
            raise FileNotFoundError(filename)

        inst = DFLPNG()
        inst.data = data
        inst.length = len(data)

        if data[0:8] != PNG_HEADER:
            msg = "No Valid PNG header"
            raise ValueError(msg)

        chunk_start = 8
        while chunk_start < inst.length:
            (chunk_length, chunk_name) = struct.unpack("!I4s", data[chunk_start:chunk_start+8])
            chunk_end = chunk_start + chunk_length + 12

            chunk = chunk_map.get(chunk_name, Chunk).load(data[chunk_start:chunk_end])
            inst.chunks.append(chunk)
            chunk_start = chunk_end

        return inst

    @staticmethod
    def load(filename):
        try:
            inst = DFLPNG.load_raw (filename)
            inst.dfl_dict = inst.getDFLDictData()

            if inst.dfl_dict is not None:
                if 'face_type' not in inst.dfl_dict:
                    inst.dfl_dict['face_type'] = FaceType.toString (FaceType.FULL)

                if 'fanseg_mask' in inst.dfl_dict:
                    fanseg_mask = inst.dfl_dict['fanseg_mask']
                    if fanseg_mask is not None:
                        numpyarray = np.asarray( inst.dfl_dict['fanseg_mask'], dtype=np.uint8)
                        inst.dfl_dict['fanseg_mask'] = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)

            if inst.dfl_dict == None:
                return None

            return inst
        except Exception as e:
            print(e)
            return None

    @staticmethod
    def embed_data(filename, face_type=None,
                             landmarks=None,
                             ie_polys=None,
                             source_filename=None,
                             source_rect=None,
                             source_landmarks=None,
                             image_to_face_mat=None,
                             fanseg_mask=None,
                             pitch_yaw_roll=None,
                             eyebrows_expand_mod=None,
                             **kwargs
                   ):

        if fanseg_mask is not None:
            fanseg_mask = np.clip ( (fanseg_mask*255).astype(np.uint8), 0, 255 )

            ret, buf = cv2.imencode( '.jpg', fanseg_mask, [int(cv2.IMWRITE_JPEG_QUALITY), 85] )

            if ret and len(buf) < 60000:
                fanseg_mask = buf
            else:
                # io.log_err("Unable to encode fanseg_mask for %s" % (filename) )
                fanseg_mask = None

        inst = DFLPNG.load_raw (filename)
        inst.setDFLDictData ({
                                'face_type': face_type,
                                'landmarks': landmarks,
                                'source_filename': source_filename,
                                'source_rect': source_rect,
                                'source_landmarks': source_landmarks,
                                'image_to_face_mat':image_to_face_mat,
                                'eyebrows_expand_mod' : eyebrows_expand_mod,
                             })

        try:
            with open(filename, "wb") as f:
                f.write ( inst.dump() )
        except:
            raise Exception( 'cannot save %s' % (filename) )

    def embed_and_set(self, filename,   face_type=None,
                                        landmarks=None,
                                        source_filename=None,
                                        source_rect=None,
                                        source_landmarks=None,
                                        image_to_face_mat=None,
                                        eyebrows_expand_mod=None,
                                        seg_ie_polys=None,
                                        **kwargs
                        ):
        if face_type is None: face_type = self.get_face_type()
        if landmarks is None: landmarks = self.get_landmarks()
        if source_filename is None: source_filename = self.get_source_filename()
        if source_rect is None: source_rect = self.get_source_rect()
        if source_landmarks is None: source_landmarks = self.get_source_landmarks()
        if image_to_face_mat is None: image_to_face_mat = self.get_image_to_face_mat()
        if eyebrows_expand_mod is None: eyebrows_expand_mod = self.get_eyebrows_expand_mod()

        DFLPNG.embed_data (filename, face_type=face_type,
                                     landmarks=landmarks,
                                     source_filename=source_filename,
                                     source_rect=source_rect,
                                     source_landmarks=source_landmarks,
                                     image_to_face_mat=image_to_face_mat,
                                     eyebrows_expand_mod=eyebrows_expand_mod)

    def dump(self):
        data = PNG_HEADER
        for chunk in self.chunks:
            data += chunk.dump()
        return data

    def get_shape(self):
        for chunk in self.chunks:
            if type(chunk) == IHDR:
                c = 3 if chunk.color_type == IHDR.COLOR_TYPE_RGB else 4
                w = chunk.width
                h = chunk.height
                return (h,w,c)
        return (0,0,0)

    def get_height(self):
        for chunk in self.chunks:
            if type(chunk) == IHDR:
                return chunk.height
        return 0

    def getDFLDictData(self):
        for chunk in self.chunks:
            if type(chunk) == DFLChunk:
                return chunk.getDictData()
        return None

    def setDFLDictData (self, dict_data=None):
        for chunk in self.chunks:
            if type(chunk) == DFLChunk:
                self.chunks.remove(chunk)
                break

        if not dict_data is None:
            chunk = DFLChunk(dict_data)
            self.chunks.insert(-1, chunk)

    def get_face_type(self): return self.dfl_dict['face_type']
    def get_landmarks(self): return np.array ( self.dfl_dict['landmarks'] )
    def get_source_filename(self): return self.dfl_dict['source_filename']
    def get_source_rect(self): return self.dfl_dict['source_rect']
    def get_source_landmarks(self): return np.array ( self.dfl_dict['source_landmarks'] )
    def get_image_to_face_mat(self):
        mat = self.dfl_dict.get ('image_to_face_mat', None)
        if mat is not None:
            return np.array (mat)
        return None

    def get_eyebrows_expand_mod(self):
        return self.dfl_dict.get ('eyebrows_expand_mod', None)

    def get_seg_ie_polys(self):
        return None

    def get_xseg_mask(self):
        return None

    def __str__(self):
        return "<PNG length={length} chunks={}>".format(len(self.chunks), **self.__dict__)

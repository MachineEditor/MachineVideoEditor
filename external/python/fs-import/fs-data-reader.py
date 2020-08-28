
# Code copied from faceswap\lib\serializer.py

import json
import logging
import os
import pickle
import zlib
import argparse

from io import BytesIO

from zlib import compress, decompress

import numpy as np

class Serializer():
    """ A convenience class for various serializers.

    This class should not be called directly as it acts as the parent for various serializers.
    All serializers should be called from :func:`get_serializer` or
    :func:`get_serializer_from_filename`

    Example
    -------
    >>> from lib.serializer import get_serializer
    >>> serializer = get_serializer('json')
    >>> json_file = '/path/to/json/file.json'
    >>> data = serializer.load(json_file)
    >>> serializer.save(json_file, data)

    """
    def __init__(self):
        self._file_extension = None
        self._write_option = "wb"
        self._read_option = "rb"

    @property
    def file_extension(self):
        """ str: The file extension of the serializer """
        return self._file_extension

    def save(self, filename, data):
        """ Serialize data and save to a file

        Parameters
        ----------
        filename: str
            The path to where the serialized file should be saved
        data: varies
            The data that is to be serialized to file

        Example
        ------
        >>> serializer = get_serializer('json')
        >>> data ['foo', 'bar']
        >>> json_file = '/path/to/json/file.json'
        >>> serializer.save(json_file, data)
        """
        filename = self._check_extension(filename)
        try:
            with open(filename, self._write_option) as s_file:
                s_file.write(self.marshal(data))
        except IOError as err:
            msg = "Error writing to '{}': {}".format(filename, err.strerror)
            raise FaceswapError(msg) from err

    def _check_extension(self, filename):
        """ Check the filename has an extension. If not add the correct one for the serializer """
        extension = os.path.splitext(filename)[1]
        retval = filename if extension else "{}.{}".format(filename, self.file_extension)
        return retval

    def load(self, filename):
        """ Load data from an existing serialized file

        Parameters
        ----------
        filename: str
            The path to the serialized file

        Returns
        ----------
        data: varies
            The data in a python object format

        Example
        ------
        >>> serializer = get_serializer('json')
        >>> json_file = '/path/to/json/file.json'
        >>> data = serializer.load(json_file)
        """
        try:
            with open(filename, self._read_option) as s_file:
                data = s_file.read()
                retval = self.unmarshal(data)

        except IOError as err:
            msg = "Error reading from '{}': {}".format(filename, err.strerror)
            raise FaceswapError(msg) from err
        return retval

    def marshal(self, data):
        """ Serialize an object

        Parameters
        ----------
        data: varies
            The data that is to be serialized

        Returns
        -------
        data: varies
            The data in a the serialized data format

        Example
        ------
        >>> serializer = get_serializer('json')
        >>> data ['foo', 'bar']
        >>> json_data = serializer.marshal(data)
        """
        try:
            retval = self._marshal(data)
        except Exception as err:
            msg = "Error serializing data for type {}: {}".format(type(data), str(err))
            raise FaceswapError(msg) from err
        return retval

    def unmarshal(self, serialized_data):
        """ Unserialize data to its original object type

        Parameters
        ----------
        serialized_data: varies
            Data in serializer format that is to be unmarshalled to its original object

        Returns
        -------
        data: varies
            The data in a python object format

        Example
        ------
        >>> serializer = get_serializer('json')
        >>> json_data = <json object>
        >>> data = serializer.unmarshal(json_data)
        """
        try:
            retval = self._unmarshal(serialized_data)
        except Exception as err:
            msg = "Error unserializing data for type {}: {}".format(type(serialized_data),
                                                                    str(err))
            raise FaceswapError(msg) from err
        return retval

    @classmethod
    def _marshal(cls, data):
        """ Override for serializer specific marshalling """
        raise NotImplementedError()

    @classmethod
    def _unmarshal(cls, data):
        """ Override for serializer specific unmarshalling """
        raise NotImplementedError()


class _CompressedSerializer(Serializer):
    """ A compressed pickle serializer for Faceswap """
    def __init__(self):
        super().__init__()
        self._file_extension = "fsa"
        self._child = _PickleSerializer()

    def _marshal(self, data):
        """ Pickle and compress data """
        data = self._child._marshal(data)  # pylint: disable=protected-access
        return zlib.compress(data)

    def _unmarshal(self, data):
        """ Decompress and unpicke data """
        data = zlib.decompress(data)
        return self._child._unmarshal(data)  # pylint: disable=protected-access

class _PickleSerializer(Serializer):
    """ Pickle Serializer """
    def __init__(self):
        super().__init__()
        self._file_extension = "pickle"

    @classmethod
    def _marshal(cls, data):
        return pickle.dumps(data)

    @classmethod
    def _unmarshal(cls, data):
        return pickle.loads(data)

# Code from faceswap\lib\faces_detect.py

class DetectedFace():
    """ Detected face and landmark information

    Holds information about a detected face, it's location in a source image
    and the face's 68 point landmarks.

    Methods for aligning a face are also callable from here.

    Parameters
    ----------
    image: numpy.ndarray, optional
        Original frame that holds this face. Optional (not required if just storing coordinates)
    x: int
        The left most point (in pixels) of the face's bounding box as discovered in
        :mod:`plugins.extract.detect`
    w: int
        The width (in pixels) of the face's bounding box as discovered in
        :mod:`plugins.extract.detect`
    y: int
        The top most point (in pixels) of the face's bounding box as discovered in
        :mod:`plugins.extract.detect`
    h: int
        The height (in pixels) of the face's bounding box as discovered in
        :mod:`plugins.extract.detect`
    landmarks_xy: list
        The 68 point landmarks as discovered in :mod:`plugins.extract.align`. Should be a ``list``
        of 68 `(x, y)` ``tuples`` with each of the landmark co-ordinates.
    mask: dict
        The generated mask(s) for the face as generated in :mod:`plugins.extract.mask`. Must be a
        dict of {**name** (`str`): :class:`Mask`}.

    Attributes
    ----------
    image: numpy.ndarray, optional
        This is a generic image placeholder that should not be relied on to be holding a particular
        image. It may hold the source frame that holds the face, a cropped face or a scaled image
        depending on the method using this object.
    x: int
        The left most point (in pixels) of the face's bounding box as discovered in
        :mod:`plugins.extract.detect`
    w: int
        The width (in pixels) of the face's bounding box as discovered in
        :mod:`plugins.extract.detect`
    y: int
        The top most point (in pixels) of the face's bounding box as discovered in
        :mod:`plugins.extract.detect`
    h: int
        The height (in pixels) of the face's bounding box as discovered in
        :mod:`plugins.extract.detect`
    landmarks_xy: list
        The 68 point landmarks as discovered in :mod:`plugins.extract.align`.
    mask: dict
        The generated mask(s) for the face as generated in :mod:`plugins.extract.mask`. Is a
        dict of {**name** (`str`): :class:`Mask`}.
    hash: str
        The hash of the face. This cannot be set until the file is saved due to image compression,
        but will be set if loading data from :func:`from_alignment`
    """
    def __init__(self, image=None, x=None, w=None, y=None, h=None,
                 landmarks_xy=None, mask=None, filename=None):
        self.image = image
        self.x = x  # pylint:disable=invalid-name
        self.w = w  # pylint:disable=invalid-name
        self.y = y  # pylint:disable=invalid-name
        self.h = h  # pylint:disable=invalid-name
        self.landmarks_xy = landmarks_xy
        self.mask = dict() if mask is None else mask
        self.hash = None

        self.aligned = dict()
        self.feed = dict()
        self.reference = dict()

    def to_alignment(self):
        """  Return the detected face formatted for an alignments file

        returns
        -------
        alignment: dict
            The alignment dict will be returned with the keys ``x``, ``w``, ``y``, ``h``,
            ``landmarks_xy``, ``mask``, ``hash``.
        """

        alignment = dict()
        alignment["x"] = self.x
        alignment["w"] = self.w
        alignment["y"] = self.y
        alignment["h"] = self.h
        alignment["landmarks_xy"] = self.landmarks_xy
        alignment["hash"] = self.hash
        alignment["mask"] = {name: mask.to_dict() for name, mask in self.mask.items()}
        return alignment

    def from_alignment(self, alignment, image=None):
        """ Set the attributes of this class from an alignments file and optionally load the face
        into the ``image`` attribute.

        Parameters
        ----------
        alignment: dict
            A dictionary entry for a face from an alignments file containing the keys
            ``x``, ``w``, ``y``, ``h``, ``landmarks_xy``.
            Optionally the key ``hash`` will be provided, but not all use cases will know the
            face hash at this time.
            Optionally the key ``mask`` will be provided, but legacy alignments will not have
            this key.
        image: numpy.ndarray, optional
            If an image is passed in, then the ``image`` attribute will
            be set to the cropped face based on the passed in bounding box co-ordinates
        """

        self.x = alignment["x"]
        self.w = alignment["w"]
        self.y = alignment["y"]
        self.h = alignment["h"]
        landmarks = alignment["landmarks_xy"]
        if not isinstance(landmarks, np.ndarray):
            landmarks = np.array(landmarks, dtype="float32")
        self.landmarks_xy = landmarks
        # Manual tool does not know the final hash so default to None
        self.hash = alignment.get("hash", None)
        # Manual tool and legacy alignments will not have a mask
        if alignment.get("mask", None) is not None:
            self.mask = dict()
            for name, mask_dict in alignment["mask"].items():
                self.mask[name] = Mask()
                self.mask[name].from_dict(mask_dict)
        if image is not None and image.any():
            self._image_to_face(image)

# note data format is data[frame][face-id]
def get_alignment_data(path):
    serializer = _CompressedSerializer()
    data = serializer.load(path)

    single_data = list(data.values())[0]
    # data frame name key, frame, faces array, mask array,mask key, mask
    #print (single_data[0]['mask']['unet-dfl']['stored_size'])
    dims = (single_data[0]['mask']['unet-dfl']['stored_size'], single_data[0]['mask']['unet-dfl']['stored_size'], 1)
    #print (list(np.frombuffer(decompress(single_data[0]['mask']['unet-dfl']['mask'])), dtype="uint8")))
    #print (decompress(single_data[0]['mask']['unet-dfl']['mask']))
    print (single_data)

    return data

def save_alignment_data(data, path):
    serializer = _CompressedSerializer()
    serializer.save(path, data)

def parse_fsa_alignment(path):
    data = get_alignment_data(args.input_folder)

    for key, value in data.items():
        for face in value:
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder")
    args = parser.parse_args()

    get_alignment_data(args.input_folder)


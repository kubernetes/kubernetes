import os

from google.protobuf import message

class Error(Exception):
    pass

class WriteError(Error):
    pass

class ReadError(Error):
    pass

class FileNotFoundError(ReadError):
    pass

class CorruptStateError(ReadError):
    pass

class StateKeeper(object):
    def __init__(self, state_file):
        self.__state_file = state_file

    def write(self, state):
        """Write a protocol buffer to the state file."""
        serialized_state = state.SerializeToString()
        try:
            with open(self.__state_file, "wb") as f:
                f.write(serialized_state)
        except IOError as e:
            raise WriteError(e)

    def read(self, state_type):
        """Read a protocol buffer from the state file."""
        if not os.path.exists(self.__state_file):
            raise FileNotFoundError("No such file: %s" % self.__state_file )
        try:
            with open(self.__state_file, "rb") as f:
                serialized_state = f.read()
        except IOError as e:
            raise ReadError(e)
        state = state_type()
        try:
            state.ParseFromString(serialized_state)
            return state
        except message.DecodeError as e:
            raise CorruptStateError(e)

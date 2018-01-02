# Utilities for printing ASN.1 values

def bits_to_hex(bit_array, delimiter=":"):
    """Convert a bit array to a prettily formated hex string. If the array
    length is not a multiple of 8, it is padded with 0-bits from the left.
    For example, [1,0,0,1,1,0,1,0,0,1,0] becomes 04:d2.
    Args:
        bit_array: the bit array to convert
    Returns:
        the formatted hex string."""
    # Pad the first partial byte.
    partial_bits = len(bit_array) % 8
    pad_length = 8 - partial_bits if partial_bits else 0

    bitstring = "0"*pad_length + "".join(map(str, bit_array))
    byte_array = [int(bitstring[i:i+8], 2) for i in range(0, len(bitstring), 8)]
    return delimiter.join(map(lambda x: "%02x" % x, byte_array)) 

    return bytes_to_hex(byte_array, delimiter=delimiter)

def bytes_to_hex(byte_string, delimiter=":"):
    """Convert a bytestring to a prettily formated hex string: for example,
    '\x04\xd2' becomes 04:d2.
    Args:
        byte_string: the bytes to convert.
    Returns:
        the formatted hex string."""
    return delimiter.join([("%02x" % ord(b)) for b in byte_string])

def int_to_hex(int_value, delimiter=":"):
    """Convert an integer to a prettily formated hex string: for example,
    1234 (0x4d2) becomes 04:d2 and -1234 becomes ' -:04:d2'
    Args:
        int_value: the value to convert.
    Returns:
        the formatted hex string."""
    hex_string = "%x" % int_value
    ret = ""
    pos = 0
    # Accommodate for negative integers.
    if hex_string[0] == '-':
        ret += ' -' + delimiter
        hex_string = hex_string[1:]
    # If the first digit is a half-byte, pad with a 0.
    remaining_len = len(hex_string) - pos
    hex_string = hex_string.zfill(remaining_len + remaining_len % 2)
    byte_values = [hex_string[i:i+2] for i in range(pos, len(hex_string), 2)]
    return ret + delimiter.join(byte_values)

def wrap_lines(long_string, wrap):
    """Split the long string into line chunks according to the wrap limit and
    existing newlines.
    Args:
        long_string: a long, possibly multiline string
        wrap:        maximum number of characters per line. 0 or negative
                     wrap means no limit.
    Returns:
       a list of lines of at most |wrap| characters each."""
    if not long_string:
        return []
    long_lines = long_string.split('\n')
    if wrap <= 0:
        return long_lines
    ret = []
    for line in long_lines:
        if not line:
            # Empty line
            ret += [line]
        else:
            ret += [line[i:i+wrap] for i in range(0, len(line), wrap)]
    return ret

def append_lines(lines, wrap, buf):
    """Append lines to the buffer. If the first line can be appended to the last
    line of the buf without exceeding wrap characters, the two lines are merged.
    Args:
        lines: an iterable of lines to append
        wrap:  maximum number of characters per line. 0 or negative wrap means
               no limit.
        buf:   an iterable of lines to append to"""
    if not lines:
        return
    if not buf or wrap > 0 and len(buf[-1]) + len(lines[0]) > wrap:
        buf += lines
    else:
        buf[-1] += lines[0]
        buf += lines[1:]

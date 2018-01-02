"""Read and write PEM files and strings."""

import base64
import StringIO

from ct.crypto import error


class PemError(error.EncodingError):
    pass

_START_TEMPLATE = "-----BEGIN %s-----"
_END_TEMPLATE = "-----END %s-----"


class PemReader(object):
    """A reader class for iteratively reading PEM files."""

    def __init__(self, fileobj, markers, skip_invalid_blobs=True):
        """Create a PemReader from a file object.

        When used as a context manager, the file object is closed
        upon exit.

        Args:
            fileobj: the file object to read from.
            markers: an iterable of markers accepted by the reader, e.g.,
                CERTIFICATE, RSA PUBLIC KEY, etc.
            skip_invalid_blobs: if False, invalid PEM blobs cause a PemError.
                If True, invalid blobs are skipped. In non-skip mode, an
                immediate StopIteration before any valid blocks are found, also
                causes a PemError exception.

        Raises:
            PemError: invalid PEM contents.
        """
        self.__f = fileobj
        self.__marker_dict = ({(_START_TEMPLATE % m): m for m in markers})
        self.__valid_blobs_read = 0
        self.__eof = False
        self.__skip_invalid_blobs = skip_invalid_blobs

    def __iter__(self):
        """Iterate over file contents.

        Returns:
            a generator function that yields decoded (blob, marker)
                tuples.
        """
        return self.read_blocks()

    def close(self):
        """Close the underlying file object."""
        self.__f.close()

    def __enter__(self):
        return self

    def __exit__(self, unused_type, unused_value, traceback):
        self.close()

    @classmethod
    def from_file(cls, pem_file, markers, skip_invalid_blobs=True):
        """Create a PemReader for reading a file.

        Caller is responsible for closing the reader afterwards.

        Args:
            pem_file: the file to read from.
            markers: an iterable of markers accepted by the reader,e.g.,
                CERTIFICATE, RSA PUBLIC KEY, etc.
            skip_invalid_blobs: if False, invalid PEM blobs cause a PemError.
                If True, invalid blobs are skipped. In non-skip mode, an
                immediate StopIteration before any valid blocks are found, also
                causes a PemError exception.

        Returns:
            a PemReader object.

        Raises:
            IOError, ValueError: the fileobject could not be operated on.
        """
        return cls(open(pem_file, "r"), markers, skip_invalid_blobs)

    @classmethod
    def from_string(cls, pem_string, markers, skip_invalid_blobs=True):
        """Create a PemReader for reading a string.

        Args:
            pem_string: the string to read from.
            markers: an iterable of markers accepted by the reader, e.g.,
                CERTIFICATE, RSA PUBLIC KEY, etc.
            skip_invalid_blobs: if False, invalid PEM blobs cause a PemError.
                If True, invalid blobs are skipped. In non-skip mode, an
                immediate StopIteration before any valid blocks are found, also
                causes a PemError exception.

        Returns:
            a PemReader object.
        """
        f = StringIO.StringIO(pem_string)
        return cls(f, markers, skip_invalid_blobs)

    def read_blocks(self):
        """Read the next PEM blob.

        Yields:
            (raw_string, marker) tuples containing the decoded blob and the
            marker used to detect the blob.

        Raises:
            PemError: a PEM block was invalid (in skip_invalid_blobs mode).
            IOError, ValueError: the file object could not be operated on.
            StopIteration: EOF was reached.
        """
        while not self.__eof:
            marker = None
            for line in self.__f:
                line = line.rstrip("\r\n")
                # PEM (RFC 1421) allows arbitrary comments between PEM blocks
                # so we skip over those
                if line in self.__marker_dict:
                    marker = self.__marker_dict[line]
                    break

            if not marker:
                self.__eof = True
                if (not self.__skip_invalid_blobs and
                    not self.__valid_blobs_read):
                    raise PemError("No PEM header")
                raise StopIteration

            ret = ""
            footer = _END_TEMPLATE % marker
            footer_found = False

            for line in self.__f:
                line = line.rstrip("\r\n")
                if line == footer:
                    footer_found = True
                    break
                ret += line

            # Here, we assume that each header is exactly matched by a footer.
            # TODO(ekasper): determine if this assumption is overly strict,
            # i.e., whether blocks such as BEGIN RSA PUBLIC KEY...END PUBLIC KEY
            # are commonly used in applications.
            if not footer_found:
                self.__eof = True
                if not self.__skip_invalid_blobs:
                    raise PemError("No PEM footer line to match the header")
                raise StopIteration

            try:
                # We don't use ret.decode('base64') here as the exceptions from
                # this method are not properly documented.
                yield base64.b64decode(ret), marker
                self.__valid_blobs_read += 1
            except TypeError:
                if not self.__skip_invalid_blobs:
                    # We do not set EOF here so caller can resume - even though
                    # this can normally be transparently handled by setting
                    # skip_invalid_blobs to True upon init.
                    raise PemError("Invalid base64 encoding")
                # Else just continue the loop
        raise StopIteration


class PemWriter(object):
    """A class for writing PEM blobs."""

    def __init__(self, fileobj, marker):
        """Create a writer.

        When used as a context manager, the underlying file object is closed
        upon exit.

        Args:
            fileobj: the file object to write to. Must be open for writing AND
                reading, and must be positioned at the writing position. Rather
                than initializing directly from a file object, it is recommended
                to use the from_file() constructor.
            marker: the marker to use in headers.
        """
        self.__f = fileobj
        self.__header = _START_TEMPLATE % marker
        self.__footer = _END_TEMPLATE % marker

    def close(self):
        self.__f.close()

    def __enter__(self):
        return self

    def __exit__(self, unused_type, unused_value, traceback):
        self.close()

    @classmethod
    def from_file(cls, filename, marker, append=False):
        """Construct a writer for writing to a file.

        Caller is responsible for closing the writer afterwards.

        Args:
            filename: the file to write to.
            marker: the marker to use in headers/footers.
            append: if True, file will be opened in append mode.

        Returns:
            A PemWriter object.

        Raises:
            IOError: the file could not be opened.
        """
        mode = "a+" if append else "w+"
        f = open(filename, mode)
        if append:
            f.seek(0, 2)
        return cls(f, marker)

    def write(self, blob, check_newline=True):
        """Write a single PEM blob.

        Args:
            blob: a binary blob.
            check_newline: if True, check whether the current position is at
                the beginning of a new line and add a newline if not.

        Raises:
            IOError: the file could not be written to.
        """
        # Header must start on a new line, so we try to be helpful and add one
        # if it's missing.
        # Note that a file open'ed in a+ mode will report its current reading
        # (rather than writing) position - we deem it the caller's
        # responsibility to seek to the write position. Failing that, the worst
        # that can happen is either we fail to heal it, or add an extra newline.
        if check_newline:
          if self.__f.tell() != 0:
              self.__f.seek(-1, 1)
              if self.__f.read(1) != "\n":
                  self.__f.write("\n")

        self.__f.write(self.__header)
        pem_blob = base64.b64encode(blob)
        for i in range(0, len(pem_blob), 64):
            self.__f.write("\n")
            self.__f.write(pem_blob[i:i+64])

        self.__f.write("\n")
        self.__f.write(self.__footer)
        self.__f.write("\n")

    def write_blocks(self, blobs):
        """Write PEM blobs.

        Args:
            blobs: an iterable of binary blobs.

        Raises:
            IOError: the file could not be written to.
        """
        check_newline = True
        for b in blobs:
            self.write(b, check_newline=check_newline)
            check_newline = False

    @classmethod
    def pem_string(cls, blob, marker):
        """Convert a binary blob to a PEM string.

        Args:
            blob: a single binary blob.
            marker: the marker to use in headers/footers.

        Returns:
            a string of concatenated PEM blobs.
        """
        stringio = StringIO.StringIO()
        with cls(stringio, marker) as writer:
            writer.write(blob)
            return stringio.getvalue()

    @classmethod
    def blocks_to_pem_string(cls, blobs, marker):
        """Convert a binary blob to a PEM string.

        Args:
            blobs: an iterable of binary blobs.
            marker: the marker to use in headers/footers.

        Returns:
            a string of concatenated PEM blobs.
        """
        stringio = StringIO.StringIO()
        with cls(stringio, marker) as writer:
            writer.write_blocks(blobs)
            return stringio.getvalue()


def from_pem(pem_string, markers):
    """Read a single PEM blob from a string.

    Ignores everything before and after the first blob with valid markers.

    Args:
        pem_string: the PEM string.
        markers: a single marker string or an iterable containing all
            accepted markers, such as CERTIFICATE, RSA PUBLIC KEY,
            PUBLIC KEY, etc.

    Returns:
        A (raw_string, marker) tuple containing the decoded blob and the
        marker used to detect the blob.

    Raises:
        PemError: a PEM block was invalid or no valid PEM block was found.
    """
    with PemReader.from_string(pem_string, markers,
                               skip_invalid_blobs=False) as reader:
        return iter(reader).next()


def from_pem_file(pem_file, markers):
    """Read a single PEM blob from a file.

    Ignores everything before and after the first blob with valid markers.

    Args:
        pem_file: the PEM file.
        markers: a single marker string or an iterable containing all
            accepted markers, such as CERTIFICATE, RSA PUBLIC KEY,
            PUBLIC KEY, etc.

    Returns:
        A (raw_string, marker) tuple containing the decoded blob and the
        marker used to detect the blob.

    Raises:
        PemError: a PEM block was invalid or no valid PEM block was found.
        IOError: the file could not be read.
    """
    with PemReader.from_file(pem_file, markers,
                             skip_invalid_blobs=False) as reader:
        return iter(reader).next()


def pem_blocks(pem_string, markers, skip_invalid_blobs=True):
    """Read PEM blobs from a string.

    Args:
        pem_string: the PEM string.
        markers: a single marker string or an iterable containing all
            accepted markers, such as CERTIFICATE, RSA PUBLIC KEY,
            PUBLIC KEY, etc.
        skip_invalid_blobs: if False, invalid PEM blobs cause a PemError.
            If True, invalid blobs are skipped. In non-skip mode, an immediate
            StopIteration before any valid blocks are found, also causes a
            a PemError exception.

    Yields:
        (raw_string, marker) tuples containing the decoded blob and the marker
        used to detect the blob.

    Raises:
        PemError: a PEM block was invalid.
    """
    with PemReader.from_string(pem_string, markers,
                               skip_invalid_blobs=skip_invalid_blobs) as reader:
        for block in reader:
            yield block


def pem_blocks_from_file(pem_file, markers, skip_invalid_blobs=True):
    """Read PEM blobs from a file.

    Args:
        pem_file: the PEM file.
        markers: a single marker string or an iterable containing all accepted
            markers, such as CERTIFICATE, RSA PUBLIC KEY, PUBLIC KEY, etc.
        skip_invalid_blobs: if False, invalid PEM blobs cause a PemError.
            If True, invalid blobs are skipped. In non-skip mode, an immediate
            StopIteration before any valid blocks are found, also causes a
            PemError exception.

    Yields:
        (raw_string, marker) tuples containing the decoded blob and the marker
        used to detect the blob.

    Raises:
        PemError: a PEM block was invalid.
    """
    with PemReader.from_file(pem_file, markers,
                             skip_invalid_blobs=skip_invalid_blobs) as reader:
        for block in reader:
            yield block


def to_pem(blob, marker):
    """Convert a binary blob to a PEM-formatted string.

    Args:
        blob: a binary blob.
        marker: the marker to use, e.g., CERTIFICATE.

    Returns:
        the PEM string.
    """
    return PemWriter.pem_string(blob, marker)


def blocks_to_pem(blobs, marker):
    """Convert binary blobs to a string of concatenated PEM-formatted blocks.

    Args:
        blobs: an iterable of binary blobs
        marker: the marker to use, e.g., CERTIFICATE

    Returns:
        the PEM string.
    """
    return PemWriter.blocks_to_pem_string(blobs, marker)


def to_pem_file(blob, filename, marker):
    """Convert a binary blob to PEM format and write to file.

    Args:
        blob: a binary blob.
        filename: the file to write to.
        marker: the marker to use, e.g., CERTIFICATE.

    Raises:
        IOError: the file could not be written to.
    """
    with PemWriter.from_file(filename, marker) as writer:
        writer.write(blob)


def blocks_to_pem_file(blobs, filename, marker):
    """Convert binary blobs to PEM format and write to file.

    Blobs must all be of one and the same type.

    Args:
        blobs: an iterable of binary blobs.
        filename: the file to write to.
        marker: the marker to use, e.g., CERTIFICATE.

    Raises:
        IOError: the file could not be written to.
    """
    with PemWriter.from_file(filename, marker) as writer:
        writer.write_blocks(blobs)

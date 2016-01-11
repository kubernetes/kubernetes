package stdcopy

import (
	"encoding/binary"
	"errors"
	"io"

	"github.com/fsouza/go-dockerclient/external/github.com/Sirupsen/logrus"
)

const (
	stdWriterPrefixLen = 8
	stdWriterFdIndex   = 0
	stdWriterSizeIndex = 4

	startingBufLen = 32*1024 + stdWriterPrefixLen + 1
)

// StdType prefixes type and length to standard stream.
type StdType [stdWriterPrefixLen]byte

var (
	// Stdin represents standard input stream type.
	Stdin = StdType{0: 0}
	// Stdout represents standard output stream type.
	Stdout = StdType{0: 1}
	// Stderr represents standard error steam type.
	Stderr = StdType{0: 2}
)

// StdWriter is wrapper of io.Writer with extra customized info.
type StdWriter struct {
	io.Writer
	prefix  StdType
	sizeBuf []byte
}

func (w *StdWriter) Write(buf []byte) (n int, err error) {
	var n1, n2 int
	if w == nil || w.Writer == nil {
		return 0, errors.New("Writer not instantiated")
	}
	binary.BigEndian.PutUint32(w.prefix[4:], uint32(len(buf)))
	n1, err = w.Writer.Write(w.prefix[:])
	if err != nil {
		n = n1 - stdWriterPrefixLen
	} else {
		n2, err = w.Writer.Write(buf)
		n = n1 + n2 - stdWriterPrefixLen
	}
	if n < 0 {
		n = 0
	}
	return
}

// NewStdWriter instantiates a new Writer.
// Everything written to it will be encapsulated using a custom format,
// and written to the underlying `w` stream.
// This allows multiple write streams (e.g. stdout and stderr) to be muxed into a single connection.
// `t` indicates the id of the stream to encapsulate.
// It can be stdcopy.Stdin, stdcopy.Stdout, stdcopy.Stderr.
func NewStdWriter(w io.Writer, t StdType) *StdWriter {
	return &StdWriter{
		Writer:  w,
		prefix:  t,
		sizeBuf: make([]byte, 4),
	}
}

var errInvalidStdHeader = errors.New("Unrecognized input header")

// StdCopy is a modified version of io.Copy.
//
// StdCopy will demultiplex `src`, assuming that it contains two streams,
// previously multiplexed together using a StdWriter instance.
// As it reads from `src`, StdCopy will write to `dstout` and `dsterr`.
//
// StdCopy will read until it hits EOF on `src`. It will then return a nil error.
// In other words: if `err` is non nil, it indicates a real underlying error.
//
// `written` will hold the total number of bytes written to `dstout` and `dsterr`.
func StdCopy(dstout, dsterr io.Writer, src io.Reader) (written int64, err error) {
	var (
		buf       = make([]byte, startingBufLen)
		bufLen    = len(buf)
		nr, nw    int
		er, ew    error
		out       io.Writer
		frameSize int
	)

	for {
		// Make sure we have at least a full header
		for nr < stdWriterPrefixLen {
			var nr2 int
			nr2, er = src.Read(buf[nr:])
			nr += nr2
			if er == io.EOF {
				if nr < stdWriterPrefixLen {
					logrus.Debugf("Corrupted prefix: %v", buf[:nr])
					return written, nil
				}
				break
			}
			if er != nil {
				logrus.Debugf("Error reading header: %s", er)
				return 0, er
			}
		}

		// Check the first byte to know where to write
		switch buf[stdWriterFdIndex] {
		case 0:
			fallthrough
		case 1:
			// Write on stdout
			out = dstout
		case 2:
			// Write on stderr
			out = dsterr
		default:
			logrus.Debugf("Error selecting output fd: (%d)", buf[stdWriterFdIndex])
			return 0, errInvalidStdHeader
		}

		// Retrieve the size of the frame
		frameSize = int(binary.BigEndian.Uint32(buf[stdWriterSizeIndex : stdWriterSizeIndex+4]))
		logrus.Debugf("framesize: %d", frameSize)

		// Check if the buffer is big enough to read the frame.
		// Extend it if necessary.
		if frameSize+stdWriterPrefixLen > bufLen {
			logrus.Debugf("Extending buffer cap by %d (was %d)", frameSize+stdWriterPrefixLen-bufLen+1, len(buf))
			buf = append(buf, make([]byte, frameSize+stdWriterPrefixLen-bufLen+1)...)
			bufLen = len(buf)
		}

		// While the amount of bytes read is less than the size of the frame + header, we keep reading
		for nr < frameSize+stdWriterPrefixLen {
			var nr2 int
			nr2, er = src.Read(buf[nr:])
			nr += nr2
			if er == io.EOF {
				if nr < frameSize+stdWriterPrefixLen {
					logrus.Debugf("Corrupted frame: %v", buf[stdWriterPrefixLen:nr])
					return written, nil
				}
				break
			}
			if er != nil {
				logrus.Debugf("Error reading frame: %s", er)
				return 0, er
			}
		}

		// Write the retrieved frame (without header)
		nw, ew = out.Write(buf[stdWriterPrefixLen : frameSize+stdWriterPrefixLen])
		if ew != nil {
			logrus.Debugf("Error writing frame: %s", ew)
			return 0, ew
		}
		// If the frame has not been fully written: error
		if nw != frameSize {
			logrus.Debugf("Error Short Write: (%d on %d)", nw, frameSize)
			return 0, io.ErrShortWrite
		}
		written += int64(nw)

		// Move the rest of the buffer to the beginning
		copy(buf, buf[frameSize+stdWriterPrefixLen:])
		// Move the index
		nr -= frameSize + stdWriterPrefixLen
	}
}

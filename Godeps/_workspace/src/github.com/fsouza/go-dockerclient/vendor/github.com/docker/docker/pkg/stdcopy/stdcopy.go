package stdcopy

import (
	"encoding/binary"
	"errors"
	"io"

	"github.com/fsouza/go-dockerclient/vendor/github.com/Sirupsen/logrus"
)

const (
	StdWriterPrefixLen = 8
	StdWriterFdIndex   = 0
	StdWriterSizeIndex = 4
)

type StdType [StdWriterPrefixLen]byte

var (
	Stdin  StdType = StdType{0: 0}
	Stdout StdType = StdType{0: 1}
	Stderr StdType = StdType{0: 2}
)

type StdWriter struct {
	io.Writer
	prefix  StdType
	sizeBuf []byte
}

func (w *StdWriter) Write(buf []byte) (n int, err error) {
	var n1, n2 int
	if w == nil || w.Writer == nil {
		return 0, errors.New("Writer not instanciated")
	}
	binary.BigEndian.PutUint32(w.prefix[4:], uint32(len(buf)))
	n1, err = w.Writer.Write(w.prefix[:])
	if err != nil {
		n = n1 - StdWriterPrefixLen
	} else {
		n2, err = w.Writer.Write(buf)
		n = n1 + n2 - StdWriterPrefixLen
	}
	if n < 0 {
		n = 0
	}
	return
}

// NewStdWriter instanciates a new Writer.
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

var ErrInvalidStdHeader = errors.New("Unrecognized input header")

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
		buf       = make([]byte, 32*1024+StdWriterPrefixLen+1)
		bufLen    = len(buf)
		nr, nw    int
		er, ew    error
		out       io.Writer
		frameSize int
	)

	for {
		// Make sure we have at least a full header
		for nr < StdWriterPrefixLen {
			var nr2 int
			nr2, er = src.Read(buf[nr:])
			nr += nr2
			if er == io.EOF {
				if nr < StdWriterPrefixLen {
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
		switch buf[StdWriterFdIndex] {
		case 0:
			fallthrough
		case 1:
			// Write on stdout
			out = dstout
		case 2:
			// Write on stderr
			out = dsterr
		default:
			logrus.Debugf("Error selecting output fd: (%d)", buf[StdWriterFdIndex])
			return 0, ErrInvalidStdHeader
		}

		// Retrieve the size of the frame
		frameSize = int(binary.BigEndian.Uint32(buf[StdWriterSizeIndex : StdWriterSizeIndex+4]))
		logrus.Debugf("framesize: %d", frameSize)

		// Check if the buffer is big enough to read the frame.
		// Extend it if necessary.
		if frameSize+StdWriterPrefixLen > bufLen {
			logrus.Debugf("Extending buffer cap by %d (was %d)", frameSize+StdWriterPrefixLen-bufLen+1, len(buf))
			buf = append(buf, make([]byte, frameSize+StdWriterPrefixLen-bufLen+1)...)
			bufLen = len(buf)
		}

		// While the amount of bytes read is less than the size of the frame + header, we keep reading
		for nr < frameSize+StdWriterPrefixLen {
			var nr2 int
			nr2, er = src.Read(buf[nr:])
			nr += nr2
			if er == io.EOF {
				if nr < frameSize+StdWriterPrefixLen {
					logrus.Debugf("Corrupted frame: %v", buf[StdWriterPrefixLen:nr])
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
		nw, ew = out.Write(buf[StdWriterPrefixLen : frameSize+StdWriterPrefixLen])
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
		copy(buf, buf[frameSize+StdWriterPrefixLen:])
		// Move the index
		nr -= frameSize + StdWriterPrefixLen
	}
}

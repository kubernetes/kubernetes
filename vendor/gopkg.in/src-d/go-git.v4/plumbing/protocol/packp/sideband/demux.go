package sideband

import (
	"errors"
	"fmt"
	"io"

	"gopkg.in/src-d/go-git.v4/plumbing/format/pktline"
)

// ErrMaxPackedExceeded returned by Read, if the maximum packed size is exceeded
var ErrMaxPackedExceeded = errors.New("max. packed size exceeded")

// Progress where the progress information is stored
type Progress interface {
	io.Writer
}

// Demuxer demultiplexes the progress reports and error info interleaved with the
// packfile itself.
//
// A sideband has three different channels the main one, called PackData, contains
// the packfile data; the ErrorMessage channel, that contains server errors; and
// the last one, ProgressMessage channel, containing information about the ongoing
// task happening in the server (optional, can be suppressed sending NoProgress
// or Quiet capabilities to the server)
//
// In order to demultiplex the data stream, method `Read` should be called to
// retrieve the PackData channel, the incoming data from the ProgressMessage is
// written at `Progress` (if any), if any message is retrieved from the
// ErrorMessage channel an error is returned and we can assume that the
// connection has been closed.
type Demuxer struct {
	t Type
	r io.Reader
	s *pktline.Scanner

	max     int
	pending []byte

	// Progress is where the progress messages are stored
	Progress Progress
}

// NewDemuxer returns a new Demuxer for the given t and read from r
func NewDemuxer(t Type, r io.Reader) *Demuxer {
	max := MaxPackedSize64k
	if t == Sideband {
		max = MaxPackedSize
	}

	return &Demuxer{
		t:   t,
		r:   r,
		max: max,
		s:   pktline.NewScanner(r),
	}
}

// Read reads up to len(p) bytes from the PackData channel into p, an error can
// be return if an error happens when reading or if a message is sent in the
// ErrorMessage channel.
//
// When a ProgressMessage is read, is not copy to b, instead of this is written
// to the Progress
func (d *Demuxer) Read(b []byte) (n int, err error) {
	var read, req int

	req = len(b)
	for read < req {
		n, err := d.doRead(b[read:req])
		read += n

		if err != nil {
			return read, err
		}
	}

	return read, nil
}

func (d *Demuxer) doRead(b []byte) (int, error) {
	read, err := d.nextPackData()
	size := len(read)
	wanted := len(b)

	if size > wanted {
		d.pending = read[wanted:]
	}

	if wanted > size {
		wanted = size
	}

	size = copy(b, read[:wanted])
	return size, err
}

func (d *Demuxer) nextPackData() ([]byte, error) {
	content := d.getPending()
	if len(content) != 0 {
		return content, nil
	}

	if !d.s.Scan() {
		if err := d.s.Err(); err != nil {
			return nil, err
		}

		return nil, io.EOF
	}

	content = d.s.Bytes()

	size := len(content)
	if size == 0 {
		return nil, nil
	} else if size > d.max {
		return nil, ErrMaxPackedExceeded
	}

	switch Channel(content[0]) {
	case PackData:
		return content[1:], nil
	case ProgressMessage:
		if d.Progress != nil {
			_, err := d.Progress.Write(content[1:])
			return nil, err
		}
	case ErrorMessage:
		return nil, fmt.Errorf("unexpected error: %s", content[1:])
	default:
		return nil, fmt.Errorf("unknown channel %s", content)
	}

	return nil, nil
}

func (d *Demuxer) getPending() (b []byte) {
	if len(d.pending) == 0 {
		return nil
	}

	content := d.pending
	d.pending = nil

	return content
}

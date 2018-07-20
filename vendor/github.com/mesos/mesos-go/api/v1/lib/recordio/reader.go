package recordio

import (
	"bufio"
	"bytes"
	"io"

	logger "github.com/mesos/mesos-go/api/v1/lib/debug"
	"github.com/mesos/mesos-go/api/v1/lib/encoding/framing"
)

const debug = logger.Logger(false)

type (
	Opt func(*reader)

	reader struct {
		*bufio.Scanner
		pend   int
		splitf func(data []byte, atEOF bool) (int, []byte, error)
		maxf   int // max frame size
	}
)

// NewReader returns a reader that parses frames from a recordio stream.
func NewReader(read io.Reader, opt ...Opt) framing.Reader {
	debug.Log("new frame reader")
	r := &reader{Scanner: bufio.NewScanner(read)}
	r.Split(func(data []byte, atEOF bool) (int, []byte, error) {
		// Scanner panics if we invoke Split after scanning has started,
		// use this proxy func as a work-around.
		return r.splitf(data, atEOF)
	})
	buf := make([]byte, 16*1024)
	r.Buffer(buf, 1<<22) // 1<<22 == max protobuf size
	r.splitf = r.splitSize
	// apply options
	for _, f := range opt {
		if f != nil {
			f(r)
		}
	}
	return r
}

// MaxMessageSize returns a functional option that configures the internal Scanner's buffer and max token (message)
// length, in bytes.
func MaxMessageSize(max int) Opt {
	return func(r *reader) {
		buf := make([]byte, max>>1)
		r.Buffer(buf, max)
		r.maxf = max
	}
}

func (r *reader) splitSize(data []byte, atEOF bool) (int, []byte, error) {
	const maxTokenLength = 20 // textual length of largest uint64 number
	if atEOF {
		x := len(data)
		switch {
		case x == 0:
			debug.Log("EOF and empty frame, returning io.EOF")
			return 0, nil, io.EOF
		case x < 2: // min frame size
			debug.Log("remaining data less than min total frame length")
			return 0, nil, framing.ErrorUnderrun
		}
		// otherwise, we may have a valid frame...
	}
	debug.Log("len(data)=", len(data))
	adv := 0
	for {
		i := 0
		for ; i < maxTokenLength && i < len(data) && data[i] != '\n'; i++ {
		}
		debug.Log("i=", i)
		if i == len(data) {
			debug.Log("need more input")
			return 0, nil, nil // need more input
		}
		if i == maxTokenLength && data[i] != '\n' {
			debug.Log("frame size: max token length exceeded")
			return 0, nil, framing.ErrorBadSize
		}
		n, err := ParseUintBytes(bytes.TrimSpace(data[:i]), 10, 64)
		if err != nil {
			debug.Log("failed to parse frame size field:", err)
			return 0, nil, framing.ErrorBadSize
		}
		if r.maxf != 0 && int(n) > r.maxf {
			debug.Log("frame size max length exceeded:", n)
			return 0, nil, framing.ErrorOversizedFrame
		}
		if n == 0 {
			// special case... don't invoke splitData, just parse the next size header
			adv += i + 1
			data = data[i+1:]
			continue
		}
		r.pend = int(n)
		r.splitf = r.splitFrame
		debug.Logf("split next frame: %d, %d", n, adv+i+1)
		return adv + i + 1, data[:0], nil // returning a nil token screws up the Scanner, so return empty
	}
}

func (r *reader) splitFrame(data []byte, atEOF bool) (advance int, token []byte, err error) {
	x := len(data)
	debug.Log("splitFrame:x=", x, ",eof=", atEOF)
	if atEOF {
		if x < r.pend {
			return 0, nil, framing.ErrorUnderrun
		}
	}
	if r.pend == 0 {
		panic("asked to read frame data, but no data left in frame")
	}
	if x < int(r.pend) {
		// need more data
		return 0, nil, nil
	}
	r.splitf = r.splitSize
	adv := int(r.pend)
	r.pend = 0
	return adv, data[:adv], nil
}

// ReadFrame implements framing.Reader
func (r *reader) ReadFrame() (tok []byte, err error) {
	for r.Scan() {
		b := r.Bytes()
		if len(b) == 0 {
			continue
		}
		tok = b
		debug.Log("len(tok)", len(tok))
		break
	}
	// either scan failed, or it succeeded and we have a token...
	err = r.Err()
	if err == nil && len(tok) == 0 {
		err = io.EOF
	}
	return
}

package http3

import (
	"fmt"
	"io"

	"github.com/lucas-clemente/quic-go"
)

// The body of a http.Request or http.Response.
type body struct {
	str quic.Stream

	// only set for the http.Response
	// The channel is closed when the user is done with this response:
	// either when Read() errors, or when Close() is called.
	reqDone       chan<- struct{}
	reqDoneClosed bool

	onFrameError func()

	bytesRemainingInFrame uint64
}

var _ io.ReadCloser = &body{}

func newRequestBody(str quic.Stream, onFrameError func()) *body {
	return &body{
		str:          str,
		onFrameError: onFrameError,
	}
}

func newResponseBody(str quic.Stream, done chan<- struct{}, onFrameError func()) *body {
	return &body{
		str:          str,
		onFrameError: onFrameError,
		reqDone:      done,
	}
}

func (r *body) Read(b []byte) (int, error) {
	n, err := r.readImpl(b)
	if err != nil {
		r.requestDone()
	}
	return n, err
}

func (r *body) readImpl(b []byte) (int, error) {
	if r.bytesRemainingInFrame == 0 {
	parseLoop:
		for {
			frame, err := parseNextFrame(r.str)
			if err != nil {
				return 0, err
			}
			switch f := frame.(type) {
			case *headersFrame:
				// skip HEADERS frames
				continue
			case *dataFrame:
				r.bytesRemainingInFrame = f.Length
				break parseLoop
			default:
				r.onFrameError()
				// parseNextFrame skips over unknown frame types
				// Therefore, this condition is only entered when we parsed another known frame type.
				return 0, fmt.Errorf("peer sent an unexpected frame: %T", f)
			}
		}
	}

	var n int
	var err error
	if r.bytesRemainingInFrame < uint64(len(b)) {
		n, err = r.str.Read(b[:r.bytesRemainingInFrame])
	} else {
		n, err = r.str.Read(b)
	}
	r.bytesRemainingInFrame -= uint64(n)
	return n, err
}

func (r *body) requestDone() {
	if r.reqDoneClosed || r.reqDone == nil {
		return
	}
	close(r.reqDone)
	r.reqDoneClosed = true
}

func (r *body) Close() error {
	r.requestDone()
	// If the EOF was read, CancelRead() is a no-op.
	r.str.CancelRead(quic.StreamErrorCode(errorRequestCanceled))
	return nil
}

// +build go1.8

package autorest

import (
	"bytes"
	"io"
	"net/http"
)

// RetriableRequest provides facilities for retrying an HTTP request.
type RetriableRequest struct {
	req   *http.Request
	rc    io.ReadCloser
	br    *bytes.Reader
	reset bool
}

// Prepare signals that the request is about to be sent.
func (rr *RetriableRequest) Prepare() (err error) {
	// preserve the request body; this is to support retry logic as
	// the underlying transport will always close the reqeust body
	if rr.req.Body != nil {
		if rr.reset {
			if rr.rc != nil {
				rr.req.Body = rr.rc
			} else if rr.br != nil {
				_, err = rr.br.Seek(0, io.SeekStart)
			}
			rr.reset = false
			if err != nil {
				return err
			}
		}
		if rr.req.GetBody != nil {
			// this will allow us to preserve the body without having to
			// make a copy.  note we need to do this on each iteration
			rr.rc, err = rr.req.GetBody()
			if err != nil {
				return err
			}
		} else if rr.br == nil {
			// fall back to making a copy (only do this once)
			err = rr.prepareFromByteReader()
		}
		// indicates that the request body needs to be reset
		rr.reset = true
	}
	return err
}

func removeRequestBody(req *http.Request) {
	req.Body = nil
	req.GetBody = nil
	req.ContentLength = 0
}

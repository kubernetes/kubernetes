package autorest

import (
	"bytes"
	"io"
	"io/ioutil"
	"net/http"
)

// NewRetriableRequest returns a wrapper around an HTTP request that support retry logic.
func NewRetriableRequest(req *http.Request) *RetriableRequest {
	return &RetriableRequest{req: req}
}

// Request returns the wrapped HTTP request.
func (rr *RetriableRequest) Request() *http.Request {
	return rr.req
}

func (rr *RetriableRequest) prepareFromByteReader() (err error) {
	// fall back to making a copy (only do this once)
	b := []byte{}
	if rr.req.ContentLength > 0 {
		b = make([]byte, rr.req.ContentLength)
		_, err = io.ReadFull(rr.req.Body, b)
		if err != nil {
			return err
		}
	} else {
		b, err = ioutil.ReadAll(rr.req.Body)
		if err != nil {
			return err
		}
	}
	rr.br = bytes.NewReader(b)
	rr.req.Body = ioutil.NopCloser(rr.br)
	return err
}

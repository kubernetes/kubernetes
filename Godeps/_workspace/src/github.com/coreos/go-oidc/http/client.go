package http

import (
	"io/ioutil"
	"net/http"
	"net/http/httptest"
)

type Client interface {
	Do(*http.Request) (*http.Response, error)
}

type HandlerClient struct {
	Handler http.Handler
}

func (hc *HandlerClient) Do(r *http.Request) (*http.Response, error) {
	w := httptest.NewRecorder()
	hc.Handler.ServeHTTP(w, r)

	resp := http.Response{
		StatusCode: w.Code,
		Header:     w.Header(),
		Body:       ioutil.NopCloser(w.Body),
	}

	return &resp, nil
}

type RequestRecorder struct {
	Response *http.Response
	Error    error

	Request *http.Request
}

func (rr *RequestRecorder) Do(req *http.Request) (*http.Response, error) {
	rr.Request = req

	if rr.Response == nil && rr.Error == nil {
		panic("RequestRecorder Response and Error cannot both be nil")
	} else if rr.Response != nil && rr.Error != nil {
		panic("RequestRecorder Response and Error cannot both be non-nil")
	}

	return rr.Response, rr.Error
}

func (rr *RequestRecorder) RoundTrip(req *http.Request) (*http.Response, error) {
	return rr.Do(req)
}

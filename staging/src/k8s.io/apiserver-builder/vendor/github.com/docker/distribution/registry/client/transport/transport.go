package transport

import (
	"io"
	"net/http"
	"sync"
)

// RequestModifier represents an object which will do an inplace
// modification of an HTTP request.
type RequestModifier interface {
	ModifyRequest(*http.Request) error
}

type headerModifier http.Header

// NewHeaderRequestModifier returns a new RequestModifier which will
// add the given headers to a request.
func NewHeaderRequestModifier(header http.Header) RequestModifier {
	return headerModifier(header)
}

func (h headerModifier) ModifyRequest(req *http.Request) error {
	for k, s := range http.Header(h) {
		req.Header[k] = append(req.Header[k], s...)
	}

	return nil
}

// NewTransport creates a new transport which will apply modifiers to
// the request on a RoundTrip call.
func NewTransport(base http.RoundTripper, modifiers ...RequestModifier) http.RoundTripper {
	return &transport{
		Modifiers: modifiers,
		Base:      base,
	}
}

// transport is an http.RoundTripper that makes HTTP requests after
// copying and modifying the request
type transport struct {
	Modifiers []RequestModifier
	Base      http.RoundTripper

	mu     sync.Mutex                      // guards modReq
	modReq map[*http.Request]*http.Request // original -> modified
}

// RoundTrip authorizes and authenticates the request with an
// access token. If no token exists or token is expired,
// tries to refresh/fetch a new token.
func (t *transport) RoundTrip(req *http.Request) (*http.Response, error) {
	req2 := cloneRequest(req)
	for _, modifier := range t.Modifiers {
		if err := modifier.ModifyRequest(req2); err != nil {
			return nil, err
		}
	}

	t.setModReq(req, req2)
	res, err := t.base().RoundTrip(req2)
	if err != nil {
		t.setModReq(req, nil)
		return nil, err
	}
	res.Body = &onEOFReader{
		rc: res.Body,
		fn: func() { t.setModReq(req, nil) },
	}
	return res, nil
}

// CancelRequest cancels an in-flight request by closing its connection.
func (t *transport) CancelRequest(req *http.Request) {
	type canceler interface {
		CancelRequest(*http.Request)
	}
	if cr, ok := t.base().(canceler); ok {
		t.mu.Lock()
		modReq := t.modReq[req]
		delete(t.modReq, req)
		t.mu.Unlock()
		cr.CancelRequest(modReq)
	}
}

func (t *transport) base() http.RoundTripper {
	if t.Base != nil {
		return t.Base
	}
	return http.DefaultTransport
}

func (t *transport) setModReq(orig, mod *http.Request) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.modReq == nil {
		t.modReq = make(map[*http.Request]*http.Request)
	}
	if mod == nil {
		delete(t.modReq, orig)
	} else {
		t.modReq[orig] = mod
	}
}

// cloneRequest returns a clone of the provided *http.Request.
// The clone is a shallow copy of the struct and its Header map.
func cloneRequest(r *http.Request) *http.Request {
	// shallow copy of the struct
	r2 := new(http.Request)
	*r2 = *r
	// deep copy of the Header
	r2.Header = make(http.Header, len(r.Header))
	for k, s := range r.Header {
		r2.Header[k] = append([]string(nil), s...)
	}

	return r2
}

type onEOFReader struct {
	rc io.ReadCloser
	fn func()
}

func (r *onEOFReader) Read(p []byte) (n int, err error) {
	n, err = r.rc.Read(p)
	if err == io.EOF {
		r.runFunc()
	}
	return
}

func (r *onEOFReader) Close() error {
	err := r.rc.Close()
	r.runFunc()
	return err
}

func (r *onEOFReader) runFunc() {
	if fn := r.fn; fn != nil {
		fn()
		r.fn = nil
	}
}

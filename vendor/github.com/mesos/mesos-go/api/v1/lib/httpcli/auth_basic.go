package httpcli

import (
	"net/http"
)

// roundTripperFunc is the functional adaptation of http.RoundTripper
type roundTripperFunc func(*http.Request) (*http.Response, error)

// RoundTrip implements RoundTripper for roundTripperFunc
func (f roundTripperFunc) RoundTrip(req *http.Request) (*http.Response, error) { return f(req) }

// BasicAuth generates a functional config option that sets HTTP Basic authentication for a Client
func BasicAuth(username, passwd string) ConfigOpt {
	// TODO(jdef) this could be more efficient. according to the stdlib we're not supposed to
	// mutate the original Request, so we copy here (including headers). another approach would
	// be to generate a functional RequestOpt that adds the right header.
	return WrapRoundTripper(func(rt http.RoundTripper) http.RoundTripper {
		return roundTripperFunc(func(req *http.Request) (*http.Response, error) {
			var h http.Header
			if req.Header != nil {
				h = make(http.Header, len(req.Header))
				for k, v := range req.Header {
					h[k] = append(make([]string, 0, len(v)), v...)
				}
			}
			clonedReq := *req
			clonedReq.Header = h
			clonedReq.SetBasicAuth(username, passwd)
			return rt.RoundTrip(&clonedReq)
		})
	})
}

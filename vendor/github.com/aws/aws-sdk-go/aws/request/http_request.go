package request

import (
	"io"
	"net/http"
	"net/url"
)

func copyHTTPRequest(r *http.Request, body io.ReadCloser) *http.Request {
	req := new(http.Request)
	*req = *r
	req.URL = &url.URL{}
	*req.URL = *r.URL
	req.Body = body

	req.Header = http.Header{}
	for k, v := range r.Header {
		for _, vv := range v {
			req.Header.Add(k, vv)
		}
	}

	return req
}

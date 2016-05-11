// +build !go1.5

package request

import (
	"io"
	"net/http"
)

func copyHTTPRequest(r *http.Request, body io.ReadCloser) *http.Request {
	return &http.Request{
		URL:           r.URL,
		Header:        r.Header,
		Close:         r.Close,
		Form:          r.Form,
		PostForm:      r.PostForm,
		Body:          body,
		MultipartForm: r.MultipartForm,
		Host:          r.Host,
		Method:        r.Method,
		Proto:         r.Proto,
		ContentLength: r.ContentLength,
	}
}

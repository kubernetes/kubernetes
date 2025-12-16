//go:build !js

package websocket

import (
	"net/http"
)

type rwUnwrapper interface {
	Unwrap() http.ResponseWriter
}

// hijacker returns the Hijacker interface of the http.ResponseWriter.
// It follows the Unwrap method of the http.ResponseWriter if available,
// matching the behavior of http.ResponseController. If the Hijacker
// interface is not found, it returns false.
//
// Since the http.ResponseController is not available in Go 1.19, and
// does not support checking the presence of the Hijacker interface,
// this function is used to provide a consistent way to check for the
// Hijacker interface across Go versions.
func hijacker(rw http.ResponseWriter) (http.Hijacker, bool) {
	for {
		switch t := rw.(type) {
		case http.Hijacker:
			return t, true
		case rwUnwrapper:
			rw = t.Unwrap()
		default:
			return nil, false
		}
	}
}

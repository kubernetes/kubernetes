// +build go1.8

package request

import "net/http"

// Is a http.NoBody reader instructing Go HTTP client to not include
// and body in the HTTP request.
var noBodyReader = http.NoBody

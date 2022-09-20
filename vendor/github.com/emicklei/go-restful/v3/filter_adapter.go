package restful

import (
	"net/http"
)

// HttpMiddlewareHandler is a function that takes a http.Handler and returns a http.Handler
type HttpMiddlewareHandler func(http.Handler) http.Handler

// HttpMiddlewareHandlerToFilter converts a HttpMiddlewareHandler to a FilterFunction.
func HttpMiddlewareHandlerToFilter(middleware HttpMiddlewareHandler) FilterFunction {
	return func(req *Request, resp *Response, chain *FilterChain) {
		next := http.HandlerFunc(func(rw http.ResponseWriter, r *http.Request) {
			req.Request = r
			resp.ResponseWriter = rw
			chain.ProcessFilter(req, resp)
		})

		middleware(next).ServeHTTP(resp.ResponseWriter, req.Request)
	}
}

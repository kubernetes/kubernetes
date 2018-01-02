package denco

import (
	"net/http"
)

// Mux represents a multiplexer for HTTP request.
type Mux struct{}

// NewMux returns a new Mux.
func NewMux() *Mux {
	return &Mux{}
}

// GET is shorthand of Mux.Handler("GET", path, handler).
func (m *Mux) GET(path string, handler HandlerFunc) Handler {
	return m.Handler("GET", path, handler)
}

// POST is shorthand of Mux.Handler("POST", path, handler).
func (m *Mux) POST(path string, handler HandlerFunc) Handler {
	return m.Handler("POST", path, handler)
}

// PUT is shorthand of Mux.Handler("PUT", path, handler).
func (m *Mux) PUT(path string, handler HandlerFunc) Handler {
	return m.Handler("PUT", path, handler)
}

// HEAD is shorthand of Mux.Handler("HEAD", path, handler).
func (m *Mux) HEAD(path string, handler HandlerFunc) Handler {
	return m.Handler("HEAD", path, handler)
}

// Handler returns a handler for HTTP method.
func (m *Mux) Handler(method, path string, handler HandlerFunc) Handler {
	return Handler{
		Method: method,
		Path:   path,
		Func:   handler,
	}
}

// Build builds a http.Handler.
func (m *Mux) Build(handlers []Handler) (http.Handler, error) {
	recordMap := make(map[string][]Record)
	for _, h := range handlers {
		recordMap[h.Method] = append(recordMap[h.Method], NewRecord(h.Path, h.Func))
	}
	mux := newServeMux()
	for m, records := range recordMap {
		router := New()
		if err := router.Build(records); err != nil {
			return nil, err
		}
		mux.routers[m] = router
	}
	return mux, nil
}

// Handler represents a handler of HTTP request.
type Handler struct {
	// Method is an HTTP method.
	Method string

	// Path is a routing path for handler.
	Path string

	// Func is a function of handler of HTTP request.
	Func HandlerFunc
}

// The HandlerFunc type is aliased to type of handler function.
type HandlerFunc func(w http.ResponseWriter, r *http.Request, params Params)

type serveMux struct {
	routers map[string]*Router
}

func newServeMux() *serveMux {
	return &serveMux{
		routers: make(map[string]*Router),
	}
}

// ServeHTTP implements http.Handler interface.
func (mux *serveMux) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	handler, params := mux.handler(r.Method, r.URL.Path)
	handler(w, r, params)
}

func (mux *serveMux) handler(method, path string) (HandlerFunc, []Param) {
	if router, found := mux.routers[method]; found {
		if handler, params, found := router.Lookup(path); found {
			return handler.(HandlerFunc), params
		}
	}
	return NotFound, nil
}

// NotFound replies to the request with an HTTP 404 not found error.
// NotFound is called when unknown HTTP method or a handler not found.
// If you want to use the your own NotFound handler, please overwrite this variable.
var NotFound = func(w http.ResponseWriter, r *http.Request, _ Params) {
	http.NotFound(w, r)
}

package dns

import (
	"strings"
	"sync"
)

// ServeMux is an DNS request multiplexer. It matches the zone name of
// each incoming request against a list of registered patterns add calls
// the handler for the pattern that most closely matches the zone name.
//
// ServeMux is DNSSEC aware, meaning that queries for the DS record are
// redirected to the parent zone (if that is also registered), otherwise
// the child gets the query.
//
// ServeMux is also safe for concurrent access from multiple goroutines.
//
// The zero ServeMux is empty and ready for use.
type ServeMux struct {
	z map[string]Handler
	m sync.RWMutex
}

// NewServeMux allocates and returns a new ServeMux.
func NewServeMux() *ServeMux {
	return new(ServeMux)
}

// DefaultServeMux is the default ServeMux used by Serve.
var DefaultServeMux = NewServeMux()

func (mux *ServeMux) match(q string, t uint16) Handler {
	mux.m.RLock()
	defer mux.m.RUnlock()
	if mux.z == nil {
		return nil
	}

	var handler Handler

	// TODO(tmthrgd): Once https://go-review.googlesource.com/c/go/+/137575
	// lands in a go release, replace the following with strings.ToLower.
	var sb strings.Builder
	for i := 0; i < len(q); i++ {
		c := q[i]
		if !(c >= 'A' && c <= 'Z') {
			continue
		}

		sb.Grow(len(q))
		sb.WriteString(q[:i])

		for ; i < len(q); i++ {
			c := q[i]
			if c >= 'A' && c <= 'Z' {
				c += 'a' - 'A'
			}

			sb.WriteByte(c)
		}

		q = sb.String()
		break
	}

	for off, end := 0, false; !end; off, end = NextLabel(q, off) {
		if h, ok := mux.z[q[off:]]; ok {
			if t != TypeDS {
				return h
			}
			// Continue for DS to see if we have a parent too, if so delegate to the parent
			handler = h
		}
	}

	// Wildcard match, if we have found nothing try the root zone as a last resort.
	if h, ok := mux.z["."]; ok {
		return h
	}

	return handler
}

// Handle adds a handler to the ServeMux for pattern.
func (mux *ServeMux) Handle(pattern string, handler Handler) {
	if pattern == "" {
		panic("dns: invalid pattern " + pattern)
	}
	mux.m.Lock()
	if mux.z == nil {
		mux.z = make(map[string]Handler)
	}
	mux.z[Fqdn(pattern)] = handler
	mux.m.Unlock()
}

// HandleFunc adds a handler function to the ServeMux for pattern.
func (mux *ServeMux) HandleFunc(pattern string, handler func(ResponseWriter, *Msg)) {
	mux.Handle(pattern, HandlerFunc(handler))
}

// HandleRemove deregisters the handler specific for pattern from the ServeMux.
func (mux *ServeMux) HandleRemove(pattern string) {
	if pattern == "" {
		panic("dns: invalid pattern " + pattern)
	}
	mux.m.Lock()
	delete(mux.z, Fqdn(pattern))
	mux.m.Unlock()
}

// ServeDNS dispatches the request to the handler whose pattern most
// closely matches the request message.
//
// ServeDNS is DNSSEC aware, meaning that queries for the DS record
// are redirected to the parent zone (if that is also registered),
// otherwise the child gets the query.
//
// If no handler is found, or there is no question, a standard SERVFAIL
// message is returned
func (mux *ServeMux) ServeDNS(w ResponseWriter, req *Msg) {
	var h Handler
	if len(req.Question) >= 1 { // allow more than one question
		h = mux.match(req.Question[0].Name, req.Question[0].Qtype)
	}

	if h != nil {
		h.ServeDNS(w, req)
	} else {
		HandleFailed(w, req)
	}
}

// Handle registers the handler with the given pattern
// in the DefaultServeMux. The documentation for
// ServeMux explains how patterns are matched.
func Handle(pattern string, handler Handler) { DefaultServeMux.Handle(pattern, handler) }

// HandleRemove deregisters the handle with the given pattern
// in the DefaultServeMux.
func HandleRemove(pattern string) { DefaultServeMux.HandleRemove(pattern) }

// HandleFunc registers the handler function with the given pattern
// in the DefaultServeMux.
func HandleFunc(pattern string, handler func(ResponseWriter, *Msg)) {
	DefaultServeMux.HandleFunc(pattern, handler)
}

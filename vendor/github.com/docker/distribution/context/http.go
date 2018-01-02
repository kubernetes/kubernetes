package context

import (
	"errors"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/docker/distribution/uuid"
	"github.com/gorilla/mux"
	log "github.com/sirupsen/logrus"
)

// Common errors used with this package.
var (
	ErrNoRequestContext        = errors.New("no http request in context")
	ErrNoResponseWriterContext = errors.New("no http response in context")
)

func parseIP(ipStr string) net.IP {
	ip := net.ParseIP(ipStr)
	if ip == nil {
		log.Warnf("invalid remote IP address: %q", ipStr)
	}
	return ip
}

// RemoteAddr extracts the remote address of the request, taking into
// account proxy headers.
func RemoteAddr(r *http.Request) string {
	if prior := r.Header.Get("X-Forwarded-For"); prior != "" {
		proxies := strings.Split(prior, ",")
		if len(proxies) > 0 {
			remoteAddr := strings.Trim(proxies[0], " ")
			if parseIP(remoteAddr) != nil {
				return remoteAddr
			}
		}
	}
	// X-Real-Ip is less supported, but worth checking in the
	// absence of X-Forwarded-For
	if realIP := r.Header.Get("X-Real-Ip"); realIP != "" {
		if parseIP(realIP) != nil {
			return realIP
		}
	}

	return r.RemoteAddr
}

// RemoteIP extracts the remote IP of the request, taking into
// account proxy headers.
func RemoteIP(r *http.Request) string {
	addr := RemoteAddr(r)

	// Try parsing it as "IP:port"
	if ip, _, err := net.SplitHostPort(addr); err == nil {
		return ip
	}

	return addr
}

// WithRequest places the request on the context. The context of the request
// is assigned a unique id, available at "http.request.id". The request itself
// is available at "http.request". Other common attributes are available under
// the prefix "http.request.". If a request is already present on the context,
// this method will panic.
func WithRequest(ctx Context, r *http.Request) Context {
	if ctx.Value("http.request") != nil {
		// NOTE(stevvooe): This needs to be considered a programming error. It
		// is unlikely that we'd want to have more than one request in
		// context.
		panic("only one request per context")
	}

	return &httpRequestContext{
		Context:   ctx,
		startedAt: time.Now(),
		id:        uuid.Generate().String(),
		r:         r,
	}
}

// GetRequest returns the http request in the given context. Returns
// ErrNoRequestContext if the context does not have an http request associated
// with it.
func GetRequest(ctx Context) (*http.Request, error) {
	if r, ok := ctx.Value("http.request").(*http.Request); r != nil && ok {
		return r, nil
	}
	return nil, ErrNoRequestContext
}

// GetRequestID attempts to resolve the current request id, if possible. An
// error is return if it is not available on the context.
func GetRequestID(ctx Context) string {
	return GetStringValue(ctx, "http.request.id")
}

// WithResponseWriter returns a new context and response writer that makes
// interesting response statistics available within the context.
func WithResponseWriter(ctx Context, w http.ResponseWriter) (Context, http.ResponseWriter) {
	if closeNotifier, ok := w.(http.CloseNotifier); ok {
		irwCN := &instrumentedResponseWriterCN{
			instrumentedResponseWriter: instrumentedResponseWriter{
				ResponseWriter: w,
				Context:        ctx,
			},
			CloseNotifier: closeNotifier,
		}

		return irwCN, irwCN
	}

	irw := instrumentedResponseWriter{
		ResponseWriter: w,
		Context:        ctx,
	}
	return &irw, &irw
}

// GetResponseWriter returns the http.ResponseWriter from the provided
// context. If not present, ErrNoResponseWriterContext is returned. The
// returned instance provides instrumentation in the context.
func GetResponseWriter(ctx Context) (http.ResponseWriter, error) {
	v := ctx.Value("http.response")

	rw, ok := v.(http.ResponseWriter)
	if !ok || rw == nil {
		return nil, ErrNoResponseWriterContext
	}

	return rw, nil
}

// getVarsFromRequest let's us change request vars implementation for testing
// and maybe future changes.
var getVarsFromRequest = mux.Vars

// WithVars extracts gorilla/mux vars and makes them available on the returned
// context. Variables are available at keys with the prefix "vars.". For
// example, if looking for the variable "name", it can be accessed as
// "vars.name". Implementations that are accessing values need not know that
// the underlying context is implemented with gorilla/mux vars.
func WithVars(ctx Context, r *http.Request) Context {
	return &muxVarsContext{
		Context: ctx,
		vars:    getVarsFromRequest(r),
	}
}

// GetRequestLogger returns a logger that contains fields from the request in
// the current context. If the request is not available in the context, no
// fields will display. Request loggers can safely be pushed onto the context.
func GetRequestLogger(ctx Context) Logger {
	return GetLogger(ctx,
		"http.request.id",
		"http.request.method",
		"http.request.host",
		"http.request.uri",
		"http.request.referer",
		"http.request.useragent",
		"http.request.remoteaddr",
		"http.request.contenttype")
}

// GetResponseLogger reads the current response stats and builds a logger.
// Because the values are read at call time, pushing a logger returned from
// this function on the context will lead to missing or invalid data. Only
// call this at the end of a request, after the response has been written.
func GetResponseLogger(ctx Context) Logger {
	l := getLogrusLogger(ctx,
		"http.response.written",
		"http.response.status",
		"http.response.contenttype")

	duration := Since(ctx, "http.request.startedat")

	if duration > 0 {
		l = l.WithField("http.response.duration", duration.String())
	}

	return l
}

// httpRequestContext makes information about a request available to context.
type httpRequestContext struct {
	Context

	startedAt time.Time
	id        string
	r         *http.Request
}

// Value returns a keyed element of the request for use in the context. To get
// the request itself, query "request". For other components, access them as
// "request.<component>". For example, r.RequestURI
func (ctx *httpRequestContext) Value(key interface{}) interface{} {
	if keyStr, ok := key.(string); ok {
		if keyStr == "http.request" {
			return ctx.r
		}

		if !strings.HasPrefix(keyStr, "http.request.") {
			goto fallback
		}

		parts := strings.Split(keyStr, ".")

		if len(parts) != 3 {
			goto fallback
		}

		switch parts[2] {
		case "uri":
			return ctx.r.RequestURI
		case "remoteaddr":
			return RemoteAddr(ctx.r)
		case "method":
			return ctx.r.Method
		case "host":
			return ctx.r.Host
		case "referer":
			referer := ctx.r.Referer()
			if referer != "" {
				return referer
			}
		case "useragent":
			return ctx.r.UserAgent()
		case "id":
			return ctx.id
		case "startedat":
			return ctx.startedAt
		case "contenttype":
			ct := ctx.r.Header.Get("Content-Type")
			if ct != "" {
				return ct
			}
		}
	}

fallback:
	return ctx.Context.Value(key)
}

type muxVarsContext struct {
	Context
	vars map[string]string
}

func (ctx *muxVarsContext) Value(key interface{}) interface{} {
	if keyStr, ok := key.(string); ok {
		if keyStr == "vars" {
			return ctx.vars
		}

		if strings.HasPrefix(keyStr, "vars.") {
			keyStr = strings.TrimPrefix(keyStr, "vars.")
		}

		if v, ok := ctx.vars[keyStr]; ok {
			return v
		}
	}

	return ctx.Context.Value(key)
}

// instrumentedResponseWriterCN provides response writer information in a
// context. It implements http.CloseNotifier so that users can detect
// early disconnects.
type instrumentedResponseWriterCN struct {
	instrumentedResponseWriter
	http.CloseNotifier
}

// instrumentedResponseWriter provides response writer information in a
// context. This variant is only used in the case where CloseNotifier is not
// implemented by the parent ResponseWriter.
type instrumentedResponseWriter struct {
	http.ResponseWriter
	Context

	mu      sync.Mutex
	status  int
	written int64
}

func (irw *instrumentedResponseWriter) Write(p []byte) (n int, err error) {
	n, err = irw.ResponseWriter.Write(p)

	irw.mu.Lock()
	irw.written += int64(n)

	// Guess the likely status if not set.
	if irw.status == 0 {
		irw.status = http.StatusOK
	}

	irw.mu.Unlock()

	return
}

func (irw *instrumentedResponseWriter) WriteHeader(status int) {
	irw.ResponseWriter.WriteHeader(status)

	irw.mu.Lock()
	irw.status = status
	irw.mu.Unlock()
}

func (irw *instrumentedResponseWriter) Flush() {
	if flusher, ok := irw.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}

func (irw *instrumentedResponseWriter) Value(key interface{}) interface{} {
	if keyStr, ok := key.(string); ok {
		if keyStr == "http.response" {
			return irw
		}

		if !strings.HasPrefix(keyStr, "http.response.") {
			goto fallback
		}

		parts := strings.Split(keyStr, ".")

		if len(parts) != 3 {
			goto fallback
		}

		irw.mu.Lock()
		defer irw.mu.Unlock()

		switch parts[2] {
		case "written":
			return irw.written
		case "status":
			return irw.status
		case "contenttype":
			contentType := irw.Header().Get("Content-Type")
			if contentType != "" {
				return contentType
			}
		}
	}

fallback:
	return irw.Context.Value(key)
}

func (irw *instrumentedResponseWriterCN) Value(key interface{}) interface{} {
	if keyStr, ok := key.(string); ok {
		if keyStr == "http.response" {
			return irw
		}
	}

	return irw.instrumentedResponseWriter.Value(key)
}

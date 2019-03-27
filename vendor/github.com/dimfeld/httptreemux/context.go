// +build go1.7

package httptreemux

import (
	"context"
	"net/http"
)

// ContextGroup is a wrapper around Group, with the purpose of mimicking its API, but with the use of http.HandlerFunc-based handlers.
// Instead of passing a parameter map via the handler (i.e. httptreemux.HandlerFunc), the path parameters are accessed via the request
// object's context.
type ContextGroup struct {
	group *Group
}

// UsingContext wraps the receiver to return a new instance of a ContextGroup.
// The returned ContextGroup is a sibling to its wrapped Group, within the parent TreeMux.
// The choice of using a *Group as the receiver, as opposed to a function parameter, allows chaining
// while method calls between a TreeMux, Group, and ContextGroup. For example:
//
//              tree := httptreemux.New()
//              group := tree.NewGroup("/api")
//
//              group.GET("/v1", func(w http.ResponseWriter, r *http.Request, params map[string]string) {
//                  w.Write([]byte(`GET /api/v1`))
//              })
//
//              group.UsingContext().GET("/v2", func(w http.ResponseWriter, r *http.Request) {
//                  w.Write([]byte(`GET /api/v2`))
//              })
//
//              http.ListenAndServe(":8080", tree)
//
func (g *Group) UsingContext() *ContextGroup {
	return &ContextGroup{g}
}

// NewContextGroup adds a child context group to its path.
func (cg *ContextGroup) NewContextGroup(path string) *ContextGroup {
	return &ContextGroup{cg.group.NewGroup(path)}
}

func (cg *ContextGroup) NewGroup(path string) *ContextGroup {
	return cg.NewContextGroup(path)
}

// Handle allows handling HTTP requests via an http.HandlerFunc, as opposed to an httptreemux.HandlerFunc.
// Any parameters from the request URL are stored in a map[string]string in the request's context.
func (cg *ContextGroup) Handle(method, path string, handler http.HandlerFunc) {
	cg.group.Handle(method, path, func(w http.ResponseWriter, r *http.Request, params map[string]string) {
		if params != nil {
			r = r.WithContext(AddParamsToContext(r.Context(), params))
		}
		handler(w, r)
	})
}

// Handler allows handling HTTP requests via an http.Handler interface, as opposed to an httptreemux.HandlerFunc.
// Any parameters from the request URL are stored in a map[string]string in the request's context.
func (cg *ContextGroup) Handler(method, path string, handler http.Handler) {
	cg.group.Handle(method, path, func(w http.ResponseWriter, r *http.Request, params map[string]string) {
		if params != nil {
			r = r.WithContext(AddParamsToContext(r.Context(), params))
		}
		handler.ServeHTTP(w, r)
	})
}

// GET is convenience method for handling GET requests on a context group.
func (cg *ContextGroup) GET(path string, handler http.HandlerFunc) {
	cg.Handle("GET", path, handler)
}

// POST is convenience method for handling POST requests on a context group.
func (cg *ContextGroup) POST(path string, handler http.HandlerFunc) {
	cg.Handle("POST", path, handler)
}

// PUT is convenience method for handling PUT requests on a context group.
func (cg *ContextGroup) PUT(path string, handler http.HandlerFunc) {
	cg.Handle("PUT", path, handler)
}

// DELETE is convenience method for handling DELETE requests on a context group.
func (cg *ContextGroup) DELETE(path string, handler http.HandlerFunc) {
	cg.Handle("DELETE", path, handler)
}

// PATCH is convenience method for handling PATCH requests on a context group.
func (cg *ContextGroup) PATCH(path string, handler http.HandlerFunc) {
	cg.Handle("PATCH", path, handler)
}

// HEAD is convenience method for handling HEAD requests on a context group.
func (cg *ContextGroup) HEAD(path string, handler http.HandlerFunc) {
	cg.Handle("HEAD", path, handler)
}

// OPTIONS is convenience method for handling OPTIONS requests on a context group.
func (cg *ContextGroup) OPTIONS(path string, handler http.HandlerFunc) {
	cg.Handle("OPTIONS", path, handler)
}

// ContextParams returns the params map associated with the given context if one exists. Otherwise, an empty map is returned.
func ContextParams(ctx context.Context) map[string]string {
	if p, ok := ctx.Value(paramsContextKey).(map[string]string); ok {
		return p
	}
	return map[string]string{}
}

// AddParamsToContext inserts a parameters map into a context using
// the package's internal context key. Clients of this package should
// really only use this for unit tests.
func AddParamsToContext(ctx context.Context, params map[string]string) context.Context {
	return context.WithValue(ctx, paramsContextKey, params)
}

type contextKey int

// paramsContextKey is used to retrieve a path's params map from a request's context.
const paramsContextKey contextKey = 0

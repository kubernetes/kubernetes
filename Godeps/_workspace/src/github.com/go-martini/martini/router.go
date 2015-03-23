package martini

import (
	"fmt"
	"net/http"
	"reflect"
	"regexp"
	"strconv"
	"sync"
)

// Params is a map of name/value pairs for named routes. An instance of martini.Params is available to be injected into any route handler.
type Params map[string]string

// Router is Martini's de-facto routing interface. Supports HTTP verbs, stacked handlers, and dependency injection.
type Router interface {
	Routes

	// Group adds a group where related routes can be added.
	Group(string, func(Router), ...Handler)
	// Get adds a route for a HTTP GET request to the specified matching pattern.
	Get(string, ...Handler) Route
	// Patch adds a route for a HTTP PATCH request to the specified matching pattern.
	Patch(string, ...Handler) Route
	// Post adds a route for a HTTP POST request to the specified matching pattern.
	Post(string, ...Handler) Route
	// Put adds a route for a HTTP PUT request to the specified matching pattern.
	Put(string, ...Handler) Route
	// Delete adds a route for a HTTP DELETE request to the specified matching pattern.
	Delete(string, ...Handler) Route
	// Options adds a route for a HTTP OPTIONS request to the specified matching pattern.
	Options(string, ...Handler) Route
	// Head adds a route for a HTTP HEAD request to the specified matching pattern.
	Head(string, ...Handler) Route
	// Any adds a route for any HTTP method request to the specified matching pattern.
	Any(string, ...Handler) Route
	// AddRoute adds a route for a given HTTP method request to the specified matching pattern.
	AddRoute(string, string, ...Handler) Route

	// NotFound sets the handlers that are called when a no route matches a request. Throws a basic 404 by default.
	NotFound(...Handler)

	// Handle is the entry point for routing. This is used as a martini.Handler
	Handle(http.ResponseWriter, *http.Request, Context)
}

type router struct {
	routes     []*route
	notFounds  []Handler
	groups     []group
	routesLock sync.RWMutex
}

type group struct {
	pattern  string
	handlers []Handler
}

// NewRouter creates a new Router instance.
// If you aren't using ClassicMartini, then you can add Routes as a
// service with:
//
//	m := martini.New()
//	r := martini.NewRouter()
//	m.MapTo(r, (*martini.Routes)(nil))
//
// If you are using ClassicMartini, then this is done for you.
func NewRouter() Router {
	return &router{notFounds: []Handler{http.NotFound}, groups: make([]group, 0)}
}

func (r *router) Group(pattern string, fn func(Router), h ...Handler) {
	r.groups = append(r.groups, group{pattern, h})
	fn(r)
	r.groups = r.groups[:len(r.groups)-1]
}

func (r *router) Get(pattern string, h ...Handler) Route {
	return r.addRoute("GET", pattern, h)
}

func (r *router) Patch(pattern string, h ...Handler) Route {
	return r.addRoute("PATCH", pattern, h)
}

func (r *router) Post(pattern string, h ...Handler) Route {
	return r.addRoute("POST", pattern, h)
}

func (r *router) Put(pattern string, h ...Handler) Route {
	return r.addRoute("PUT", pattern, h)
}

func (r *router) Delete(pattern string, h ...Handler) Route {
	return r.addRoute("DELETE", pattern, h)
}

func (r *router) Options(pattern string, h ...Handler) Route {
	return r.addRoute("OPTIONS", pattern, h)
}

func (r *router) Head(pattern string, h ...Handler) Route {
	return r.addRoute("HEAD", pattern, h)
}

func (r *router) Any(pattern string, h ...Handler) Route {
	return r.addRoute("*", pattern, h)
}

func (r *router) AddRoute(method, pattern string, h ...Handler) Route {
	return r.addRoute(method, pattern, h)
}

func (r *router) Handle(res http.ResponseWriter, req *http.Request, context Context) {
	for _, route := range r.getRoutes() {
		ok, vals := route.Match(req.Method, req.URL.Path)
		if ok {
			params := Params(vals)
			context.Map(params)
			route.Handle(context, res)
			return
		}
	}

	// no routes exist, 404
	c := &routeContext{context, 0, r.notFounds}
	context.MapTo(c, (*Context)(nil))
	c.run()
}

func (r *router) NotFound(handler ...Handler) {
	r.notFounds = handler
}

func (r *router) addRoute(method string, pattern string, handlers []Handler) *route {
	if len(r.groups) > 0 {
		groupPattern := ""
		h := make([]Handler, 0)
		for _, g := range r.groups {
			groupPattern += g.pattern
			h = append(h, g.handlers...)
		}

		pattern = groupPattern + pattern
		h = append(h, handlers...)
		handlers = h
	}

	route := newRoute(method, pattern, handlers)
	route.Validate()
	r.appendRoute(route)
	return route
}

func (r *router) appendRoute(rt *route) {
	r.routesLock.Lock()
	defer r.routesLock.Unlock()
	r.routes = append(r.routes, rt)
}

func (r *router) getRoutes() []*route {
	r.routesLock.RLock()
	defer r.routesLock.RUnlock()
	return r.routes[:]
}

func (r *router) findRoute(name string) *route {
	for _, route := range r.getRoutes() {
		if route.name == name {
			return route
		}
	}

	return nil
}

// Route is an interface representing a Route in Martini's routing layer.
type Route interface {
	// URLWith returns a rendering of the Route's url with the given string params.
	URLWith([]string) string
	// Name sets a name for the route.
	Name(string)
	// GetName returns the name of the route.
	GetName() string
	// Pattern returns the pattern of the route.
	Pattern() string
	// Method returns the method of the route.
	Method() string
}

type route struct {
	method   string
	regex    *regexp.Regexp
	handlers []Handler
	pattern  string
	name     string
}

var routeReg1 = regexp.MustCompile(`:[^/#?()\.\\]+`)
var routeReg2 = regexp.MustCompile(`\*\*`)

func newRoute(method string, pattern string, handlers []Handler) *route {
	route := route{method, nil, handlers, pattern, ""}
	pattern = routeReg1.ReplaceAllStringFunc(pattern, func(m string) string {
		return fmt.Sprintf(`(?P<%s>[^/#?]+)`, m[1:])
	})
	var index int
	pattern = routeReg2.ReplaceAllStringFunc(pattern, func(m string) string {
		index++
		return fmt.Sprintf(`(?P<_%d>[^#?]*)`, index)
	})
	pattern += `\/?`
	route.regex = regexp.MustCompile(pattern)
	return &route
}

func (r route) MatchMethod(method string) bool {
	return r.method == "*" || method == r.method || (method == "HEAD" && r.method == "GET")
}

func (r route) Match(method string, path string) (bool, map[string]string) {
	// add Any method matching support
	if !r.MatchMethod(method) {
		return false, nil
	}

	matches := r.regex.FindStringSubmatch(path)
	if len(matches) > 0 && matches[0] == path {
		params := make(map[string]string)
		for i, name := range r.regex.SubexpNames() {
			if len(name) > 0 {
				params[name] = matches[i]
			}
		}
		return true, params
	}
	return false, nil
}

func (r *route) Validate() {
	for _, handler := range r.handlers {
		validateHandler(handler)
	}
}

func (r *route) Handle(c Context, res http.ResponseWriter) {
	context := &routeContext{c, 0, r.handlers}
	c.MapTo(context, (*Context)(nil))
	c.MapTo(r, (*Route)(nil))
	context.run()
}

var urlReg = regexp.MustCompile(`:[^/#?()\.\\]+|\(\?P<[a-zA-Z0-9]+>.*\)`)

// URLWith returns the url pattern replacing the parameters for its values
func (r *route) URLWith(args []string) string {
	if len(args) > 0 {
		argCount := len(args)
		i := 0
		url := urlReg.ReplaceAllStringFunc(r.pattern, func(m string) string {
			var val interface{}
			if i < argCount {
				val = args[i]
			} else {
				val = m
			}
			i += 1
			return fmt.Sprintf(`%v`, val)
		})

		return url
	}
	return r.pattern
}

func (r *route) Name(name string) {
	r.name = name
}

func (r *route) GetName() string {
	return r.name
}

func (r *route) Pattern() string {
	return r.pattern
}

func (r *route) Method() string {
	return r.method
}

// Routes is a helper service for Martini's routing layer.
type Routes interface {
	// URLFor returns a rendered URL for the given route. Optional params can be passed to fulfill named parameters in the route.
	URLFor(name string, params ...interface{}) string
	// MethodsFor returns an array of methods available for the path
	MethodsFor(path string) []string
	// All returns an array with all the routes in the router.
	All() []Route
}

// URLFor returns the url for the given route name.
func (r *router) URLFor(name string, params ...interface{}) string {
	route := r.findRoute(name)

	if route == nil {
		panic("route not found")
	}

	var args []string
	for _, param := range params {
		switch v := param.(type) {
		case int:
			args = append(args, strconv.FormatInt(int64(v), 10))
		case string:
			args = append(args, v)
		default:
			if v != nil {
				panic("Arguments passed to URLFor must be integers or strings")
			}
		}
	}

	return route.URLWith(args)
}

func (r *router) All() []Route {
	routes := r.getRoutes()
	var ri = make([]Route, len(routes))

	for i, route := range routes {
		ri[i] = Route(route)
	}

	return ri
}

func hasMethod(methods []string, method string) bool {
	for _, v := range methods {
		if v == method {
			return true
		}
	}
	return false
}

// MethodsFor returns all methods available for path
func (r *router) MethodsFor(path string) []string {
	methods := []string{}
	for _, route := range r.getRoutes() {
		matches := route.regex.FindStringSubmatch(path)
		if len(matches) > 0 && matches[0] == path && !hasMethod(methods, route.method) {
			methods = append(methods, route.method)
		}
	}
	return methods
}

type routeContext struct {
	Context
	index    int
	handlers []Handler
}

func (r *routeContext) Next() {
	r.index += 1
	r.run()
}

func (r *routeContext) run() {
	for r.index < len(r.handlers) {
		handler := r.handlers[r.index]
		vals, err := r.Invoke(handler)
		if err != nil {
			panic(err)
		}
		r.index += 1

		// if the handler returned something, write it to the http response
		if len(vals) > 0 {
			ev := r.Get(reflect.TypeOf(ReturnHandler(nil)))
			handleReturn := ev.Interface().(ReturnHandler)
			handleReturn(r, vals)
		}

		if r.Written() {
			return
		}
	}
}

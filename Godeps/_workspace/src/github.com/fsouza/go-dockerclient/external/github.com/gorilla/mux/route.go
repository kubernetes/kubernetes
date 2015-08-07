// Copyright 2012 The Gorilla Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mux

import (
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"regexp"
	"strings"
)

// Route stores information to match a request and build URLs.
type Route struct {
	// Parent where the route was registered (a Router).
	parent parentRoute
	// Request handler for the route.
	handler http.Handler
	// List of matchers.
	matchers []matcher
	// Manager for the variables from host and path.
	regexp *routeRegexpGroup
	// If true, when the path pattern is "/path/", accessing "/path" will
	// redirect to the former and vice versa.
	strictSlash bool
	// If true, this route never matches: it is only used to build URLs.
	buildOnly bool
	// The name used to build URLs.
	name string
	// Error resulted from building a route.
	err error

	buildVarsFunc BuildVarsFunc
}

// Match matches the route against the request.
func (r *Route) Match(req *http.Request, match *RouteMatch) bool {
	if r.buildOnly || r.err != nil {
		return false
	}
	// Match everything.
	for _, m := range r.matchers {
		if matched := m.Match(req, match); !matched {
			return false
		}
	}
	// Yay, we have a match. Let's collect some info about it.
	if match.Route == nil {
		match.Route = r
	}
	if match.Handler == nil {
		match.Handler = r.handler
	}
	if match.Vars == nil {
		match.Vars = make(map[string]string)
	}
	// Set variables.
	if r.regexp != nil {
		r.regexp.setMatch(req, match, r)
	}
	return true
}

// ----------------------------------------------------------------------------
// Route attributes
// ----------------------------------------------------------------------------

// GetError returns an error resulted from building the route, if any.
func (r *Route) GetError() error {
	return r.err
}

// BuildOnly sets the route to never match: it is only used to build URLs.
func (r *Route) BuildOnly() *Route {
	r.buildOnly = true
	return r
}

// Handler --------------------------------------------------------------------

// Handler sets a handler for the route.
func (r *Route) Handler(handler http.Handler) *Route {
	if r.err == nil {
		r.handler = handler
	}
	return r
}

// HandlerFunc sets a handler function for the route.
func (r *Route) HandlerFunc(f func(http.ResponseWriter, *http.Request)) *Route {
	return r.Handler(http.HandlerFunc(f))
}

// GetHandler returns the handler for the route, if any.
func (r *Route) GetHandler() http.Handler {
	return r.handler
}

// Name -----------------------------------------------------------------------

// Name sets the name for the route, used to build URLs.
// If the name was registered already it will be overwritten.
func (r *Route) Name(name string) *Route {
	if r.name != "" {
		r.err = fmt.Errorf("mux: route already has name %q, can't set %q",
			r.name, name)
	}
	if r.err == nil {
		r.name = name
		r.getNamedRoutes()[name] = r
	}
	return r
}

// GetName returns the name for the route, if any.
func (r *Route) GetName() string {
	return r.name
}

// ----------------------------------------------------------------------------
// Matchers
// ----------------------------------------------------------------------------

// matcher types try to match a request.
type matcher interface {
	Match(*http.Request, *RouteMatch) bool
}

// addMatcher adds a matcher to the route.
func (r *Route) addMatcher(m matcher) *Route {
	if r.err == nil {
		r.matchers = append(r.matchers, m)
	}
	return r
}

// addRegexpMatcher adds a host or path matcher and builder to a route.
func (r *Route) addRegexpMatcher(tpl string, matchHost, matchPrefix, matchQuery bool) error {
	if r.err != nil {
		return r.err
	}
	r.regexp = r.getRegexpGroup()
	if !matchHost && !matchQuery {
		if len(tpl) == 0 || tpl[0] != '/' {
			return fmt.Errorf("mux: path must start with a slash, got %q", tpl)
		}
		if r.regexp.path != nil {
			tpl = strings.TrimRight(r.regexp.path.template, "/") + tpl
		}
	}
	rr, err := newRouteRegexp(tpl, matchHost, matchPrefix, matchQuery, r.strictSlash)
	if err != nil {
		return err
	}
	for _, q := range r.regexp.queries {
		if err = uniqueVars(rr.varsN, q.varsN); err != nil {
			return err
		}
	}
	if matchHost {
		if r.regexp.path != nil {
			if err = uniqueVars(rr.varsN, r.regexp.path.varsN); err != nil {
				return err
			}
		}
		r.regexp.host = rr
	} else {
		if r.regexp.host != nil {
			if err = uniqueVars(rr.varsN, r.regexp.host.varsN); err != nil {
				return err
			}
		}
		if matchQuery {
			r.regexp.queries = append(r.regexp.queries, rr)
		} else {
			r.regexp.path = rr
		}
	}
	r.addMatcher(rr)
	return nil
}

// Headers --------------------------------------------------------------------

// headerMatcher matches the request against header values.
type headerMatcher map[string]string

func (m headerMatcher) Match(r *http.Request, match *RouteMatch) bool {
	return matchMapWithString(m, r.Header, true)
}

// Headers adds a matcher for request header values.
// It accepts a sequence of key/value pairs to be matched. For example:
//
//     r := mux.NewRouter()
//     r.Headers("Content-Type", "application/json",
//               "X-Requested-With", "XMLHttpRequest")
//
// The above route will only match if both request header values match.
// Alternatively, you can provide a regular expression and match the header as follows:
//
//     r.Headers("Content-Type", "application/(text|json)",
//               "X-Requested-With", "XMLHttpRequest")
//
// The above route will the same as the previous example, with the addition of matching
// application/text as well.
//
// It the value is an empty string, it will match any value if the key is set.
func (r *Route) Headers(pairs ...string) *Route {
	if r.err == nil {
		var headers map[string]string
		headers, r.err = mapFromPairsToString(pairs...)
		return r.addMatcher(headerMatcher(headers))
	}
	return r
}

// headerRegexMatcher matches the request against the route given a regex for the header
type headerRegexMatcher map[string]*regexp.Regexp

func (m headerRegexMatcher) Match(r *http.Request, match *RouteMatch) bool {
	return matchMapWithRegex(m, r.Header, true)
}

// Regular expressions can be used with headers as well.
// It accepts a sequence of key/value pairs, where the value has regex support. For example
//     r := mux.NewRouter()
//     r.HeadersRegexp("Content-Type", "application/(text|json)",
//               "X-Requested-With", "XMLHttpRequest")
//
// The above route will only match if both the request header matches both regular expressions.
// It the value is an empty string, it will match any value if the key is set.
func (r *Route) HeadersRegexp(pairs ...string) *Route {
	if r.err == nil {
		var headers map[string]*regexp.Regexp
		headers, r.err = mapFromPairsToRegex(pairs...)
		return r.addMatcher(headerRegexMatcher(headers))
	}
	return r
}

// Host -----------------------------------------------------------------------

// Host adds a matcher for the URL host.
// It accepts a template with zero or more URL variables enclosed by {}.
// Variables can define an optional regexp pattern to be matched:
//
// - {name} matches anything until the next dot.
//
// - {name:pattern} matches the given regexp pattern.
//
// For example:
//
//     r := mux.NewRouter()
//     r.Host("www.domain.com")
//     r.Host("{subdomain}.domain.com")
//     r.Host("{subdomain:[a-z]+}.domain.com")
//
// Variable names must be unique in a given route. They can be retrieved
// calling mux.Vars(request).
func (r *Route) Host(tpl string) *Route {
	r.err = r.addRegexpMatcher(tpl, true, false, false)
	return r
}

// MatcherFunc ----------------------------------------------------------------

// MatcherFunc is the function signature used by custom matchers.
type MatcherFunc func(*http.Request, *RouteMatch) bool

func (m MatcherFunc) Match(r *http.Request, match *RouteMatch) bool {
	return m(r, match)
}

// MatcherFunc adds a custom function to be used as request matcher.
func (r *Route) MatcherFunc(f MatcherFunc) *Route {
	return r.addMatcher(f)
}

// Methods --------------------------------------------------------------------

// methodMatcher matches the request against HTTP methods.
type methodMatcher []string

func (m methodMatcher) Match(r *http.Request, match *RouteMatch) bool {
	return matchInArray(m, r.Method)
}

// Methods adds a matcher for HTTP methods.
// It accepts a sequence of one or more methods to be matched, e.g.:
// "GET", "POST", "PUT".
func (r *Route) Methods(methods ...string) *Route {
	for k, v := range methods {
		methods[k] = strings.ToUpper(v)
	}
	return r.addMatcher(methodMatcher(methods))
}

// Path -----------------------------------------------------------------------

// Path adds a matcher for the URL path.
// It accepts a template with zero or more URL variables enclosed by {}. The
// template must start with a "/".
// Variables can define an optional regexp pattern to be matched:
//
// - {name} matches anything until the next slash.
//
// - {name:pattern} matches the given regexp pattern.
//
// For example:
//
//     r := mux.NewRouter()
//     r.Path("/products/").Handler(ProductsHandler)
//     r.Path("/products/{key}").Handler(ProductsHandler)
//     r.Path("/articles/{category}/{id:[0-9]+}").
//       Handler(ArticleHandler)
//
// Variable names must be unique in a given route. They can be retrieved
// calling mux.Vars(request).
func (r *Route) Path(tpl string) *Route {
	r.err = r.addRegexpMatcher(tpl, false, false, false)
	return r
}

// PathPrefix -----------------------------------------------------------------

// PathPrefix adds a matcher for the URL path prefix. This matches if the given
// template is a prefix of the full URL path. See Route.Path() for details on
// the tpl argument.
//
// Note that it does not treat slashes specially ("/foobar/" will be matched by
// the prefix "/foo") so you may want to use a trailing slash here.
//
// Also note that the setting of Router.StrictSlash() has no effect on routes
// with a PathPrefix matcher.
func (r *Route) PathPrefix(tpl string) *Route {
	r.err = r.addRegexpMatcher(tpl, false, true, false)
	return r
}

// Query ----------------------------------------------------------------------

// Queries adds a matcher for URL query values.
// It accepts a sequence of key/value pairs. Values may define variables.
// For example:
//
//     r := mux.NewRouter()
//     r.Queries("foo", "bar", "id", "{id:[0-9]+}")
//
// The above route will only match if the URL contains the defined queries
// values, e.g.: ?foo=bar&id=42.
//
// It the value is an empty string, it will match any value if the key is set.
//
// Variables can define an optional regexp pattern to be matched:
//
// - {name} matches anything until the next slash.
//
// - {name:pattern} matches the given regexp pattern.
func (r *Route) Queries(pairs ...string) *Route {
	length := len(pairs)
	if length%2 != 0 {
		r.err = fmt.Errorf(
			"mux: number of parameters must be multiple of 2, got %v", pairs)
		return nil
	}
	for i := 0; i < length; i += 2 {
		if r.err = r.addRegexpMatcher(pairs[i]+"="+pairs[i+1], false, false, true); r.err != nil {
			return r
		}
	}

	return r
}

// Schemes --------------------------------------------------------------------

// schemeMatcher matches the request against URL schemes.
type schemeMatcher []string

func (m schemeMatcher) Match(r *http.Request, match *RouteMatch) bool {
	return matchInArray(m, r.URL.Scheme)
}

// Schemes adds a matcher for URL schemes.
// It accepts a sequence of schemes to be matched, e.g.: "http", "https".
func (r *Route) Schemes(schemes ...string) *Route {
	for k, v := range schemes {
		schemes[k] = strings.ToLower(v)
	}
	return r.addMatcher(schemeMatcher(schemes))
}

// BuildVarsFunc --------------------------------------------------------------

// BuildVarsFunc is the function signature used by custom build variable
// functions (which can modify route variables before a route's URL is built).
type BuildVarsFunc func(map[string]string) map[string]string

// BuildVarsFunc adds a custom function to be used to modify build variables
// before a route's URL is built.
func (r *Route) BuildVarsFunc(f BuildVarsFunc) *Route {
	r.buildVarsFunc = f
	return r
}

// Subrouter ------------------------------------------------------------------

// Subrouter creates a subrouter for the route.
//
// It will test the inner routes only if the parent route matched. For example:
//
//     r := mux.NewRouter()
//     s := r.Host("www.domain.com").Subrouter()
//     s.HandleFunc("/products/", ProductsHandler)
//     s.HandleFunc("/products/{key}", ProductHandler)
//     s.HandleFunc("/articles/{category}/{id:[0-9]+}"), ArticleHandler)
//
// Here, the routes registered in the subrouter won't be tested if the host
// doesn't match.
func (r *Route) Subrouter() *Router {
	router := &Router{parent: r, strictSlash: r.strictSlash}
	r.addMatcher(router)
	return router
}

// ----------------------------------------------------------------------------
// URL building
// ----------------------------------------------------------------------------

// URL builds a URL for the route.
//
// It accepts a sequence of key/value pairs for the route variables. For
// example, given this route:
//
//     r := mux.NewRouter()
//     r.HandleFunc("/articles/{category}/{id:[0-9]+}", ArticleHandler).
//       Name("article")
//
// ...a URL for it can be built using:
//
//     url, err := r.Get("article").URL("category", "technology", "id", "42")
//
// ...which will return an url.URL with the following path:
//
//     "/articles/technology/42"
//
// This also works for host variables:
//
//     r := mux.NewRouter()
//     r.Host("{subdomain}.domain.com").
//       HandleFunc("/articles/{category}/{id:[0-9]+}", ArticleHandler).
//       Name("article")
//
//     // url.String() will be "http://news.domain.com/articles/technology/42"
//     url, err := r.Get("article").URL("subdomain", "news",
//                                      "category", "technology",
//                                      "id", "42")
//
// All variables defined in the route are required, and their values must
// conform to the corresponding patterns.
func (r *Route) URL(pairs ...string) (*url.URL, error) {
	if r.err != nil {
		return nil, r.err
	}
	if r.regexp == nil {
		return nil, errors.New("mux: route doesn't have a host or path")
	}
	values, err := r.prepareVars(pairs...)
	if err != nil {
		return nil, err
	}
	var scheme, host, path string
	if r.regexp.host != nil {
		// Set a default scheme.
		scheme = "http"
		if host, err = r.regexp.host.url(values); err != nil {
			return nil, err
		}
	}
	if r.regexp.path != nil {
		if path, err = r.regexp.path.url(values); err != nil {
			return nil, err
		}
	}
	return &url.URL{
		Scheme: scheme,
		Host:   host,
		Path:   path,
	}, nil
}

// URLHost builds the host part of the URL for a route. See Route.URL().
//
// The route must have a host defined.
func (r *Route) URLHost(pairs ...string) (*url.URL, error) {
	if r.err != nil {
		return nil, r.err
	}
	if r.regexp == nil || r.regexp.host == nil {
		return nil, errors.New("mux: route doesn't have a host")
	}
	values, err := r.prepareVars(pairs...)
	if err != nil {
		return nil, err
	}
	host, err := r.regexp.host.url(values)
	if err != nil {
		return nil, err
	}
	return &url.URL{
		Scheme: "http",
		Host:   host,
	}, nil
}

// URLPath builds the path part of the URL for a route. See Route.URL().
//
// The route must have a path defined.
func (r *Route) URLPath(pairs ...string) (*url.URL, error) {
	if r.err != nil {
		return nil, r.err
	}
	if r.regexp == nil || r.regexp.path == nil {
		return nil, errors.New("mux: route doesn't have a path")
	}
	values, err := r.prepareVars(pairs...)
	if err != nil {
		return nil, err
	}
	path, err := r.regexp.path.url(values)
	if err != nil {
		return nil, err
	}
	return &url.URL{
		Path: path,
	}, nil
}

// prepareVars converts the route variable pairs into a map. If the route has a
// BuildVarsFunc, it is invoked.
func (r *Route) prepareVars(pairs ...string) (map[string]string, error) {
	m, err := mapFromPairsToString(pairs...)
	if err != nil {
		return nil, err
	}
	return r.buildVars(m), nil
}

func (r *Route) buildVars(m map[string]string) map[string]string {
	if r.parent != nil {
		m = r.parent.buildVars(m)
	}
	if r.buildVarsFunc != nil {
		m = r.buildVarsFunc(m)
	}
	return m
}

// ----------------------------------------------------------------------------
// parentRoute
// ----------------------------------------------------------------------------

// parentRoute allows routes to know about parent host and path definitions.
type parentRoute interface {
	getNamedRoutes() map[string]*Route
	getRegexpGroup() *routeRegexpGroup
	buildVars(map[string]string) map[string]string
}

// getNamedRoutes returns the map where named routes are registered.
func (r *Route) getNamedRoutes() map[string]*Route {
	if r.parent == nil {
		// During tests router is not always set.
		r.parent = NewRouter()
	}
	return r.parent.getNamedRoutes()
}

// getRegexpGroup returns regexp definitions from this route.
func (r *Route) getRegexpGroup() *routeRegexpGroup {
	if r.regexp == nil {
		if r.parent == nil {
			// During tests router is not always set.
			r.parent = NewRouter()
		}
		regexp := r.parent.getRegexpGroup()
		if regexp == nil {
			r.regexp = new(routeRegexpGroup)
		} else {
			// Copy.
			r.regexp = &routeRegexpGroup{
				host:    regexp.host,
				path:    regexp.path,
				queries: regexp.queries,
			}
		}
	}
	return r.regexp
}

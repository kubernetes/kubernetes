// Copyright 2012 The Gorilla Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package gorilla/mux implements a request router and dispatcher.

The name mux stands for "HTTP request multiplexer". Like the standard
http.ServeMux, mux.Router matches incoming requests against a list of
registered routes and calls a handler for the route that matches the URL
or other conditions. The main features are:

	* Requests can be matched based on URL host, path, path prefix, schemes,
	  header and query values, HTTP methods or using custom matchers.
	* URL hosts and paths can have variables with an optional regular
	  expression.
	* Registered URLs can be built, or "reversed", which helps maintaining
	  references to resources.
	* Routes can be used as subrouters: nested routes are only tested if the
	  parent route matches. This is useful to define groups of routes that
	  share common conditions like a host, a path prefix or other repeated
	  attributes. As a bonus, this optimizes request matching.
	* It implements the http.Handler interface so it is compatible with the
	  standard http.ServeMux.

Let's start registering a couple of URL paths and handlers:

	func main() {
		r := mux.NewRouter()
		r.HandleFunc("/", HomeHandler)
		r.HandleFunc("/products", ProductsHandler)
		r.HandleFunc("/articles", ArticlesHandler)
		http.Handle("/", r)
	}

Here we register three routes mapping URL paths to handlers. This is
equivalent to how http.HandleFunc() works: if an incoming request URL matches
one of the paths, the corresponding handler is called passing
(http.ResponseWriter, *http.Request) as parameters.

Paths can have variables. They are defined using the format {name} or
{name:pattern}. If a regular expression pattern is not defined, the matched
variable will be anything until the next slash. For example:

	r := mux.NewRouter()
	r.HandleFunc("/products/{key}", ProductHandler)
	r.HandleFunc("/articles/{category}/", ArticlesCategoryHandler)
	r.HandleFunc("/articles/{category}/{id:[0-9]+}", ArticleHandler)

The names are used to create a map of route variables which can be retrieved
calling mux.Vars():

	vars := mux.Vars(request)
	category := vars["category"]

And this is all you need to know about the basic usage. More advanced options
are explained below.

Routes can also be restricted to a domain or subdomain. Just define a host
pattern to be matched. They can also have variables:

	r := mux.NewRouter()
	// Only matches if domain is "www.domain.com".
	r.Host("www.domain.com")
	// Matches a dynamic subdomain.
	r.Host("{subdomain:[a-z]+}.domain.com")

There are several other matchers that can be added. To match path prefixes:

	r.PathPrefix("/products/")

...or HTTP methods:

	r.Methods("GET", "POST")

...or URL schemes:

	r.Schemes("https")

...or header values:

	r.Headers("X-Requested-With", "XMLHttpRequest")

...or query values:

	r.Queries("key", "value")

...or to use a custom matcher function:

	r.MatcherFunc(func(r *http.Request, rm *RouteMatch) bool {
		return r.ProtoMajor == 0
    })

...and finally, it is possible to combine several matchers in a single route:

	r.HandleFunc("/products", ProductsHandler).
	  Host("www.domain.com").
	  Methods("GET").
	  Schemes("http")

Setting the same matching conditions again and again can be boring, so we have
a way to group several routes that share the same requirements.
We call it "subrouting".

For example, let's say we have several URLs that should only match when the
host is "www.domain.com". Create a route for that host and get a "subrouter"
from it:

	r := mux.NewRouter()
	s := r.Host("www.domain.com").Subrouter()

Then register routes in the subrouter:

	s.HandleFunc("/products/", ProductsHandler)
	s.HandleFunc("/products/{key}", ProductHandler)
	s.HandleFunc("/articles/{category}/{id:[0-9]+}"), ArticleHandler)

The three URL paths we registered above will only be tested if the domain is
"www.domain.com", because the subrouter is tested first. This is not
only convenient, but also optimizes request matching. You can create
subrouters combining any attribute matchers accepted by a route.

Subrouters can be used to create domain or path "namespaces": you define
subrouters in a central place and then parts of the app can register its
paths relatively to a given subrouter.

There's one more thing about subroutes. When a subrouter has a path prefix,
the inner routes use it as base for their paths:

	r := mux.NewRouter()
	s := r.PathPrefix("/products").Subrouter()
	// "/products/"
	s.HandleFunc("/", ProductsHandler)
	// "/products/{key}/"
	s.HandleFunc("/{key}/", ProductHandler)
	// "/products/{key}/details"
	s.HandleFunc("/{key}/details", ProductDetailsHandler)

Now let's see how to build registered URLs.

Routes can be named. All routes that define a name can have their URLs built,
or "reversed". We define a name calling Name() on a route. For example:

	r := mux.NewRouter()
	r.HandleFunc("/articles/{category}/{id:[0-9]+}", ArticleHandler).
	  Name("article")

To build a URL, get the route and call the URL() method, passing a sequence of
key/value pairs for the route variables. For the previous route, we would do:

	url, err := r.Get("article").URL("category", "technology", "id", "42")

...and the result will be a url.URL with the following path:

	"/articles/technology/42"

This also works for host variables:

	r := mux.NewRouter()
	r.Host("{subdomain}.domain.com").
	  Path("/articles/{category}/{id:[0-9]+}").
	  HandlerFunc(ArticleHandler).
	  Name("article")

	// url.String() will be "http://news.domain.com/articles/technology/42"
	url, err := r.Get("article").URL("subdomain", "news",
									 "category", "technology",
									 "id", "42")

All variables defined in the route are required, and their values must
conform to the corresponding patterns. These requirements guarantee that a
generated URL will always match a registered route -- the only exception is
for explicitly defined "build-only" routes which never match.

There's also a way to build only the URL host or path for a route:
use the methods URLHost() or URLPath() instead. For the previous route,
we would do:

	// "http://news.domain.com/"
	host, err := r.Get("article").URLHost("subdomain", "news")

	// "/articles/technology/42"
	path, err := r.Get("article").URLPath("category", "technology", "id", "42")

And if you use subrouters, host and path defined separately can be built
as well:

	r := mux.NewRouter()
	s := r.Host("{subdomain}.domain.com").Subrouter()
	s.Path("/articles/{category}/{id:[0-9]+}").
	  HandlerFunc(ArticleHandler).
	  Name("article")

	// "http://news.domain.com/articles/technology/42"
	url, err := r.Get("article").URL("subdomain", "news",
									 "category", "technology",
									 "id", "42")
*/
package mux

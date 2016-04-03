# Negroni [![GoDoc](https://godoc.org/github.com/codegangsta/negroni?status.svg)](http://godoc.org/github.com/codegangsta/negroni) [![wercker status](https://app.wercker.com/status/13688a4a94b82d84a0b8d038c4965b61/s "wercker status")](https://app.wercker.com/project/bykey/13688a4a94b82d84a0b8d038c4965b61)

Negroni is an idiomatic approach to web middleware in Go. It is tiny, non-intrusive, and encourages use of `net/http` Handlers.

If you like the idea of [Martini](http://github.com/go-martini/martini), but you think it contains too much magic, then Negroni is a great fit.


Language Translations:
* [Português Brasileiro (pt_BR)](translations/README_pt_br.md)

## Getting Started

After installing Go and setting up your [GOPATH](http://golang.org/doc/code.html#GOPATH), create your first `.go` file. We'll call it `server.go`.

~~~ go
package main

import (
  "github.com/codegangsta/negroni"
  "net/http"
  "fmt"
)

func main() {
  mux := http.NewServeMux()
  mux.HandleFunc("/", func(w http.ResponseWriter, req *http.Request) {
    fmt.Fprintf(w, "Welcome to the home page!")
  })

  n := negroni.Classic()
  n.UseHandler(mux)
  n.Run(":3000")
}
~~~

Then install the Negroni package (**go 1.1** and greater is required):
~~~
go get github.com/codegangsta/negroni
~~~

Then run your server:
~~~
go run server.go
~~~

You will now have a Go net/http webserver running on `localhost:3000`.

## Need Help?
If you have a question or feature request, [go ask the mailing list](https://groups.google.com/forum/#!forum/negroni-users). The GitHub issues for Negroni will be used exclusively for bug reports and pull requests.

## Is Negroni a Framework?
Negroni is **not** a framework. It is a library that is designed to work directly with net/http.

## Routing?
Negroni is BYOR (Bring your own Router). The Go community already has a number of great http routers available, Negroni tries to play well with all of them by fully supporting `net/http`. For instance, integrating with [Gorilla Mux](http://github.com/gorilla/mux) looks like so:

~~~ go
router := mux.NewRouter()
router.HandleFunc("/", HomeHandler)

n := negroni.New(Middleware1, Middleware2)
// Or use a middleware with the Use() function
n.Use(Middleware3)
// router goes last
n.UseHandler(router)

n.Run(":3000")
~~~

## `negroni.Classic()`
`negroni.Classic()` provides some default middleware that is useful for most applications:

* `negroni.Recovery` - Panic Recovery Middleware.
* `negroni.Logging` - Request/Response Logging Middleware.
* `negroni.Static` - Static File serving under the "public" directory.

This makes it really easy to get started with some useful features from Negroni.

## Handlers
Negroni provides a bidirectional middleware flow. This is done through the `negroni.Handler` interface:

~~~ go
type Handler interface {
  ServeHTTP(rw http.ResponseWriter, r *http.Request, next http.HandlerFunc)
}
~~~

If a middleware hasn't already written to the ResponseWriter, it should call the next `http.HandlerFunc` in the chain to yield to the next middleware handler. This can be used for great good:

~~~ go
func MyMiddleware(rw http.ResponseWriter, r *http.Request, next http.HandlerFunc) {
  // do some stuff before
  next(rw, r)
  // do some stuff after
}
~~~

And you can map it to the handler chain with the `Use` function:

~~~ go
n := negroni.New()
n.Use(negroni.HandlerFunc(MyMiddleware))
~~~

You can also map plain old `http.Handler`s:

~~~ go
n := negroni.New()

mux := http.NewServeMux()
// map your routes

n.UseHandler(mux)

n.Run(":3000")
~~~

## `Run()`
Negroni has a convenience function called `Run`. `Run` takes an addr string identical to [http.ListenAndServe](http://golang.org/pkg/net/http#ListenAndServe).

~~~ go
n := negroni.Classic()
// ...
log.Fatal(http.ListenAndServe(":8080", n))
~~~

## Route Specific Middleware
If you have a route group of routes that need specific middleware to be executed, you can simply create a new Negroni instance and use it as your route handler.

~~~ go
router := mux.NewRouter()
adminRoutes := mux.NewRouter()
// add admin routes here

// Create a new negroni for the admin middleware
router.Handle("/admin", negroni.New(
  Middleware1,
  Middleware2,
  negroni.Wrap(adminRoutes),
))
~~~

## Third Party Middleware

Here is a current list of Negroni compatible middlware. Feel free to put up a PR linking your middleware if you have built one:


| Middleware | Author | Description |
| -----------|--------|-------------|
| [RestGate](https://github.com/pjebs/restgate) | [Prasanga Siripala](https://github.com/pjebs) | Secure authentication for REST API endpoints |
| [Graceful](https://github.com/stretchr/graceful) | [Tyler Bunnell](https://github.com/tylerb) | Graceful HTTP Shutdown |
| [secure](https://github.com/unrolled/secure) | [Cory Jacobsen](https://github.com/unrolled) | Middleware that implements a few quick security wins |
| [JWT Middleware](https://github.com/auth0/go-jwt-middleware) | [Auth0](https://github.com/auth0) | Middleware checks for a JWT on the `Authorization` header on incoming requests and decodes it|
| [binding](https://github.com/mholt/binding) | [Matt Holt](https://github.com/mholt) | Data binding from HTTP requests into structs |
| [logrus](https://github.com/meatballhat/negroni-logrus) | [Dan Buch](https://github.com/meatballhat) | Logrus-based logger |
| [render](https://github.com/unrolled/render) | [Cory Jacobsen](https://github.com/unrolled) | Render JSON, XML and HTML templates |
| [gorelic](https://github.com/jingweno/negroni-gorelic) | [Jingwen Owen Ou](https://github.com/jingweno) | New Relic agent for Go runtime |
| [gzip](https://github.com/phyber/negroni-gzip) | [phyber](https://github.com/phyber) | GZIP response compression |
| [oauth2](https://github.com/goincremental/negroni-oauth2) | [David Bochenski](https://github.com/bochenski) | oAuth2 middleware |
| [sessions](https://github.com/goincremental/negroni-sessions) | [David Bochenski](https://github.com/bochenski) | Session Management |
| [permissions2](https://github.com/xyproto/permissions2) | [Alexander Rødseth](https://github.com/xyproto) | Cookies, users and permissions |
| [onthefly](https://github.com/xyproto/onthefly) | [Alexander Rødseth](https://github.com/xyproto) | Generate TinySVG, HTML and CSS on the fly |
| [cors](https://github.com/rs/cors) | [Olivier Poitrey](https://github.com/rs) | [Cross Origin Resource Sharing](http://www.w3.org/TR/cors/) (CORS) support |
| [xrequestid](https://github.com/pilu/xrequestid) | [Andrea Franz](https://github.com/pilu) | Middleware that assigns a random X-Request-Id header to each request |
| [VanGoH](https://github.com/auroratechnologies/vangoh) | [Taylor Wrobel](https://github.com/twrobel3) | Configurable [AWS-Style](http://docs.aws.amazon.com/AmazonS3/latest/dev/RESTAuthentication.html) HMAC authentication middleware |
| [stats](https://github.com/thoas/stats) | [Florent Messa](https://github.com/thoas) | Store information about your web application (response time, etc.) |

## Examples
[Alexander Rødseth](https://github.com/xyproto) created [mooseware](https://github.com/xyproto/mooseware), a skeleton for writing a Negroni middleware handler.

## Live code reload?
[gin](https://github.com/codegangsta/gin) and [fresh](https://github.com/pilu/fresh) both live reload negroni apps.

## Essential Reading for Beginners of Go & Negroni

* [Using a Context to pass information from middleware to end handler](http://elithrar.github.io/article/map-string-interface/)
* [Understanding middleware](http://mattstauffer.co/blog/laravel-5.0-middleware-replacing-filters)

## About

Negroni is obsessively designed by none other than the [Code Gangsta](http://codegangsta.io/)

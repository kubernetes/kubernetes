# Martini  [![wercker status](https://app.wercker.com/status/9b7dbc6e2654b604cd694d191c3d5487/s/master "wercker status")](https://app.wercker.com/project/bykey/9b7dbc6e2654b604cd694d191c3d5487)[![GoDoc](https://godoc.org/github.com/go-martini/martini?status.png)](http://godoc.org/github.com/go-martini/martini)

Martini is a powerful package for quickly writing modular web applications/services in Golang.

Language Translations:
* [繁體中文](translations/README_zh_tw.md)
* [简体中文](translations/README_zh_cn.md)
* [Português Brasileiro (pt_BR)](translations/README_pt_br.md)
* [Español](translations/README_es_ES.md)
* [한국어 번역](translations/README_ko_kr.md)
* [Русский](translations/README_ru_RU.md)
* [日本語](translations/README_ja_JP.md)
* [French](translations/README_fr_FR.md)
* [Turkish](translations/README_tr_TR.md)
* [German](translations/README_de_DE.md)

## Getting Started

After installing Go and setting up your [GOPATH](http://golang.org/doc/code.html#GOPATH), create your first `.go` file. We'll call it `server.go`.

~~~ go
package main

import "github.com/go-martini/martini"

func main() {
  m := martini.Classic()
  m.Get("/", func() string {
    return "Hello world!"
  })
  m.Run()
}
~~~

Then install the Martini package (**go 1.1** or greater is required):
~~~
go get github.com/go-martini/martini
~~~

Then run your server:
~~~
go run server.go
~~~

You will now have a Martini webserver running on `localhost:3000`.

## Getting Help

Join the [Mailing list](https://groups.google.com/forum/#!forum/martini-go)

Watch the [Demo Video](http://martini.codegangsta.io/#demo)

Ask questions on Stackoverflow using the [martini tag](http://stackoverflow.com/questions/tagged/martini)

GoDoc [documentation](http://godoc.org/github.com/go-martini/martini)


## Features
* Extremely simple to use.
* Non-intrusive design.
* Plays nice with other Golang packages.
* Awesome path matching and routing.
* Modular design - Easy to add functionality, easy to rip stuff out.
* Lots of good handlers/middlewares to use.
* Great 'out of the box' feature set.
* **Fully compatible with the [http.HandlerFunc](http://godoc.org/net/http#HandlerFunc) interface.**
* Default document serving (e.g., for serving AngularJS apps in HTML5 mode).

## More Middleware
For more middleware and functionality, check out the repositories in the  [martini-contrib](https://github.com/martini-contrib) organization.

## Table of Contents
* [Classic Martini](#classic-martini)
  * [Handlers](#handlers)
  * [Routing](#routing)
  * [Services](#services)
  * [Serving Static Files](#serving-static-files)
* [Middleware Handlers](#middleware-handlers)
  * [Next()](#next)
* [Martini Env](#martini-env)
* [FAQ](#faq)

## Classic Martini
To get up and running quickly, [martini.Classic()](http://godoc.org/github.com/go-martini/martini#Classic) provides some reasonable defaults that work well for most web applications:
~~~ go
  m := martini.Classic()
  // ... middleware and routing goes here
  m.Run()
~~~

Below is some of the functionality [martini.Classic()](http://godoc.org/github.com/go-martini/martini#Classic) pulls in automatically:
  * Request/Response Logging - [martini.Logger](http://godoc.org/github.com/go-martini/martini#Logger)
  * Panic Recovery - [martini.Recovery](http://godoc.org/github.com/go-martini/martini#Recovery)
  * Static File serving - [martini.Static](http://godoc.org/github.com/go-martini/martini#Static)
  * Routing - [martini.Router](http://godoc.org/github.com/go-martini/martini#Router)

### Handlers
Handlers are the heart and soul of Martini. A handler is basically any kind of callable function:
~~~ go
m.Get("/", func() {
  println("hello world")
})
~~~

#### Return Values
If a handler returns something, Martini will write the result to the current [http.ResponseWriter](http://godoc.org/net/http#ResponseWriter) as a string:
~~~ go
m.Get("/", func() string {
  return "hello world" // HTTP 200 : "hello world"
})
~~~

You can also optionally return a status code:
~~~ go
m.Get("/", func() (int, string) {
  return 418, "i'm a teapot" // HTTP 418 : "i'm a teapot"
})
~~~

#### Service Injection
Handlers are invoked via reflection. Martini makes use of *Dependency Injection* to resolve dependencies in a Handlers argument list. **This makes Martini completely  compatible with golang's `http.HandlerFunc` interface.**

If you add an argument to your Handler, Martini will search its list of services and attempt to resolve the dependency via type assertion:
~~~ go
m.Get("/", func(res http.ResponseWriter, req *http.Request) { // res and req are injected by Martini
  res.WriteHeader(200) // HTTP 200
})
~~~

The following services are included with [martini.Classic()](http://godoc.org/github.com/go-martini/martini#Classic):
  * [*log.Logger](http://godoc.org/log#Logger) - Global logger for Martini.
  * [martini.Context](http://godoc.org/github.com/go-martini/martini#Context) - http request context.
  * [martini.Params](http://godoc.org/github.com/go-martini/martini#Params) - `map[string]string` of named params found by route matching.
  * [martini.Routes](http://godoc.org/github.com/go-martini/martini#Routes) - Route helper service.
  * [martini.Route](http://godoc.org/github.com/go-martini/martini#Route) - Current active route.
  * [http.ResponseWriter](http://godoc.org/net/http/#ResponseWriter) - http Response writer interface.
  * [*http.Request](http://godoc.org/net/http/#Request) - http Request.

### Routing
In Martini, a route is an HTTP method paired with a URL-matching pattern.
Each route can take one or more handler methods:
~~~ go
m.Get("/", func() {
  // show something
})

m.Patch("/", func() {
  // update something
})

m.Post("/", func() {
  // create something
})

m.Put("/", func() {
  // replace something
})

m.Delete("/", func() {
  // destroy something
})

m.Options("/", func() {
  // http options
})

m.NotFound(func() {
  // handle 404
})
~~~

Routes are matched in the order they are defined. The first route that
matches the request is invoked.

Route patterns may include named parameters, accessible via the [martini.Params](http://godoc.org/github.com/go-martini/martini#Params) service:
~~~ go
m.Get("/hello/:name", func(params martini.Params) string {
  return "Hello " + params["name"]
})
~~~

Routes can be matched with globs:
~~~ go
m.Get("/hello/**", func(params martini.Params) string {
  return "Hello " + params["_1"]
})
~~~

Regular expressions can be used as well:
~~~go
m.Get("/hello/(?P<name>[a-zA-Z]+)", func(params martini.Params) string {
  return fmt.Sprintf ("Hello %s", params["name"])
})
~~~
Take a look at the [Go documentation](http://golang.org/pkg/regexp/syntax/) for more info about regular expressions syntax .

Route handlers can be stacked on top of each other, which is useful for things like authentication and authorization:
~~~ go
m.Get("/secret", authorize, func() {
  // this will execute as long as authorize doesn't write a response
})
~~~

Route groups can be added too using the Group method.
~~~ go
m.Group("/books", func(r martini.Router) {
    r.Get("/:id", GetBooks)
    r.Post("/new", NewBook)
    r.Put("/update/:id", UpdateBook)
    r.Delete("/delete/:id", DeleteBook)
})
~~~

Just like you can pass middlewares to a handler you can pass middlewares to groups.
~~~ go
m.Group("/books", func(r martini.Router) {
    r.Get("/:id", GetBooks)
    r.Post("/new", NewBook)
    r.Put("/update/:id", UpdateBook)
    r.Delete("/delete/:id", DeleteBook)
}, MyMiddleware1, MyMiddleware2)
~~~

### Services
Services are objects that are available to be injected into a Handler's argument list. You can map a service on a *Global* or *Request* level.

#### Global Mapping
A Martini instance implements the inject.Injector interface, so mapping a service is easy:
~~~ go
db := &MyDatabase{}
m := martini.Classic()
m.Map(db) // the service will be available to all handlers as *MyDatabase
// ...
m.Run()
~~~

#### Request-Level Mapping
Mapping on the request level can be done in a handler via [martini.Context](http://godoc.org/github.com/go-martini/martini#Context):
~~~ go
func MyCustomLoggerHandler(c martini.Context, req *http.Request) {
  logger := &MyCustomLogger{req}
  c.Map(logger) // mapped as *MyCustomLogger
}
~~~

#### Mapping values to Interfaces
One of the most powerful parts about services is the ability to map a service to an interface. For instance, if you wanted to override the [http.ResponseWriter](http://godoc.org/net/http#ResponseWriter) with an object that wrapped it and performed extra operations, you can write the following handler:
~~~ go
func WrapResponseWriter(res http.ResponseWriter, c martini.Context) {
  rw := NewSpecialResponseWriter(res)
  c.MapTo(rw, (*http.ResponseWriter)(nil)) // override ResponseWriter with our wrapper ResponseWriter
}
~~~

### Serving Static Files
A [martini.Classic()](http://godoc.org/github.com/go-martini/martini#Classic) instance automatically serves static files from the "public" directory in the root of your server.
You can serve from more directories by adding more [martini.Static](http://godoc.org/github.com/go-martini/martini#Static) handlers.
~~~ go
m.Use(martini.Static("assets")) // serve from the "assets" directory as well
~~~

#### Serving a Default Document
You can specify the URL of a local file to serve when the requested URL is not
found. You can also specify an exclusion prefix so that certain URLs are ignored.
This is useful for servers that serve both static files and have additional
handlers defined (e.g., REST API). When doing so, it's useful to define the
static handler as a part of the NotFound chain.

The following example serves the `/index.html` file whenever any URL is
requested that does not match any local file and does not start with `/api/v`:
~~~ go
static := martini.Static("assets", martini.StaticOptions{Fallback: "/index.html", Exclude: "/api/v"})
m.NotFound(static, http.NotFound)
~~~

## Middleware Handlers
Middleware Handlers sit between the incoming http request and the router. In essence they are no different than any other Handler in Martini. You can add a middleware handler to the stack like so:
~~~ go
m.Use(func() {
  // do some middleware stuff
})
~~~

You can have full control over the middleware stack with the `Handlers` function. This will replace any handlers that have been previously set:
~~~ go
m.Handlers(
  Middleware1,
  Middleware2,
  Middleware3,
)
~~~

Middleware Handlers work really well for things like logging, authorization, authentication, sessions, gzipping, error pages and any other operations that must happen before or after an http request:
~~~ go
// validate an api key
m.Use(func(res http.ResponseWriter, req *http.Request) {
  if req.Header.Get("X-API-KEY") != "secret123" {
    res.WriteHeader(http.StatusUnauthorized)
  }
})
~~~

### Next()
[Context.Next()](http://godoc.org/github.com/go-martini/martini#Context) is an optional function that Middleware Handlers can call to yield the until after the other Handlers have been executed. This works really well for any operations that must happen after an http request:
~~~ go
// log before and after a request
m.Use(func(c martini.Context, log *log.Logger){
  log.Println("before a request")

  c.Next()

  log.Println("after a request")
})
~~~

## Martini Env

Some Martini handlers make use of the `martini.Env` global variable to provide special functionality for development environments vs production environments. It is recommended that the `MARTINI_ENV=production` environment variable to be set when deploying a Martini server into a production environment.

## FAQ

### Where do I find middleware X?

Start by looking in the [martini-contrib](https://github.com/martini-contrib) projects. If it is not there feel free to contact a martini-contrib team member about adding a new repo to the organization.

* [auth](https://github.com/martini-contrib/auth) - Handlers for authentication.
* [binding](https://github.com/martini-contrib/binding) - Handler for mapping/validating a raw request into a structure.
* [gzip](https://github.com/martini-contrib/gzip) - Handler for adding gzip compress to requests
* [render](https://github.com/martini-contrib/render) - Handler that provides a service for easily rendering JSON and HTML templates.
* [acceptlang](https://github.com/martini-contrib/acceptlang) - Handler for parsing the `Accept-Language` HTTP header.
* [sessions](https://github.com/martini-contrib/sessions) - Handler that provides a Session service.
* [strip](https://github.com/martini-contrib/strip) - URL Prefix stripping.
* [method](https://github.com/martini-contrib/method) - HTTP method overriding via Header or form fields.
* [secure](https://github.com/martini-contrib/secure) - Implements a few quick security wins.
* [encoder](https://github.com/martini-contrib/encoder) - Encoder service for rendering data in several formats and content negotiation.
* [cors](https://github.com/martini-contrib/cors) - Handler that enables CORS support.
* [oauth2](https://github.com/martini-contrib/oauth2) - Handler that provides OAuth 2.0 login for Martini apps. Google Sign-in, Facebook Connect and Github login is supported.
* [vauth](https://github.com/rafecolton/vauth) - Handlers for vender webhook authentication (currently GitHub and TravisCI)
* [permissions2](https://github.com/xyproto/permissions2) - Handler for keeping track of users, login states and permissions.

### How do I integrate with existing servers?

A Martini instance implements `http.Handler`, so it can easily be used to serve subtrees
on existing Go servers. For example this is a working Martini app for Google App Engine:

~~~ go
package hello

import (
  "net/http"
  "github.com/go-martini/martini"
)

func init() {
  m := martini.Classic()
  m.Get("/", func() string {
    return "Hello world!"
  })
  http.Handle("/", m)
}
~~~

### How do I change the port/host?

Martini's `Run` function looks for the PORT and HOST environment variables and uses those. Otherwise Martini will default to localhost:3000.
To have more flexibility over port and host, use the `martini.RunOnAddr` function instead.

~~~ go
  m := martini.Classic()
  // ...
  log.Fatal(m.RunOnAddr(":8080"))
~~~

### Live code reload?

[gin](https://github.com/codegangsta/gin) and [fresh](https://github.com/pilu/fresh) both live reload martini apps.

## Contributing
Martini is meant to be kept tiny and clean. Most contributions should end up in a repository in the [martini-contrib](https://github.com/martini-contrib) organization. If you do have a contribution for the core of Martini feel free to put up a Pull Request.

## About

Inspired by [express](https://github.com/visionmedia/express) and [sinatra](https://github.com/sinatra/sinatra)

Martini is obsessively designed by none other than the [Code Gangsta](http://codegangsta.io/)

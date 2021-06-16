go-restful
==========
package for building REST-style Web Services using Google Go

[![Build Status](https://travis-ci.org/emicklei/go-restful.png)](https://travis-ci.org/emicklei/go-restful)
[![Go Report Card](https://goreportcard.com/badge/github.com/emicklei/go-restful)](https://goreportcard.com/report/github.com/emicklei/go-restful)
[![GoDoc](https://godoc.org/github.com/emicklei/go-restful?status.svg)](https://pkg.go.dev/github.com/emicklei/go-restful)
[![codecov](https://codecov.io/gh/emicklei/go-restful/branch/master/graph/badge.svg)](https://codecov.io/gh/emicklei/go-restful)

- [Code examples using v3](https://github.com/emicklei/go-restful/tree/master/examples)

REST asks developers to use HTTP methods explicitly and in a way that's consistent with the protocol definition. This basic REST design principle establishes a one-to-one mapping between create, read, update, and delete (CRUD) operations and HTTP methods. According to this mapping:

- GET = Retrieve a representation of a resource
- POST = Create if you are sending content to the server to create a subordinate of the specified resource collection, using some server-side algorithm.
- PUT = Create if you are sending the full content of the specified resource (URI).
- PUT = Update if you are updating the full content of the specified resource.
- DELETE = Delete if you are requesting the server to delete the resource
- PATCH = Update partial content of a resource
- OPTIONS = Get information about the communication options for the request URI
    
### Usage

#### Using Go Modules

As of version `v3.0.0` (on the v3 branch), this package supports Go modules.

```
import (
	restful "github.com/emicklei/go-restful/v3"
)
```

#### Without Go Modules

All versions up to `v2.*.*` (on the master) are not supporting Go modules.

```
import (
	restful "github.com/emicklei/go-restful"
)
```

### Example

```Go
ws := new(restful.WebService)
ws.
	Path("/users").
	Consumes(restful.MIME_XML, restful.MIME_JSON).
	Produces(restful.MIME_JSON, restful.MIME_XML)

ws.Route(ws.GET("/{user-id}").To(u.findUser).
	Doc("get a user").
	Param(ws.PathParameter("user-id", "identifier of the user").DataType("string")).
	Writes(User{}))		
...
	
func (u UserResource) findUser(request *restful.Request, response *restful.Response) {
	id := request.PathParameter("user-id")
	...
}
```

[Full API of a UserResource](https://github.com/emicklei/go-restful/tree/master/examples/user-resource/restful-user-resource.go)

### Features

- Routes for request &#8594; function mapping with path parameter (e.g. {id} but also prefix_{var} and {var}_suffix) support
- Configurable router:
	- (default) Fast routing algorithm that allows static elements, [google custom method](https://cloud.google.com/apis/design/custom_methods), regular expressions and dynamic parameters in the URL path (e.g. /resource/name:customVerb, /meetings/{id} or /static/{subpath:*})
	- Routing algorithm after [JSR311](http://jsr311.java.net/nonav/releases/1.1/spec/spec.html) that is implemented using (but does **not** accept) regular expressions
- Request API for reading structs from JSON/XML and accesing parameters (path,query,header)
- Response API for writing structs to JSON/XML and setting headers
- Customizable encoding using EntityReaderWriter registration
- Filters for intercepting the request &#8594; response flow on Service or Route level
- Request-scoped variables using attributes
- Containers for WebServices on different HTTP endpoints
- Content encoding (gzip,deflate) of request and response payloads
- Automatic responses on OPTIONS (using a filter)
- Automatic CORS request handling (using a filter)
- API declaration for Swagger UI ([go-restful-openapi](https://github.com/emicklei/go-restful-openapi), see [go-restful-swagger12](https://github.com/emicklei/go-restful-swagger12))
- Panic recovery to produce HTTP 500, customizable using RecoverHandler(...)
- Route errors produce HTTP 404/405/406/415 errors, customizable using ServiceErrorHandler(...)
- Configurable (trace) logging
- Customizable gzip/deflate readers and writers using CompressorProvider registration

## How to customize
There are several hooks to customize the behavior of the go-restful package.

- Router algorithm
- Panic recovery
- JSON decoder
- Trace logging
- Compression
- Encoders for other serializers
- Use [jsoniter](https://github.com/json-iterator/go) by build this package using a tag, e.g. `go build -tags=jsoniter .`

TODO: write examples of these.

## Resources

- [Example posted on blog](http://ernestmicklei.com/2012/11/go-restful-first-working-example/)
- [Design explained on blog](http://ernestmicklei.com/2012/11/go-restful-api-design/)
- [sourcegraph](https://sourcegraph.com/github.com/emicklei/go-restful)
- [showcase: Zazkia - tcp proxy for testing resiliency](https://github.com/emicklei/zazkia)
- [showcase: Mora - MongoDB REST Api server](https://github.com/emicklei/mora)

Type ```git shortlog -s``` for a full list of contributors.

© 2012 - 2020, http://ernestmicklei.com. MIT License. Contributions are welcome.

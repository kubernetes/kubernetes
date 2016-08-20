// Copyright 2009 The Go Authors. All rights reserved.
// Copyright 2012 The Gorilla Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package gorilla/rpc is a foundation for RPC over HTTP services, providing
access to the exported methods of an object through HTTP requests.

This package derives from the standard net/rpc package but uses a single HTTP
request per call instead of persistent connections. Other differences
compared to net/rpc:

	- Multiple codecs can be registered in the same server.
	- A codec is chosen based on the "Content-Type" header from the request.
	- Service methods also receive http.Request as parameter.
	- This package can be used on Google App Engine.

Let's setup a server and register a codec and service:

	import (
		"http"
		"github.com/gorilla/rpc/v2"
		"github.com/gorilla/rpc/v2/json"
	)

	func init() {
		s := rpc.NewServer()
		s.RegisterCodec(json.NewCodec(), "application/json")
		s.RegisterService(new(HelloService), "")
		http.Handle("/rpc", s)
	}

This server handles requests to the "/rpc" path using a JSON codec.
A codec is tied to a content type. In the example above, the JSON codec is
registered to serve requests with "application/json" as the value for the
"Content-Type" header. If the header includes a charset definition, it is
ignored; only the media-type part is taken into account.

A service can be registered using a name. If the name is empty, like in the
example above, it will be inferred from the service type.

That's all about the server setup. Now let's define a simple service:

	type HelloArgs struct {
		Who string
	}

	type HelloReply struct {
		Message string
	}

	type HelloService struct {}

	func (h *HelloService) Say(r *http.Request, args *HelloArgs, reply *HelloReply) error {
		reply.Message = "Hello, " + args.Who + "!"
		return nil
	}

The example above defines a service with a method "HelloService.Say" and
the arguments and reply related to that method.

The service must be exported (begin with an upper case letter) or local
(defined in the package registering the service).

When a service is registered, the server inspects the service methods
and make available the ones that follow these rules:

	- The method name is exported.
	- The method has three arguments: *http.Request, *args, *reply.
	- All three arguments are pointers.
	- The second and third arguments are exported or local.
	- The method has return type error.

All other methods are ignored.

Gorilla has packages with common RPC codecs. Check out their documentation:

	JSON: http://gorilla-web.appspot.com/pkg/rpc/json
*/
package rpc

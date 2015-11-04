// Copyright 2012 The Gorilla Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package context stores values shared during a request lifetime.

For example, a router can set variables extracted from the URL and later
application handlers can access those values, or it can be used to store
sessions values to be saved at the end of a request. There are several
others common uses.

The idea was posted by Brad Fitzpatrick to the go-nuts mailing list:

	http://groups.google.com/group/golang-nuts/msg/e2d679d303aa5d53

Here's the basic usage: first define the keys that you will need. The key
type is interface{} so a key can be of any type that supports equality.
Here we define a key using a custom int type to avoid name collisions:

	package foo

	import (
		"github.com/gorilla/context"
	)

	type key int

	const MyKey key = 0

Then set a variable. Variables are bound to an http.Request object, so you
need a request instance to set a value:

	context.Set(r, MyKey, "bar")

The application can later access the variable using the same key you provided:

	func MyHandler(w http.ResponseWriter, r *http.Request) {
		// val is "bar".
		val := context.Get(r, foo.MyKey)

		// returns ("bar", true)
		val, ok := context.GetOk(r, foo.MyKey)
		// ...
	}

And that's all about the basic usage. We discuss some other ideas below.

Any type can be stored in the context. To enforce a given type, make the key
private and wrap Get() and Set() to accept and return values of a specific
type:

	type key int

	const mykey key = 0

	// GetMyKey returns a value for this package from the request values.
	func GetMyKey(r *http.Request) SomeType {
		if rv := context.Get(r, mykey); rv != nil {
			return rv.(SomeType)
		}
		return nil
	}

	// SetMyKey sets a value for this package in the request values.
	func SetMyKey(r *http.Request, val SomeType) {
		context.Set(r, mykey, val)
	}

Variables must be cleared at the end of a request, to remove all values
that were stored. This can be done in an http.Handler, after a request was
served. Just call Clear() passing the request:

	context.Clear(r)

...or use ClearHandler(), which conveniently wraps an http.Handler to clear
variables at the end of a request lifetime.

The Routers from the packages gorilla/mux and gorilla/pat call Clear()
so if you are using either of them you don't need to clear the context manually.
*/
package context

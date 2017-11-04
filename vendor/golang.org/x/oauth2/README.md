# OAuth2 for Go

[![Build Status](https://travis-ci.org/golang/oauth2.svg?branch=master)](https://travis-ci.org/golang/oauth2)
[![GoDoc](https://godoc.org/golang.org/x/oauth2?status.svg)](https://godoc.org/golang.org/x/oauth2)

oauth2 package contains a client implementation for OAuth 2.0 spec.

## Installation

~~~~
go get golang.org/x/oauth2
~~~~

See godoc for further documentation and examples.

* [godoc.org/golang.org/x/oauth2](http://godoc.org/golang.org/x/oauth2)
* [godoc.org/golang.org/x/oauth2/google](http://godoc.org/golang.org/x/oauth2/google)


## App Engine

In change 96e89be (March 2015) we removed the `oauth2.Context2` type in favor
of the [`context.Context`](https://golang.org/x/net/context#Context) type from
the `golang.org/x/net/context` package

This means its no longer possible to use the "Classic App Engine"
`appengine.Context` type with the `oauth2` package. (You're using
Classic App Engine if you import the package `"appengine"`.)

To work around this, you may use the new `"google.golang.org/appengine"`
package. This package has almost the same API as the `"appengine"` package,
but it can be fetched with `go get` and used on "Managed VMs" and well as
Classic App Engine.

See the [new `appengine` package's readme](https://github.com/golang/appengine#updating-a-go-app-engine-app)
for information on updating your app.

If you don't want to update your entire app to use the new App Engine packages,
you may use both sets of packages in parallel, using only the new packages
with the `oauth2` package.

	import (
		"golang.org/x/net/context"
		"golang.org/x/oauth2"
		"golang.org/x/oauth2/google"
		newappengine "google.golang.org/appengine"
		newurlfetch "google.golang.org/appengine/urlfetch"

		"appengine"
	)

	func handler(w http.ResponseWriter, r *http.Request) {
		var c appengine.Context = appengine.NewContext(r)
		c.Infof("Logging a message with the old package")

		var ctx context.Context = newappengine.NewContext(r)
		client := &http.Client{
			Transport: &oauth2.Transport{
				Source: google.AppEngineTokenSource(ctx, "scope"),
				Base:   &newurlfetch.Transport{Context: ctx},
			},
		}
		client.Get("...")
	}

## Contributing

We appreciate your help!

To contribute, please read the contribution guidelines:
	https://golang.org/doc/contribute.html

Note that the Go project does not use GitHub pull requests but
uses Gerrit for code reviews. See the contribution guide for details.

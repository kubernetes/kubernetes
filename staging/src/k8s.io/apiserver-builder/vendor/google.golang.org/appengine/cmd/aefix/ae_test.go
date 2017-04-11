// Copyright 2016 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package main

func init() {
	addTestCases(aeTests, nil)
}

var aeTests = []testCase{
	// Collection of fixes:
	//	- imports
	//	- appengine.Timeout -> context.WithTimeout
	//	- add ctx arg to appengine.Datacenter
	//	- logging API
	{
		Name: "ae.0",
		In: `package foo

import (
	"net/http"
	"time"

	"appengine"
	"appengine/datastore"
)

func f(w http.ResponseWriter, r *http.Request) {
	c := appengine.NewContext(r)

	c = appengine.Timeout(c, 5*time.Second)
	err := datastore.ErrNoSuchEntity
	c.Errorf("Something interesting happened: %v", err)
	_ = appengine.Datacenter()
}
`,
		Out: `package foo

import (
	"net/http"
	"time"

	"golang.org/x/net/context"
	"google.golang.org/appengine"
	"google.golang.org/appengine/datastore"
	"google.golang.org/appengine/log"
)

func f(w http.ResponseWriter, r *http.Request) {
	c := appengine.NewContext(r)

	c, _ = context.WithTimeout(c, 5*time.Second)
	err := datastore.ErrNoSuchEntity
	log.Errorf(c, "Something interesting happened: %v", err)
	_ = appengine.Datacenter(c)
}
`,
	},

	// Updating a function that takes an appengine.Context arg.
	{
		Name: "ae.1",
		In: `package foo

import (
	"appengine"
)

func LogSomething(c2 appengine.Context) {
	c2.Warningf("Stand back! I'm going to try science!")
}
`,
		Out: `package foo

import (
	"golang.org/x/net/context"
	"google.golang.org/appengine/log"
)

func LogSomething(c2 context.Context) {
	log.Warningf(c2, "Stand back! I'm going to try science!")
}
`,
	},

	// Less widely used API changes:
	//	- drop maxTasks arg to taskqueue.QueueStats
	{
		Name: "ae.2",
		In: `package foo

import (
	"appengine"
	"appengine/taskqueue"
)

func f(ctx appengine.Context) {
	stats, err := taskqueue.QueueStats(ctx, []string{"one", "two"}, 0)
}
`,
		Out: `package foo

import (
	"golang.org/x/net/context"
	"google.golang.org/appengine/taskqueue"
)

func f(ctx context.Context) {
	stats, err := taskqueue.QueueStats(ctx, []string{"one", "two"})
}
`,
	},

	// Check that the main "appengine" import will not be dropped
	// if an appengine.Context -> context.Context change happens
	// but the appengine package is still referenced.
	{
		Name: "ae.3",
		In: `package foo

import (
	"appengine"
	"io"
)

func f(ctx appengine.Context, w io.Writer) {
	_ = appengine.IsDevAppServer()
}
`,
		Out: `package foo

import (
	"golang.org/x/net/context"
	"google.golang.org/appengine"
	"io"
)

func f(ctx context.Context, w io.Writer) {
	_ = appengine.IsDevAppServer()
}
`,
	},
}

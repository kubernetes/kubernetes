/*
Package aetest provides an API for running dev_appserver for use in tests.

An example test file:

	package foo_test

	import (
		"testing"

		"google.golang.org/appengine/memcache"
		"google.golang.org/appengine/aetest"
	)

	func TestFoo(t *testing.T) {
		ctx, done, err := aetest.NewContext()
		if err != nil {
			t.Fatal(err)
		}
		defer done()

		it := &memcache.Item{
			Key:   "some-key",
			Value: []byte("some-value"),
		}
		err = memcache.Set(ctx, it)
		if err != nil {
			t.Fatalf("Set err: %v", err)
		}
		it, err = memcache.Get(ctx, "some-key")
		if err != nil {
			t.Fatalf("Get err: %v; want no error", err)
		}
		if g, w := string(it.Value), "some-value" ; g != w {
			t.Errorf("retrieved Item.Value = %q, want %q", g, w)
		}
	}

The environment variable APPENGINE_DEV_APPSERVER specifies the location of the
dev_appserver.py executable to use. If unset, the system PATH is consulted.
*/
package aetest

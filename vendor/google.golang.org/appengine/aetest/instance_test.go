package aetest

import (
	"os"
	"testing"

	"google.golang.org/appengine"
	"google.golang.org/appengine/datastore"
	"google.golang.org/appengine/memcache"
	"google.golang.org/appengine/user"
)

func TestBasicAPICalls(t *testing.T) {
	// Only run the test if APPENGINE_DEV_APPSERVER is explicitly set.
	if os.Getenv("APPENGINE_DEV_APPSERVER") == "" {
		t.Skip("APPENGINE_DEV_APPSERVER not set")
	}

	inst, err := NewInstance(nil)
	if err != nil {
		t.Fatalf("NewInstance: %v", err)
	}
	defer inst.Close()

	req, err := inst.NewRequest("GET", "http://example.com/page", nil)
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	ctx := appengine.NewContext(req)

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
	if g, w := string(it.Value), "some-value"; g != w {
		t.Errorf("retrieved Item.Value = %q, want %q", g, w)
	}

	type Entity struct{ Value string }
	e := &Entity{Value: "foo"}
	k := datastore.NewIncompleteKey(ctx, "Entity", nil)
	k, err = datastore.Put(ctx, k, e)
	if err != nil {
		t.Fatalf("datastore.Put: %v", err)
	}
	e = new(Entity)
	if err := datastore.Get(ctx, k, e); err != nil {
		t.Fatalf("datastore.Get: %v", err)
	}
	if g, w := e.Value, "foo"; g != w {
		t.Errorf("retrieved Entity.Value = %q, want %q", g, w)
	}
}

func TestContext(t *testing.T) {
	// Only run the test if APPENGINE_DEV_APPSERVER is explicitly set.
	if os.Getenv("APPENGINE_DEV_APPSERVER") == "" {
		t.Skip("APPENGINE_DEV_APPSERVER not set")
	}

	// Check that the context methods work.
	_, done, err := NewContext()
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	done()
}

func TestUsers(t *testing.T) {
	// Only run the test if APPENGINE_DEV_APPSERVER is explicitly set.
	if os.Getenv("APPENGINE_DEV_APPSERVER") == "" {
		t.Skip("APPENGINE_DEV_APPSERVER not set")
	}

	inst, err := NewInstance(nil)
	if err != nil {
		t.Fatalf("NewInstance: %v", err)
	}
	defer inst.Close()

	req, err := inst.NewRequest("GET", "http://example.com/page", nil)
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	ctx := appengine.NewContext(req)

	if user := user.Current(ctx); user != nil {
		t.Errorf("user.Current initially %v, want nil", user)
	}

	u := &user.User{
		Email: "gopher@example.com",
		Admin: true,
	}
	Login(u, req)

	if got := user.Current(ctx); got.Email != u.Email {
		t.Errorf("user.Current: %v, want %v", got, u)
	}
	if admin := user.IsAdmin(ctx); !admin {
		t.Errorf("user.IsAdmin: %t, want true", admin)
	}

	Logout(req)
	if user := user.Current(ctx); user != nil {
		t.Errorf("user.Current after logout %v, want nil", user)
	}
}

// +build !appengine

package memcache

import (
	"bytes"
	"net"
	"testing"
)

const testServer = "localhost:11211"

func TestMemCache(t *testing.T) {
	conn, err := net.Dial("tcp", testServer)
	if err != nil {
		// TODO: rather than skip the test, fall back to a faked memcached server
		t.Skipf("skipping test; no server running at %s", testServer)
	}
	conn.Write([]byte("flush_all\r\n")) // flush memcache
	conn.Close()

	cache := New(testServer)

	key := "testKey"
	_, ok := cache.Get(key)
	if ok {
		t.Fatal("retrieved key before adding it")
	}

	val := []byte("some bytes")
	cache.Set(key, val)

	retVal, ok := cache.Get(key)
	if !ok {
		t.Fatal("could not retrieve an element we just added")
	}
	if !bytes.Equal(retVal, val) {
		t.Fatal("retrieved a different value than what we put in")
	}

	cache.Delete(key)

	_, ok = cache.Get(key)
	if ok {
		t.Fatal("deleted key still present")
	}
}

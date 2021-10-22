// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package autocert

import (
	"context"
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// make sure DirCache satisfies Cache interface
var _ Cache = DirCache("/")

func TestDirCache(t *testing.T) {
	dir, err := ioutil.TempDir("", "autocert")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)
	dir = filepath.Join(dir, "certs") // a nonexistent dir
	cache := DirCache(dir)
	ctx := context.Background()

	// test cache miss
	if _, err := cache.Get(ctx, "nonexistent"); err != ErrCacheMiss {
		t.Errorf("get: %v; want ErrCacheMiss", err)
	}

	// test put/get
	b1 := []byte{1}
	if err := cache.Put(ctx, "dummy", b1); err != nil {
		t.Fatalf("put: %v", err)
	}
	b2, err := cache.Get(ctx, "dummy")
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	if !reflect.DeepEqual(b1, b2) {
		t.Errorf("b1 = %v; want %v", b1, b2)
	}
	name := filepath.Join(dir, "dummy")
	if _, err := os.Stat(name); err != nil {
		t.Error(err)
	}

	// test put deletes temp file
	tmp, err := filepath.Glob(name + "?*")
	if err != nil {
		t.Error(err)
	}
	if tmp != nil {
		t.Errorf("temp file exists: %s", tmp)
	}

	// test delete
	if err := cache.Delete(ctx, "dummy"); err != nil {
		t.Fatalf("delete: %v", err)
	}
	if _, err := cache.Get(ctx, "dummy"); err != ErrCacheMiss {
		t.Errorf("get: %v; want ErrCacheMiss", err)
	}
}

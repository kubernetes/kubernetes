// Copyright (c) 2025 Bart Venter <bartventer@proton.me>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package acceptance provides a suite of acceptance tests for Cache implementations.
package acceptance

import (
	"bytes"
	"slices"
	"testing"

	"github.com/bartventer/httpcache/internal/testutil"
	"github.com/bartventer/httpcache/store/driver"
	"github.com/bartventer/httpcache/store/expapi"
)

type Factory interface {
	Make() (cache driver.Conn, cleanup func())
}

type FactoryFunc func() (driver.Conn, func())

func (f FactoryFunc) Make() (driver.Conn, func()) { return f() }

// Run runs a standard suite of tests against the provided Cache implementation.
// The factory function must return a new, empty Cache for each test.
func Run(t *testing.T, factory Factory) {
	t.Helper()
	t.Run("SetAndGet", func(t *testing.T) { testSetAndGet(t, factory.Make) })
	t.Run("Overwrite", func(t *testing.T) { testOverwrite(t, factory.Make) })
	t.Run("Delete", func(t *testing.T) { testDelete(t, factory.Make) })
	t.Run("GetNonexistent", func(t *testing.T) { testGetNonexistent(t, factory.Make) })
	t.Run("DeleteNonexistent", func(t *testing.T) { testDeleteNonexistent(t, factory.Make) })
	t.Run("Keys", func(t *testing.T) { testKeys(t, factory.Make) })
}

func testSetAndGet(t *testing.T, factory FactoryFunc) {
	cache, cleanup := factory.Make()
	t.Cleanup(cleanup)

	key := "foo" //nolint:goconst // "foo" is a common test key
	value := []byte("bar")
	testutil.RequireNoError(t, cache.Set(key, value), "Set failed")
	got, err := cache.Get(key)
	testutil.RequireNoError(t, err, "Get failed")
	testutil.AssertTrue(t, bytes.Equal(got, value), "Get returned unexpected value")
}

func testOverwrite(t *testing.T, factory FactoryFunc) {
	cache, cleanup := factory.Make()
	t.Cleanup(cleanup)

	key := "foo"
	value1 := []byte("bar")
	value2 := []byte("baz")
	testutil.RequireNoError(t, cache.Set(key, value1), "Set failed")
	testutil.RequireNoError(t, cache.Set(key, value2), "Set overwrite failed")
	got, err := cache.Get(key)
	testutil.RequireNoError(t, err, "Get failed")
	testutil.AssertTrue(
		t,
		bytes.Equal(got, value2),
		"Get after overwrite returned unexpected value",
	)
}

func testDelete(t *testing.T, factory FactoryFunc) {
	cache, cleanup := factory.Make()
	t.Cleanup(cleanup)

	key := "foo"
	value := []byte("bar")
	testutil.RequireNoError(t, cache.Set(key, value), "Set failed")
	testutil.RequireNoError(t, cache.Delete(key), "Delete failed")
	_, err := cache.Get(key)
	testutil.RequireErrorIs(
		t,
		err,
		driver.ErrNotExist,
		"Get after delete did not return ErrNotExist",
	)
}

func testGetNonexistent(t *testing.T, factory FactoryFunc) {
	cache, cleanup := factory.Make()
	t.Cleanup(cleanup)

	_, err := cache.Get("doesnotexist")
	testutil.RequireErrorIs(
		t,
		err,
		driver.ErrNotExist,
		"Get non-existent key did not return ErrNotExist",
	)
}

func testDeleteNonexistent(t *testing.T, factory FactoryFunc) {
	cache, cleanup := factory.Make()
	t.Cleanup(cleanup)

	err := cache.Delete("doesnotexist")
	testutil.RequireErrorIs(
		t,
		err,
		driver.ErrNotExist,
		"Delete non-existent key did not return ErrNotExist",
	)
}

func testKeys(t *testing.T, factory FactoryFunc) {
	cache, cleanup := factory.Make()
	t.Cleanup(cleanup)

	kl, ok := cache.(expapi.KeyLister)
	if !ok {
		t.Skip("Cache implementation does not support key listing")
	}
	keys := []string{"foo", "bar", "baz"}
	for _, key := range keys {
		value := []byte("value for " + key)
		testutil.RequireNoError(t, cache.Set(key, value), "Set failed for key "+key)
	}
	gotKeys, err := kl.Keys("")
	testutil.RequireNoError(t, err, "Keys retrieval failed")
	slices.Sort(gotKeys)
	slices.Sort(keys)
	testutil.AssertTrue(
		t,
		slices.Equal(gotKeys, keys),
		"Keys did not match expected keys",
	)
}

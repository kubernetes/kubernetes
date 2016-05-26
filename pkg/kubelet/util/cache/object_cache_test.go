/*
Copyright 2016 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package cache

import (
	"fmt"
	"testing"
	"time"

	expirationCache "k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/util/clock"
)

type testObject struct {
	key string
	val string
}

// A fake objectCache for unit test.
func NewFakeObjectCache(f func() (interface{}, error), ttl time.Duration, clock clock.Clock) *ObjectCache {
	ttlPolicy := &expirationCache.TTLPolicy{Ttl: ttl, Clock: clock}
	deleteChan := make(chan string, 1)
	return &ObjectCache{
		updater: f,
		cache:   expirationCache.NewFakeExpirationStore(stringKeyFunc, deleteChan, ttlPolicy, clock),
	}
}

func TestAddAndGet(t *testing.T) {
	testObj := testObject{
		key: "foo",
		val: "bar",
	}
	objectCache := NewFakeObjectCache(func() (interface{}, error) {
		return nil, fmt.Errorf("Unexpected Error: updater should never be called in this test!")
	}, 1*time.Hour, clock.NewFakeClock(time.Now()))

	err := objectCache.Add(testObj.key, testObj.val)
	if err != nil {
		t.Errorf("Unable to add obj %#v by key: %s", testObj, testObj.key)
	}
	value, err := objectCache.Get(testObj.key)
	if err != nil {
		t.Errorf("Unable to get obj %#v by key: %s", testObj, testObj.key)
	}
	if value.(string) != testObj.val {
		t.Errorf("Expected to get cached value: %#v, but got: %s", testObj.val, value.(string))
	}

}

func TestExpirationBasic(t *testing.T) {
	unexpectedVal := "bar"
	expectedVal := "bar2"

	testObj := testObject{
		key: "foo",
		val: unexpectedVal,
	}

	fakeClock := clock.NewFakeClock(time.Now())

	objectCache := NewFakeObjectCache(func() (interface{}, error) {
		return expectedVal, nil
	}, 1*time.Second, fakeClock)

	err := objectCache.Add(testObj.key, testObj.val)
	if err != nil {
		t.Errorf("Unable to add obj %#v by key: %s", testObj, testObj.key)
	}

	// sleep 2s so cache should be expired.
	fakeClock.Sleep(2 * time.Second)

	value, err := objectCache.Get(testObj.key)
	if err != nil {
		t.Errorf("Unable to get obj %#v by key: %s", testObj, testObj.key)
	}
	if value.(string) != expectedVal {
		t.Errorf("Expected to get cached value: %#v, but got: %s", expectedVal, value.(string))
	}
}

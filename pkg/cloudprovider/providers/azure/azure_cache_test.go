/*
Copyright 2017 The Kubernetes Authors.

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

package azure

import (
	"sync/atomic"
	"testing"
	"time"
)

func TestCacheReturnsSameObject(t *testing.T) {
	type cacheTestingStruct struct{}
	c := newTimedcache(1 * time.Minute)
	o1 := cacheTestingStruct{}
	get1, _ := c.GetOrCreate("b1", func() interface{} {
		return o1
	})
	o2 := cacheTestingStruct{}
	get2, _ := c.GetOrCreate("b1", func() interface{} {
		return o2
	})
	if get1 != get2 {
		t.Error("Get not equal")
	}
}

func TestCacheCallsCreateFuncOnce(t *testing.T) {
	var callsCount uint32
	f1 := func() interface{} {
		atomic.AddUint32(&callsCount, 1)
		return 1
	}
	c := newTimedcache(500 * time.Millisecond)
	for index := 0; index < 20; index++ {
		_, _ = c.GetOrCreate("b1", f1)
	}

	if callsCount != 1 {
		t.Error("Count not match")
	}
	time.Sleep(500 * time.Millisecond)
	c.GetOrCreate("b1", f1)
	if callsCount != 2 {
		t.Error("Count not match")
	}
}

func TestCacheExpires(t *testing.T) {
	f1 := func() interface{} {
		return 1
	}
	c := newTimedcache(500 * time.Millisecond)
	get1, _ := c.GetOrCreate("b1", f1)
	if get1 != 1 {
		t.Error("Value not equal")
	}
	time.Sleep(500 * time.Millisecond)
	get1, _ = c.GetOrCreate("b1", nil)
	if get1 != nil {
		t.Error("value not expired")
	}
}

func TestCacheDelete(t *testing.T) {
	f1 := func() interface{} {
		return 1
	}
	c := newTimedcache(500 * time.Millisecond)
	get1, _ := c.GetOrCreate("b1", f1)
	if get1 != 1 {
		t.Error("Value not equal")
	}
	get1, _ = c.GetOrCreate("b1", nil)
	if get1 != 1 {
		t.Error("Value not equal")
	}
	c.Delete("b1")
	get1, _ = c.GetOrCreate("b1", nil)
	if get1 != nil {
		t.Error("value not deleted")
	}
}

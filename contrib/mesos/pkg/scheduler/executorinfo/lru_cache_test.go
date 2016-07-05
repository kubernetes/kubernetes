/*
Copyright 2015 The Kubernetes Authors.

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

package executorinfo

import (
	"testing"

	"github.com/mesos/mesos-go/mesosproto"
)

func TestLruCache(t *testing.T) {
	c, err := NewCache(2)
	if err != nil {
		t.Fatal(err)
	}

	e := &mesosproto.ExecutorInfo{}

	c.Add("foo", e)
	c.Add("bar", e)

	if _, ok := c.Get("bar"); !ok {
		t.Fatal(`expected "bar" but got none`)
	}

	if _, ok := c.Get("foo"); !ok {
		t.Fatal(`expected "foo" but got none`)
	}

	c.Add("foo", e)
	c.Add("baz", e)

	if _, ok := c.Get("bar"); ok {
		t.Fatal(`expected none but got "bar"`)
	}

	c.Remove("foo")
	if _, ok := c.Get("foo"); ok {
		t.Fatal(`expected none but got "foo"`)
	}
}

/*
Copyright 2014 The Kubernetes Authors.

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

package watch_test

import (
	"io"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	. "k8s.io/apimachinery/pkg/watch"
)

type fakeDecoder struct {
	items chan Event
}

func (f fakeDecoder) Decode() (action EventType, object runtime.Object, err error) {
	item, open := <-f.items
	if !open {
		return action, nil, io.EOF
	}
	return item.Type, item.Object, nil
}

func (f fakeDecoder) Close() {
	close(f.items)
}

func TestStreamWatcher(t *testing.T) {
	table := []Event{
		{Type: Added, Object: testType("foo")},
	}

	fd := fakeDecoder{make(chan Event, 5)}
	sw := NewStreamWatcher(fd)

	for _, item := range table {
		fd.items <- item
		got, open := <-sw.ResultChan()
		if !open {
			t.Errorf("unexpected early close")
		}
		if e, a := item, got; !reflect.DeepEqual(e, a) {
			t.Errorf("expected %v, got %v", e, a)
		}
	}

	sw.Stop()
	_, open := <-sw.ResultChan()
	if open {
		t.Errorf("Unexpected failure to close")
	}
}

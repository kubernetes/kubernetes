/*
Copyright 2018 The Kubernetes Authors.

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

package request

import (
	"net/http"
	"sync"
	"testing"
)

func TestRequestContextMapperGet(t *testing.T) {
	mapper := NewRequestContextMapper()
	context := NewContext()
	req, _ := http.NewRequest("GET", "/api/version/resource", nil)

	// empty mapper
	if _, ok := mapper.Get(req); ok {
		t.Fatalf("got unexpected context")
	}

	// init mapper
	mapper.(*requestContextMap).init(req, context)
	if _, ok := mapper.Get(req); !ok {
		t.Fatalf("got no context")
	}

	// remove request context
	mapper.(*requestContextMap).remove(req)
	if _, ok := mapper.Get(req); ok {
		t.Fatalf("got unexpected context")
	}

}
func TestRequestContextMapperUpdate(t *testing.T) {
	mapper := NewRequestContextMapper()
	context := NewContext()
	req, _ := http.NewRequest("GET", "/api/version/resource", nil)

	// empty mapper
	if err := mapper.Update(req, context); err == nil {
		t.Fatalf("got no error")
	}

	// init mapper
	if !mapper.(*requestContextMap).init(req, context) {
		t.Fatalf("unexpected error, should init mapper")
	}

	context = WithNamespace(context, "default")
	if err := mapper.Update(req, context); err != nil {
		t.Fatalf("unexpected error")
	}

	if context, ok := mapper.Get(req); !ok {
		t.Fatalf("go no context")
	} else {
		if ns, _ := NamespaceFrom(context); ns != "default" {
			t.Fatalf("unexpected namespace %s", ns)
		}
	}
}

func TestRequestContextMapperConcurrent(t *testing.T) {
	mapper := NewRequestContextMapper()

	testCases := []struct{ url, namespace string }{
		{"/api/version/resource1", "ns1"},
		{"/api/version/resource2", "ns2"},
		{"/api/version/resource3", "ns3"},
		{"/api/version/resource4", "ns4"},
		{"/api/version/resource5", "ns5"},
	}

	wg := sync.WaitGroup{}
	for _, testcase := range testCases {
		wg.Add(1)
		go func(testcase struct{ url, namespace string }) {
			defer wg.Done()
			context := NewContext()
			req, _ := http.NewRequest("GET", testcase.url, nil)

			if !mapper.(*requestContextMap).init(req, context) {
				t.Errorf("unexpected init error")
				return
			}
			if _, ok := mapper.Get(req); !ok {
				t.Errorf("got no context")
				return
			}
			context2 := WithNamespace(context, testcase.namespace)
			if err := mapper.Update(req, context2); err != nil {
				t.Errorf("unexpected update error")
				return
			}
			if context, ok := mapper.Get(req); !ok {
				t.Errorf("got no context")
				return
			} else {
				if ns, _ := NamespaceFrom(context); ns != testcase.namespace {
					t.Errorf("unexpected namespace %s", ns)
					return
				}
			}
		}(testcase)
	}
	wg.Wait()
}

func BenchmarkRequestContextMapper(b *testing.B) {
	mapper := NewRequestContextMapper()

	b.SetParallelism(500)
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			context := NewContext()
			req, _ := http.NewRequest("GET", "/api/version/resource", nil)

			// 1 init
			mapper.(*requestContextMap).init(req, context)

			// 5 Get + 4 Update
			mapper.Get(req)
			context = WithNamespace(context, "default1")
			mapper.Update(req, context)
			mapper.Get(req)
			context = WithNamespace(context, "default2")
			mapper.Update(req, context)
			mapper.Get(req)
			context = WithNamespace(context, "default3")
			mapper.Update(req, context)
			mapper.Get(req)
			context = WithNamespace(context, "default4")
			mapper.Update(req, context)
			mapper.Get(req)

			// 1 remove
			mapper.(*requestContextMap).remove(req)
		}
	})
}

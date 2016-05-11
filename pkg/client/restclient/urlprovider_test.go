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

package restclient

import (
	"net/url"
	"testing"
)

func TestRoundRobinProvider(t *testing.T) {
	url1, _ := url.Parse("http://master:8000")
	url2, _ := url.Parse("http://master:8001")

	rr := NewRoundRobinProvider(url1, url2)

	returned := rr.Get()
	if url1 != returned {
		t.Errorf("Expected %v != Returned %v", url1, returned)
	}
	returned = rr.Next()
	if url2 != returned {
		t.Errorf("Expected %v != Returned %v", url2, returned)
	}
	returned = rr.Next()
	if url1 != returned {
		t.Errorf("Expected %v != Returned %v", url1, returned)
	}
}

func TestRoundRobinSameURLs(t *testing.T) {
	url, _ := url.Parse("http://master:8000")
	rr := NewRoundRobinProvider(url, url)

	if u := rr.Get(); u != url {
		t.Errorf("Unexpected URL %v on Get", u)
	}
	if u := rr.Next(); u != url {
		t.Errorf("Unexpected URL %v on Next", u)
	}
}

func TestRoundRobinSingleURL(t *testing.T) {
	url, _ := url.Parse("http://master:8000")
	rr := NewRoundRobinProvider(url)
	if u := rr.Next(); u != url {
		t.Errorf("Unexpected URL %v on Next", u)
	}
}

// Copyright 2015 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package flags

import (
	"testing"
)

func TestValidateURLsValueBad(t *testing.T) {
	tests := []string{
		// bad IP specification
		":2379",
		"127.0:8080",
		"123:456",
		// bad port specification
		"127.0.0.1:foo",
		"127.0.0.1:",
		// unix sockets not supported
		"unix://",
		"unix://tmp/etcd.sock",
		// bad strings
		"somewhere",
		"234#$",
		"file://foo/bar",
		"http://hello/asdf",
		"http://10.1.1.1",
	}
	for i, in := range tests {
		u := URLsValue{}
		if err := u.Set(in); err == nil {
			t.Errorf(`#%d: unexpected nil error for in=%q`, i, in)
		}
	}
}

func TestValidateURLsValueGood(t *testing.T) {
	tests := []string{
		"https://1.2.3.4:8080",
		"http://10.1.1.1:80",
		"http://localhost:80",
		"http://:80",
	}
	for i, in := range tests {
		u := URLsValue{}
		if err := u.Set(in); err != nil {
			t.Errorf("#%d: err=%v, want nil for in=%q", i, err, in)
		}
	}
}

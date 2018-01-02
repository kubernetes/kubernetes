/*
Copyright (c) 2015 VMware, Inc. All Rights Reserved.

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

package soap

import "testing"

func TestSplitHostPort(t *testing.T) {
	tests := []struct {
		url  string
		host string
		port string
	}{
		{"127.0.0.1", "127.0.0.1", ""},
		{"*:1234", "*", "1234"},
		{"127.0.0.1:80", "127.0.0.1", "80"},
		{"[::1]:6767", "[::1]", "6767"},
		{"[::1]", "[::1]", ""},
	}

	for _, test := range tests {
		host, port := splitHostPort(test.url)
		if host != test.host {
			t.Errorf("(%s) %s != %s", test.url, host, test.host)
		}
		if port != test.port {
			t.Errorf("(%s) %s != %s", test.url, port, test.port)
		}
	}
}

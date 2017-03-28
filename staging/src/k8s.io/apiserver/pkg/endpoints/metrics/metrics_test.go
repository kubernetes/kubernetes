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

package metrics

import "testing"

func TestCleanUserAgent(t *testing.T) {
	for _, tc := range []struct {
		In  string
		Out string
	}{
		{
			In:  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36",
			Out: "Browser",
		},
		{
			In:  "kubectl/v1.2.4",
			Out: "kubectl/v1.2.4",
		},
	} {
		if cleanUserAgent(tc.In) != tc.Out {
			t.Errorf("Failed to clean User-Agent: %s", tc.In)
		}
	}
}

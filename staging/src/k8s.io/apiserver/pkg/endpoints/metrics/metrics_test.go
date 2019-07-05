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
	panicBuf := []byte{198, 73, 129, 133, 90, 216, 104, 29, 13, 134, 209, 233, 30, 0, 22}

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
		{
			In:  `C:\Users\Kubernetes\kubectl.exe/v1.5.4`,
			Out: "kubectl.exe/v1.5.4",
		},
		{
			In:  `C:\Program Files\kubectl.exe/v1.5.4`,
			Out: "kubectl.exe/v1.5.4",
		},
		{
			// This malicious input courtesy of enisoc.
			In:  string(panicBuf) + "kubectl.exe",
			Out: "kubectl.exe",
		},
	} {
		if cleanUserAgent(tc.In) != tc.Out {
			t.Errorf("Failed to clean User-Agent: %s", tc.In)
		}
	}
}

/*
Copyright The Kubernetes Authors.

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

package httpresponse

import "net/http"

type wrapper interface {
	Unwrap() http.ResponseWriter
}

// GetOriginal walks any response writer wrapper chain and returns the first writer.
func GetOriginal(w http.ResponseWriter) http.ResponseWriter {
	for {
		decorator, ok := w.(wrapper)
		if !ok {
			return w
		}
		inner := decorator.Unwrap()
		if inner == w {
			panic("http.ResponseWriter decorator chain has a cycle")
		}
		w = inner
	}
}

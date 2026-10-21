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

package rest

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

// APIResultAdapter lets a rest.Result implement the metav1.APIResult interface.
type APIResultAdapter struct {
	Result Result
}

var _ metav1.APIResult = APIResultAdapter{}

// Get implements metav1.APIResult.
func (a APIResultAdapter) Get() (runtime.Object, error) {
	return a.Result.Get()
}

// StatusCode returns the HTTP status code of the response and any error from the underlying Result.
func (a APIResultAdapter) StatusCode() (int, error) {
	return a.Result.statusCode, a.Result.Error()
}

// Error implements metav1.APIResult.
func (a APIResultAdapter) Error() error {
	return a.Result.Error()
}

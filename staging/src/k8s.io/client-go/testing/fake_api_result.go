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

package testing

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

// FakeAPIResult implements metav1.APIResult for testing.
type FakeAPIResult struct {
	Obj  runtime.Object
	Err  error
	Code int
}

var _ metav1.APIResult = FakeAPIResult{}

// Get implements metav1.APIResult.
func (f FakeAPIResult) Get() (runtime.Object, error) {
	if f.Err != nil {
		return nil, f.Err
	}
	return f.Obj, nil
}

// StatusCode returns the configured HTTP status code and error.
func (f FakeAPIResult) StatusCode() (int, error) {
	return f.Code, f.Err
}

// Error implements metav1.APIResult.
func (f FakeAPIResult) Error() error {
	return f.Err
}

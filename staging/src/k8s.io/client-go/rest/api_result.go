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
)

// RestResultWrapper wraps a rest.Result to implement metav1.APIResult.
type RestResultWrapper struct {
	Result
}

var _ metav1.APIResult = RestResultWrapper{}

// StatusCode overrides Result.StatusCode to return metav1.APIResult.
func (w RestResultWrapper) StatusCode(statusCode *int) metav1.APIResult {
	w.Result.StatusCode(statusCode)
	return w
}

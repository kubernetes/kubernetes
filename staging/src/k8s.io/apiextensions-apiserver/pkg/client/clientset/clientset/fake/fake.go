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

package fake

import (
	fake "k8s.io/apiextensions/pkg/client/clientset/clientset/fake"
	"k8s.io/apimachinery/pkg/runtime"
)

//go:fix inline
type Clientset = fake.Clientset

var (
	AddToScheme = fake.AddToScheme
)

//go:fix inline
func NewClientset(objects ...runtime.Object) *fake.Clientset {
	return fake.NewClientset(objects...)
}

//go:fix inline
func NewSimpleClientset(objects ...runtime.Object) *fake.Clientset {
	return fake.NewSimpleClientset(objects...)
}

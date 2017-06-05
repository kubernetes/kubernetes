/*
Copyright 2017 The Kubernetes Authors.

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

package v1beta1

import (
	rest "k8s.io/client-go/rest"
)

// ScalesGetter has a method to return a ScaleInterface.
// A group's client should implement this interface.
type ScalesGetter interface {
	Scales(namespace string) ScaleInterface
}

// ScaleInterface has methods to work with Scale resources.
type ScaleInterface interface {
	ScaleExpansion
}

// scales implements ScaleInterface
type scales struct {
	client rest.Interface
	ns     string
}

// newScales returns a Scales
func newScales(c *ExtensionsV1beta1Client, namespace string) *scales {
	return &scales{
		client: c.RESTClient(),
		ns:     namespace,
	}
}

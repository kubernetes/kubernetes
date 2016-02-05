/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package unversioned

// ScaleTwosGetter has a method to return a ScaleTwoInterface.
// A group's client should implement this interface.
type ScaleTwosGetter interface {
	ScaleTwos(namespace string) ScaleTwoInterface
}

// ScaleTwoInterface has methods to work with ScaleTwo resources.
type ScaleTwoInterface interface {
	ScaleTwoExpansion
}

// scaleTwos implements ScaleTwoInterface
type scaleTwos struct {
	client *ExtensionsClient
	ns     string
}

// newScaleTwos returns a ScaleTwos
func newScaleTwos(c *ExtensionsClient, namespace string) *scaleTwos {
	return &scaleTwos{
		client: c,
		ns:     namespace,
	}
}

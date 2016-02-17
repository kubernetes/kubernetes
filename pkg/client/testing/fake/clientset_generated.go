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

package fake

import (
	unversionedcore "k8s.io/kubernetes/pkg/client/typed/generated/core/unversioned"
	fakeunversionedcore "k8s.io/kubernetes/pkg/client/typed/generated/core/unversioned/fake"
	unversionedextensions "k8s.io/kubernetes/pkg/client/typed/generated/extensions/unversioned"
	fakeunversionedextensions "k8s.io/kubernetes/pkg/client/typed/generated/extensions/unversioned/fake"
)

// Core retrieves the CoreClient
func (c *Clientset) Core() unversionedcore.CoreInterface {
	return &fakeunversionedcore.FakeCore{&c.Fake}
}

// Extensions retrieves the ExtensionsClient
func (c *Clientset) Extensions() unversionedextensions.ExtensionsInterface {
	return &fakeunversionedextensions.FakeExtensions{&c.Fake}
}

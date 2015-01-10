/*
Copyright 2015 Google Inc. All rights reserved.

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

package cache

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// ListWatch knows how to list and watch a set of apiserver resources.  It satisfies the ListerWatcher interface.
// It is a convenience function for users of NewReflector, etc.
type ListWatch struct {
	Client        *client.Client
	FieldSelector labels.Selector
	Resource      string
	Namespace     string
}

// ListWatch knows how to list and watch a set of apiserver resources.
func (lw *ListWatch) List() (runtime.Object, error) {
	return lw.Client.
		Get().
		Namespace(lw.Namespace).
		Resource(lw.Resource).
		SelectorParam("fields", lw.FieldSelector).
		Do().
		Get()
}

func (lw *ListWatch) Watch(resourceVersion string) (watch.Interface, error) {
	return lw.Client.
		Get().
		Prefix("watch").
		Namespace(lw.Namespace).
		Resource(lw.Resource).
		SelectorParam("fields", lw.FieldSelector).
		Param("resourceVersion", resourceVersion).
		Watch()
}

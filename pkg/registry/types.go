/*
Copyright 2014 Google Inc. All rights reserved.

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

package registry

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	_ "github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	_ "github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

type Store interface {
	List(ctx api.Context, out runtime.Object) error
	Get(ctx api.Context, name string, out runtime.Object) error
	Update(ctx api.Context, name string, obj runtime.Object) error
	//	Watch(ctx api.Context, labels, fields labels.Selector, resourceVersion string) (watch.Interface, error)
}

type REST interface {
}

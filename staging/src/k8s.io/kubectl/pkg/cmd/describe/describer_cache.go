/*
Copyright 2022 The Kubernetes Authors.

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

package describe

import (
	"github.com/davecgh/go-spew/spew"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubectl/pkg/describe"
)

// describerCache can map from a meta.RESTMapping to a describe.ResourceDescriber.
//
// As of right now, this does not need to be synced since it is function-local, but
// it certainly could be updated to be thread-safe in the future.
type describerCache struct {
	cache map[string]describe.ResourceDescriber
}

func newDescriberCache() *describerCache {
	return &describerCache{cache: make(map[string]describe.ResourceDescriber)}
}

func (d *describerCache) key(mapping *meta.RESTMapping) string {
	key := struct {
		gvr   schema.GroupVersionResource
		gvk   schema.GroupVersionKind
		scope string
	}{
		gvr:   mapping.Resource,
		gvk:   mapping.GroupVersionKind,
		scope: string(mapping.Scope.Name()),
	}
	return (&spew.ConfigState{DisableMethods: true, Indent: " "}).Sprint(key)
}

func (d *describerCache) get(mapping *meta.RESTMapping) describe.ResourceDescriber {
	return d.cache[d.key(mapping)]
}

func (d *describerCache) put(mapping *meta.RESTMapping, describer describe.ResourceDescriber) {
	d.cache[d.key(mapping)] = describer
}

/*
Copyright 2021 The Kubernetes Authors.

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

package events

import (
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// Inspired by k8s.io/cli-runtime/pkg/resource splitResourceTypeName()

// splitResourceTypeName handles type/name resource formats and returns a resource tuple
// (empty or not), whether it successfully found one, and an error
func decodeResourceTypeName(mapper meta.RESTMapper, s string) (gvk schema.GroupVersionKind, name string, found bool, err error) {
	if !strings.Contains(s, "/") {
		return
	}
	seg := strings.Split(s, "/")
	if len(seg) != 2 {
		err = fmt.Errorf("arguments in resource/name form may not have more than one slash")
		return
	}
	resource, name := seg[0], seg[1]

	var gvr schema.GroupVersionResource
	gvr, err = mapper.ResourceFor(schema.GroupVersionResource{Resource: resource})
	if err != nil {
		return
	}
	gvk, err = mapper.KindFor(gvr)
	if err != nil {
		return
	}
	found = true

	return
}

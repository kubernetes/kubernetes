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

package testing

import (
	"github.com/googleapis/gnostic/OpenAPIv2"
	"k8s.io/kube-openapi/pkg/util/proto"
)

// tesingExtensionKey is the key used only for testing.
// We use some kubernetes openapi spec for testing that has this key.
const tesingExtensionKey = "x-kubernetes-group-version-kind"

// Get and parse GroupVersionKind from the extension. Returns empty if it doesn't have one.
func ParseGroupVersionKind(s *openapi_v2.Schema) string {
	extensionMap := proto.VendorExtensionToMap(s.GetVendorExtension())

	// Get the extensions
	gvkExtension, ok := extensionMap[tesingExtensionKey]
	if !ok {
		return ""
	}

	// gvk extension must be a list of 1 element.
	gvkList, ok := gvkExtension.([]interface{})
	if !ok {
		return ""
	}
	if len(gvkList) != 1 {
		return ""

	}
	gvk := gvkList[0]

	// gvk extension list must be a map with group, version, and
	// kind fields
	gvkMap, ok := gvk.(map[interface{}]interface{})
	if !ok {
		return ""
	}
	group, ok := gvkMap["group"].(string)
	if !ok {
		return ""
	}
	version, ok := gvkMap["version"].(string)
	if !ok {
		return ""
	}
	kind, ok := gvkMap["kind"].(string)
	if !ok {
		return ""
	}

	return GvkString(group, version, kind)
}

func GvkString(group, version, kind string) string {
	return group + "/" + version + ", Kind=" + kind
}


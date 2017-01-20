/*
Copyright 2015 The Kubernetes Authors.

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

package thirdpartyresourcedata

import (
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/apis/extensions"
)

func ExtractGroupVersionKind(list *extensions.ThirdPartyResourceList) ([]schema.GroupVersion, []schema.GroupVersionKind, error) {
	gvs := []schema.GroupVersion{}
	gvks := []schema.GroupVersionKind{}
	for ix := range list.Items {
		rsrc := &list.Items[ix]
		kind, group, err := ExtractApiGroupAndKind(rsrc)
		if err != nil {
			return nil, nil, err
		}
		for _, version := range rsrc.Versions {
			gv := schema.GroupVersion{Group: group, Version: version.Name}
			gvs = append(gvs, gv)
			gvks = append(gvks, schema.GroupVersionKind{Group: group, Version: version.Name, Kind: kind})
		}
	}
	return gvs, gvks, nil
}

func convertToCamelCase(input string) string {
	result := ""
	toUpper := true
	for ix := range input {
		char := input[ix]
		if toUpper {
			result = result + string([]byte{(char - 32)})
			toUpper = false
		} else if char == '-' {
			toUpper = true
		} else {
			result = result + string([]byte{char})
		}
	}
	return result
}

func ExtractApiGroupAndKind(rsrc *extensions.ThirdPartyResource) (kind string, group string, err error) {
	parts := strings.Split(rsrc.Name, ".")
	if len(parts) < 3 {
		return "", "", fmt.Errorf("unexpectedly short resource name: %s, expected at least <kind>.<domain>.<tld>", rsrc.Name)
	}
	return convertToCamelCase(parts[0]), strings.Join(parts[1:], "."), nil
}

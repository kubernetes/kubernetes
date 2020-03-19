/*
Copyright 2020 The Kubernetes Authors.

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

package v1

import (
	"strings"

	"k8s.io/apimachinery/pkg/util/sets"
)

func GetNodeRoles(labels map[string]string) []string {
	roles := sets.NewString()
	for k, v := range labels {
		switch {
		case strings.HasPrefix(k, LabelNodeRolePrefix):
			if role := strings.TrimPrefix(k, LabelNodeRolePrefix); len(role) > 0 {
				roles.Insert(role)
			}

		case k == LabelLegacyNodeLabelRole && v != "":
			roles.Insert(v)
		}
	}
	return roles.List()
}

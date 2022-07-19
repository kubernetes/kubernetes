/*
Copyright 2022 The KCP Authors.

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

package builder

import (
	"strings"

	"k8s.io/apimachinery/pkg/util/sets"
)

var dotlessKubeGroups = sets.NewString(
	"",
	"apps",
	"batch",
	"extensions",
	"policy",
)

// HACK: support the case when we add "dotless" built-in API groups
func packagePrefix(group string) string {
	if strings.Contains(group, ".") {
		return group
	}

	if !dotlessKubeGroups.Has(group) {
		// Shouldn't really be possible...
		return group
	}

	if group == "" {
		group = "core"
	}

	return "k8s.io/api/" + group
}

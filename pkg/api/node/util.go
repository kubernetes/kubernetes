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

package persistentvolume

import (
	"context"
	"fmt"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/node"
)

var DeprecatedNodeLabels = map[string]string{
	`beta.kubernetes.io/arch`:                  `deprecated since v1.14; use "kubernetes.io/arch" instead`,
	`beta.kubernetes.io/os`:                    `deprecated since v1.14; use "kubernetes.io/os" instead`,
	`failure-domain.beta.kubernetes.io/region`: `deprecated since v1.17; use "topology.kubernetes.io/region" instead`,
	`failure-domain.beta.kubernetes.io/zone`:   `deprecated since v1.17; use "topology.kubernetes.io/zone" instead`,
	`beta.kubernetes.io/instance-type`:         `deprecated since v1.17; use "node.kubernetes.io/instance-type" instead`,
}

func GetWarningsForRuntimeClass(ctx context.Context, rc *node.RuntimeClass) []string {
	var warnings []string

	if rc != nil {
		// use of deprecated node labels in scheduling's node affinity
		for key, _ := range rc.Scheduling.NodeSelector {
			if msg, deprecated := DeprecatedNodeLabels[key]; deprecated {
				warnings = append(warnings, fmt.Sprintf("%s: %s", field.NewPath("scheduling", "nodeSelector"), msg))
			}
		}
	}

	return warnings
}

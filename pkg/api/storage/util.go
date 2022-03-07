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

package storage

import (
	"context"
	"fmt"

	"k8s.io/apimachinery/pkg/util/validation/field"
	nodeapi "k8s.io/kubernetes/pkg/api/node"
	"k8s.io/kubernetes/pkg/apis/storage"
)

func GetWarningsForStorageClass(ctx context.Context, sc *storage.StorageClass) []string {
	var warnings []string

	if sc != nil {
		// use of deprecated node labels in allowedTopologies's matchLabelExpressions
		for i, topo := range sc.AllowedTopologies {
			for j, expression := range topo.MatchLabelExpressions {
				if msg, deprecated := nodeapi.DeprecatedNodeLabels[expression.Key]; deprecated {
					warnings = append(warnings, fmt.Sprintf("%s: %s", field.NewPath("allowedTopologies").Index(i).Child("matchLabelExpressions").Index(j), msg))
				}
			}
		}
	}

	return warnings
}

func GetWarningsForCSIStorageCapacity(ctx context.Context, csc *storage.CSIStorageCapacity) []string {
	var warnings []string

	if csc != nil {
		// use of deprecated node labels in allowedTopologies's matchLabelExpressions
		for i, expression := range csc.NodeTopology.MatchExpressions {
			if msg, deprecated := nodeapi.DeprecatedNodeLabels[expression.Key]; deprecated {
				warnings = append(
					warnings,
					fmt.Sprintf(
						"%s: %s is %s",
						field.NewPath("nodeTopology").Child("matchExpressions").Index(i),
						expression.Key,
						msg,
					),
				)
			}
		}

		// use of deprecated node labels in allowedTopologies's matchLabels
		for label, _ := range csc.NodeTopology.MatchLabels {
			if msg, deprecated := nodeapi.DeprecatedNodeLabels[label]; deprecated {
				warnings = append(warnings, fmt.Sprintf("%s: %s", field.NewPath("nodeTopology").Child("matchLabels").Child(label), msg))
			}
		}
	}

	return warnings
}

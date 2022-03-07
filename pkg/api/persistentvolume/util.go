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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	nodeapi "k8s.io/kubernetes/pkg/api/node"
	api "k8s.io/kubernetes/pkg/apis/core"
)

func GetWarningsForPersistentVolume(ctx context.Context, pv *api.PersistentVolume) []string {
	if pv == nil {
		return nil
	}
	return warningsForPersistentVolumeSpecAndMeta(nil, &pv.Spec, &pv.ObjectMeta)
}

func warningsForPersistentVolumeSpecAndMeta(fieldPath *field.Path, pvSpec *api.PersistentVolumeSpec, meta *metav1.ObjectMeta) []string {
	var warnings []string

	// use of deprecated node labels in node affinity
	for i, k := range pvSpec.NodeAffinity.Required.NodeSelectorTerms {
		expressions := k.MatchExpressions
		for j, e := range expressions {
			if msg, deprecated := nodeapi.DeprecatedNodeLabels[e.Key]; deprecated {
				warnings = append(
					warnings,
					fmt.Sprintf(
						"%s: %s is %s",
						fieldPath.Child("spec", "NodeAffinity").Child("Required").Child("NodeSelectorTerms").Index(i).Child("MatchExpressions").Index(j).Child("key"),
						e.Key,
						msg,
					),
				)
			}
		}
	}

	return warnings
}

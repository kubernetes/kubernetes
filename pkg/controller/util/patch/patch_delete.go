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

package patch

import (
	"encoding/json"
	"k8s.io/apimachinery/pkg/types"
)

type objectForDeleteOwnerRefStrategicMergePatch struct {
	Metadata objectMetaForMergePatch `json:"metadata"`
}

type objectMetaForMergePatch struct {
	UID              types.UID           `json:"uid"`
	OwnerReferences  []map[string]string `json:"ownerReferences"`
	DeleteFinalizers []string            `json:"$deleteFromPrimitiveList/finalizers,omitempty"`
}

func GenerateDeleteOwnerRefStrategicMergeBytes(dependentUID types.UID, ownerUIDs []types.UID, finalizers ...string) ([]byte, error) {
	var ownerReferences []map[string]string
	for _, ownerUID := range ownerUIDs {
		ownerReferences = append(ownerReferences, ownerReference(ownerUID, "delete"))
	}
	patch := objectForDeleteOwnerRefStrategicMergePatch{
		Metadata: objectMetaForMergePatch{
			UID:              dependentUID,
			OwnerReferences:  ownerReferences,
			DeleteFinalizers: finalizers,
		},
	}
	patchBytes, err := json.Marshal(&patch)
	if err != nil {
		return nil, err
	}
	return patchBytes, nil
}

func ownerReference(uid types.UID, patchType string) map[string]string {
	return map[string]string{
		"$patch": patchType,
		"uid":    string(uid),
	}
}

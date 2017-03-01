/*
Copyright 2016 The Kubernetes Authors.

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

package garbagecollector

import (
	"encoding/json"
	"fmt"
	"strings"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/controller/garbagecollector/metaonly"
)

func deleteOwnerRefPatch(dependentUID types.UID, ownerUIDs ...types.UID) []byte {
	var pieces []string
	for _, ownerUID := range ownerUIDs {
		pieces = append(pieces, fmt.Sprintf(`{"$patch":"delete","uid":"%s"}`, ownerUID))
	}
	patch := fmt.Sprintf(`{"metadata":{"ownerReferences":[%s],"uid":"%s"}}`, strings.Join(pieces, ","), dependentUID)
	return []byte(patch)
}

// generate a patch that unsets the BlockOwnerDeletion field of all
// ownerReferences of node.
func (n *node) patchToUnblockOwnerReferences() ([]byte, error) {
	var dummy metaonly.MetadataOnlyObject
	var blockingRefs []metav1.OwnerReference
	falseVar := false
	for _, owner := range n.owners {
		if owner.BlockOwnerDeletion != nil && *owner.BlockOwnerDeletion {
			ref := owner
			ref.BlockOwnerDeletion = &falseVar
			blockingRefs = append(blockingRefs, ref)
		}
	}
	dummy.ObjectMeta.SetOwnerReferences(blockingRefs)
	dummy.ObjectMeta.UID = n.identity.UID
	return json.Marshal(dummy)
}

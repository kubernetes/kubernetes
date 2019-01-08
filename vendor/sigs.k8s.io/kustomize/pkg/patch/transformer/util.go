/*
Copyright 2018 The Kubernetes Authors.

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

package transformer

import (
	"fmt"

	"sigs.k8s.io/kustomize/pkg/resid"
	"sigs.k8s.io/kustomize/pkg/resmap"
	"sigs.k8s.io/kustomize/pkg/resource"
)

func findTargetObj(m resmap.ResMap, targetId resid.ResId) (*resource.Resource, error) {
	matchedIds := m.FindByGVKN(targetId)
	if targetId.Namespace() != "" {
		var ids []resid.ResId
		for _, id := range matchedIds {
			if id.Namespace() == targetId.Namespace() {
				ids = append(ids, id)
			}
		}
		matchedIds = ids
	}

	if len(matchedIds) == 0 {
		return nil, fmt.Errorf("couldn't find any object to apply the json patch %v", targetId)
	}
	if len(matchedIds) > 1 {
		return nil, fmt.Errorf("found multiple objects that the patch can apply %v", matchedIds)
	}
	return m[matchedIds[0]], nil
}

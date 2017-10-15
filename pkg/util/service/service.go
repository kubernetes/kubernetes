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

package service

import (
	"encoding/json"
	"fmt"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
)

// Patch patches service spec and metadata.
func Patch(c v1core.CoreV1Interface, oldSvc *v1.Service, newSvc *v1.Service) (*v1.Service, error) {
	patchBytes, err := getPatchBytes(oldSvc, newSvc)
	if err != nil {
		return nil, err
	}

	updatedSvc, err := c.Services(oldSvc.Namespace).Patch(oldSvc.Name, types.StrategicMergePatchType, patchBytes)
	if err != nil {
		return nil, fmt.Errorf("failed to patch %q for svc %s/%s: %v", patchBytes, oldSvc.Namespace, oldSvc.Name, err)
	}
	return updatedSvc, nil
}

// PatchStatus patches services status.
func PatchStatus(c v1core.CoreV1Interface, oldSvc *v1.Service, newSvc *v1.Service) (*v1.Service, error) {
	// Reset spec to make sure only patch for Status or ObjectMeta is generated.
	// Note that we don't reset ObjectMeta here, because:
	// 1. This aligns with Services().UpdateStatus().
	// 2. Some components do use this to update service annotations.
	newSvc.Spec = oldSvc.Spec

	patchBytes, err := getPatchBytes(oldSvc, newSvc)
	if err != nil {
		return nil, err
	}

	updatedSvc, err := c.Services(oldSvc.Namespace).Patch(oldSvc.Name, types.StrategicMergePatchType, patchBytes, "status")
	if err != nil {
		return nil, fmt.Errorf("failed to patch status %q for svc %s/%s: %v", patchBytes, oldSvc.Namespace, oldSvc.Name, err)
	}
	return updatedSvc, nil
}

func getPatchBytes(oldSvc *v1.Service, newSvc *v1.Service) ([]byte, error) {
	oldData, err := json.Marshal(oldSvc)
	if err != nil {
		return nil, fmt.Errorf("failed to Marshal oldData for svc %s/%s: %v", oldSvc.Namespace, oldSvc.Name, err)
	}

	newData, err := json.Marshal(newSvc)
	if err != nil {
		return nil, fmt.Errorf("failed to Marshal newData for svc %s/%s: %v", newSvc.Namespace, newSvc.Name, err)
	}

	patchBytes, err := strategicpatch.CreateTwoWayMergePatch(oldData, newData, v1.Service{})
	if err != nil {
		return nil, fmt.Errorf("failed to CreateTwoWayMergePatch for svc %s/%s: %v", oldSvc.Namespace, oldSvc.Name, err)
	}
	return patchBytes, nil
}

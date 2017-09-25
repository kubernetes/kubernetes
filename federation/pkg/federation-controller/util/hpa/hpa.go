/*
Copyright 2017 The Kubernetes Authors.

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

package hpa

import (
	"encoding/json"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
)

const (
	// FederatedAnnotationOnHpaTargetObj as key, is used by hpa controller to
	// set selected cluster name list as annotation on the target object.
	FederatedAnnotationOnHpaTargetObj = "federation.kubernetes.io/hpa-target-cluster-list"
)

// ClusterNames stores the list of clusters represented by names as appearing on federation
// cluster objects. This is set by federation hpa and used by target objects federation
// controller to restrict that target object to only these clusters.
type ClusterNames struct {
	Names []string
}

func (cn *ClusterNames) String() string {
	annotationBytes, _ := json.Marshal(cn)
	return string(annotationBytes[:])
}

// GetHpaTargetClusterList is used to get the list of clusters from the target object
// annotations.
func GetHpaTargetClusterList(obj runtime.Object) (*ClusterNames, error) {
	accessor, _ := meta.Accessor(obj)
	targetObjAnno := accessor.GetAnnotations()
	if targetObjAnno == nil {
		return nil, nil
	}
	targetObjAnnoString, exists := targetObjAnno[FederatedAnnotationOnHpaTargetObj]
	if !exists {
		return nil, nil
	}

	clusterNames := &ClusterNames{}
	if err := json.Unmarshal([]byte(targetObjAnnoString), clusterNames); err != nil {
		return nil, err
	}
	return clusterNames, nil
}

// SetHpaTargetClusterList is used to set the list of clusters on the target object
// annotations.
func SetHpaTargetClusterList(obj runtime.Object, clusterNames ClusterNames) runtime.Object {
	accessor, _ := meta.Accessor(obj)
	anno := accessor.GetAnnotations()
	if anno == nil {
		anno = make(map[string]string)
		accessor.SetAnnotations(anno)
	}
	anno[FederatedAnnotationOnHpaTargetObj] = clusterNames.String()
	return obj
}

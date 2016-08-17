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

package util

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api/meta"
	pkg_runtime "k8s.io/kubernetes/pkg/runtime"
)

const (
	//TODO: This will be removed once cluster name field is added to ObjectMeta.
	ClusterNameAnnotation = "federation.io/name"
)

// TODO: This will be refactored once cluster name field is added to ObjectMeta.
func GetClusterName(obj pkg_runtime.Object) (string, error) {
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return "", err
	}
	annotations := accessor.GetAnnotations()
	if annotations != nil {
		if value, found := annotations[ClusterNameAnnotation]; found {
			return value, nil
		}
	}
	return "", fmt.Errorf("Cluster information not available")
}

// TODO: This will be removed once cluster name field is added to ObjectMeta.
func SetClusterName(obj pkg_runtime.Object, clusterName string) error {
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return err
	}
	annotations := accessor.GetAnnotations()
	if annotations == nil {
		annotations = make(map[string]string)
		accessor.SetAnnotations(annotations)
	}
	annotations[ClusterNameAnnotation] = clusterName
	return nil
}

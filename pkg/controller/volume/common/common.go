/*
Copyright 2020 The Kubernetes Authors.

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

package common

import (
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/tools/cache"
)

const (
	// PodPVCIndex is the lookup name for the index function, which is to index by pod pvcs.
	PodPVCIndex = "pod-pvc-index"
)

// PodPVCIndexFunc returns PVC keys for given pod
func PodPVCIndexFunc(obj interface{}) ([]string, error) {
	pod, ok := obj.(*v1.Pod)
	if !ok {
		return []string{}, nil
	}
	keys := []string{}
	for _, podVolume := range pod.Spec.Volumes {
		if pvcSource := podVolume.VolumeSource.PersistentVolumeClaim; pvcSource != nil {
			keys = append(keys, fmt.Sprintf("%s/%s", pod.Namespace, pvcSource.ClaimName))
		}
	}
	return keys, nil
}

// AddIndexerIfNotPresent adds the index function with the name into the cache indexer if not present
func AddIndexerIfNotPresent(indexer cache.Indexer, indexName string, indexFunc cache.IndexFunc) error {
	indexers := indexer.GetIndexers()
	if _, ok := indexers[indexName]; ok {
		return nil
	}
	return indexer.AddIndexers(cache.Indexers{indexName: indexFunc})
}

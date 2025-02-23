/*
Copyright 2019 The Kubernetes Authors.

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

package componentconfigs

import (
	"crypto/sha256"
	"fmt"
	"maps"
	"slices"

	v1 "k8s.io/api/core/v1"

	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

// ChecksumForConfigMap calculates a checksum for the supplied config map. The exact algorithm depends on hash and prefix parameters
func ChecksumForConfigMap(cm *v1.ConfigMap) string {
	hash := sha256.New()

	for _, key := range slices.Sorted(maps.Keys(cm.Data)) {
		hash.Write([]byte(cm.Data[key]))
	}

	for _, key := range slices.Sorted(maps.Keys(cm.BinaryData)) {
		hash.Write(cm.BinaryData[key])
	}

	return fmt.Sprintf("sha256:%x", hash.Sum(nil))
}

// SignConfigMap calculates the supplied config map checksum and annotates it with it
func SignConfigMap(cm *v1.ConfigMap) {
	if cm.Annotations == nil {
		cm.Annotations = map[string]string{}
	}
	cm.Annotations[constants.ComponentConfigHashAnnotationKey] = ChecksumForConfigMap(cm)
}

// VerifyConfigMapSignature returns true if the config map has checksum annotation and it matches; false otherwise
func VerifyConfigMapSignature(cm *v1.ConfigMap) bool {
	signature, ok := cm.Annotations[constants.ComponentConfigHashAnnotationKey]
	if !ok {
		return false
	}
	return signature == ChecksumForConfigMap(cm)
}

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
	"reflect"

	api_v1 "k8s.io/kubernetes/pkg/api/v1"
)

// Checks if cluster-independent, user provided data in two given ConfigMaps are equal. If in
// the future the ConfigMap structure is expanded then any field that is not populated.
// by the api server should be included here.
func ConfigMapEquivalent(s1, s2 *api_v1.ConfigMap) bool {
	return ObjectMetaEquivalent(s1.ObjectMeta, s2.ObjectMeta) &&
		reflect.DeepEqual(s1.Data, s2.Data)
}

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

package internaldeps

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/api"
)

// ToInternalList implements resource.ToList and is used by callers of
// Result.Object to convert multiple returned Objects into an internal
// List containing the objects.
func ToInternalList(objects []runtime.Object, version string) runtime.Object {
	return &api.List{
		ListMeta: metav1.ListMeta{
			ResourceVersion: version,
		},
		Items: objects,
	}
}

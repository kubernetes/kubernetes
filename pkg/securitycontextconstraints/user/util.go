/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package user

import (
	"fmt"
	"strconv"

	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
)

func AnnotationToIntPtr(sUID string) (*int64, error) {
	uid, err := strconv.ParseInt(sUID, 10, 64)
	if err != nil {
		return nil, err
	}
	return &uid, nil
}

func GetAllocatedID(kClient client.Interface, pod *api.Pod, annotation string) (*int64, error) {
	if len(pod.Spec.ServiceAccountName) > 0 {
		sa, err := kClient.ServiceAccounts(pod.Namespace).Get(pod.Spec.ServiceAccountName)
		if err != nil {
			return nil, err
		}
		sUID, ok := sa.Annotations[annotation]
		if !ok {
			return nil, fmt.Errorf("Unable to find annotation %s on service account %s", annotation, pod.Spec.ServiceAccountName)
		}
		return AnnotationToIntPtr(sUID)
	} else {
		ns, err := kClient.Namespaces().Get(pod.Namespace)
		if err != nil {
			return nil, err
		}
		sUID, ok := ns.Annotations[annotation]
		if !ok {
			return nil, fmt.Errorf("Unable to find annotation %s on namespace %s", annotation, pod.Namespace)
		}
		return AnnotationToIntPtr(sUID)
	}
}

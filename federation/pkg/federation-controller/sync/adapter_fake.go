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

package sync

import (
	pkgruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/federation/pkg/federatedtypes"
)

type fakeSchedulerAdapter struct {
	federatedtypes.SecretAdapter
}

func (f *fakeSchedulerAdapter) IsScheduler() bool {
	return true
}

func (f *fakeSchedulerAdapter) Schedule(fedObj pkgruntime.Object, currentObjs map[string]pkgruntime.Object) (map[string]pkgruntime.Object, error) {
	scheduleObjs := make(map[string]pkgruntime.Object)
	for cluster, obj := range currentObjs {
		differingObj := f.Copy(obj)
		federatedtypes.SetAnnotation(f, differingObj, "A", "B")
		scheduleObjs[cluster] = differingObj
	}
	return scheduleObjs, nil
}

func (f *fakeSchedulerAdapter) FedUpdateStatus(fedObj pkgruntime.Object, currentObjs map[string]pkgruntime.Object) error {
	return nil
}

/*
Copyright 2022 The Kubernetes Authors.

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

package daemonset

import (
	"context"
	"fmt"
	"sync"

	"k8s.io/apiserver/pkg/admission"
	api "k8s.io/kubernetes/pkg/apis/core"
)

var _ admission.ValidationInterface = &fakePodFailAdmission{}

type fakePodFailAdmission struct {
	lock             sync.Mutex
	limitedPodNumber int
	succeedPodsCount int
}

func (f *fakePodFailAdmission) Handles(operation admission.Operation) bool {
	return operation == admission.Create
}

func (f *fakePodFailAdmission) Validate(ctx context.Context, attr admission.Attributes, o admission.ObjectInterfaces) (err error) {
	if attr.GetKind().GroupKind() != api.Kind("Pod") {
		return nil
	}

	f.lock.Lock()
	defer f.lock.Unlock()

	if f.succeedPodsCount >= f.limitedPodNumber {
		return fmt.Errorf("fakePodFailAdmission error")
	}
	f.succeedPodsCount++
	return nil
}

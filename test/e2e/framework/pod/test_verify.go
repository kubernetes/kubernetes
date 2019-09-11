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

package pod

import (
	"fmt"

	"github.com/onsi/ginkgo"

	v1 "k8s.io/api/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
)

// TestPodSuccessOrFail tests whether the pod's exit code is zero.
func TestPodSuccessOrFail(c clientset.Interface, ns string, pod *v1.Pod) error {
	ginkgo.By("Pod should terminate with exitcode 0 (success)")
	if err := WaitForPodSuccessInNamespace(c, pod.Name, ns); err != nil {
		return fmt.Errorf("pod %q failed to reach Success: %v", pod.Name, err)
	}
	e2elog.Logf("Pod %v succeeded ", pod.Name)
	return nil
}

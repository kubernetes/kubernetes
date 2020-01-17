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

package upgrades

import (
	"fmt"

	"github.com/onsi/ginkgo"

	"k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/kubelet/sysctl"

	"k8s.io/kubernetes/test/e2e/framework"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

// SysctlUpgradeTest tests that a pod with sysctls runs before and after an upgrade. During
// a master upgrade, the exact pod is expected to stay running. A pod with unsafe sysctls is
// expected to keep failing before and after the upgrade.
type SysctlUpgradeTest struct {
	validPod   *v1.Pod
	invalidPod *v1.Pod
}

// Setup creates two pods: one with safe sysctls, one with unsafe sysctls. It checks that the former
// launched and the later is rejected.
func (t *SysctlUpgradeTest) Setup(f *framework.Framework) {
	t.validPod = t.verifySafeSysctlWork(f)
	t.invalidPod = t.verifyUnsafeSysctlsAreRejected(f)
}

// Test waits for the upgrade to complete, and then verifies that a
// pod can still consume the ConfigMap.
func (t *SysctlUpgradeTest) Test(f *framework.Framework, done <-chan struct{}, upgrade UpgradeType) {
	<-done
	switch upgrade {
	case MasterUpgrade, ClusterUpgrade:
		ginkgo.By("Checking the safe sysctl pod keeps running on master upgrade")
		pod, err := f.ClientSet.CoreV1().Pods(t.validPod.Namespace).Get(t.validPod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		framework.ExpectEqual(pod.Status.Phase, v1.PodRunning)
	}

	ginkgo.By("Checking the old unsafe sysctl pod was not suddenly started during an upgrade")
	pod, err := f.ClientSet.CoreV1().Pods(t.invalidPod.Namespace).Get(t.invalidPod.Name, metav1.GetOptions{})
	if err != nil && !apierrors.IsNotFound(err) {
		framework.ExpectNoError(err)
	}
	if err == nil {
		framework.ExpectNotEqual(pod.Status.Phase, v1.PodRunning)
	}

	t.verifySafeSysctlWork(f)
	t.verifyUnsafeSysctlsAreRejected(f)
}

// Teardown cleans up any remaining resources.
func (t *SysctlUpgradeTest) Teardown(f *framework.Framework) {
	// rely on the namespace deletion to clean up everything
}

func (t *SysctlUpgradeTest) verifySafeSysctlWork(f *framework.Framework) *v1.Pod {
	ginkgo.By("Creating a pod with safe sysctls")
	safeSysctl := "net.ipv4.ip_local_port_range"
	safeSysctlValue := "1024 1042"
	sysctlTestPod("valid-sysctls", map[string]string{safeSysctl: safeSysctlValue})
	validPod := f.PodClient().Create(t.validPod)

	ginkgo.By("Making sure the valid pod launches")
	ev, err := f.PodClient().WaitForErrorEventOrSuccess(t.validPod)
	framework.ExpectNoError(err)
	if ev != nil && ev.Reason == sysctl.UnsupportedReason {
		e2eskipper.Skipf("No sysctl support in Docker <1.12")
	}
	f.TestContainerOutput("pod with safe sysctl launched", t.validPod, 0, []string{fmt.Sprintf("%s = %s", safeSysctl, safeSysctlValue)})

	return validPod
}

func (t *SysctlUpgradeTest) verifyUnsafeSysctlsAreRejected(f *framework.Framework) *v1.Pod {
	ginkgo.By("Creating a pod with unsafe sysctls")
	invalidPod := sysctlTestPod("valid-sysctls-"+string(uuid.NewUUID()), map[string]string{
		"fs.mount-max": "1000000",
	})
	invalidPod = f.PodClient().Create(invalidPod)

	ginkgo.By("Making sure the invalid pod failed")
	ev, err := f.PodClient().WaitForErrorEventOrSuccess(invalidPod)
	framework.ExpectNoError(err)
	if ev != nil && ev.Reason == sysctl.UnsupportedReason {
		e2eskipper.Skipf("No sysctl support in Docker <1.12")
	}
	framework.ExpectEqual(ev.Reason, sysctl.ForbiddenReason)

	return invalidPod
}

func sysctlTestPod(name string, sysctls map[string]string) *v1.Pod {
	sysctlList := []v1.Sysctl{}
	keys := []string{}
	for k, v := range sysctls {
		sysctlList = append(sysctlList, v1.Sysctl{Name: k, Value: v})
		keys = append(keys, k)
	}
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    "test-container",
					Image:   imageutils.GetE2EImage(imageutils.BusyBox),
					Command: append([]string{"/bin/sysctl"}, keys...),
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
			SecurityContext: &v1.PodSecurityContext{
				Sysctls: sysctlList,
			},
		},
	}
}

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

/* This test check that setHostnameAsFQDN PodSpec field works as
 * expected.
 */

package e2enode

import (
	"crypto/rand"
	"fmt"
	"math/big"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/kubernetes/pkg/kubelet/events"

	"k8s.io/kubernetes/test/e2e/framework"
	e2eevents "k8s.io/kubernetes/test/e2e/framework/events"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
)

func generatePodName(base string) string {
	id, err := rand.Int(rand.Reader, big.NewInt(214748))
	if err != nil {
		return base
	}
	return fmt.Sprintf("%s-%d", base, id)
}

func testPod(podnamebase string) *v1.Pod {
	podName := generatePodName(podnamebase)
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        podName,
			Labels:      map[string]string{"name": podName},
			Annotations: map[string]string{},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "test-container",
					Image: imageutils.GetE2EImage(imageutils.BusyBox),
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
		},
	}

	return pod
}

var _ = SIGDescribe("Hostname of Pod [Feature:SetHostnameAsFQDN][NodeFeature:SetHostnameAsFQDN]", func() {
	f := framework.NewDefaultFramework("hostfqdn")
	/*
	   Release: v1.19
	   Testname: Create Pod without fully qualified domain name (FQDN)
	   Description: A Pod that does not define the subdomain field in it spec, does not have FQDN.
	*/
	ginkgo.It("a pod without subdomain field does not have FQDN", func() {
		pod := testPod("hostfqdn")
		pod.Spec.Containers[0].Command = []string{"sh", "-c", "echo $(hostname)';'$(hostname -f)';'"}
		output := []string{fmt.Sprintf("%s;%s;", pod.ObjectMeta.Name, pod.ObjectMeta.Name)}
		// Create Pod
		f.TestContainerOutput("shortname only", pod, 0, output)
	})

	/*
	   Release: v1.19
	   Testname: Create Pod without FQDN, setHostnameAsFQDN field set to true
	   Description: A Pod that does not define the subdomain field in it spec, does not have FQDN.
	                Hence, SetHostnameAsFQDN feature has no effect.
	*/
	ginkgo.It("a pod without FQDN is not affected by SetHostnameAsFQDN field", func() {
		pod := testPod("hostfqdn")
		// Setting setHostnameAsFQDN field to true should have no effect.
		setHostnameAsFQDN := true
		pod.Spec.SetHostnameAsFQDN = &setHostnameAsFQDN
		pod.Spec.Containers[0].Command = []string{"sh", "-c", "echo $(hostname)';'$(hostname -f)';'"}
		output := []string{fmt.Sprintf("%s;%s;", pod.ObjectMeta.Name, pod.ObjectMeta.Name)}
		// Create Pod
		f.TestContainerOutput("shortname only", pod, 0, output)
	})

	/*
	   Release: v1.19
	   Testname: Create Pod with FQDN, setHostnameAsFQDN field not defined.
	   Description: A Pod that defines the subdomain field in it spec has FQDN.
	                hostname command returns shortname (pod name in this case), and hostname -f returns FQDN.
	*/
	ginkgo.It("a pod with subdomain field has FQDN, hostname is shortname", func() {
		pod := testPod("hostfqdn")
		pod.Spec.Containers[0].Command = []string{"sh", "-c", "echo $(hostname)';'$(hostname -f)';'"}
		subdomain := "t"
		// Set PodSpec subdomain field to generate FQDN for pod
		pod.Spec.Subdomain = subdomain
		// Expected Pod FQDN
		hostFQDN := fmt.Sprintf("%s.%s.%s.svc.%s", pod.ObjectMeta.Name, subdomain, f.Namespace.Name, framework.TestContext.ClusterDNSDomain)
		output := []string{fmt.Sprintf("%s;%s;", pod.ObjectMeta.Name, hostFQDN)}
		// Create Pod
		f.TestContainerOutput("shortname and fqdn", pod, 0, output)
	})

	/*
	   Release: v1.19
	   Testname: Create Pod with FQDN, setHostnameAsFQDN field set to true.
	   Description: A Pod that defines the subdomain field in it spec has FQDN. When setHostnameAsFQDN: true, the
	                hostname is set to be the FQDN. In this case, both commands hostname and hostname -f return the FQDN of the Pod.
	*/
	ginkgo.It("a pod with subdomain field has FQDN, when setHostnameAsFQDN is set to true, the FQDN is set as hostname", func() {
		pod := testPod("hostfqdn")
		pod.Spec.Containers[0].Command = []string{"sh", "-c", "echo $(hostname)';'$(hostname -f)';'"}
		subdomain := "t"
		// Set PodSpec subdomain field to generate FQDN for pod
		pod.Spec.Subdomain = subdomain
		// Set PodSpec setHostnameAsFQDN to set FQDN as hostname
		setHostnameAsFQDN := true
		pod.Spec.SetHostnameAsFQDN = &setHostnameAsFQDN
		// Expected Pod FQDN
		hostFQDN := fmt.Sprintf("%s.%s.%s.svc.%s", pod.ObjectMeta.Name, subdomain, f.Namespace.Name, framework.TestContext.ClusterDNSDomain)
		// Fail if FQDN is longer than 64 characters, otherwise the Pod will remain pending until test timeout.
		// In Linux, 64 characters is the limit of the hostname kernel field, which this test sets to the pod FQDN.
		framework.ExpectEqual(len(hostFQDN) < 65, true, fmt.Sprintf("The FQDN of the Pod cannot be longer than 64 characters, requested %s which is %d characters long.", hostFQDN, len(hostFQDN)))
		output := []string{fmt.Sprintf("%s;%s;", hostFQDN, hostFQDN)}
		// Create Pod
		f.TestContainerOutput("fqdn and fqdn", pod, 0, output)
	})

	/*
	   Release: v1.20
	   Testname: Fail to Create Pod with longer than 64 bytes FQDN when setHostnameAsFQDN field set to true.
	   Description: A Pod that defines the subdomain field in it spec has FQDN.
	                 When setHostnameAsFQDN: true, the hostname is set to be
	                 the FQDN. Since kernel limit is 64 bytes for hostname field,
	                 if pod FQDN is longer than 64 bytes it will generate events
	                 regarding FailedCreatePodSandBox.
	*/

	ginkgo.It("a pod configured to set FQDN as hostname will remain in Pending "+
		"state generating FailedCreatePodSandBox events when the FQDN is "+
		"longer than 64 bytes", func() {
		// 55 characters for name plus -<int>.t.svc.cluster.local is way more than 64 bytes
		pod := testPod("hostfqdnveryveryveryverylongforfqdntobemorethan64bytes")
		pod.Spec.Containers[0].Command = []string{"sh", "-c", "echo $(hostname)';'$(hostname -f)';'"}
		subdomain := "t"
		// Set PodSpec subdomain field to generate FQDN for pod
		pod.Spec.Subdomain = subdomain
		// Set PodSpec setHostnameAsFQDN to set FQDN as hostname
		setHostnameAsFQDN := true
		pod.Spec.SetHostnameAsFQDN = &setHostnameAsFQDN
		// Create Pod
		launchedPod := f.PodClient().Create(pod)
		// Ensure we delete pod
		defer f.PodClient().DeleteSync(launchedPod.Name, metav1.DeleteOptions{}, framework.DefaultPodDeletionTimeout)

		// Pod should remain in the pending state generating events with reason FailedCreatePodSandBox
		// Expected Message Error Event
		expectedMessage := "Failed to create pod sandbox: failed " +
			"to construct FQDN from pod hostname and cluster domain, FQDN "
		framework.Logf("Waiting for Pod to generate FailedCreatePodSandBox event.")
		// Wait for event with reason FailedCreatePodSandBox
		expectSandboxFailureEvent(f, launchedPod, expectedMessage)
		// Check Pod is in Pending Phase
		err := checkPodIsPending(f, launchedPod.ObjectMeta.Name, launchedPod.ObjectMeta.Namespace)
		framework.ExpectNoError(err)

	})
})

// expectSandboxFailureEvent polls for an event with reason "FailedCreatePodSandBox" containing the
// expected message string.
func expectSandboxFailureEvent(f *framework.Framework, pod *v1.Pod, msg string) {
	eventSelector := fields.Set{
		"involvedObject.kind":      "Pod",
		"involvedObject.name":      pod.Name,
		"involvedObject.namespace": f.Namespace.Name,
		"reason":                   events.FailedCreatePodSandBox,
	}.AsSelector().String()
	framework.ExpectNoError(e2eevents.WaitTimeoutForEvent(
		f.ClientSet, f.Namespace.Name, eventSelector, msg, framework.PodEventTimeout))
}

func checkPodIsPending(f *framework.Framework, podName, namespace string) error {
	c := f.ClientSet
	// we call this functoin after we saw event failing to create Pod, hence
	// pod has already been created and it should be in Pending status. Giving
	// 30 seconds to fetch the pod to avoid failing for transient issues getting
	// pods.
	fetchPodTimeout := 30 * time.Second
	return e2epod.WaitForPodCondition(c, namespace, podName, "Failed to Create Pod", fetchPodTimeout, func(pod *v1.Pod) (bool, error) {
		// We are looking for the pod to be scheduled and in Pending state
		if pod.Status.Phase == v1.PodPending {
			for _, cond := range pod.Status.Conditions {
				if cond.Type == v1.PodScheduled && cond.Status == v1.ConditionTrue {
					return true, nil
				}
			}
		}
		// If pod gets to this status, either FQDN is shorter than 64bytes
		// or setHostnameAsFQDN feature is not enable/in use.
		if pod.Status.Phase == v1.PodRunning || pod.Status.Phase == v1.PodSucceeded || pod.Status.Phase == v1.PodFailed {
			return true, fmt.Errorf("Expected pod %q in namespace %q to be in phase Pending, but got phase: %v", podName, namespace, pod.Status.Phase)
		}
		return false, nil
	})
}

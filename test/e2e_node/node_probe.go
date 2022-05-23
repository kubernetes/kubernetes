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

package e2enode

import (
	"context"
	"strings"
	"time"

	"github.com/onsi/ginkgo"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("Node Probe [NodeConformance]", func() {
	ginkgo.Context("Kubelet", func() {
		f := framework.NewDefaultFramework("node-probe")
		f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

		ginkgo.It("should run pod with readiness status set to false on termination", func() {
			podName := "probe-test-" + string(uuid.NewUUID())
			podClient := f.PodClient()
			terminationGracePeriod := int64(30)
			script := `
			_term() {
				rm -f /tmp/ready
				sleep 30
				exit 0
			}
			trap _term SIGTERM

			touch /tmp/ready

			while true; do
			  echo \"hello\"
			  sleep 10
			done
			`

			// Create Pod
			podClient.Create(&v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: podName,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Image:   imageutils.GetE2EImage(imageutils.Agnhost),
							Name:    podName,
							Command: []string{"/bin/bash"},
							Args:    []string{"-c", script},
							ReadinessProbe: &v1.Probe{
								ProbeHandler: v1.ProbeHandler{
									Exec: &v1.ExecAction{
										Command: []string{"cat", "/tmp/ready"},
									},
								},
								FailureThreshold:    1,
								InitialDelaySeconds: 5,
								PeriodSeconds:       2,
							},
						},
					},
					TerminationGracePeriodSeconds: &terminationGracePeriod,
				},
			})

			// verify pods are running and ready
			err := e2epod.WaitForPodsRunningReady(f.ClientSet, f.Namespace.Name, 1, 0, 30*time.Second, map[string]string{})
			framework.ExpectNoError(err)

			// Shutdown pod. Readiness should change to false
			podClient.Delete(context.Background(), podName, metav1.DeleteOptions{})
			err = wait.PollImmediate(2*time.Second, 30*time.Second, func() (bool, error) {
				pod, err := podClient.Get(context.Background(), podName, metav1.GetOptions{})
				if err != nil {
					return false, err
				}
				// verify the pod ready status has reported not ready
				return podutil.IsPodReady(pod) == false, nil
			})
			framework.ExpectNoError(err)
		})

		ginkgo.It("should run pod with stopped liveness probes", func() {
			podName := "probe-test-" + string(uuid.NewUUID())
			podClient := f.PodClient()
			terminationGracePeriod := int64(30)
			script := `
_term() { 
	rm -f /tmp/ready
	rm -f /tmp/liveness
	sleep 20
	exit 0
}
trap _term SIGTERM

touch /tmp/ready
touch /tmp/liveness

while true; do 
  echo \"hello\"
  sleep 10
done
`

			// Create Pod
			podClient.Create(&v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: podName,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Image:   imageutils.GetE2EImage(imageutils.Agnhost),
							Name:    podName,
							Command: []string{"/bin/bash"},
							Args:    []string{"-c", script},
							ReadinessProbe: &v1.Probe{
								ProbeHandler: v1.ProbeHandler{
									Exec: &v1.ExecAction{
										Command: []string{"cat", "/tmp/ready"},
									},
								},
								FailureThreshold:    1,
								InitialDelaySeconds: 5,
								PeriodSeconds:       2,
							},
							LivenessProbe: &v1.Probe{
								ProbeHandler: v1.ProbeHandler{
									Exec: &v1.ExecAction{
										Command: []string{"cat", "/tmp/liveness"},
									},
								},
								FailureThreshold:    1,
								InitialDelaySeconds: 6,
								PeriodSeconds:       1,
							},
						},
					},
					TerminationGracePeriodSeconds: &terminationGracePeriod,
				},
			})

			// verify pods are running and ready
			err := e2epod.WaitForPodsRunningReady(f.ClientSet, f.Namespace.Name, 1, 0, 30*time.Second, map[string]string{})
			framework.ExpectNoError(err)

			// Shutdown pod. Readiness should change to false
			podClient.Delete(context.Background(), podName, metav1.DeleteOptions{})

			// Wait for pod to go unready
			err = wait.PollImmediate(2*time.Second, 30*time.Second, func() (bool, error) {
				pod, err := podClient.Get(context.Background(), podName, metav1.GetOptions{})
				if err != nil {
					return false, err
				}
				// verify the pod ready status has reported not ready
				return podutil.IsPodReady(pod) == false, nil
			})
			framework.ExpectNoError(err)

			// Verify there are zero liveness failures since they are turned off
			// during pod termination
			items, _ := f.ClientSet.EventsV1().Events(f.Namespace.Name).List(context.Background(), metav1.ListOptions{})
			for _, event := range items.Items {
				if strings.Contains(event.Note, "failed liveness probe") {
					framework.Fail("should not see liveness probe failures")
				}
			}
		})
	})
})

/*
Copyright 2014 The Kubernetes Authors.

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

package node

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/cluster/ports"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ekubelet "k8s.io/kubernetes/test/e2e/framework/kubelet"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
)

// State partially cloned from webserver.go
type State struct {
	Received map[string]int
}

func testPreStop(c clientset.Interface, ns string) {
	// This is the server that will receive the preStop notification
	podDescr := e2epod.NewAgnhostPod(ns, "server", nil, nil, []v1.ContainerPort{{ContainerPort: 8080}}, "nettest")
	ginkgo.By(fmt.Sprintf("Creating server pod %s in namespace %s", podDescr.Name, ns))
	podDescr, err := c.CoreV1().Pods(ns).Create(context.TODO(), podDescr, metav1.CreateOptions{})
	framework.ExpectNoError(err, fmt.Sprintf("creating pod %s", podDescr.Name))

	// At the end of the test, clean up by removing the pod.
	defer func() {
		ginkgo.By("Deleting the server pod")
		c.CoreV1().Pods(ns).Delete(context.TODO(), podDescr.Name, metav1.DeleteOptions{})
	}()

	ginkgo.By("Waiting for pods to come up.")
	err = e2epod.WaitForPodRunningInNamespace(c, podDescr)
	framework.ExpectNoError(err, "waiting for server pod to start")

	val := "{\"Source\": \"prestop\"}"

	podOut, err := c.CoreV1().Pods(ns).Get(context.TODO(), podDescr.Name, metav1.GetOptions{})
	framework.ExpectNoError(err, "getting pod info")

	podURL := net.JoinHostPort(podOut.Status.PodIP, "8080")

	preStopDescr := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "tester",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    "tester",
					Image:   imageutils.GetE2EImage(imageutils.BusyBox),
					Command: []string{"sleep", "600"},
					Lifecycle: &v1.Lifecycle{
						PreStop: &v1.Handler{
							Exec: &v1.ExecAction{
								Command: []string{
									"wget", "-O-", "--post-data=" + val, fmt.Sprintf("http://%s/write", podURL),
								},
							},
						},
					},
				},
			},
		},
	}

	ginkgo.By(fmt.Sprintf("Creating tester pod %s in namespace %s", preStopDescr.Name, ns))
	preStopDescr, err = c.CoreV1().Pods(ns).Create(context.TODO(), preStopDescr, metav1.CreateOptions{})
	framework.ExpectNoError(err, fmt.Sprintf("creating pod %s", preStopDescr.Name))
	deletePreStop := true

	// At the end of the test, clean up by removing the pod.
	defer func() {
		if deletePreStop {
			ginkgo.By("Deleting the tester pod")
			c.CoreV1().Pods(ns).Delete(context.TODO(), preStopDescr.Name, metav1.DeleteOptions{})
		}
	}()

	err = e2epod.WaitForPodRunningInNamespace(c, preStopDescr)
	framework.ExpectNoError(err, "waiting for tester pod to start")

	// Delete the pod with the preStop handler.
	ginkgo.By("Deleting pre-stop pod")
	if err := c.CoreV1().Pods(ns).Delete(context.TODO(), preStopDescr.Name, metav1.DeleteOptions{}); err == nil {
		deletePreStop = false
	}
	framework.ExpectNoError(err, fmt.Sprintf("deleting pod: %s", preStopDescr.Name))

	// Validate that the server received the web poke.
	err = wait.Poll(time.Second*5, time.Second*60, func() (bool, error) {

		ctx, cancel := context.WithTimeout(context.Background(), framework.SingleCallTimeout)
		defer cancel()

		var body []byte
		body, err = c.CoreV1().RESTClient().Get().
			Namespace(ns).
			Resource("pods").
			SubResource("proxy").
			Name(podDescr.Name).
			Suffix("read").
			DoRaw(ctx)

		if err != nil {
			if ctx.Err() != nil {
				framework.Failf("Error validating prestop: %v", err)
				return true, err
			}
			ginkgo.By(fmt.Sprintf("Error validating prestop: %v", err))
		} else {
			framework.Logf("Saw: %s", string(body))
			state := State{}
			err := json.Unmarshal(body, &state)
			if err != nil {
				framework.Logf("Error parsing: %v", err)
				return false, nil
			}
			if state.Received["prestop"] != 0 {
				return true, nil
			}
		}
		return false, nil
	})
	framework.ExpectNoError(err, "validating pre-stop.")
}

var _ = SIGDescribe("PreStop", func() {
	f := framework.NewDefaultFramework("prestop")
	var podClient *framework.PodClient
	ginkgo.BeforeEach(func() {
		podClient = f.PodClient()
	})

	/*
		Release: v1.9
		Testname: Pods, prestop hook
		Description: Create a server pod with a rest endpoint '/write' that changes state.Received field. Create a Pod with a pre-stop handle that posts to the /write endpoint on the server Pod. Verify that the Pod with pre-stop hook is running. Delete the Pod with the pre-stop hook. Before the Pod is deleted, pre-stop handler MUST be called when configured. Verify that the Pod is deleted and a call to prestop hook is verified by checking the status received on the server Pod.
	*/
	framework.ConformanceIt("should call prestop when killing a pod ", func() {
		testPreStop(f.ClientSet, f.Namespace.Name)
	})

	ginkgo.It("graceful pod terminated should wait until preStop hook completes the process", func() {
		gracefulTerminationPeriodSeconds := int64(30)
		ginkgo.By("creating the pod")
		name := "pod-prestop-hook-" + string(uuid.NewUUID())
		pod := getPodWithpreStopLifeCycle(name)

		ginkgo.By("submitting the pod to kubernetes")
		podClient.Create(pod)

		ginkgo.By("waiting for pod running")
		framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(f.ClientSet, pod.Name, f.Namespace.Name))

		var err error
		pod, err = podClient.Get(context.TODO(), pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to GET scheduled pod")

		ginkgo.By("deleting the pod gracefully")
		err = podClient.Delete(context.TODO(), pod.Name, *metav1.NewDeleteOptions(gracefulTerminationPeriodSeconds))
		framework.ExpectNoError(err, "failed to delete pod")

		// wait for less than the gracePeriod termination ensuring the
		// preStop hook is still executing.
		time.Sleep(15 * time.Second)

		ginkgo.By("verifying the pod is running while in the graceful period termination")
		result := &v1.PodList{}
		err = wait.Poll(time.Second*5, time.Second*60, func() (bool, error) {
			client, err := e2ekubelet.ProxyRequest(f.ClientSet, pod.Spec.NodeName, "pods", ports.KubeletPort)
			framework.ExpectNoError(err, "failed to get the pods of the node")
			err = client.Into(result)
			framework.ExpectNoError(err, "failed to parse the pods of the node")

			for _, kubeletPod := range result.Items {
				if pod.Name != kubeletPod.Name {
					continue
				} else if kubeletPod.Status.Phase == v1.PodRunning {
					framework.Logf("pod is running")
					return true, err
				}
			}
			return false, err
		})

		framework.ExpectNoError(err, "validate-pod-is-running")
	})
})

func getPodWithpreStopLifeCycle(name string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "nginx",
					Image: imageutils.GetE2EImage(imageutils.Nginx),
					Lifecycle: &v1.Lifecycle{
						PreStop: &v1.Handler{
							Exec: &v1.ExecAction{
								Command: []string{"sh", "-c", "while true; do echo preStop; sleep 1; done"},
							},
						},
					},
				},
			},
		},
	}
}

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

package e2e

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/api/v1"
	rbacv1alpha1 "k8s.io/kubernetes/pkg/apis/rbac/v1alpha1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/serviceaccount"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

// partially cloned from webserver.go
type State struct {
	Received map[string]int
}

func testPreStop(c clientset.Interface, ns string) {
	// This is the server that will receive the preStop notification
	podDescr := &v1.Pod{
		ObjectMeta: v1.ObjectMeta{
			Name: "server",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "server",
					Image: "gcr.io/google_containers/nettest:1.7",
					Ports: []v1.ContainerPort{{ContainerPort: 8080}},
				},
			},
		},
	}
	By(fmt.Sprintf("Creating server pod %s in namespace %s", podDescr.Name, ns))
	podDescr, err := c.Core().Pods(ns).Create(podDescr)
	framework.ExpectNoError(err, fmt.Sprintf("creating pod %s", podDescr.Name))

	// At the end of the test, clean up by removing the pod.
	defer func() {
		By("Deleting the server pod")
		c.Core().Pods(ns).Delete(podDescr.Name, nil)
	}()

	By("Waiting for pods to come up.")
	err = framework.WaitForPodRunningInNamespace(c, podDescr)
	framework.ExpectNoError(err, "waiting for server pod to start")

	val := "{\"Source\": \"prestop\"}"

	podOut, err := c.Core().Pods(ns).Get(podDescr.Name, metav1.GetOptions{})
	framework.ExpectNoError(err, "getting pod info")

	preStopDescr := &v1.Pod{
		ObjectMeta: v1.ObjectMeta{
			Name: "tester",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    "tester",
					Image:   "gcr.io/google_containers/busybox:1.24",
					Command: []string{"sleep", "600"},
					Lifecycle: &v1.Lifecycle{
						PreStop: &v1.Handler{
							Exec: &v1.ExecAction{
								Command: []string{
									"wget", "-O-", "--post-data=" + val, fmt.Sprintf("http://%s:8080/write", podOut.Status.PodIP),
								},
							},
						},
					},
				},
			},
		},
	}

	By(fmt.Sprintf("Creating tester pod %s in namespace %s", preStopDescr.Name, ns))
	preStopDescr, err = c.Core().Pods(ns).Create(preStopDescr)
	framework.ExpectNoError(err, fmt.Sprintf("creating pod %s", preStopDescr.Name))
	deletePreStop := true

	// At the end of the test, clean up by removing the pod.
	defer func() {
		if deletePreStop {
			By("Deleting the tester pod")
			c.Core().Pods(ns).Delete(preStopDescr.Name, nil)
		}
	}()

	err = framework.WaitForPodRunningInNamespace(c, preStopDescr)
	framework.ExpectNoError(err, "waiting for tester pod to start")

	// Delete the pod with the preStop handler.
	By("Deleting pre-stop pod")
	if err := c.Core().Pods(ns).Delete(preStopDescr.Name, nil); err == nil {
		deletePreStop = false
	}
	framework.ExpectNoError(err, fmt.Sprintf("deleting pod: %s", preStopDescr.Name))

	// Validate that the server received the web poke.
	err = wait.Poll(time.Second*5, time.Second*60, func() (bool, error) {
		subResourceProxyAvailable, err := framework.ServerVersionGTE(framework.SubResourcePodProxyVersion, c.Discovery())
		if err != nil {
			return false, err
		}

		ctx, cancel := context.WithTimeout(context.Background(), framework.SingleCallTimeout)
		defer cancel()

		var body []byte
		if subResourceProxyAvailable {
			body, err = c.Core().RESTClient().Get().
				Context(ctx).
				Namespace(ns).
				Resource("pods").
				SubResource("proxy").
				Name(podDescr.Name).
				Suffix("read").
				DoRaw()
		} else {
			body, err = c.Core().RESTClient().Get().
				Context(ctx).
				Prefix("proxy").
				Namespace(ns).
				Resource("pods").
				Name(podDescr.Name).
				Suffix("read").
				DoRaw()
		}
		if err != nil {
			if ctx.Err() != nil {
				framework.Failf("Error validating prestop: %v", err)
				return true, err
			}
			By(fmt.Sprintf("Error validating prestop: %v", err))
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

var _ = framework.KubeDescribe("PreStop", func() {
	f := framework.NewDefaultFramework("prestop")

	BeforeEach(func() {
		// this test wants extra permissions.  Since the namespace names are unique, we can leave this
		// lying around so we don't have to race any caches
		framework.BindClusterRole(f.ClientSet.Rbac(), "cluster-admin", f.Namespace.Name,
			rbacv1alpha1.Subject{Kind: rbacv1alpha1.ServiceAccountKind, Namespace: f.Namespace.Name, Name: "default"})

		err := framework.WaitForAuthorizationUpdate(f.ClientSet.Authorization(),
			serviceaccount.MakeUsername(f.Namespace.Name, "default"),
			"", "create", schema.GroupResource{Resource: "pods"}, true)
		framework.ExpectNoError(err)
	})

	It("should call prestop when killing a pod [Conformance]", func() {
		testPreStop(f.ClientSet, f.Namespace.Name)
	})
})

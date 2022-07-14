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
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/test/e2e/framework"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

const (
	// timeout for proxy requests.
	proxyTimeout = 2 * time.Minute
)

// proxyPostRequest performs a post on a node proxy endpoint given the nodename and rest client.
func proxyPostRequest(c clientset.Interface, node, endpoint string, port int) (restclient.Result, error) {
	// proxy tends to hang in some cases when Node is not ready. Add an artificial timeout for this call. #22165
	var result restclient.Result
	finished := make(chan struct{}, 1)
	go func() {
		result = c.CoreV1().RESTClient().Post().
			Resource("nodes").
			SubResource("proxy").
			Name(fmt.Sprintf("%v:%v", node, port)).
			Suffix(endpoint).
			Do(context.TODO())

		finished <- struct{}{}
	}()
	select {
	case <-finished:
		return result, nil
	case <-time.After(proxyTimeout):
		return restclient.Result{}, nil
	}
}

var _ = SIGDescribe("Checkpoint Container [NodeFeature:CheckpointContainer]", func() {
	f := framework.NewDefaultFramework("checkpoint-container-test")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelBaseline
	ginkgo.It("will checkpoint a container out of a pod", func() {
		ginkgo.By("creating a target pod")
		podClient := f.PodClient()
		pod := podClient.CreateSync(&v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "checkpoint-container-pod"},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    "test-container-1",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"/bin/sleep"},
						Args:    []string{"10000"},
					},
				},
			},
		})

		p, err := podClient.Get(
			context.TODO(),
			pod.Name,
			metav1.GetOptions{},
		)

		framework.ExpectNoError(err)
		isReady, err := testutils.PodRunningReady(p)
		framework.ExpectNoError(err)
		framework.ExpectEqual(
			isReady,
			true,
			"pod should be ready",
		)

		framework.Logf(
			"About to checkpoint container %q on %q",
			pod.Spec.Containers[0].Name,
			pod.Spec.NodeName,
		)
		result, err := proxyPostRequest(
			f.ClientSet,
			pod.Spec.NodeName,
			fmt.Sprintf(
				"checkpoint/%s/%s/%s",
				f.Namespace.Name,
				pod.Name,
				pod.Spec.Containers[0].Name,
			),
			framework.KubeletPort,
		)

		framework.ExpectNoError(err)

		err = result.Error()
		if err != nil {
			statusError, ok := err.(*apierrors.StatusError)
			if !ok {
				framework.Failf("got error %#v, expected StatusError", err)
			}
			// If we are testing against a kubelet with ContainerCheckpoint == false
			// we should get a 404. So a 404 is (also) a good sign.
			if (int(statusError.ErrStatus.Code)) == http.StatusNotFound {
				ginkgo.Skip("Feature 'ContainerCheckpoint' is not enabled and not available")
				return
			}

			// If the container engine has not implemented the Checkpoint CRI API
			// we will get 500 and a message with
			// '(rpc error: code = Unimplemented desc = unknown method CheckpointContainer'
			if (int(statusError.ErrStatus.Code)) == http.StatusInternalServerError {
				if strings.Contains(
					statusError.ErrStatus.Message,
					"(rpc error: code = Unimplemented desc = unknown method CheckpointContainer",
				) {
					ginkgo.Skip("Container engine does not implement 'CheckpointContainer'")
					return
				}
			}
			framework.Failf("Unexpected status code (%d) during 'CheckpointContainer'", statusError.ErrStatus.Code)
		}

		framework.ExpectNoError(err)
		// TODO: once a container engine implements the Checkpoint CRI API this needs
		// to be extended to handle it.
		//
		// Checkpointing actually worked. Verify that the checkpoint exists and that
		// it is a checkpoint.
	})
})

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
	"archive/tar"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2emetrics "k8s.io/kubernetes/test/e2e/framework/metrics"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/gomega"
)

const (
	// timeout for proxy requests.
	proxyTimeout = 2 * time.Minute
)

type checkpointResult struct {
	Items []string `json:"items"`
}

// proxyPostRequest performs a post on a node proxy endpoint given the nodename and rest client.
func proxyPostRequest(ctx context.Context, c clientset.Interface, node, endpoint string, port int) (restclient.Result, error) {
	// proxy tends to hang in some cases when Node is not ready. Add an artificial timeout for this call. #22165
	var result restclient.Result
	finished := make(chan struct{}, 1)
	go func() {
		result = c.CoreV1().RESTClient().Post().
			Resource("nodes").
			SubResource("proxy").
			Name(fmt.Sprintf("%v:%v", node, port)).
			Suffix(endpoint).
			Do(ctx)

		finished <- struct{}{}
	}()
	select {
	case <-finished:
		return result, nil
	case <-ctx.Done():
		return restclient.Result{}, nil
	case <-time.After(proxyTimeout):
		return restclient.Result{}, nil
	}
}

func getCheckpointContainerMetric(ctx context.Context, f *framework.Framework, pod *v1.Pod) (int, error) {
	framework.Logf("Getting 'checkpoint_container' metrics from %q", pod.Spec.NodeName)
	ms, err := e2emetrics.GetKubeletMetrics(
		ctx,
		f.ClientSet,
		pod.Spec.NodeName,
	)
	if err != nil {
		return 0, err
	}

	runtimeOperationsTotal, ok := ms["runtime_operations_total"]
	if !ok {
		// If the metric was not found it was probably not written to, yet.
		return 0, nil
	}

	for _, item := range runtimeOperationsTotal {
		if item.Metric["__name__"] == "kubelet_runtime_operations_total" && item.Metric["operation_type"] == "checkpoint_container" {
			return int(item.Value), nil
		}
	}
	// If the metric was not found it was probably not written to, yet.
	return 0, nil
}

func getCheckpointContainerErrorMetric(ctx context.Context, f *framework.Framework, pod *v1.Pod) (int, error) {
	framework.Logf("Getting 'checkpoint_container' error metrics from %q", pod.Spec.NodeName)
	ms, err := e2emetrics.GetKubeletMetrics(
		ctx,
		f.ClientSet,
		pod.Spec.NodeName,
	)
	if err != nil {
		return 0, err
	}

	runtimeOperationsErrorsTotal, ok := ms["runtime_operations_errors_total"]
	if !ok {
		// If the metric was not found it was probably not written to, yet.
		return 0, nil
	}

	for _, item := range runtimeOperationsErrorsTotal {
		if item.Metric["__name__"] == "kubelet_runtime_operations_errors_total" && item.Metric["operation_type"] == "checkpoint_container" {
			return int(item.Value), nil
		}
	}
	// If the metric was not found it was probably not written to, yet.
	return 0, nil
}

var _ = SIGDescribe("Checkpoint Container", feature.CheckpointContainer, func() {
	f := framework.NewDefaultFramework("checkpoint-container-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline
	ginkgo.It("will checkpoint a container out of a pod", func(ctx context.Context) {
		ginkgo.By("creating a target pod")
		podClient := e2epod.NewPodClient(f)
		pod := podClient.CreateSync(ctx, &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "checkpoint-container-pod",
				Namespace: f.Namespace.Name,
			},
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
			ctx,
			pod.Name,
			metav1.GetOptions{},
		)

		framework.ExpectNoError(err)
		isReady, err := testutils.PodRunningReady(p)
		framework.ExpectNoError(err)
		if !isReady {
			framework.Failf("pod %q should be ready", p.Name)
		}

		// No checkpoint operation should have been logged
		checkpointContainerMetric, err := getCheckpointContainerMetric(ctx, f, pod)
		framework.ExpectNoError(err)
		gomega.Expect(checkpointContainerMetric).To(gomega.Equal(0))
		// No error should have been logged
		checkpointContainerErrorMetric, err := getCheckpointContainerErrorMetric(ctx, f, pod)
		framework.ExpectNoError(err)
		gomega.Expect(checkpointContainerErrorMetric).To(gomega.Equal(0))

		framework.Logf(
			"About to checkpoint container %q on %q",
			pod.Spec.Containers[0].Name,
			pod.Spec.NodeName,
		)
		result, err := proxyPostRequest(
			ctx,
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
			// or
			// '(rpc error: code = Unimplemented desc = method CheckpointContainer not implemented)'
			// if the container engine returns that it explicitly has disabled support for it.
			// or
			// '(rpc error: code = Unknown desc = checkpoint/restore support not available)'
			// if the container engine explicitly disabled the checkpoint/restore support
			// or
			// '(rpc error: code = Unknown desc = CRIU binary not found or too old (<31600). Failed to checkpoint container'
			// if the CRIU binary was not found if it is too old
			if (int(statusError.ErrStatus.Code)) == http.StatusInternalServerError {
				if strings.Contains(
					statusError.ErrStatus.Message,
					"(rpc error: code = Unimplemented desc = unknown method CheckpointContainer",
				) {
					ginkgo.Skip("Container engine does not implement 'CheckpointContainer'")
					return
				}
				if strings.Contains(
					statusError.ErrStatus.Message,
					"(rpc error: code = Unimplemented desc = method CheckpointContainer not implemented)",
				) {
					ginkgo.Skip("Container engine does not implement 'CheckpointContainer'")
					return
				}
				if strings.Contains(
					statusError.ErrStatus.Message,
					"(rpc error: code = Unknown desc = checkpoint/restore support not available)",
				) {
					ginkgo.Skip("Container engine does not implement 'CheckpointContainer'")
					return
				}
				if strings.Contains(
					statusError.ErrStatus.Message,
					"(rpc error: code = Unknown desc = CRIU binary not found or too old (<31600). Failed to checkpoint container",
				) {
					ginkgo.Skip("Container engine reports missing or too old CRIU binary")
					return
				}
			}
			framework.Failf(
				"Unexpected status code (%d) during 'CheckpointContainer': %q",
				statusError.ErrStatus.Code,
				statusError.ErrStatus.Message,
			)
		}

		framework.ExpectNoError(err)

		// Checkpointing actually worked. Verify that the checkpoint exists and that
		// it is a checkpoint.

		raw, err := result.Raw()
		framework.ExpectNoError(err)
		answer := checkpointResult{}
		err = json.Unmarshal(raw, &answer)
		framework.ExpectNoError(err)

		for _, item := range answer.Items {
			// Check that the file exists
			_, err := os.Stat(item)
			framework.ExpectNoError(err)
			// Check the content of the tar file
			// At least looking for the following files
			//  * spec.dump
			//  * config.dump
			//  * checkpoint/inventory.img
			// If these files exist in the checkpoint archive it is
			// probably a complete checkpoint.
			checkForFiles := map[string]bool{
				"spec.dump":                false,
				"config.dump":              false,
				"checkpoint/inventory.img": false,
			}
			fileReader, err := os.Open(item)
			framework.ExpectNoError(err)
			tr := tar.NewReader(fileReader)
			for {
				hdr, err := tr.Next()
				if err == io.EOF {
					// End of archive
					break
				}
				framework.ExpectNoError(err)
				if _, key := checkForFiles[hdr.Name]; key {
					checkForFiles[hdr.Name] = true
				}
			}
			for fileName := range checkForFiles {
				if !checkForFiles[fileName] {
					framework.Failf("File %q not found in checkpoint archive %q", fileName, item)
				}
			}
			// cleanup checkpoint archive
			os.RemoveAll(item)
		}
		// Exactly one checkpoint operation should have happened
		checkpointContainerMetric, err = getCheckpointContainerMetric(ctx, f, pod)
		framework.ExpectNoError(err)
		gomega.Expect(checkpointContainerMetric).To(gomega.Equal(1))
		// No error should have been logged
		checkpointContainerErrorMetric, err = getCheckpointContainerErrorMetric(ctx, f, pod)
		framework.ExpectNoError(err)
		gomega.Expect(checkpointContainerErrorMetric).To(gomega.Equal(0))
	})
})

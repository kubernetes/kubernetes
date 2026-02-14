/*
Copyright 2025 The Kubernetes Authors.

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

	"github.com/onsi/ginkgo/v2"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("Checkpoint Pod", feature.KubeletLocalPodCheckpointRestore, func() {
	f := framework.NewDefaultFramework("checkpoint-pod-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline
	ginkgo.It("will checkpoint a pod", func(ctx context.Context) {
		ginkgo.By("creating a target pod")
		podClient := e2epod.NewPodClient(f)
		pod := podClient.CreateSync(ctx, &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "checkpoint-pod-test",
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

		framework.Logf(
			"About to checkpoint pod %q on %q",
			pod.Name,
			pod.Spec.NodeName,
		)
		result, err := proxyPostRequest(
			ctx,
			f.ClientSet,
			pod.Spec.NodeName,
			fmt.Sprintf(
				"checkpointpod/%s/%s",
				f.Namespace.Name,
				pod.Name,
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
			// If we are testing against a kubelet with KubeletLocalPodCheckpointRestore == false
			// we should get a 404. So a 404 is (also) a good sign.
			if (int(statusError.ErrStatus.Code)) == http.StatusNotFound {
				ginkgo.Skip("Feature 'KubeletLocalPodCheckpointRestore' is not enabled and not available")
				return
			}

			// If the container engine has not implemented the CheckpointPod CRI API
			// we will get 500 and a message with
			// '(rpc error: code = Unimplemented desc = unknown method CheckpointPod'
			// or
			// '(rpc error: code = Unimplemented desc = method CheckpointPod not implemented)'
			// if the container engine returns that it explicitly has disabled support for it.
			// or
			// '(rpc error: code = Unknown desc = checkpoint/restore support not available)'
			// if the container engine explicitly disabled the checkpoint/restore support
			// or
			// '(rpc error: code = Unknown desc = CRIU binary not found or too old'
			// if the CRIU binary was not found or if it is too old
			if (int(statusError.ErrStatus.Code)) == http.StatusInternalServerError {
				if strings.Contains(
					statusError.ErrStatus.Message,
					"(rpc error: code = Unimplemented desc = unknown method CheckpointPod",
				) {
					ginkgo.Skip("Container engine does not implement 'CheckpointPod'")
					return
				}
				if strings.Contains(
					statusError.ErrStatus.Message,
					"(rpc error: code = Unimplemented desc = method CheckpointPod not implemented)",
				) {
					ginkgo.Skip("Container engine does not implement 'CheckpointPod'")
					return
				}
				if strings.Contains(
					statusError.ErrStatus.Message,
					"(rpc error: code = Unknown desc = checkpoint/restore support not available)",
				) {
					ginkgo.Skip("Container engine does not implement 'CheckpointPod'")
					return
				}
				if strings.Contains(
					statusError.ErrStatus.Message,
					"(rpc error: code = Unknown desc = CRIU binary not found or too old",
				) {
					ginkgo.Skip("Container engine reports missing or too old CRIU binary")
					return
				}
			}
			framework.Failf(
				"Unexpected status code (%d) during 'CheckpointPod': %q",
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

			// For pod checkpoints, the tar layout is:
			//   pod.options                                          (pod metadata)
			//   <containerID>-<containerName>/checkpoint/inventory.img  (per container)
			//
			// Parse pod.options to discover container directories, then verify
			// each one contains checkpoint/inventory.img.
			fileReader, err := os.Open(item)
			framework.ExpectNoError(err)

			type podOptionsFile struct {
				Version    int      `json:"version"`
				Containers []string `json:"containers"`
			}
			var podOpts podOptionsFile
			foundPodOptions := false
			allFiles := make(map[string]bool)

			tr := tar.NewReader(fileReader)
			for {
				hdr, err := tr.Next()
				if err == io.EOF {
					break
				}
				framework.ExpectNoError(err)
				name := strings.TrimPrefix(hdr.Name, "./")
				allFiles[name] = true
				if name == "pod.options" {
					data, err := io.ReadAll(tr)
					framework.ExpectNoError(err)
					err = json.Unmarshal(data, &podOpts)
					framework.ExpectNoError(err)
					foundPodOptions = true
				}
			}
			fileReader.Close()

			if !foundPodOptions {
				framework.Failf("pod.options not found in checkpoint archive %q", item)
			}
			if len(podOpts.Containers) == 0 {
				framework.Failf("pod.options contains no containers in checkpoint archive %q", item)
			}

			for _, containerDir := range podOpts.Containers {
				inventoryPath := containerDir + "/checkpoint/inventory.img"
				if !allFiles[inventoryPath] {
					framework.Failf(
						"File %q not found in checkpoint archive %q",
						inventoryPath,
						item,
					)
				}
			}

			// cleanup checkpoint archive
			os.RemoveAll(item)
		}
	})
})

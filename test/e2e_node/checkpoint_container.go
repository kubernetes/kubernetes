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
	"path"
	"path/filepath"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

const (
	// timeout for proxy requests.
	proxyTimeout = 2 * time.Minute

	maxCheckpointsPerContainer = 2
)

type checkpointResult struct {
	Items []string `json:"items"`
}

func triggerCheckpoint(ctx context.Context, f *framework.Framework, pod *v1.Pod) restclient.Result {
	result, err := proxyPostRequest(
		ctx,
		f.ClientSet,
		pod.Spec.NodeName,
		fmt.Sprintf("checkpoint/%s/%s/%s", f.Namespace.Name, pod.Name, pod.Spec.Containers[0].Name),
		framework.KubeletPort,
	)

	framework.ExpectNoError(err)
	return result
}

func decodeCheckpointResult(result restclient.Result) checkpointResult {
	raw, err := result.Raw()
	framework.ExpectNoError(err)
	answer := checkpointResult{}
	err = json.Unmarshal(raw, &answer)
	framework.ExpectNoError(err)

	return answer
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

var _ = SIGDescribe("Checkpoint Container [NodeFeature:CheckpointContainer]", func() {
	f := framework.NewDefaultFramework("checkpoint-container-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline
	ginkgo.Context("checkpoint", func() {
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			initialConfig.MaxCheckpointsPerContainer = maxCheckpointsPerContainer
		})

		ginkgo.It("will checkpoint a container out of a pod", func(ctx context.Context) {
			ginkgo.By("creating a target pod")
			podClient := e2epod.NewPodClient(f)
			pod := podClient.CreateSync(ctx, &v1.Pod{
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
				ctx,
				pod.Name,
				metav1.GetOptions{},
			)

			framework.ExpectNoError(err)
			isReady, err := testutils.PodRunningReady(p)
			framework.ExpectNoError(err)
			gomega.Expect(isReady).To(gomega.BeTrue())

			framework.Logf(
				"About to checkpoint container %q on %q",
				pod.Spec.Containers[0].Name,
				pod.Spec.NodeName,
			)

			result := triggerCheckpoint(ctx, f, pod)

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

			// Checkpointing actually worked. Verify that the checkpoint exists and that
			// it is a checkpoint.

			answer := decodeCheckpointResult(result)

			// To check if MaxCheckpointsPerContainer works correctly
			// all checkpoints are tracked in this slice.
			var checkpointArchives []string

			checkpointDirectory := ""

			for _, item := range answer.Items {
				// Check that the file exists
				_, err := os.Stat(item)
				framework.ExpectNoError(err)
				checkpointArchives = append(checkpointArchives, item)
				if checkpointDirectory == "" {
					checkpointDirectory = path.Dir(item)
				}
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
					gomega.Expect(checkForFiles[fileName]).To(gomega.BeTrue())
				}

			}
			// Create a second checkpoint
			result = triggerCheckpoint(ctx, f, pod)
			err = result.Error()
			framework.ExpectNoError(err)

			answer = decodeCheckpointResult(result)

			for _, item := range answer.Items {
				// Check that the file exists
				_, err := os.Stat(item)
				framework.ExpectNoError(err)
				checkpointArchives = append(checkpointArchives, item)
			}

			// And we need a third checkpoint to see if MaxCheckpointsPerContainer
			// works as expected.
			result = triggerCheckpoint(ctx, f, pod)
			err = result.Error()
			framework.ExpectNoError(err)

			answer = decodeCheckpointResult(result)

			for _, item := range answer.Items {
				// Check that the file exists
				_, err := os.Stat(item)
				framework.ExpectNoError(err)
				checkpointArchives = append(checkpointArchives, item)
			}

			// At this point three checkpoints were created but only two
			// should exist on disk because of MaxCheckpointsPerContainer == 2

			gomega.Expect(checkpointArchives).To(gomega.HaveLen(3))

			podFullName := fmt.Sprintf(
				"%s_%s",
				pod.Name,
				f.Namespace.Name,
			)
			globPattern := filepath.Join(
				checkpointDirectory,
				fmt.Sprintf(
					"checkpoint-%s-%s-*.tar",
					podFullName,
					pod.Spec.Containers[0].Name,
				),
			)

			checkpointArchivesOnDisk, err := filepath.Glob(globPattern)
			framework.ExpectNoError(err)

			// Only two checkpoint archives should be on disk
			gomega.Expect(checkpointArchivesOnDisk).To(gomega.HaveLen(maxCheckpointsPerContainer))

			// cleanup all checkpoint archives
			for _, item := range checkpointArchives {
				os.Remove(item)
			}
		})
	})
})

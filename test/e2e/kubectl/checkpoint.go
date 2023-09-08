/*
Copyright 2023 The Kubernetes Authors.

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

// OWNER = sig/cli

package kubectl

import (
	"archive/tar"
	"context"
	"encoding/json"
	"io"
	"os"
	"strconv"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubectl/pkg/cmd/util/podcmd"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ekubectl "k8s.io/kubernetes/test/e2e/framework/kubectl"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

func checkpointTestingPod(name, value, defaultContainerName string) v1.Pod {
	return v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				"name": "foo",
				"time": value,
			},
			Annotations: map[string]string{
				podcmd.DefaultContainerAnnotationName: defaultContainerName,
			},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "container-1",
					Image: agnhostImage,
					Args:  []string{"test-webserver", "--port", "81"},
				},
				{
					Name:  defaultContainerName,
					Image: agnhostImage,
					Args:  []string{"test-webserver", "--port", "82"},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
		},
	}
}

var _ = SIGDescribe("Kubectl:checkpoint", func() {
	f := framework.NewDefaultFramework("kubectl-checkpoint")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline
	defer ginkgo.GinkgoRecover()

	var c clientset.Interface
	var ns string
	ginkgo.BeforeEach(func() {
		c = f.ClientSet
		ns = f.Namespace.Name
	})

	ginkgo.Describe("checkpoint", func() {
		podName := "checkpoint-pod"
		ginkgo.BeforeEach(func() {
			ginkgo.By("creating an pod")
			e2ekubectl.RunKubectlOrDie(
				ns,
				"run",
				podName,
				"--image="+agnhostImage,
				"--restart=Never",
				podRunningTimeoutArg,
				"--",
				"test-webserver",
			)
		})
		ginkgo.AfterEach(func() {
			e2ekubectl.RunKubectlOrDie(ns, "delete", "pod", podName)
		})
	})

	ginkgo.Describe("default container checkpoint", func() {
		ginkgo.Describe("the second container is the default-container by annotation", func() {
			var pod *v1.Pod
			podName := "pod" + string(uuid.NewUUID())
			defaultContainerName := "container-2"
			ginkgo.BeforeEach(func(ctx context.Context) {
				podClient := f.ClientSet.CoreV1().Pods(ns)
				ginkgo.By("constructing the pod")
				value := strconv.Itoa(time.Now().Nanosecond())
				podCopy := checkpointTestingPod(podName, value, defaultContainerName)
				pod = &podCopy
				ginkgo.By("creating the pod")
				_, err := podClient.Create(ctx, pod, metav1.CreateOptions{})
				if err != nil {
					framework.Failf("Failed to create pod: %v", err)
				}
			})
			ginkgo.AfterEach(func() {
				e2ekubectl.RunKubectlOrDie(ns, "delete", "pod", podName)
			})

			ginkgo.It("should checkpoint default container if not specified", func(ctx context.Context) {
				ginkgo.By("Waiting for pod to start.")
				if err := e2epod.WaitForPodNameRunningInNamespace(ctx, c, pod.Name, ns); err != nil {
					framework.Failf("Pod %s did not start: %v", podName, err)
				}
				ginkgo.By("checkpoint default container")
				out, err := e2ekubectl.RunKubectl(ns, "checkpoint", podName)
				if err != nil {
					if strings.Contains(err.Error(), "the server could not find the requested resource") {
						// This is the error returned if the corresponding feature gate is not enabled.
						ginkgo.Skip("Feature 'ContainerCheckpoint' is not enabled and not available")
					}
					if strings.Contains(err.Error(), "rpc error: code = Unknown desc = checkpoint/restore support not available") ||
						strings.Contains(err.Error(), "rpc error: code = Unimplemented desc = unknown method CheckpointContainer") {
						// This is the error returned if the underlying container engine does not support the CheckpointContainer CRI RPC.
						ginkgo.Skip("Container engine does not implement 'CheckpointContainer'")
					}
					framework.Failf("unexpected error during 'kubectl checkpoint %s': %v", podName, err)
				}
				framework.Logf("got output %q", out)
				framework.ExpectNoError(err)
				gomega.Expect(out).NotTo(gomega.BeEmpty())
				gomega.Expect(out).To(gomega.ContainSubstring(defaultContainerName))
				type checkpointResult struct {
					Items []string `json:"items"`
				}
				answer := checkpointResult{}
				err = json.Unmarshal([]byte(out), &answer)
				framework.ExpectNoError(err)
				gomega.Expect(answer.Items).To(gomega.HaveLen(1))

				item := answer.Items[0]

				// Check that the file exists
				_, err = os.Stat(item)
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
				defer fileReader.Close()
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
				// cleanup checkpoint archive
				os.RemoveAll(item)
			})
		})
	})

})

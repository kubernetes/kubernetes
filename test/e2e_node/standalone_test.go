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

package e2enode

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/pkg/cluster/ports"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var _ = SIGDescribe("[Feature:StandaloneMode] ", func() {
	f := framework.NewDefaultFramework("static-pod")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged
	ginkgo.It("can create a static Pod ", func(ctx context.Context) {

		var ns, podPath, staticPodName string

		ns = f.Namespace.Name
		staticPodName = "static-pod-" + string(uuid.NewUUID())

		podPath = framework.TestContext.KubeletConfig.StaticPodPath

		err := createBasicStaticPod(podPath, staticPodName, ns,
			imageutils.GetE2EImage(imageutils.Nginx), v1.RestartPolicyAlways)
		framework.ExpectNoError(err)

		file := staticPodPath(podPath, staticPodName, ns)
		defer os.Remove(file)

		gomega.Eventually(ctx, func() error {
			pod, err := getPodStatus(ctx, staticPodName)
			if err != nil {
				return err
			}
			if pod.Status.Phase != v1.PodRunning {
				return fmt.Errorf("pod %s is not running. Status: %v", staticPodName, pod.Status.Phase)
			}
			return nil
		}, 1*time.Minute, 5*time.Second).Should(gomega.Succeed())
	})
})

func createBasicStaticPod(dir, name, namespace, image string, restart v1.RestartPolicy) error {
	podSpec := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyAlways,
			InitContainers: []v1.Container{
				{
					Name:  "init-1",
					Image: busyboxImage,
					Command: ExecCommand("init-1", execCommand{
						Delay:    1,
						ExitCode: 0,
					}),
				},
			},
			Containers: []v1.Container{
				{
					Name:  "regular1",
					Image: busyboxImage,
					Command: ExecCommand("regular1", execCommand{
						Delay:    1000,
						ExitCode: 0,
					}),
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceMemory: resource.MustParse("15Mi"),
						},
						Limits: v1.ResourceList{
							v1.ResourceMemory: resource.MustParse("15Mi"),
						},
					},
				},
			},
		},
	}

	podYaml, err := kubeadmutil.MarshalToYaml(podSpec, v1.SchemeGroupVersion)
	if err != nil {
		return err
	}

	file := staticPodPath(dir, name, namespace)

	f, err := os.OpenFile(file, os.O_RDWR|os.O_TRUNC|os.O_CREATE, 0666)
	if err != nil {
		return err
	}
	defer f.Close()

	_, err = f.Write(podYaml)
	return err
}

// returns a status 200 response from the /configz endpoint or nil if fails
func getPodStatus(ctx context.Context, name string) (*v1.Pod, error) {
	endpoint := fmt.Sprintf("http://127.0.0.1:%d/pods", ports.KubeletReadOnlyPort)
	// TODO: we do not need TLS and bearer token for this test
	tr := &http.Transport{
		TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
	}
	client := &http.Client{Transport: tr}
	req, err := http.NewRequest("GET", endpoint, nil)
	framework.ExpectNoError(err)
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", framework.TestContext.BearerToken))
	req.Header.Add("Accept", "application/json")

	var pod *v1.Pod

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("/pods response status not 200. Response was: %+v", resp)
	}

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read body from /pods response %w", err)
	}

	pods, err := decodePods(respBody)
	framework.ExpectNoError(err)

	found := false
	for _, p := range pods.Items {
		// Static pods has a node name suffix so comparing as substring
		if strings.Contains(p.Name, name) {
			found = true
			pod = &p
		}
	}

	if !found {
		return nil, fmt.Errorf("pod %s not found in /pods response. Pods were: %v", name, string(respBody))
	}

	return pod, nil
}

// Decodes the http response from /configz and returns a kubeletconfig.KubeletConfiguration (internal type).
func decodePods(respBody []byte) (*v1.PodList, error) {
	// This hack because /pods reports the following structure:
	// {"kind":"PodList","apiVersion":"v1","metadata":{},"items":[{"metadata":{"name":"kube-dns-autoscaler-758c4689b9-htpqj","generateName":"kube-dns-autoscaler-758c4689b9-",

	var pods v1.PodList
	err := json.Unmarshal(respBody, &pods)
	if err != nil {
		return nil, err
	}

	return &pods, nil
}

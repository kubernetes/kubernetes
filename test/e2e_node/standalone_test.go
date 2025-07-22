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
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/cli-runtime/pkg/printers"
	"k8s.io/kubernetes/pkg/cluster/ports"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	testutils "k8s.io/kubernetes/test/utils"
)

var _ = SIGDescribe(feature.StandaloneMode, func() {
	f := framework.NewDefaultFramework("static-pod")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline
	ginkgo.Context("when creating a static pod", func() {
		var ns, podPath, staticPodName string

		ginkgo.It("the pod should be running", func(ctx context.Context) {
			ns = f.Namespace.Name
			staticPodName = "static-pod-" + string(uuid.NewUUID())
			podPath = kubeletCfg.StaticPodPath

			err := scheduleStaticPod(podPath, staticPodName, ns, createBasicStaticPodSpec(staticPodName, ns))
			framework.ExpectNoError(err)

			gomega.Eventually(ctx, func(ctx context.Context) error {
				pod, err := getPodFromStandaloneKubelet(ctx, ns, staticPodName)
				if err != nil {
					return fmt.Errorf("error getting pod(%v/%v) from standalone kubelet: %w", ns, staticPodName, err)
				}

				isReady, err := testutils.PodRunningReady(pod)
				if err != nil {
					return fmt.Errorf("error checking if pod (%v/%v) is running ready: %w", ns, staticPodName, err)
				}
				if !isReady {
					return fmt.Errorf("pod (%v/%v) is not running", ns, staticPodName)
				}
				return nil
			}, f.Timeouts.PodStart, time.Second*5).Should(gomega.BeNil())
		})

		ginkgo.It("the pod with the port on host network should be running and can be upgraded when the container name changes", func(ctx context.Context) {
			ns = f.Namespace.Name
			staticPodName = "static-pod-" + string(uuid.NewUUID())
			podPath = kubeletCfg.StaticPodPath

			podSpec := createBasicStaticPodSpec(staticPodName, ns)
			podSpec.Spec.HostNetwork = true
			podSpec.Spec.Containers[0].Ports = []v1.ContainerPort{
				{
					Name:          "tcp",
					ContainerPort: 4534,
					Protocol:      v1.ProtocolTCP,
				},
			}
			err := scheduleStaticPod(podPath, staticPodName, ns, podSpec)
			framework.ExpectNoError(err)

			gomega.Eventually(ctx, func(ctx context.Context) error {
				pod, err := getPodFromStandaloneKubelet(ctx, ns, staticPodName)
				if err != nil {
					return fmt.Errorf("error getting pod(%v/%v) from standalone kubelet: %w", ns, staticPodName, err)
				}

				isReady, err := testutils.PodRunningReady(pod)
				if err != nil {
					return fmt.Errorf("error checking if pod (%v/%v) is running ready: %w", ns, staticPodName, err)
				}
				if !isReady {
					return fmt.Errorf("pod (%v/%v) is not running", ns, staticPodName)
				}
				return nil
			}, f.Timeouts.PodStart, time.Second*5).Should(gomega.BeNil())

			// Upgrade the pod
			podSpec.Spec.Containers[0].Name = "upgraded"
			err = scheduleStaticPod(podPath, staticPodName, ns, podSpec)
			framework.ExpectNoError(err)

			gomega.Eventually(ctx, func(ctx context.Context) error {
				pod, err := getPodFromStandaloneKubelet(ctx, ns, staticPodName)
				if err != nil {
					return fmt.Errorf("error getting pod(%v/%v) from standalone kubelet: %w", ns, staticPodName, err)
				}

				if pod.Spec.Containers[0].Name != "upgraded" {
					return fmt.Errorf("pod (%v/%v) is not upgraded", ns, staticPodName)
				}

				isReady, err := testutils.PodRunningReady(pod)
				if err != nil {
					return fmt.Errorf("error checking if pod (%v/%v) is running ready: %w", ns, staticPodName, err)
				}
				if !isReady {
					return fmt.Errorf("pod (%v/%v) is not running", ns, staticPodName)
				}
				return nil
			}, f.Timeouts.PodStart, time.Second*5).Should(gomega.BeNil())
		})

		// the test below is not working - pod update fails with the "Predicate NodePorts failed: node(s) didn't have free ports for the requested pod ports"
		// ginkgo.It("the pod with the port on host network should be running and can be upgraded when namespace of a Pod changes", func(ctx context.Context) {
		// 	ns = f.Namespace.Name
		// 	staticPodName = "static-pod-" + string(uuid.NewUUID())
		// 	podPath = kubeletCfg.StaticPodPath

		// 	podSpec := createBasicStaticPodSpec(staticPodName, ns)
		// 	podSpec.Spec.HostNetwork = true
		// 	podSpec.Spec.Containers[0].Ports = []v1.ContainerPort{
		// 		{
		// 			Name:          "tcp",
		// 			ContainerPort: 4534,
		// 			Protocol:      v1.ProtocolTCP,
		// 		},
		// 	}
		// 	err := scheduleStaticPod(podPath, staticPodName, ns, podSpec)
		// 	framework.ExpectNoError(err)

		// 	gomega.Eventually(ctx, func(ctx context.Context) error {
		// 		pod, err := getPodFromStandaloneKubelet(ctx, ns, staticPodName)
		// 		if err != nil {
		// 			return fmt.Errorf("error getting pod(%v/%v) from standalone kubelet: %w", ns, staticPodName, err)
		// 		}

		// 		isReady, err := testutils.PodRunningReady(pod)
		// 		if err != nil {
		// 			return fmt.Errorf("error checking if pod (%v/%v) is running ready: %w", ns, staticPodName, err)
		// 		}
		// 		if !isReady {
		// 			return fmt.Errorf("pod (%v/%v) is not running", ns, staticPodName)
		// 		}
		// 		return nil
		// 	}, f.Timeouts.PodStart, time.Second*5).Should(gomega.BeNil())

		// 	// Upgrade the pod
		// 	upgradedNs := ns + "-upgraded"
		// 	podSpec.Namespace = upgradedNs

		// 	// use old namespace as it uses ns in a file name
		// 	err = scheduleStaticPod(podPath, staticPodName, ns, podSpec)
		// 	framework.ExpectNoError(err)

		// 	gomega.Eventually(ctx, func(ctx context.Context) error {
		// 		pod, err := getPodFromStandaloneKubelet(ctx, upgradedNs, staticPodName)
		// 		if err != nil {
		// 			return fmt.Errorf("error getting pod(%v/%v) from standalone kubelet: %w", upgradedNs, staticPodName, err)
		// 		}

		// 		isReady, err := testutils.PodRunningReady(pod)
		// 		if err != nil {
		// 			return fmt.Errorf("error checking if pod (%v/%v) is running ready: %w", upgradedNs, staticPodName, err)
		// 		}
		// 		if !isReady {
		// 			return fmt.Errorf("pod (%v/%v) is not running", upgradedNs, staticPodName)
		// 		}
		// 		return nil
		// 	}, f.Timeouts.PodStart, time.Second*5).Should(gomega.BeNil())
		// })

		ginkgo.AfterEach(func(ctx context.Context) {
			ginkgo.By(fmt.Sprintf("delete the static pod (%v/%v)", ns, staticPodName))
			err := deleteStaticPod(podPath, staticPodName, ns)
			framework.ExpectNoError(err)

			ginkgo.By(fmt.Sprintf("wait for pod to disappear (%v/%v)", ns, staticPodName))
			gomega.Eventually(ctx, func(ctx context.Context) error {
				_, err := getPodFromStandaloneKubelet(ctx, ns, staticPodName)

				if apierrors.IsNotFound(err) {
					return nil
				}
				return fmt.Errorf("pod (%v/%v) still exists", ns, staticPodName)
			}).Should(gomega.Succeed())
		})
	})
})

func createBasicStaticPodSpec(name, namespace string) *v1.Pod {
	podSpec := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyAlways,
			Containers: []v1.Container{
				{
					Name:  "regular1",
					Image: imageutils.GetE2EImage(imageutils.BusyBox),
					Command: []string{
						"/bin/sh", "-c", "touch /tmp/healthy; sleep 10000",
					},
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceMemory: resource.MustParse("15Mi"),
						},
						Limits: v1.ResourceList{
							v1.ResourceMemory: resource.MustParse("15Mi"),
						},
					},
					ReadinessProbe: &v1.Probe{
						InitialDelaySeconds: 2,
						TimeoutSeconds:      2,
						ProbeHandler: v1.ProbeHandler{
							Exec: &v1.ExecAction{
								Command: []string{"/bin/sh", "-c", "cat /tmp/healthy"},
							},
						},
					},
				},
			},
		},
	}

	return podSpec
}

func scheduleStaticPod(dir, name, namespace string, podSpec *v1.Pod) error {
	file := staticPodPath(dir, name, namespace)
	f, err := os.OpenFile(file, os.O_RDWR|os.O_TRUNC|os.O_CREATE, 0666)
	if err != nil {
		return err
	}
	defer f.Close()

	y := printers.YAMLPrinter{}
	y.PrintObj(podSpec, f)

	return nil
}

func getPodFromStandaloneKubelet(ctx context.Context, podNamespace string, podName string) (*v1.Pod, error) {
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

	resp, err := client.Do(req)
	if err != nil {
		framework.Logf("Failed to get /pods: %v", err)
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		framework.Logf("/pods response status not 200. Response was: %+v", resp)
		return nil, fmt.Errorf("/pods response was not 200: %v", err)
	}

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("/pods response was unable to be read: %v", err)
	}

	pods, err := decodePods(respBody)
	if err != nil {
		return nil, fmt.Errorf("unable to decode /pods: %v", err)
	}

	for _, p := range pods.Items {
		// Static pods has a node name suffix so comparing as substring
		p := p
		if strings.Contains(p.Name, podName) && strings.Contains(p.Namespace, podNamespace) {
			return &p, nil
		}
	}

	return nil, apierrors.NewNotFound(schema.GroupResource{Resource: "pods"}, podName)
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

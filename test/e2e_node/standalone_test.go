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
	"path/filepath"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/cli-runtime/pkg/printers"
	"k8s.io/kubernetes/pkg/cluster/ports"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe(feature.StandaloneMode, framework.WithFeatureGate(features.EnvFiles), func() {
	f := framework.NewDefaultFramework("static-pod-envfiles")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline
	ginkgo.Context("when creating a static pod with EnvFiles", func() {
		var ns, podPath, staticPodName string

		ginkgo.It("the pod should be running and consume variables", func(ctx context.Context) {
			ns = f.Namespace.Name
			staticPodName = "static-pod-envfiles-" + string(uuid.NewUUID())
			podPath = kubeletCfg.StaticPodPath

			podSpec := &v1.Pod{
				TypeMeta: metav1.TypeMeta{
					Kind:       "Pod",
					APIVersion: "v1",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      staticPodName,
					Namespace: ns,
				},
				Spec: v1.PodSpec{
					InitContainers: []v1.Container{
						{
							Name:    "setup-envfile",
							Image:   imageutils.GetE2EImage(imageutils.BusyBox),
							Command: []string{"sh", "-c", `echo CONFIG_1=\'value1\' > /data/config.env && echo CONFIG_2=\'value2\' >> /data/config.env`},
							VolumeMounts: []v1.VolumeMount{
								{
									Name:      "config",
									MountPath: "/data",
								},
							},
						},
					},
					Containers: []v1.Container{
						{
							Name:    "use-envfile",
							Image:   imageutils.GetE2EImage(imageutils.BusyBox),
							Command: []string{"sh", "-c", "env | grep -E '(CONFIG_1|CONFIG_2)' | sort"},
							Env: []v1.EnvVar{
								{
									Name: "CONFIG_1",
									ValueFrom: &v1.EnvVarSource{
										FileKeyRef: &v1.FileKeySelector{
											VolumeName: "config",
											Path:       "config.env",
											Key:        "CONFIG_1",
										},
									},
								},
								{
									Name: "CONFIG_2",
									ValueFrom: &v1.EnvVarSource{
										FileKeyRef: &v1.FileKeySelector{
											VolumeName: "config",
											Path:       "config.env",
											Key:        "CONFIG_2",
										},
									},
								},
							},
						},
					},
					RestartPolicy: v1.RestartPolicyNever,
					Volumes: []v1.Volume{
						{
							Name: "config",
							VolumeSource: v1.VolumeSource{
								EmptyDir: &v1.EmptyDirVolumeSource{},
							},
						},
					},
				},
			}

			err := scheduleStaticPod(podPath, staticPodName, ns, podSpec)
			framework.ExpectNoError(err)

			gomega.Eventually(ctx, func(ctx context.Context) error {
				pod, err := getPodFromStandaloneKubelet(ctx, ns, staticPodName)
				if err != nil {
					return fmt.Errorf("error getting pod(%v/%v) from standalone kubelet: %w", ns, staticPodName, err)
				}
				if pod.Status.Phase == v1.PodSucceeded {
					return nil
				}
				if pod.Status.Phase == v1.PodFailed {
					logs, err := getPodLogsFromStandaloneKubelet(ctx, ns, staticPodName, "use-envfile")
					if err != nil {
						framework.Logf("failed to get logs on pod failure: %v", err)
					}
					return fmt.Errorf("pod (%v/%v) failed, logs: %s", ns, staticPodName, logs)
				}
				return fmt.Errorf("pod (%v/%v) is not succeeded, phase: %s", ns, staticPodName, pod.Status.Phase)
			}, f.Timeouts.PodStart, time.Second*5).Should(gomega.Succeed())

			logs, err := getPodLogsFromStandaloneKubelet(ctx, ns, staticPodName, "use-envfile")
			framework.ExpectNoError(err)

			gomega.Expect(logs).To(gomega.ContainSubstring("CONFIG_1=value1"))
			gomega.Expect(logs).To(gomega.ContainSubstring("CONFIG_2=value2"))
		})

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

		f.Context("when the static pod has init container", f.WithSerial(), func() {
			f.It("should be ready after init container is removed and kubelet restarts", func(ctx context.Context) {
				ginkgo.By("create static pod")
				staticPod := &v1.Pod{
					TypeMeta: metav1.TypeMeta{
						Kind:       "Pod",
						APIVersion: "v1",
					},
					ObjectMeta: metav1.ObjectMeta{
						Name:      "static",
						Namespace: f.Namespace.Name,
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Name:  "main",
								Image: imageutils.GetE2EImage(imageutils.Pause),
							},
						},
						InitContainers: []v1.Container{
							{
								Name:    "init",
								Image:   imageutils.GetE2EImage(imageutils.BusyBox),
								Command: []string{"ls"},
							},
						},
					},
				}
				staticPodName = staticPod.Name
				podPath = kubeletCfg.StaticPodPath
				ns = staticPod.Namespace
				err := scheduleStaticPod(podPath, staticPod.Name, ns, staticPod)
				framework.ExpectNoError(err)

				var initCtrID string
				var startTime *metav1.Time
				ginkgo.By("wait for the mirror pod to be updated")
				gomega.Eventually(ctx, func(g gomega.Gomega) {
					pod, err := getPodFromStandaloneKubelet(ctx, staticPod.Namespace, staticPod.Name)
					g.Expect(err).Should(gomega.Succeed())
					g.Expect(pod.Status.InitContainerStatuses).To(gomega.HaveLen(1))
					cstatus := pod.Status.InitContainerStatuses[0]
					// Wait until the init container is terminated.
					g.Expect(cstatus.State.Terminated).NotTo(gomega.BeNil())
					g.Expect(cstatus.State.Terminated.ContainerID).NotTo(gomega.BeEmpty())
					initCtrID = cstatus.ContainerID
					startTime = pod.Status.StartTime
				}, 2*time.Minute, 5*time.Second).Should(gomega.Succeed())

				ginkgo.By("remove init container")
				removeInitContainer(ctx, initCtrID)

				ginkgo.By("restart kubelet")
				restartKubelet(ctx, true)
				gomega.Eventually(ctx, func() bool {
					return kubeletHealthCheck(kubeletHealthCheckURL)
				}, f.Timeouts.PodStart, f.Timeouts.Poll).Should(gomega.BeTrueBecause("kubelet should be started"))

				ginkgo.By("wait for the mirror pod to be updated")
				gomega.Eventually(ctx, func(g gomega.Gomega) {
					pod, err := getPodFromStandaloneKubelet(ctx, staticPod.Namespace, staticPod.Name)
					g.Expect(pod.Status.StartTime).NotTo(gomega.Equal(startTime))
					g.Expect(err).Should(gomega.Succeed())
					g.Expect(pod.Status.InitContainerStatuses).To(gomega.HaveLen(1))
					cstatus := pod.Status.InitContainerStatuses[0]
					// Init container should be completed.
					g.Expect(cstatus.State.Terminated).NotTo(gomega.BeNil())
					g.Expect(cstatus.State.Terminated.Reason).To(gomega.Equal("Completed"))
					g.Expect(cstatus.State.Terminated.ExitCode).To(gomega.BeZero())
				}, 2*time.Minute, 5*time.Second).Should(gomega.Succeed())
			})
		})
	})
})

var _ = SIGDescribe("Pod Extended (RestartAllContainers)",
	feature.StandaloneMode,
	framework.WithFeatureGate(features.ContainerRestartRules),
	framework.WithFeatureGate(features.RestartAllContainersOnContainerExits),
	func() {
		f := framework.NewDefaultFramework("static-pod")
		f.NamespacePodSecurityLevel = admissionapi.LevelBaseline
		ginkgo.It("should restart all containers on regular container exit", func(ctx context.Context) {
			ns := f.Namespace.Name
			staticPodName := "static-pod-" + string(uuid.NewUUID())
			podPath := kubeletCfg.StaticPodPath
			var (
				containerRestartPolicyAlways = v1.ContainerRestartPolicyAlways
				containerRestartPolicyNever  = v1.ContainerRestartPolicyNever
			)
			restartAllContainersRules := []v1.ContainerRestartRule{
				{
					Action: v1.ContainerRestartRuleActionRestartAllContainers,
					ExitCodes: &v1.ContainerRestartRuleOnExitCodes{
						Operator: v1.ContainerRestartRuleOnExitCodesOpIn,
						Values:   []int32{42},
					},
				},
			}
			pod := &v1.Pod{
				TypeMeta: metav1.TypeMeta{
					Kind:       "Pod",
					APIVersion: "v1",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      staticPodName,
					Namespace: ns,
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					InitContainers: []v1.Container{
						{
							Name:    "init",
							Image:   imageutils.GetE2EImage(imageutils.BusyBox),
							Command: []string{"/bin/sh", "-c", "exit 0"},
						},
						{
							Name:          "sidecar",
							Image:         imageutils.GetE2EImage(imageutils.BusyBox),
							Command:       []string{"/bin/sh", "-c", "sleep 10000"},
							RestartPolicy: &containerRestartPolicyAlways,
						},
					},
					Containers: []v1.Container{
						{
							Name:               "source-container",
							Image:              imageutils.GetE2EImage(imageutils.BusyBox),
							Command:            []string{"/bin/sh", "-c", "sleep 60; exit 42"},
							RestartPolicy:      &containerRestartPolicyNever,
							RestartPolicyRules: restartAllContainersRules,
						},
						{
							Name:    "regular",
							Image:   imageutils.GetE2EImage(imageutils.BusyBox),
							Command: []string{"/bin/sh", "-c", "sleep 10000"},
						},
					},
				},
			}

			err := scheduleStaticPod(podPath, staticPodName, ns, pod)
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
			}, f.Timeouts.PodStart, f.Timeouts.Poll).Should(gomega.Succeed())

			gomega.Eventually(ctx, func(ctx context.Context) error {
				pod, err := getPodFromStandaloneKubelet(ctx, ns, staticPodName)
				if err != nil {
					return fmt.Errorf("error getting pod(%v/%v) from standalone kubelet: %w", ns, staticPodName, err)
				}
				for _, c := range pod.Status.InitContainerStatuses {
					if c.RestartCount == 0 {
						return fmt.Errorf("init container %v has not restarted", c.Name)
					}
				}
				for _, c := range pod.Status.ContainerStatuses {
					if c.RestartCount == 0 {
						return fmt.Errorf("container %v has not restarted", c.Name)
					}
				}
				return nil
			}, 10*time.Minute, f.Timeouts.Poll).Should(gomega.Succeed())
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
		if strings.Contains(p.Name, podName) && strings.Contains(p.Namespace, podNamespace) {
			return &p, nil
		}
	}

	return nil, apierrors.NewNotFound(schema.GroupResource{Resource: "pods"}, podName)
}

func getPodLogsFromStandaloneKubelet(ctx context.Context, podNamespace string, podName string, containerName string) (string, error) {
	pod, err := getPodFromStandaloneKubelet(ctx, podNamespace, podName)
	if err != nil {
		return "", fmt.Errorf("failed to get pod %s/%s: %w", podNamespace, podName, err)
	}

	logCRIDir := "/var/log/pods"
	podLogDir := filepath.Join(logCRIDir, fmt.Sprintf("%s_%s_%s", pod.Namespace, pod.Name, pod.UID))
	logFile := filepath.Join(podLogDir, containerName, "0.log")

	var content []byte
	err = wait.PollUntilContextTimeout(ctx, time.Second, time.Minute, true, func(ctx context.Context) (bool, error) {
		var errRead error
		content, errRead = os.ReadFile(logFile)
		if errRead != nil {
			if os.IsNotExist(errRead) {
				return false, nil
			}
			return false, errRead
		}
		return true, nil
	})

	if err != nil {
		return "", fmt.Errorf("could not read log file %s: %w", logFile, err)
	}

	return string(content), nil
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

var _ = SIGDescribe(feature.StandaloneMode, framework.WithSerial(), func() {
	f := framework.NewDefaultFramework("static-pod-serial")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline
	ginkgo.Context("when creating a static pod and restarting kubelet", func() {
		var ns, podPath, staticPodName string

		ginkgo.BeforeEach(func() {
			ns = f.Namespace.Name
			staticPodName = "static-pod-" + string(uuid.NewUUID())
			podPath = kubeletCfg.StaticPodPath
		})

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

		ginkgo.It("the pod should be running and kubelet not panic", func(ctx context.Context) {
			err := scheduleStaticPod(podPath, staticPodName, ns, createBasicStaticPodSpec(staticPodName, ns))
			framework.ExpectNoError(err)

			ginkgo.By("Waiting for the pod to be running")
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
			}, f.Timeouts.PodStart, time.Second*5).Should(gomega.Succeed())

			ginkgo.By("restarting the kubelet")
			restartKubelet(ctx, true)

			gomega.Eventually(ctx, func() bool {
				return kubeletHealthCheck(kubeletHealthCheckURL)
			}, f.Timeouts.PodStart, f.Timeouts.Poll).Should(gomega.BeTrueBecause("kubelet should be started"))

			ginkgo.By("ensuring that pod is running")
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
			}, f.Timeouts.PodStart, time.Second*30).Should(gomega.Succeed())
		})
	})
})

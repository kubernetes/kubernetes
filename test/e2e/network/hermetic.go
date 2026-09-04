/*
Copyright 2026 The Kubernetes Authors.

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

package network

import (
	"context"
	"strings"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/network/common"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var _ = common.SIGDescribe("Hermetic", feature.Hermetic, func() {
	f := framework.NewDefaultFramework("hermetic")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	var cs clientset.Interface
	var podClient *e2epod.PodClient

	ginkgo.BeforeEach(func() {
		cs = f.ClientSet
		podClient = e2epod.NewPodClient(f)
	})

	ginkgo.It("should start a hermetic pod without IP address, with DNSPolicy None and EnableServiceLinks false", func(ctx context.Context) {
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "hermetic-lifecycle",
			},
			Spec: v1.PodSpec{
				Hermetic: ptr.To(true),
				Containers: []v1.Container{{
					Name:    "agnhost",
					Image:   imageutils.GetE2EImage(imageutils.Agnhost),
					Command: []string{"sleep", "3600"},
				}},
			},
		}

		ginkgo.By("Creating hermetic pod")
		createdPod := podClient.Create(ctx, pod)

		ginkgo.By("Waiting for pod to be running")
		err := e2epod.WaitForPodNameRunningInNamespace(ctx, cs, createdPod.Name, createdPod.Namespace)
		framework.ExpectNoError(err, "pod failed to reach Running state")

		ginkgo.By("Verifying pod status and spec")
		fetchedPod, err := podClient.Get(ctx, createdPod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to get pod")

		gomega.Expect(fetchedPod.Status.PodIP).To(gomega.BeEmpty(), "podIP should be empty for hermetic pod")
		gomega.Expect(fetchedPod.Status.PodIPs).To(gomega.BeEmpty(), "podIPs should be empty for hermetic pod")
		gomega.Expect(fetchedPod.Spec.DNSPolicy).To(gomega.Equal(v1.DNSNone), "DNSPolicy should default to None")
		gomega.Expect(fetchedPod.Spec.EnableServiceLinks).To(gomega.Equal(ptr.To(false)), "EnableServiceLinks should default to false")
	})

	ginkgo.It("should only have loopback interface inside the hermetic container and allow local loopback traffic", func(ctx context.Context) {
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "hermetic-netns",
			},
			Spec: v1.PodSpec{
				Hermetic: ptr.To(true),
				Containers: []v1.Container{{
					Name:    "agnhost",
					Image:   imageutils.GetE2EImage(imageutils.Agnhost),
					Command: []string{"sleep", "3600"},
				}},
			},
		}

		ginkgo.By("Creating hermetic pod")
		createdPod := podClient.Create(ctx, pod)

		ginkgo.By("Waiting for pod to be running")
		err := e2epod.WaitForPodNameRunningInNamespace(ctx, cs, createdPod.Name, createdPod.Namespace)
		framework.ExpectNoError(err, "pod failed to reach Running state")

		ginkgo.By("Verifying network interfaces inside container via ip link")
		stdout := e2epod.ExecShellInPod(ctx, f, createdPod.Name, "ip -o link show")
		lines := strings.Split(strings.TrimSpace(stdout), "\n")
		// In a sealed hermetic sandbox, only the loopback interface "lo" should exist.
		for _, line := range lines {
			trimmed := strings.TrimSpace(line)
			if len(trimmed) == 0 {
				continue
			}
			gomega.Expect(trimmed).To(gomega.ContainSubstring("lo:"), "unexpected network interface found in hermetic sandbox: %s", trimmed)
		}

		ginkgo.By("Verifying loopback connectivity inside container")
		stdout = e2epod.ExecShellInPod(ctx, f, createdPod.Name, "ping -c 1 -W 2 127.0.0.1")
		gomega.Expect(stdout).To(gomega.ContainSubstring("1 packets transmitted, 1 packets received"), "loopback ping failed")
	})

	ginkgo.It("should not be included in Endpoints or EndpointSlices when matching a Service selector", func(ctx context.Context) {
		labels := map[string]string{"app": "hermetic-svc-test"}

		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "hermetic-service-member",
				Labels: labels,
			},
			Spec: v1.PodSpec{
				Hermetic: ptr.To(true),
				Containers: []v1.Container{{
					Name:    "agnhost",
					Image:   imageutils.GetE2EImage(imageutils.Agnhost),
					Command: []string{"sleep", "3600"},
					Ports: []v1.ContainerPort{{
						Name:          "http",
						ContainerPort: 8080,
						Protocol:      v1.ProtocolTCP,
					}},
				}},
			},
		}

		ginkgo.By("Creating hermetic pod with service selector label")
		createdPod := podClient.Create(ctx, pod)

		ginkgo.By("Waiting for pod to be running")
		err := e2epod.WaitForPodNameRunningInNamespace(ctx, cs, createdPod.Name, createdPod.Namespace)
		framework.ExpectNoError(err, "pod failed to reach Running state")

		ginkgo.By("Creating Service selecting the hermetic pod")
		svc := &v1.Service{
			ObjectMeta: metav1.ObjectMeta{
				Name: "hermetic-test-svc",
			},
			Spec: v1.ServiceSpec{
				Selector: labels,
				Ports: []v1.ServicePort{{
					Name:       "http",
					Port:       80,
					TargetPort: intstr.FromInt32(8080),
					Protocol:   v1.ProtocolTCP,
				}},
			},
		}
		createdSvc, err := cs.CoreV1().Services(f.Namespace.Name).Create(ctx, svc, metav1.CreateOptions{})
		framework.ExpectNoError(err, "error creating Service")

		ginkgo.By("Verifying EndpointSlice contains no endpoints for the hermetic pod")
		err = wait.PollUntilContextTimeout(ctx, 1*time.Second, 15*time.Second, true, func(ctx context.Context) (bool, error) {
			endpointSliceList, err := cs.DiscoveryV1().EndpointSlices(createdSvc.Namespace).List(ctx, metav1.ListOptions{
				LabelSelector: discoveryv1.LabelServiceName + "=" + createdSvc.Name,
			})
			if err != nil {
				return false, err
			}
			if len(endpointSliceList.Items) == 0 {
				return false, nil
			}
			for _, epSlice := range endpointSliceList.Items {
				if len(epSlice.Endpoints) > 0 {
					framework.Failf("Hermetic pod unexpectedly added to EndpointSlice: %+v", epSlice.Endpoints)
				}
			}
			return true, nil
		})
		framework.ExpectNoError(err, "EndpointSlice polling failed")

		ginkgo.By("Verifying Endpoints contains no addresses for the hermetic pod")
		endpoints, err := cs.CoreV1().Endpoints(createdSvc.Namespace).Get(ctx, createdSvc.Name, metav1.GetOptions{})
		if err == nil {
			for _, subset := range endpoints.Subsets {
				gomega.Expect(subset.Addresses).To(gomega.BeEmpty(), "Hermetic pod unexpectedly added to Endpoints addresses")
				gomega.Expect(subset.NotReadyAddresses).To(gomega.BeEmpty(), "Hermetic pod unexpectedly added to Endpoints notReadyAddresses")
			}
		}
	})

	ginkgo.It("should support Exec probes on hermetic pods", func(ctx context.Context) {
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "hermetic-exec-probe",
			},
			Spec: v1.PodSpec{
				Hermetic: ptr.To(true),
				Containers: []v1.Container{{
					Name:    "agnhost",
					Image:   imageutils.GetE2EImage(imageutils.Agnhost),
					Command: []string{"sleep", "3600"},
					LivenessProbe: &v1.Probe{
						ProbeHandler: v1.ProbeHandler{
							Exec: &v1.ExecAction{
								Command: []string{"echo", "healthy"},
							},
						},
						InitialDelaySeconds: 1,
						PeriodSeconds:       2,
					},
				}},
			},
		}

		ginkgo.By("Creating hermetic pod with Exec liveness probe")
		createdPod := podClient.Create(ctx, pod)

		ginkgo.By("Waiting for pod to be running and healthy")
		err := e2epod.WaitForPodNameRunningInNamespace(ctx, cs, createdPod.Name, createdPod.Namespace)
		framework.ExpectNoError(err, "pod failed to reach Running state")

		ginkgo.By("Verifying container restart count remains 0")
		time.Sleep(5 * time.Second)
		fetchedPod, err := podClient.Get(ctx, createdPod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to get pod")
		gomega.Expect(fetchedPod.Status.ContainerStatuses).To(gomega.HaveLen(1))
		gomega.Expect(fetchedPod.Status.ContainerStatuses[0].RestartCount).To(gomega.Equal(int32(0)))
	})

	ginkgo.It("should resolve local hostname to loopback in /etc/hosts, omit service env vars, and handle downward API", func(ctx context.Context) {
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "hermetic-env-hosts",
			},
			Spec: v1.PodSpec{
				Hermetic: ptr.To(true),
				Containers: []v1.Container{{
					Name:    "agnhost",
					Image:   imageutils.GetE2EImage(imageutils.Agnhost),
					Command: []string{"sleep", "3600"},
					Env: []v1.EnvVar{
						{
							Name: "POD_IP",
							ValueFrom: &v1.EnvVarSource{
								FieldRef: &v1.ObjectFieldSelector{
									FieldPath: "status.podIP",
								},
							},
						},
						{
							Name: "HOST_IP",
							ValueFrom: &v1.EnvVarSource{
								FieldRef: &v1.ObjectFieldSelector{
									FieldPath: "status.hostIP",
								},
							},
						},
					},
				}},
			},
		}

		ginkgo.By("Creating hermetic pod with downward API")
		createdPod := podClient.Create(ctx, pod)

		ginkgo.By("Waiting for pod to be running")
		err := e2epod.WaitForPodNameRunningInNamespace(ctx, cs, createdPod.Name, createdPod.Namespace)
		framework.ExpectNoError(err, "pod failed to reach Running state")

		ginkgo.By("Verifying /etc/hosts maps the pod hostname to loopback")
		stdout := e2epod.ExecShellInPod(ctx, f, createdPod.Name, "cat /etc/hosts")
		gomega.Expect(stdout).To(gomega.ContainSubstring("127.0.0.1\thermetic-env-hosts"), "expected loopback entry for hostname in /etc/hosts")

		ginkgo.By("Verifying ping $(hostname) succeeds inside container")
		stdout = e2epod.ExecShellInPod(ctx, f, createdPod.Name, "ping -c 1 -W 2 $(hostname)")
		gomega.Expect(stdout).To(gomega.ContainSubstring("1 packets transmitted, 1 packets received"), "hostname loopback ping failed")

		ginkgo.By("Verifying environment variables omit service env vars and Downward API POD_IP is empty")
		stdout = e2epod.ExecShellInPod(ctx, f, createdPod.Name, "env")
		gomega.Expect(stdout).NotTo(gomega.ContainSubstring("KUBERNETES_SERVICE_HOST"), "service env vars should be omitted for hermetic pods")
		gomega.Expect(stdout).To(gomega.ContainSubstring("POD_IP=\n"), "POD_IP should be empty")
		gomega.Expect(stdout).To(gomega.MatchRegexp(`HOST_IP=\d+\.\d+\.\d+\.\d+`), "HOST_IP should be populated")
	})

	ginkgo.It("should manage hermetic workloads in Deployments, StatefulSets, and Jobs", func(ctx context.Context) {
		ginkgo.By("Creating a hermetic Deployment")
		var replicas int32 = 2
		deploy := &appsv1.Deployment{
			ObjectMeta: metav1.ObjectMeta{
				Name: "hermetic-deploy",
			},
			Spec: appsv1.DeploymentSpec{
				Replicas: &replicas,
				Selector: &metav1.LabelSelector{
					MatchLabels: map[string]string{"app": "hermetic-deploy"},
				},
				Template: v1.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{
						Labels: map[string]string{"app": "hermetic-deploy"},
					},
					Spec: v1.PodSpec{
						Hermetic: ptr.To(true),
						Containers: []v1.Container{{
							Name:    "agnhost",
							Image:   imageutils.GetE2EImage(imageutils.Agnhost),
							Command: []string{"sleep", "3600"},
						}},
					},
				},
			},
		}
		_, err := cs.AppsV1().Deployments(f.Namespace.Name).Create(ctx, deploy, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create hermetic deployment")
		_, err = e2epod.WaitForPodsWithLabelRunningReady(ctx, cs, f.Namespace.Name, labels.SelectorFromSet(map[string]string{"app": "hermetic-deploy"}), 2, 1*time.Minute)
		framework.ExpectNoError(err, "hermetic deployment pods failed to become ready")

		ginkgo.By("Creating a hermetic StatefulSet")
		sts := &appsv1.StatefulSet{
			ObjectMeta: metav1.ObjectMeta{
				Name: "hermetic-sts",
			},
			Spec: appsv1.StatefulSetSpec{
				Replicas:    &replicas,
				ServiceName: "hermetic-sts-headless",
				Selector: &metav1.LabelSelector{
					MatchLabels: map[string]string{"app": "hermetic-sts"},
				},
				Template: v1.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{
						Labels: map[string]string{"app": "hermetic-sts"},
					},
					Spec: v1.PodSpec{
						Hermetic: ptr.To(true),
						Containers: []v1.Container{{
							Name:    "agnhost",
							Image:   imageutils.GetE2EImage(imageutils.Agnhost),
							Command: []string{"sleep", "3600"},
						}},
					},
				},
			},
		}
		_, err = cs.AppsV1().StatefulSets(f.Namespace.Name).Create(ctx, sts, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create hermetic statefulset")
		_, err = e2epod.WaitForPodsWithLabelRunningReady(ctx, cs, f.Namespace.Name, labels.SelectorFromSet(map[string]string{"app": "hermetic-sts"}), 2, 1*time.Minute)
		framework.ExpectNoError(err, "hermetic statefulset pods failed to become ready")

		ginkgo.By("Creating a hermetic Job")
		job := &batchv1.Job{
			ObjectMeta: metav1.ObjectMeta{
				Name: "hermetic-job",
			},
			Spec: batchv1.JobSpec{
				Template: v1.PodTemplateSpec{
					Spec: v1.PodSpec{
						Hermetic:      ptr.To(true),
						RestartPolicy: v1.RestartPolicyNever,
						Containers: []v1.Container{{
							Name:    "agnhost",
							Image:   imageutils.GetE2EImage(imageutils.Agnhost),
							Command: []string{"sh", "-c", "echo 'hermetic job running' && ping -c 1 127.0.0.1"},
						}},
					},
				},
			},
		}
		_, err = cs.BatchV1().Jobs(f.Namespace.Name).Create(ctx, job, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create hermetic job")
		err = wait.PollUntilContextTimeout(ctx, 1*time.Second, 1*time.Minute, true, func(ctx context.Context) (bool, error) {
			j, err := cs.BatchV1().Jobs(f.Namespace.Name).Get(ctx, "hermetic-job", metav1.GetOptions{})
			if err != nil {
				return false, err
			}
			return j.Status.Succeeded > 0, nil
		})
		framework.ExpectNoError(err, "hermetic job failed to complete")
	})
})

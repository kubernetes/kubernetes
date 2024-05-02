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
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubectl/pkg/cmd/util/podcmd"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ekubectl "k8s.io/kubernetes/test/e2e/framework/kubectl"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eoutput "k8s.io/kubernetes/test/e2e/framework/pod/output"
	e2estatefulset "k8s.io/kubernetes/test/e2e/framework/statefulset"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

func testingStatefulSet(name, ns string, numberOfPods int32) appsv1.StatefulSet {
	return appsv1.StatefulSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: ns,
			Labels: map[string]string{
				"name": name,
			},
		},
		Spec: appsv1.StatefulSetSpec{
			Replicas: &numberOfPods,
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{
					"name": name,
				},
			},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"name": name,
					},
					Annotations: map[string]string{
						podcmd.DefaultContainerAnnotationName: "container-2",
					},
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "container-1",
							Image: imageutils.GetE2EImage(imageutils.Agnhost),
							Args:  []string{"logs-generator", "--log-lines-total", "10", "--run-duration", "5s"},
						},
						{
							Name:  "container-2",
							Image: imageutils.GetE2EImage(imageutils.Agnhost),
							Args:  []string{"logs-generator", "--log-lines-total", "20", "--run-duration", "5s"},
						},
					},
					RestartPolicy: v1.RestartPolicyAlways,
				},
			},
		},
	}
}
func testingPod(name, value, defaultContainerName string) v1.Pod {
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
					Image: imageutils.GetE2EImage(imageutils.Agnhost),
					Args:  []string{"logs-generator", "--log-lines-total", "10", "--run-duration", "5s"},
				},
				{
					Name:  defaultContainerName,
					Image: imageutils.GetE2EImage(imageutils.Agnhost),
					Args:  []string{"logs-generator", "--log-lines-total", "20", "--run-duration", "5s"},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
		},
	}
}

var _ = SIGDescribe("Kubectl logs", func() {
	f := framework.NewDefaultFramework("kubectl-logs")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline
	defer ginkgo.GinkgoRecover()

	var c clientset.Interface
	var ns string
	ginkgo.BeforeEach(func() {
		c = f.ClientSet
		ns = f.Namespace.Name
	})

	// Split("something\n", "\n") returns ["something", ""], so
	// strip trailing newline first
	lines := func(out string) []string {
		return strings.Split(strings.TrimRight(out, "\n"), "\n")
	}

	ginkgo.Describe("logs", func() {

		podName := "logs-generator"
		containerName := "logs-generator"
		ginkgo.BeforeEach(func() {
			ginkgo.By("creating a pod")
			// Agnhost image generates logs for a total of 100 lines over 20s.
			e2ekubectl.RunKubectlOrDie(ns, "run", podName, "--image="+imageutils.GetE2EImage(imageutils.Agnhost), "--restart=Never", podRunningTimeoutArg, "--", "logs-generator", "--log-lines-total", "100", "--run-duration", "20s")
		})
		ginkgo.AfterEach(func() {
			e2ekubectl.RunKubectlOrDie(ns, "delete", "pod", podName)
		})

		/*
			Release: v1.9
			Testname: Kubectl, logs
			Description: When a Pod is running then it MUST generate logs.
			Starting a Pod should have a expected log line. Also log command options MUST work as expected and described below.
				'kubectl logs -tail=1' should generate a output of one line, the last line in the log.
				'kubectl --limit-bytes=1' should generate a single byte output.
				'kubectl --tail=1 --timestamp should generate one line with timestamp in RFC3339 format
				'kubectl --since=1s' should output logs that are only 1 second older from now
				'kubectl --since=24h' should output logs that are only 1 day older from now
		*/
		framework.ConformanceIt("should be able to retrieve and filter logs", func(ctx context.Context) {

			ginkgo.By("Waiting for log generator to start.")
			if !e2epod.CheckPodsRunningReadyOrSucceeded(ctx, c, ns, []string{podName}, framework.PodStartTimeout) {
				framework.Failf("Pod %s was not ready", podName)
			}

			ginkgo.By("checking for a matching strings")
			_, err := e2eoutput.LookForStringInLog(ns, podName, containerName, "/api/v1/namespaces/kube-system", framework.PodStartTimeout)
			framework.ExpectNoError(err)

			ginkgo.By("limiting log lines")
			out := e2ekubectl.RunKubectlOrDie(ns, "logs", podName, containerName, "--tail=1")
			framework.Logf("got output %q", out)
			gomega.Expect(out).NotTo(gomega.BeEmpty())
			gomega.Expect(lines(out)).To(gomega.HaveLen(1))

			ginkgo.By("limiting log bytes")
			out = e2ekubectl.RunKubectlOrDie(ns, "logs", podName, containerName, "--limit-bytes=1")
			framework.Logf("got output %q", out)
			gomega.Expect(lines(out)).To(gomega.HaveLen(1))
			gomega.Expect(out).To(gomega.HaveLen(1))

			ginkgo.By("exposing timestamps")
			out = e2ekubectl.RunKubectlOrDie(ns, "logs", podName, containerName, "--tail=1", "--timestamps")
			framework.Logf("got output %q", out)
			l := lines(out)
			gomega.Expect(l).To(gomega.HaveLen(1))
			words := strings.Split(l[0], " ")
			gomega.Expect(len(words)).To(gomega.BeNumerically(">", 1))
			if _, err := time.Parse(time.RFC3339Nano, words[0]); err != nil {
				if _, err := time.Parse(time.RFC3339, words[0]); err != nil {
					framework.Failf("expected %q to be RFC3339 or RFC3339Nano", words[0])
				}
			}

			ginkgo.By("restricting to a time range")
			// Note: we must wait at least two seconds,
			// because the granularity is only 1 second and
			// it could end up rounding the wrong way.
			time.Sleep(2500 * time.Millisecond) // ensure that startup logs on the node are seen as older than 1s
			recentOut := e2ekubectl.RunKubectlOrDie(ns, "logs", podName, containerName, "--since=1s")
			recent := len(strings.Split(recentOut, "\n"))
			olderOut := e2ekubectl.RunKubectlOrDie(ns, "logs", podName, containerName, "--since=24h")
			older := len(strings.Split(olderOut, "\n"))
			gomega.Expect(recent).To(gomega.BeNumerically("<", older), "expected recent(%v) to be less than older(%v)\nrecent lines:\n%v\nolder lines:\n%v\n", recent, older, recentOut, olderOut)
		})
	})

	ginkgo.Describe("default container logs", func() {
		ginkgo.Describe("the second container is the default-container by annotation", func() {
			var pod *v1.Pod
			podName := "pod" + string(uuid.NewUUID())
			defaultContainerName := "container-2"
			ginkgo.BeforeEach(func(ctx context.Context) {
				podClient := f.ClientSet.CoreV1().Pods(ns)
				ginkgo.By("constructing the pod")
				value := strconv.Itoa(time.Now().Nanosecond())
				podCopy := testingPod(podName, value, defaultContainerName)
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

			ginkgo.It("should log default container if not specified", func(ctx context.Context) {
				ginkgo.By("Waiting for log generator to start.")
				// we need to wait for pod completion, to check the generated number of lines
				if err := e2epod.WaitForPodSuccessInNamespaceTimeout(ctx, c, podName, ns, framework.PodStartTimeout); err != nil {
					framework.Failf("Pod %s did not finish: %v", podName, err)
				}

				ginkgo.By("specified container log lines")
				out := e2ekubectl.RunKubectlOrDie(ns, "logs", podName, "-c", "container-1")
				framework.Logf("got output %q", out)
				gomega.Expect(out).NotTo(gomega.BeEmpty())
				gomega.Expect(lines(out)).To(gomega.HaveLen(10))

				ginkgo.By("log all containers log lines")
				out = e2ekubectl.RunKubectlOrDie(ns, "logs", podName, "--all-containers")
				framework.Logf("got output %q", out)
				gomega.Expect(out).NotTo(gomega.BeEmpty())
				gomega.Expect(lines(out)).To(gomega.HaveLen(30))

				ginkgo.By("default container logs")
				out = e2ekubectl.RunKubectlOrDie(ns, "logs", podName)
				framework.Logf("got output %q", out)
				gomega.Expect(lines(out)).To(gomega.HaveLen(20))
			})
		})
	})

	ginkgo.Describe("all pod logs", func() {
		ginkgo.Describe("the StatefulSet has 2 replicas and each pod has 2 containers", func() {
			var sts *appsv1.StatefulSet
			stsName := "sts" + string(uuid.NewUUID())
			numberReplicas := int32(2)
			ginkgo.BeforeEach(func(ctx context.Context) {
				stsClient := c.AppsV1().StatefulSets(ns)
				ginkgo.By("constructing the StatefulSet")
				stsCopy := testingStatefulSet(stsName, ns, numberReplicas)
				sts = &stsCopy
				ginkgo.By("creating the StatefulSet")

				_, err := stsClient.Create(ctx, sts, metav1.CreateOptions{})
				if err != nil {
					framework.Failf("Failed to create StatefulSet: %v", err)
				}
			})

			ginkgo.AfterEach(func() {
				e2ekubectl.RunKubectlOrDie(ns, "delete", "sts", stsName)
			})

			ginkgo.It("should get logs from all pods", func(ctx context.Context) {
				ginkgo.By("Waiting for StatefulSet pods to be running.")
				e2estatefulset.WaitForStatusAvailableReplicas(ctx, c, sts, int32(numberReplicas))

				ginkgo.By("default container for each pod")
				out := e2ekubectl.RunKubectlOrDie(ns, "logs", fmt.Sprintf("sts/%s", stsName), "--all-pods")
				framework.Logf("got output %q", out)
				gomega.Expect(out).NotTo(gomega.BeEmpty())
				gomega.Expect(out).To(gomega.ContainSubstring("container-2"))
				gomega.Expect(out).NotTo(gomega.ContainSubstring("container-1"))

				ginkgo.By("all containers for each pod")
				out = e2ekubectl.RunKubectlOrDie(ns, "logs", fmt.Sprintf("sts/%s", stsName), "--all-pods", "--all-containers")
				framework.Logf("got output %q", out)
				gomega.Expect(out).NotTo(gomega.BeEmpty())
				gomega.Expect(out).To(gomega.ContainSubstring("container-2"))
				gomega.Expect(out).To(gomega.ContainSubstring("container-1"))

			})

		})
	})

})

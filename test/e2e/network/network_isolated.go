/*
Copyright The Kubernetes Authors.

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
	"k8s.io/kubernetes/pkg/features"
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

// isolatedPod returns a pod with spec.defaultNetwork "None" that sleeps forever.
func isolatedPod(name string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: v1.PodSpec{
			DefaultNetwork: ptr.To(v1.PodDefaultNetworkNone),
			Containers: []v1.Container{{
				Name:    "agnhost",
				Image:   imageutils.GetE2EImage(imageutils.Agnhost),
				Command: []string{"sleep", "3600"},
			}},
		},
	}
}

// The tests require a container runtime that reports the default_network_none
// capability; otherwise the kubelet rejects the pods with PodFeatureUnsupported.
var _ = common.SIGDescribe("Network Isolated Pods", feature.PodDefaultNetwork, framework.WithFeatureGate(features.PodDefaultNetwork), func() {
	f := framework.NewDefaultFramework("network-isolated")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	var cs clientset.Interface
	var podClient *e2epod.PodClient

	ginkgo.BeforeEach(func() {
		cs = f.ClientSet
		podClient = e2epod.NewPodClient(f)
	})

	ginkgo.It("should run and become ready without a pod IP", func(ctx context.Context) {
		ginkgo.By("Creating a network isolated pod")
		pod := podClient.CreateSync(ctx, isolatedPod("isolated-no-ip"))

		ginkgo.By("Verifying the pod is ready and has no pod IP")
		gomega.Expect(pod.Status.Phase).To(gomega.Equal(v1.PodRunning))
		gomega.Expect(pod.Status.PodIP).To(gomega.BeEmpty(), "status.podIP must be empty")
		gomega.Expect(pod.Status.PodIPs).To(gomega.BeEmpty(), "status.podIPs must be empty")
		gomega.Expect(pod.Status.HostIP).NotTo(gomega.BeEmpty(), "status.hostIP must be reported")
		gomega.Expect(pod.Spec.DNSPolicy).To(gomega.Equal(v1.DNSNone), "dnsPolicy must default to None")
		gomega.Expect(pod.Spec.EnableServiceLinks).To(gomega.Equal(ptr.To(false)), "enableServiceLinks must default to false")
	})

	ginkgo.It("should only have a loopback interface in the sandbox", func(ctx context.Context) {
		ginkgo.By("Creating a network isolated pod")
		pod := podClient.CreateSync(ctx, isolatedPod("isolated-netns"))

		ginkgo.By("Listing the network interfaces inside the container")
		stdout := e2epod.ExecShellInPod(ctx, f, pod.Name, "ip -o link show")
		for _, line := range strings.Split(strings.TrimSpace(stdout), "\n") {
			line = strings.TrimSpace(line)
			if line == "" {
				continue
			}
			gomega.Expect(line).To(gomega.ContainSubstring(" lo: "), "unexpected network interface in isolated sandbox: %s", line)
		}

		ginkgo.By("Verifying loopback connectivity")
		stdout = e2epod.ExecShellInPod(ctx, f, pod.Name, "ping -c 1 -W 2 127.0.0.1")
		gomega.Expect(stdout).To(gomega.ContainSubstring("1 packets transmitted, 1 packets received"))
	})

	ginkgo.It("should resolve its hostname to loopback and expose an empty status.podIP through the downward API", func(ctx context.Context) {
		pod := isolatedPod("isolated-env-hosts")
		pod.Spec.Containers[0].Env = []v1.EnvVar{
			{Name: "POD_IP", ValueFrom: &v1.EnvVarSource{FieldRef: &v1.ObjectFieldSelector{FieldPath: "status.podIP"}}},
			{Name: "HOST_IP", ValueFrom: &v1.EnvVarSource{FieldRef: &v1.ObjectFieldSelector{FieldPath: "status.hostIP"}}},
		}

		ginkgo.By("Creating a network isolated pod with downward API environment variables")
		pod = podClient.CreateSync(ctx, pod)

		ginkgo.By("Verifying /etc/hosts maps the hostname to loopback")
		hosts := e2epod.ExecShellInPod(ctx, f, pod.Name, "cat /etc/hosts")
		gomega.Expect(hosts).To(gomega.ContainSubstring("127.0.0.1\t" + pod.Name))
		gomega.Expect(hosts).To(gomega.ContainSubstring("::1\t" + pod.Name))

		ginkgo.By("Verifying $(hostname) resolves")
		stdout := e2epod.ExecShellInPod(ctx, f, pod.Name, "ping -c 1 -W 2 $(hostname)")
		gomega.Expect(stdout).To(gomega.ContainSubstring("1 packets transmitted, 1 packets received"))

		ginkgo.By("Verifying the environment")
		env := e2epod.ExecShellInPod(ctx, f, pod.Name, "env")
		gomega.Expect(env).NotTo(gomega.ContainSubstring("KUBERNETES_SERVICE_HOST"), "service environment variables must not be injected by default")
		gomega.Expect(env).To(gomega.MatchRegexp(`(?m)^POD_IP=$`), "status.podIP must resolve to an empty value")
		gomega.Expect(env).To(gomega.MatchRegexp(`(?m)^HOST_IP=\S+$`), "status.hostIP must be populated")
	})

	ginkgo.It("should inject service environment variables when enableServiceLinks is true", func(ctx context.Context) {
		pod := isolatedPod("isolated-service-links")
		pod.Spec.EnableServiceLinks = ptr.To(true)

		ginkgo.By("Creating a network isolated pod with enableServiceLinks: true")
		pod = podClient.CreateSync(ctx, pod)

		ginkgo.By("Verifying the environment")
		env := e2epod.ExecShellInPod(ctx, f, pod.Name, "env")
		gomega.Expect(env).To(gomega.ContainSubstring("KUBERNETES_SERVICE_HOST="), "the master service variables must be injected when asked for")
	})

	ginkgo.It("should support exec probes", func(ctx context.Context) {
		pod := isolatedPod("isolated-exec-probe")
		pod.Spec.Containers[0].LivenessProbe = &v1.Probe{
			ProbeHandler:        v1.ProbeHandler{Exec: &v1.ExecAction{Command: []string{"true"}}},
			InitialDelaySeconds: 1,
			PeriodSeconds:       1,
		}
		pod.Spec.Containers[0].ReadinessProbe = &v1.Probe{
			ProbeHandler:  v1.ProbeHandler{Exec: &v1.ExecAction{Command: []string{"true"}}},
			PeriodSeconds: 1,
		}

		ginkgo.By("Creating a network isolated pod with exec probes")
		pod = podClient.CreateSync(ctx, pod)

		ginkgo.By("Verifying the container is not restarted by the liveness probe")
		gomega.Consistently(ctx, func(ctx context.Context) (int32, error) {
			p, err := podClient.Get(ctx, pod.Name, metav1.GetOptions{})
			if err != nil {
				return 0, err
			}
			return p.Status.ContainerStatuses[0].RestartCount, nil
		}).WithTimeout(10 * time.Second).WithPolling(time.Second).Should(gomega.BeZero())
	})

	ginkgo.It("should not be selected into Endpoints or EndpointSlices", func(ctx context.Context) {
		podLabels := map[string]string{"app": "isolated-service-member"}
		pod := isolatedPod("isolated-service-member")
		pod.Labels = podLabels
		pod.Spec.Containers[0].Ports = []v1.ContainerPort{{Name: "http", ContainerPort: 8080, Protocol: v1.ProtocolTCP}}

		ginkgo.By("Creating a network isolated pod that matches a Service selector")
		pod = podClient.CreateSync(ctx, pod)

		ginkgo.By("Creating the Service")
		svc, err := cs.CoreV1().Services(f.Namespace.Name).Create(ctx, &v1.Service{
			ObjectMeta: metav1.ObjectMeta{Name: "isolated-svc"},
			Spec: v1.ServiceSpec{
				Selector: podLabels,
				Ports:    []v1.ServicePort{{Name: "http", Port: 80, TargetPort: intstr.FromInt32(8080), Protocol: v1.ProtocolTCP}},
			},
		}, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create Service")

		ginkgo.By("Waiting for the EndpointSlice and verifying it has no endpoints")
		err = wait.PollUntilContextTimeout(ctx, time.Second, 30*time.Second, true, func(ctx context.Context) (bool, error) {
			slices, err := cs.DiscoveryV1().EndpointSlices(svc.Namespace).List(ctx, metav1.ListOptions{
				LabelSelector: discoveryv1.LabelServiceName + "=" + svc.Name,
			})
			if err != nil {
				return false, err
			}
			if len(slices.Items) == 0 {
				return false, nil
			}
			for _, slice := range slices.Items {
				if len(slice.Endpoints) > 0 {
					return false, nil
				}
			}
			return true, nil
		})
		framework.ExpectNoError(err, "expected an EndpointSlice without endpoints for the isolated pod")

		// The controllers never publish the pod, so the slice must stay empty.
		gomega.Consistently(ctx, func(ctx context.Context) ([]discoveryv1.Endpoint, error) {
			slices, err := cs.DiscoveryV1().EndpointSlices(svc.Namespace).List(ctx, metav1.ListOptions{
				LabelSelector: discoveryv1.LabelServiceName + "=" + svc.Name,
			})
			if err != nil {
				return nil, err
			}
			var endpoints []discoveryv1.Endpoint
			for _, slice := range slices.Items {
				endpoints = append(endpoints, slice.Endpoints...)
			}
			return endpoints, nil
		}).WithTimeout(10*time.Second).WithPolling(time.Second).Should(gomega.BeEmpty(), "isolated pod %s must not be published as an endpoint", pod.Name)

		endpoints, err := cs.CoreV1().Endpoints(svc.Namespace).Get(ctx, svc.Name, metav1.GetOptions{})
		if err == nil {
			for _, subset := range endpoints.Subsets {
				gomega.Expect(subset.Addresses).To(gomega.BeEmpty(), "isolated pod must not be an Endpoints address")
				gomega.Expect(subset.NotReadyAddresses).To(gomega.BeEmpty(), "isolated pod must not be an Endpoints notReadyAddress")
			}
		}
	})

	ginkgo.It("should run in Deployments, StatefulSets and Jobs", func(ctx context.Context) {
		template := func(app string) v1.PodTemplateSpec {
			p := isolatedPod("")
			return v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"app": app}},
				Spec:       p.Spec,
			}
		}

		ginkgo.By("Creating a Deployment with network isolated pods")
		_, err := cs.AppsV1().Deployments(f.Namespace.Name).Create(ctx, &appsv1.Deployment{
			ObjectMeta: metav1.ObjectMeta{Name: "isolated-deploy"},
			Spec: appsv1.DeploymentSpec{
				Replicas: ptr.To[int32](2),
				Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": "isolated-deploy"}},
				Template: template("isolated-deploy"),
			},
		}, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create Deployment")
		_, err = e2epod.WaitForPodsWithLabelRunningReady(ctx, cs, f.Namespace.Name, labels.SelectorFromSet(map[string]string{"app": "isolated-deploy"}), 2, framework.PodStartTimeout)
		framework.ExpectNoError(err, "Deployment pods did not become ready")

		ginkgo.By("Creating a StatefulSet with network isolated pods")
		_, err = cs.AppsV1().StatefulSets(f.Namespace.Name).Create(ctx, &appsv1.StatefulSet{
			ObjectMeta: metav1.ObjectMeta{Name: "isolated-sts"},
			Spec: appsv1.StatefulSetSpec{
				Replicas:    ptr.To[int32](2),
				ServiceName: "isolated-sts",
				Selector:    &metav1.LabelSelector{MatchLabels: map[string]string{"app": "isolated-sts"}},
				Template:    template("isolated-sts"),
			},
		}, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create StatefulSet")
		_, err = e2epod.WaitForPodsWithLabelRunningReady(ctx, cs, f.Namespace.Name, labels.SelectorFromSet(map[string]string{"app": "isolated-sts"}), 2, framework.PodStartTimeout)
		framework.ExpectNoError(err, "StatefulSet pods did not become ready")

		ginkgo.By("Creating a Job with a network isolated pod")
		jobTemplate := template("isolated-job")
		jobTemplate.Spec.RestartPolicy = v1.RestartPolicyNever
		jobTemplate.Spec.Containers[0].Command = []string{"sh", "-c", "ping -c 1 -W 2 127.0.0.1"}
		_, err = cs.BatchV1().Jobs(f.Namespace.Name).Create(ctx, &batchv1.Job{
			ObjectMeta: metav1.ObjectMeta{Name: "isolated-job"},
			Spec:       batchv1.JobSpec{Template: jobTemplate},
		}, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create Job")
		err = wait.PollUntilContextTimeout(ctx, time.Second, framework.PodStartTimeout, true, func(ctx context.Context) (bool, error) {
			job, err := cs.BatchV1().Jobs(f.Namespace.Name).Get(ctx, "isolated-job", metav1.GetOptions{})
			if err != nil {
				return false, err
			}
			return job.Status.Succeeded > 0, nil
		})
		framework.ExpectNoError(err, "Job did not complete")
	})
})

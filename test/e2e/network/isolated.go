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
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/portforward"
	"k8s.io/client-go/transport/spdy"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eendpointslice "k8s.io/kubernetes/test/e2e/framework/endpointslice"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/network/common"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

const isolatedServerPort = "8080"

// isolatedPod returns a pod with spec.defaultNetwork "None" that serves HTTP
// on loopback and becomes Ready once the server accepts connections.
func isolatedPod(name string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: v1.PodSpec{
			DefaultNetwork: new(v1.PodDefaultNetworkNone),
			Containers: []v1.Container{{
				Name:  "agnhost",
				Image: imageutils.GetE2EImage(imageutils.Agnhost),
				Args:  []string{"netexec", "--http-port=" + isolatedServerPort},
				Ports: []v1.ContainerPort{{Name: "http", ContainerPort: 8080, Protocol: v1.ProtocolTCP}},
				ReadinessProbe: &v1.Probe{
					ProbeHandler:  v1.ProbeHandler{Exec: &v1.ExecAction{Command: []string{"/agnhost", "connect", "--timeout=2s", "127.0.0.1:" + isolatedServerPort}}},
					PeriodSeconds: 1,
				},
			}},
		},
	}
}

// createIsolatedServiceMember creates a network isolated pod, a regular pod
// as a positive control, and a Service whose selector matches both.
func createIsolatedServiceMember(ctx context.Context, f *framework.Framework, name string) (isolated, control *v1.Pod, svc *v1.Service) {
	podLabels := map[string]string{"app": name}
	podClient := e2epod.NewPodClient(f)

	ginkgo.By("Creating a network isolated pod that matches a Service selector")
	isolated = isolatedPod(name)
	isolated.Labels = podLabels
	isolated = podClient.CreateSync(ctx, isolated)

	ginkgo.By("Creating a regular pod that matches the same selector")
	control = isolatedPod(name + "-control")
	control.Labels = podLabels
	control.Spec.DefaultNetwork = nil
	control = podClient.CreateSync(ctx, control)

	ginkgo.By("Creating the Service")
	svc, err := f.ClientSet.CoreV1().Services(f.Namespace.Name).Create(ctx, &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: v1.ServiceSpec{
			Selector: podLabels,
			Ports:    []v1.ServicePort{{Name: "http", Port: 80, TargetPort: intstr.FromInt32(8080), Protocol: v1.ProtocolTCP}},
		},
	}, metav1.CreateOptions{})
	framework.ExpectNoError(err, "failed to create Service")
	return isolated, control, svc
}

// The tests require a container runtime that reports the default_network_none
// capability; otherwise the kubelet rejects the pods with PodFeatureUnsupported.
var _ = common.SIGDescribe("Network Isolated Pods", feature.NetworkIsolatedPodsSupport, framework.WithFeatureGate(features.NetworkIsolatedPods), func() {
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
		gomega.Expect(pod.Spec.EnableServiceLinks).To(gomega.Equal(new(false)), "enableServiceLinks must default to false")
	})

	ginkgo.It("should only have a loopback interface in the sandbox [LinuxOnly]", func(ctx context.Context) {
		ginkgo.By("Creating a network isolated pod")
		pod := podClient.CreateSync(ctx, isolatedPod("isolated-netns"))

		ginkgo.By("Listing the network interfaces inside the container")
		stdout := e2epod.ExecShellInPod(ctx, f, pod.Name, "ip -o link show")
		for line := range strings.SplitSeq(strings.TrimSpace(stdout), "\n") {
			line = strings.TrimSpace(line)
			if line == "" {
				continue
			}
			gomega.Expect(line).To(gomega.ContainSubstring(" lo: "), "unexpected network interface in isolated sandbox: %s", line)
		}

	})

	ginkgo.It("should accept connections on loopback", func(ctx context.Context) {
		ginkgo.By("Creating a network isolated pod that serves on loopback")
		pod := podClient.CreateSync(ctx, isolatedPod("isolated-loopback-server"))

		// The readiness probe already dials it; this makes the check explicit.
		ginkgo.By("Connecting to the server over loopback from inside the pod")
		e2epod.ExecCommandInContainer(f, pod.Name, pod.Spec.Containers[0].Name, "/agnhost", "connect", "--timeout=5s", "127.0.0.1:"+isolatedServerPort)
	})

	// Port forwarding is the only network path into an isolated pod from
	// outside the sandbox: the runtime dials localhost inside the sandbox
	// network namespace, so it does not need a pod IP.
	ginkgo.It("should support port forwarding", func(ctx context.Context) {
		ginkgo.By("Creating a network isolated pod that serves on loopback")
		pod := podClient.CreateSync(ctx, isolatedPod("isolated-port-forward"))

		ginkgo.By("Starting a port forward to the pod")
		req := cs.CoreV1().RESTClient().Post().Resource("pods").Namespace(pod.Namespace).Name(pod.Name).SubResource("portforward")
		transport, upgrader, err := spdy.RoundTripperFor(f.ClientConfig())
		framework.ExpectNoError(err, "failed to create SPDY round tripper")
		dialer := spdy.NewDialer(upgrader, &http.Client{Transport: transport}, http.MethodPost, req.URL())
		stopCh, readyCh := make(chan struct{}), make(chan struct{})
		defer close(stopCh)
		fw, err := portforward.New(dialer, []string{"0:" + isolatedServerPort}, stopCh, readyCh, io.Discard, io.Discard)
		framework.ExpectNoError(err, "failed to create port forwarder")
		errCh := make(chan error, 1)
		go func() { errCh <- fw.ForwardPorts() }()
		select {
		case <-readyCh:
		case err := <-errCh:
			framework.Failf("port forward failed before becoming ready: %v", err)
		case <-time.After(framework.PodStartShortTimeout):
			framework.Failf("timed out waiting for the port forward to become ready")
		}
		ports, err := fw.GetPorts()
		framework.ExpectNoError(err, "failed to get forwarded ports")
		gomega.Expect(ports).To(gomega.HaveLen(1))

		ginkgo.By("Requesting the hostname through the forwarded port")
		resp, err := http.Get(fmt.Sprintf("http://127.0.0.1:%d/hostname", ports[0].Local))
		framework.ExpectNoError(err, "failed to connect through the port forward")
		defer func() { _ = resp.Body.Close() }()
		body, err := io.ReadAll(resp.Body)
		framework.ExpectNoError(err, "failed to read the response")
		gomega.Expect(resp.StatusCode).To(gomega.Equal(http.StatusOK))
		gomega.Expect(strings.TrimSpace(string(body))).To(gomega.Equal(pod.Name))
	})

	ginkgo.It("should resolve its hostname and hostAliases to loopback, get an empty resolv.conf and expose an empty status.podIP through the downward API", func(ctx context.Context) {
		pod := isolatedPod("isolated-env-hosts")
		pod.Spec.Containers[0].Env = []v1.EnvVar{
			{Name: "POD_IP", ValueFrom: &v1.EnvVarSource{FieldRef: &v1.ObjectFieldSelector{FieldPath: "status.podIP"}}},
			{Name: "HOST_IP", ValueFrom: &v1.EnvVarSource{FieldRef: &v1.ObjectFieldSelector{FieldPath: "status.hostIP"}}},
		}
		pod.Spec.HostAliases = []v1.HostAlias{{IP: "127.0.0.2", Hostnames: []string{"alias.isolated.local"}}}

		ginkgo.By("Creating a network isolated pod with downward API environment variables and hostAliases")
		pod = podClient.CreateSync(ctx, pod)

		ginkgo.By("Verifying /etc/hosts maps the hostname to loopback and contains the hostAliases")
		hosts := e2epod.ExecShellInPod(ctx, f, pod.Name, "cat /etc/hosts")
		gomega.Expect(hosts).To(gomega.ContainSubstring("127.0.0.1\t" + pod.Name))
		gomega.Expect(hosts).To(gomega.ContainSubstring("::1\t" + pod.Name))
		gomega.Expect(hosts).To(gomega.ContainSubstring("127.0.0.2\talias.isolated.local"))

		ginkgo.By("Verifying $(hostname) and the alias resolve to the server on loopback")
		e2epod.ExecShellInPod(ctx, f, pod.Name, "/agnhost connect --timeout=5s $(hostname):"+isolatedServerPort)
		e2epod.ExecShellInPod(ctx, f, pod.Name, "/agnhost connect --timeout=5s alias.isolated.local:"+isolatedServerPort)

		ginkgo.By("Verifying /etc/resolv.conf is empty with the defaulted dnsPolicy None")
		resolv := e2epod.ExecShellInPod(ctx, f, pod.Name, "cat /etc/resolv.conf")
		gomega.Expect(strings.TrimSpace(resolv)).To(gomega.BeEmpty(), "resolv.conf must not fall back to the host configuration")

		ginkgo.By("Verifying the environment")
		env := e2epod.ExecShellInPod(ctx, f, pod.Name, "env")
		gomega.Expect(env).NotTo(gomega.ContainSubstring("KUBERNETES_SERVICE_HOST"), "service environment variables must not be injected by default")
		gomega.Expect(env).To(gomega.MatchRegexp(`(?m)^POD_IP=$`), "status.podIP must resolve to an empty value")
		gomega.Expect(env).To(gomega.MatchRegexp(`(?m)^HOST_IP=\S+$`), "status.hostIP must be populated")
	})

	ginkgo.It("should inject service environment variables when enableServiceLinks is true", func(ctx context.Context) {
		pod := isolatedPod("isolated-service-links")
		pod.Spec.EnableServiceLinks = new(true)

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

	ginkgo.It("should not be selected into EndpointSlices", func(ctx context.Context) {
		isolated, control, svc := createIsolatedServiceMember(ctx, f, "isolated-slice-member")

		ginkgo.By("Waiting for the EndpointSlice to publish the regular pod")
		err := e2eendpointslice.WaitForEndpointCount(ctx, cs, svc.Namespace, svc.Name, 1)
		framework.ExpectNoError(err, "expected exactly one endpoint (the regular pod) for Service %s", svc.Name)

		// The controller never publishes the isolated pod, so the regular pod
		// must stay the only endpoint.
		listEndpoints := func(ctx context.Context) ([]discoveryv1.Endpoint, error) {
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
		}
		gomega.Consistently(ctx, listEndpoints).WithTimeout(10*time.Second).WithPolling(time.Second).Should(gomega.HaveExactElements(
			gomega.HaveField("TargetRef.Name", control.Name),
		), "only the regular pod %s may be published; isolated pod %s must not", control.Name, isolated.Name)
	})

	// "EndpointsController" in the name lets jobs that run without the
	// Endpoints controller skip this test.
	ginkgo.It("should not be selected into Endpoints by the EndpointsController", func(ctx context.Context) {
		isolated, control, svc := createIsolatedServiceMember(ctx, f, "isolated-endpoints-member")

		ginkgo.By("Waiting for the Endpoints object to publish the regular pod")
		err := wait.PollUntilContextTimeout(ctx, time.Second, e2eendpointslice.ControllerUpdateTimeout, true, func(ctx context.Context) (bool, error) {
			endpoints, err := cs.CoreV1().Endpoints(svc.Namespace).Get(ctx, svc.Name, metav1.GetOptions{})
			if apierrors.IsNotFound(err) {
				return false, nil
			}
			if err != nil {
				return false, err
			}
			return len(endpoints.Subsets) == 1 && len(endpoints.Subsets[0].Addresses) == 1, nil
		})
		framework.ExpectNoError(err, "expected exactly one address (the regular pod) in Endpoints %s", svc.Name)

		// The controller never publishes the isolated pod, so the regular pod
		// must stay the only address.
		gomega.Consistently(ctx, func(ctx context.Context) ([]v1.EndpointAddress, error) {
			endpoints, err := cs.CoreV1().Endpoints(svc.Namespace).Get(ctx, svc.Name, metav1.GetOptions{})
			if err != nil {
				return nil, err
			}
			var addresses []v1.EndpointAddress
			for _, subset := range endpoints.Subsets {
				addresses = append(addresses, subset.Addresses...)
				addresses = append(addresses, subset.NotReadyAddresses...)
			}
			return addresses, nil
		}).WithTimeout(10*time.Second).WithPolling(time.Second).Should(gomega.HaveExactElements(
			gomega.HaveField("TargetRef.Name", control.Name),
		), "only the regular pod %s may be an Endpoints address; isolated pod %s must not", control.Name, isolated.Name)
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
		jobTemplate.Spec.Containers[0].ReadinessProbe = nil
		jobTemplate.Spec.Containers[0].Args = nil
		// Succeeds only if loopback is the single interface in the sandbox.
		jobTemplate.Spec.Containers[0].Command = []string{"sh", "-c", "test \"$(ip -o link show | wc -l)\" -eq 1"}
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

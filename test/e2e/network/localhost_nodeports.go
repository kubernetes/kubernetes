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

	"github.com/onsi/ginkgo/v2"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eendpointslice "k8s.io/kubernetes/test/e2e/framework/endpointslice"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eregistry "k8s.io/kubernetes/test/e2e/framework/registry"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/network/common"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = common.SIGDescribe("LocalhostNodePorts", func() {
	fr := framework.NewDefaultFramework("localhost-nodeports")
	fr.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	framework.It("should proxy localhost NodePort traffic to backends across nodes", feature.LocalhostNodePorts, func(ctx context.Context) {
		cs := fr.ClientSet
		ns := fr.Namespace.Name

		nodes, err := e2enode.GetBoundedReadySchedulableNodes(ctx, cs, 2)
		framework.ExpectNoError(err)
		if len(nodes.Items) < 2 {
			e2eskipper.Skipf("Test requires >= 2 Ready nodes, but there are only %v nodes", len(nodes.Items))
		}
		clientNodeName := nodes.Items[0].Name
		remoteNodeName := nodes.Items[1].Name

		hostExecPod := e2epod.NewExecPodSpec(ns, "host-exec-pod", true)
		e2epod.SetNodeSelection(&hostExecPod.Spec, e2epod.NodeSelection{Name: clientNodeName})
		hostExecPod = e2epod.NewPodClient(fr).CreateSync(ctx, hostExecPod)

		ginkgo.By("creating one backend pod on the client node and one on a different node")
		label := map[string]string{"app": "agnhost-localhost-nodeport"}
		servePort := []v1.ContainerPort{{ContainerPort: 9376, Protocol: v1.ProtocolTCP}}
		for _, bp := range []struct {
			name string
			node string
		}{
			{"agnhost-localhost-nodeport-local", clientNodeName},
			{"agnhost-localhost-nodeport-remote", remoteNodeName},
		} {
			pod := e2epod.NewAgnhostPod(ns, bp.name, nil, nil, servePort, "serve-hostname")
			pod.Labels = label
			e2epod.SetNodeSelection(&pod.Spec, e2epod.NodeSelection{Name: bp.node})
			e2epod.NewPodClient(fr).CreateSync(ctx, pod)
		}

		ginkgo.By("creating NodePort service")
		svc := &v1.Service{
			ObjectMeta: metav1.ObjectMeta{Name: "agnhost-localhost-nodeport"},
			Spec: v1.ServiceSpec{
				Type:     v1.ServiceTypeNodePort,
				Selector: label,
				Ports: []v1.ServicePort{{
					Protocol:   v1.ProtocolTCP,
					Port:       9000,
					TargetPort: intstr.FromInt32(9376),
				}},
			},
		}
		svc, err = cs.CoreV1().Services(ns).Create(ctx, svc, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("waiting for endpoints")
		framework.ExpectNoError(e2eendpointslice.WaitForEndpointCount(ctx, cs, ns, svc.Name, 2))

		ginkgo.By("curling the localhost NodePort and confirming traffic spreads across both backends")
		loopbackIP := "127.0.0.1"
		if framework.TestContext.ClusterIsIPv6() {
			loopbackIP = "::1"
		}
		// shouldHold=false: without session affinity, requests must spread across
		// the two backends (one local, one remote), confirming cross-node proxying.
		if !checkAffinity(ctx, cs, hostExecPod, loopbackIP, int(svc.Spec.Ports[0].NodePort), false) {
			framework.Failf("expected localhost NodePort %d traffic to be distributed across both backends", svc.Spec.Ports[0].NodePort)
		}
	})

	framework.It("should honor sessionAffinity=ClientIP for localhost NodePort", feature.LocalhostNodePorts, func(ctx context.Context) {
		cs := fr.ClientSet
		ns := fr.Namespace.Name

		nodes, err := e2enode.GetBoundedReadySchedulableNodes(ctx, cs, 1)
		framework.ExpectNoError(err)
		if len(nodes.Items) < 1 {
			e2eskipper.Skipf("Test requires >= 1 Ready nodes, but there are only %v nodes", len(nodes.Items))
		}
		nodeName := nodes.Items[0].Name

		hostExecPod := e2epod.NewExecPodSpec(ns, "host-exec-pod", true)
		e2epod.SetNodeSelection(&hostExecPod.Spec, e2epod.NodeSelection{Name: nodeName})
		hostExecPod = e2epod.NewPodClient(fr).CreateSync(ctx, hostExecPod)

		ginkgo.By("creating 3 backend pods pinned to the same node")
		label := map[string]string{"app": "agnhost-sticky-localhost"}
		servePort := []v1.ContainerPort{{ContainerPort: 9376, Protocol: v1.ProtocolTCP}}
		for i := range 3 {
			name := fmt.Sprintf("agnhost-sticky-localhost-%d", i)
			pod := e2epod.NewAgnhostPod(ns, name, nil, nil, servePort, "serve-hostname")
			pod.Labels = label
			e2epod.SetNodeSelection(&pod.Spec, e2epod.NodeSelection{Name: nodeName})
			e2epod.NewPodClient(fr).CreateSync(ctx, pod)
		}

		ginkgo.By("creating NodePort service with sessionAffinity=ClientIP")
		timeoutSeconds := int32(10800)
		svc := &v1.Service{
			ObjectMeta: metav1.ObjectMeta{Name: "agnhost-sticky-localhost"},
			Spec: v1.ServiceSpec{
				Type:            v1.ServiceTypeNodePort,
				Selector:        label,
				SessionAffinity: v1.ServiceAffinityClientIP,
				SessionAffinityConfig: &v1.SessionAffinityConfig{
					ClientIP: &v1.ClientIPConfig{TimeoutSeconds: &timeoutSeconds},
				},
				Ports: []v1.ServicePort{{
					Protocol:   v1.ProtocolTCP,
					Port:       9000,
					TargetPort: intstr.FromInt32(9376),
				}},
			},
		}
		svc, err = cs.CoreV1().Services(ns).Create(ctx, svc, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("waiting for all 3 endpoints to be ready")
		framework.ExpectNoError(e2eendpointslice.WaitForEndpointCount(ctx, cs, ns, svc.Name, 3))

		ginkgo.By("curling the localhost NodePort and confirming all requests hit the same pod")
		loopbackIP := "127.0.0.1"
		if framework.TestContext.ClusterIsIPv6() {
			loopbackIP = "::1"
		}
		// shouldHold=true: with sessionAffinity=ClientIP, all localhost requests
		// (source IP 127.0.0.1/::1) must be pinned to a single backend.
		if !checkAffinity(ctx, cs, hostExecPod, loopbackIP, int(svc.Spec.Ports[0].NodePort), true) {
			framework.Failf("expected sessionAffinity=ClientIP to pin localhost NodePort %d traffic to a single backend", svc.Spec.Ports[0].NodePort)
		}
	})

	framework.It("should let a node pull an image from a registry exposed via localhost NodePort", feature.LocalhostNodePorts, func(ctx context.Context) {
		cs := fr.ClientSet
		ns := fr.Namespace.Name

		nodes, err := e2enode.GetBoundedReadySchedulableNodes(ctx, cs, 2)
		framework.ExpectNoError(err)
		if len(nodes.Items) < 2 {
			e2eskipper.Skipf("Test requires >= 2 Ready nodes, but there are only %v nodes", len(nodes.Items))
		}
		clientNodeName := nodes.Items[0].Name
		registryNodeName := nodes.Items[1].Name

		ginkgo.By("running a private fake registry on a different node, behind a NodePort service")
		label := map[string]string{"app": "localhost-nodeport-registry"}
		registryPorts := []v1.ContainerPort{{ContainerPort: 5000, Protocol: v1.ProtocolTCP}}
		registryPod := e2epod.NewAgnhostPod(ns, "localhost-nodeport-registry", nil, nil, registryPorts, "fake-registry-server", "--private")
		registryPod.Labels = label
		e2epod.SetNodeSelection(&registryPod.Spec, e2epod.NodeSelection{Name: registryNodeName})
		e2epod.NewPodClient(fr).CreateSync(ctx, registryPod)

		svc := &v1.Service{
			ObjectMeta: metav1.ObjectMeta{Name: "localhost-nodeport-registry"},
			Spec: v1.ServiceSpec{
				Type:     v1.ServiceTypeNodePort,
				Selector: label,
				Ports: []v1.ServicePort{{
					Protocol:   v1.ProtocolTCP,
					Port:       5000,
					TargetPort: intstr.FromInt32(5000),
				}},
			},
		}
		svc, err = cs.CoreV1().Services(ns).Create(ctx, svc, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("waiting for the registry endpoint")
		framework.ExpectNoError(e2eendpointslice.WaitForEndpointCount(ctx, cs, ns, svc.Name, 1))

		nodePort := svc.Spec.Ports[0].NodePort
		registryAddress := fmt.Sprintf("localhost:%d", nodePort)

		ginkgo.By("creating the image pull secret for the localhost registry")
		secret := e2eregistry.User1DockerSecret(registryAddress)
		secret.Name = "localhost-nodeport-registry-creds-" + string(uuid.NewUUID())
		_, err = cs.CoreV1().Secrets(ns).Create(ctx, secret, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("pulling an image through the localhost NodePort and running it")
		pullPod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "localhost-nodeport-registry-client"},
			Spec: v1.PodSpec{
				Containers: []v1.Container{{
					Name:            "client",
					Image:           registryAddress + "/pause:testing",
					ImagePullPolicy: v1.PullAlways,
				}},
				ImagePullSecrets: []v1.LocalObjectReference{{Name: secret.Name}},
				RestartPolicy:    v1.RestartPolicyNever,
			},
		}
		e2epod.SetNodeSelection(&pullPod.Spec, e2epod.NodeSelection{Name: clientNodeName})
		pullPod = e2epod.NewPodClient(fr).Create(ctx, pullPod)
		framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, cs, pullPod))
	})
})

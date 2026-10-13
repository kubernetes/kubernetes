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

package netpol

import (
	"context"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/network/common"
	admissionapi "k8s.io/pod-security-admission/api"
)

// The netpol suite builds its workloads as bare pods, so NetworkPolicy
// enforcement on StatefulSet pods (which have stable identities and are
// recreated with the same name) is not covered anywhere. This verifies that an
// egress policy selecting a StatefulSet pod is enforced, and keeps being
// enforced after the pod is deleted and recreated by the StatefulSet
// controller.
var _ = common.SIGDescribe("Netpol", func() {
	f := framework.NewDefaultFramework("netpol-statefulset")
	f.SkipNamespaceCreation = true // we create our own x/y/z test namespaces
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	ginkgo.Context("NetworkPolicy on StatefulSet pods", func() {
		const port int32 = 80
		const stsName = "sts"
		const stsPod = "sts-0"

		f.It("should enforce an egress NetworkPolicy on StatefulSet pods and keep enforcing it across pod recreation", feature.NetworkPolicy, func(ctx context.Context) {
			// x/a = allowed egress target, x/b = denied egress target.
			k8s := initializeResources(ctx, f, []v1.Protocol{protocolTCP}, []int32{port}, "x/a", "x/b")
			nsX, _, _ := getK8sNamespaces(k8s)

			ginkgo.By("labeling x/a as the allowed egress target")
			AddPodLabels(ctx, k8s, nsX, "a", map[string]string{"role": "allowed"})

			ginkgo.By("creating a StatefulSet client pod serving the same agnhost workload")
			client := &Pod{Name: "client", Ports: []int32{port}, Protocols: []v1.Protocol{protocolTCP}}
			labels := client.Labels() // pod=client
			zero := int64(0)
			replicas := int32(1)

			// A StatefulSet needs a (headless) governing Service.
			headless := &v1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: stsName, Namespace: nsX},
				Spec: v1.ServiceSpec{
					ClusterIP: v1.ClusterIPNone,
					Selector:  labels,
					Ports:     []v1.ServicePort{{Name: "http", Port: port, Protocol: v1.ProtocolTCP}},
				},
			}
			_, err := k8s.clientSet.CoreV1().Services(nsX).Create(ctx, headless, metav1.CreateOptions{})
			framework.ExpectNoError(err, "creating headless service for the StatefulSet")

			sts := &appsv1.StatefulSet{
				ObjectMeta: metav1.ObjectMeta{Name: stsName, Namespace: nsX},
				Spec: appsv1.StatefulSetSpec{
					ServiceName: stsName,
					Replicas:    &replicas,
					Selector:    &metav1.LabelSelector{MatchLabels: labels},
					Template: v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{Labels: labels},
						Spec: v1.PodSpec{
							TerminationGracePeriodSeconds: &zero,
							Containers:                    client.ContainerSpecs(),
						},
					},
				},
			}
			_, err = k8s.clientSet.AppsV1().StatefulSets(nsX).Create(ctx, sts, metav1.CreateOptions{})
			framework.ExpectNoError(err, "creating StatefulSet")
			framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, k8s.clientSet, stsPod, nsX), "waiting for %s/%s to be running", nsX, stsPod)

			ginkgo.By("creating an egress policy allowing the StatefulSet pod to reach only role=allowed pods")
			protoTCP := v1.ProtocolTCP
			p80 := intstr.FromInt32(port)
			egress := networkingv1.NetworkPolicyEgressRule{
				To:    []networkingv1.NetworkPolicyPeer{{PodSelector: &metav1.LabelSelector{MatchLabels: map[string]string{"role": "allowed"}}}},
				Ports: []networkingv1.NetworkPolicyPort{{Protocol: &protoTCP, Port: &p80}},
			}
			policy := GenNetworkPolicyWithNameAndPodMatchLabel("allow-egress-to-allowed", labels, SetSpecEgressRules(egress))
			CreatePolicy(ctx, k8s, policy, nsX)

			// Probe by pod IP so the test needs no DNS.
			allowed, err := k8s.getPod(ctx, nsX, "a")
			framework.ExpectNoError(err, "getting allowed pod x/a")
			denied, err := k8s.getPod(ctx, nsX, "b")
			framework.ExpectNoError(err, "getting denied pod x/b")

			probe := func(addrTo string, expect bool) {
				connected, cmd, err := k8s.probeConnectivity(&probeConnectivityArgs{
					nsFrom:              nsX,
					podFrom:             stsPod,
					addrTo:              addrTo,
					protocol:            protocolTCP,
					toPort:              int(port),
					expectConnectivity:  expect,
					timeoutSeconds:      getProbeTimeoutSeconds(),
					pollIntervalSeconds: getPollIntervalSeconds(),
					pollTimeoutSeconds:  getPollTimeoutSeconds(),
				})
				framework.ExpectNoError(err, "probing %s -> %s", stsPod, addrTo)
				gomega.Expect(connected).To(gomega.Equal(expect),
					"%s -> %s: expected connected=%v; probe: %s", stsPod, addrTo, expect, cmd)
			}

			ginkgo.By("verifying the egress policy is enforced for the StatefulSet pod")
			probe(allowed.Status.PodIP, true)
			probe(denied.Status.PodIP, false)

			ginkgo.By("deleting the StatefulSet pod and waiting for it to be recreated")
			err = k8s.clientSet.CoreV1().Pods(nsX).Delete(ctx, stsPod, metav1.DeleteOptions{})
			framework.ExpectNoError(err, "deleting StatefulSet pod %s/%s", nsX, stsPod)
			framework.ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(ctx, k8s.clientSet, stsPod, nsX), "waiting for recreated %s/%s to be running", nsX, stsPod)

			ginkgo.By("verifying the egress policy is still enforced after pod recreation")
			probe(allowed.Status.PodIP, true)
			probe(denied.Status.PodIP, false)
		})
	})
})

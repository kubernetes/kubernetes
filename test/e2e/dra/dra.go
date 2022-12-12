/*
Copyright 2022 The Kubernetes Authors.

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

package dra

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	resourcev1alpha1 "k8s.io/api/resource/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/test/e2e/dra/test-driver/app"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
)

const (
	// podStartTimeout is how long to wait for the pod to be started.
	podStartTimeout = 5 * time.Minute
)

func networkResources() app.Resources {
	return app.Resources{
		Shareable: true,
	}
}

var _ = ginkgo.Describe("[sig-node] DRA [Feature:DynamicResourceAllocation]", func() {
	f := framework.NewDefaultFramework("dra")

	// The driver containers have to run with sufficient privileges to
	// modify /var/lib/kubelet/plugins.
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

	ginkgo.Context("kubelet", func() {
		nodes := NewNodes(f, 1, 1)
		driver := NewDriver(f, nodes, networkResources) // All tests get their own driver instance.
		b := newBuilder(f, driver)
		ginkgo.It("registers plugin", func() {
			ginkgo.By("the driver is running")
		})

		// This test does not pass at the moment because kubelet doesn't retry.
		ginkgo.It("must retry NodePrepareResource", func(ctx context.Context) {
			// We have exactly one host.
			m := MethodInstance{driver.Nodenames()[0], NodePrepareResourceMethod}

			driver.Fail(m, true)

			ginkgo.By("waiting for container startup to fail")
			parameters := b.parameters()
			pod, template := b.podInline(resourcev1alpha1.AllocationModeWaitForFirstConsumer)

			b.create(ctx, parameters, pod, template)

			ginkgo.By("wait for NodePrepareResource call")
			gomega.Eventually(ctx, func(ctx context.Context) error {
				if driver.CallCount(m) == 0 {
					return errors.New("NodePrepareResource not called yet")
				}
				return nil
			}).WithTimeout(podStartTimeout).Should(gomega.Succeed())

			ginkgo.By("allowing container startup to succeed")
			callCount := driver.CallCount(m)
			driver.Fail(m, false)
			err := e2epod.WaitForPodNameRunningInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace)
			framework.ExpectNoError(err, "start pod with inline resource claim")
			if driver.CallCount(m) == callCount {
				framework.Fail("NodePrepareResource should have been called again")
			}
		})
		ginkgo.It("must not run a pod if a claim is not reserved for it", func(ctx context.Context) {
			parameters := b.parameters()
			claim := b.externalClaim(resourcev1alpha1.AllocationModeImmediate)
			pod := b.podExternal()

			// This bypasses scheduling and therefore the pod gets
			// to run on the node although it never gets added to
			// the `ReservedFor` field of the claim.
			pod.Spec.NodeName = nodes.NodeNames[0]

			b.create(ctx, parameters, claim, pod)

			gomega.Consistently(func() error {
				testPod, err := b.f.ClientSet.CoreV1().Pods(pod.Namespace).Get(context.TODO(), pod.Name, metav1.GetOptions{})
				if err != nil {
					return fmt.Errorf("expected the test pod %s to exist: %v", pod.Name, err)
				}
				if testPod.Status.Phase != v1.PodPending {
					return fmt.Errorf("pod %s: unexpected status %s, expected status: %s", pod.Name, testPod.Status.Phase, v1.PodPending)
				}
				return nil
			}, 20*time.Second, 200*time.Millisecond).Should(gomega.BeNil())
		})
		ginkgo.It("must unprepare resources for force-deleted pod", func(ctx context.Context) {
			parameters := b.parameters()
			claim := b.externalClaim(resourcev1alpha1.AllocationModeImmediate)
			pod := b.podExternal()
			zero := int64(0)
			pod.Spec.TerminationGracePeriodSeconds = &zero

			b.create(ctx, parameters, claim, pod)

			b.testPod(ctx, f.ClientSet, pod)

			ginkgo.By(fmt.Sprintf("force delete test pod %s", pod.Name))
			err := b.f.ClientSet.CoreV1().Pods(b.f.Namespace.Name).Delete(ctx, pod.Name, metav1.DeleteOptions{GracePeriodSeconds: &zero})
			if !apierrors.IsNotFound(err) {
				framework.ExpectNoError(err, "force delete test pod")
			}

			for host, plugin := range b.driver.Nodes {
				ginkgo.By(fmt.Sprintf("waiting for resources on %s to be unprepared", host))
				gomega.Eventually(plugin.GetPreparedResources).WithTimeout(time.Minute).Should(gomega.BeEmpty(), "prepared claims on host %s", host)
			}
		})
	})

	ginkgo.Context("driver", func() {
		nodes := NewNodes(f, 1, 1)
		driver := NewDriver(f, nodes, networkResources) // All tests get their own driver instance.
		b := newBuilder(f, driver)
		// We need the parameters name *before* creating it.
		b.parametersCounter = 1
		b.classParametersName = b.parametersName()

		ginkgo.It("supports claim and class parameters", func(ctx context.Context) {
			classParameters := b.parameters("x", "y")
			claimParameters := b.parameters()
			pod, template := b.podInline(resourcev1alpha1.AllocationModeWaitForFirstConsumer)

			b.create(ctx, classParameters, claimParameters, pod, template)

			b.testPod(ctx, f.ClientSet, pod, "user_a", "b", "admin_x", "y")
		})
	})

	ginkgo.Context("cluster", func() {
		nodes := NewNodes(f, 1, 4)
		driver := NewDriver(f, nodes, networkResources)
		b := newBuilder(f, driver)

		// claimTests tries out several different combinations of pods with
		// claims, both inline and external.
		claimTests := func(allocationMode resourcev1alpha1.AllocationMode) {
			ginkgo.It("supports simple pod referencing inline resource claim", func(ctx context.Context) {
				parameters := b.parameters()
				pod, template := b.podInline(allocationMode)
				b.create(ctx, parameters, pod, template)

				b.testPod(ctx, f.ClientSet, pod)
			})

			ginkgo.It("supports inline claim referenced by multiple containers", func(ctx context.Context) {
				parameters := b.parameters()
				pod, template := b.podInlineMultiple(allocationMode)
				b.create(ctx, parameters, pod, template)

				b.testPod(ctx, f.ClientSet, pod)
			})

			ginkgo.It("supports simple pod referencing external resource claim", func(ctx context.Context) {
				parameters := b.parameters()
				pod := b.podExternal()
				b.create(ctx, parameters, b.externalClaim(allocationMode), pod)

				b.testPod(ctx, f.ClientSet, pod)
			})

			ginkgo.It("supports external claim referenced by multiple pods", func(ctx context.Context) {
				parameters := b.parameters()
				pod1 := b.podExternal()
				pod2 := b.podExternal()
				pod3 := b.podExternal()
				claim := b.externalClaim(allocationMode)
				b.create(ctx, parameters, claim, pod1, pod2, pod3)

				for _, pod := range []*v1.Pod{pod1, pod2, pod3} {
					b.testPod(ctx, f.ClientSet, pod)
				}
			})

			ginkgo.It("supports external claim referenced by multiple containers of multiple pods", func(ctx context.Context) {
				parameters := b.parameters()
				pod1 := b.podExternalMultiple()
				pod2 := b.podExternalMultiple()
				pod3 := b.podExternalMultiple()
				claim := b.externalClaim(allocationMode)
				b.create(ctx, parameters, claim, pod1, pod2, pod3)

				for _, pod := range []*v1.Pod{pod1, pod2, pod3} {
					b.testPod(ctx, f.ClientSet, pod)
				}
			})

			ginkgo.It("supports init containers", func(ctx context.Context) {
				parameters := b.parameters()
				pod, template := b.podInline(allocationMode)
				pod.Spec.InitContainers = []v1.Container{pod.Spec.Containers[0]}
				pod.Spec.InitContainers[0].Name += "-init"
				// This must succeed for the pod to start.
				pod.Spec.InitContainers[0].Command = []string{"sh", "-c", "env | grep user_a=b"}
				b.create(ctx, parameters, pod, template)

				b.testPod(ctx, f.ClientSet, pod)
			})
		}

		ginkgo.Context("with delayed allocation", func() {
			claimTests(resourcev1alpha1.AllocationModeWaitForFirstConsumer)
		})

		ginkgo.Context("with immediate allocation", func() {
			claimTests(resourcev1alpha1.AllocationModeImmediate)
		})
	})

	ginkgo.Context("multiple nodes", func() {
		nodes := NewNodes(f, 2, 8)
		ginkgo.Context("with network-attached resources", func() {
			driver := NewDriver(f, nodes, networkResources)
			b := newBuilder(f, driver)

			ginkgo.It("schedules onto different nodes", func(ctx context.Context) {
				parameters := b.parameters()
				label := "app.kubernetes.io/instance"
				instance := f.UniqueName + "-test-app"
				antiAffinity := &v1.Affinity{
					PodAntiAffinity: &v1.PodAntiAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
							{
								TopologyKey: "kubernetes.io/hostname",
								LabelSelector: &metav1.LabelSelector{
									MatchLabels: map[string]string{
										label: instance,
									},
								},
							},
						},
					},
				}
				createPod := func() *v1.Pod {
					pod := b.podExternal()
					pod.Labels[label] = instance
					pod.Spec.Affinity = antiAffinity
					return pod
				}
				pod1 := createPod()
				pod2 := createPod()
				claim := b.externalClaim(resourcev1alpha1.AllocationModeWaitForFirstConsumer)
				b.create(ctx, parameters, claim, pod1, pod2)

				for _, pod := range []*v1.Pod{pod1, pod2} {
					err := e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod)
					framework.ExpectNoError(err, "start pod")
				}
			})
		})

		ginkgo.Context("with node-local resources", func() {
			driver := NewDriver(f, nodes, func() app.Resources {
				return app.Resources{
					NodeLocal:      true,
					MaxAllocations: 1,
					Nodes:          nodes.NodeNames,
				}
			})
			b := newBuilder(f, driver)

			tests := func(allocationMode resourcev1alpha1.AllocationMode) {
				ginkgo.It("uses all resources", func(ctx context.Context) {
					var objs = []klog.KMetadata{
						b.parameters(),
					}
					var pods []*v1.Pod
					for i := 0; i < len(nodes.NodeNames); i++ {
						pod, template := b.podInline(allocationMode)
						pods = append(pods, pod)
						objs = append(objs, pod, template)
					}
					b.create(ctx, objs...)

					for _, pod := range pods {
						err := e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod)
						framework.ExpectNoError(err, "start pod")
					}

					// The pods all should run on different
					// nodes because the maximum number of
					// claims per node was limited to 1 for
					// this test.
					//
					// We cannot know for sure why the pods
					// ran on two different nodes (could
					// also be a coincidence) but if they
					// don't cover all nodes, then we have
					// a problem.
					used := make(map[string]*v1.Pod)
					for _, pod := range pods {
						pod, err := f.ClientSet.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
						framework.ExpectNoError(err, "get pod")
						nodeName := pod.Spec.NodeName
						if other, ok := used[nodeName]; ok {
							framework.Failf("Pod %s got started on the same node %s as pod %s although claim allocation should have been limited to one claim per node.", pod.Name, nodeName, other.Name)
						}
						used[nodeName] = pod
					}
				})
			}

			ginkgo.Context("with delayed allocation", func() {
				tests(resourcev1alpha1.AllocationModeWaitForFirstConsumer)
			})

			ginkgo.Context("with immediate allocation", func() {
				tests(resourcev1alpha1.AllocationModeImmediate)
			})
		})

		ginkgo.Context("reallocation", func() {
			var allocateWrapper app.AllocateWrapperType
			driver := NewDriver(f, nodes, func() app.Resources {
				return app.Resources{
					NodeLocal:      true,
					MaxAllocations: 2,
					Nodes:          nodes.NodeNames,

					AllocateWrapper: func(ctx context.Context, claim *resourcev1alpha1.ResourceClaim, claimParameters interface{}, class *resourcev1alpha1.ResourceClass, classParameters interface{}, selectedNode string,
						handler func(ctx context.Context, claim *resourcev1alpha1.ResourceClaim, claimParameters interface{}, class *resourcev1alpha1.ResourceClass, classParameters interface{}, selectedNode string) (result *resourcev1alpha1.AllocationResult, err error)) (result *resourcev1alpha1.AllocationResult, err error) {
						return allocateWrapper(ctx, claim, claimParameters, class, classParameters, selectedNode, handler)
					},
				}
			})
			b := newBuilder(f, driver)

			ginkgo.It("works", func(ctx context.Context) {
				// A pod with two claims can run on a node, but
				// only if allocation of both succeeds. This
				// tests simulates the scenario where one claim
				// gets allocated but the second doesn't
				// because of a race with some other pod.
				//
				// To ensure the right timing, allocation of the second
				// claim gets delayed while creating another pod
				// that gets the remaining resource on the node.
				ctx, cancel := context.WithCancel(ctx)
				defer cancel()

				parameters := b.parameters()
				claim1 := b.externalClaim(resourcev1alpha1.AllocationModeWaitForFirstConsumer)
				claim2 := b.externalClaim(resourcev1alpha1.AllocationModeWaitForFirstConsumer)
				pod1 := b.podExternal()
				pod1.Spec.ResourceClaims = append(pod1.Spec.ResourceClaims,
					v1.PodResourceClaim{
						Name: "claim2",
						Source: v1.ClaimSource{
							ResourceClaimName: &claim2.Name,
						},
					},
				)

				// Block on the second external claim that is to be allocated.
				blockClaim, cancelBlockClaim := context.WithCancel(ctx)
				defer cancelBlockClaim()
				var allocated int32
				allocateWrapper = func(ctx context.Context, claim *resourcev1alpha1.ResourceClaim, claimParameters interface{},
					class *resourcev1alpha1.ResourceClass, classParameters interface{}, selectedNode string,
					handler func(ctx context.Context, claim *resourcev1alpha1.ResourceClaim, claimParameters interface{},
						class *resourcev1alpha1.ResourceClass, classParameters interface{}, selectedNode string) (result *resourcev1alpha1.AllocationResult, err error),
				) (result *resourcev1alpha1.AllocationResult, err error) {
					oldAllocated := atomic.AddInt32(&allocated, 0)
					if oldAllocated == 1 && strings.HasPrefix(claim.Name, "external-claim") {
						<-blockClaim.Done()
					}
					result, err = handler(ctx, claim, claimParameters, class, classParameters, selectedNode)
					if err == nil {
						atomic.AddInt32(&allocated, 1)
					}
					return
				}
				b.create(ctx, parameters, claim1, claim2, pod1)

				ginkgo.By("waiting for one claim to be allocated")
				var nodeSelector *v1.NodeSelector
				gomega.Eventually(ctx, func(ctx context.Context) (int, error) {
					claims, err := f.ClientSet.ResourceV1alpha1().ResourceClaims(f.Namespace.Name).List(ctx, metav1.ListOptions{})
					if err != nil {
						return 0, err
					}
					allocated := 0
					for _, claim := range claims.Items {
						if claim.Status.Allocation != nil {
							allocated++
							nodeSelector = claim.Status.Allocation.AvailableOnNodes
						}
					}
					return allocated, nil
				}).WithTimeout(time.Minute).Should(gomega.Equal(1), "one claim allocated")

				// Now create a second pod which we force to
				// run on the same node that is currently being
				// considered for the first one. We know what
				// the node selector looks like and can
				// directly access the key and value from it.
				ginkgo.By(fmt.Sprintf("create second pod on the same node %s", nodeSelector))
				pod2, template2 := b.podInline(resourcev1alpha1.AllocationModeWaitForFirstConsumer)
				req := nodeSelector.NodeSelectorTerms[0].MatchExpressions[0]
				node := req.Values[0]
				pod2.Spec.NodeSelector = map[string]string{req.Key: node}
				b.create(ctx, pod2, template2)
				framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod2), "start pod 2")

				// Allow allocation of claim2 to proceed. It should fail now
				// and the other node must be used instead, after deallocating
				// the first claim.
				ginkgo.By("move first pod to other node")
				cancelBlockClaim()
				framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, pod1), "start pod 1")
				pod1, err := f.ClientSet.CoreV1().Pods(pod1.Namespace).Get(ctx, pod1.Name, metav1.GetOptions{})
				framework.ExpectNoError(err, "get first pod")
				if pod1.Spec.NodeName == "" {
					framework.Fail("first pod should be running on node, was not scheduled")
				}
				framework.ExpectNotEqual(pod1.Spec.NodeName, node, "first pod should run on different node than second one")
				framework.ExpectEqual(driver.Controller.GetNumDeallocations(), int64(1), "number of deallocations")
			})
		})
	})

	ginkgo.Context("multiple drivers", func() {
		nodes := NewNodes(f, 1, 4)
		driver1 := NewDriver(f, nodes, func() app.Resources {
			return app.Resources{
				NodeLocal:      true,
				MaxAllocations: 1,
				Nodes:          nodes.NodeNames,
			}
		})
		b1 := newBuilder(f, driver1)
		driver2 := NewDriver(f, nodes, func() app.Resources {
			return app.Resources{
				NodeLocal:      true,
				MaxAllocations: 1,
				Nodes:          nodes.NodeNames,
			}
		})
		driver2.NameSuffix = "-other"
		b2 := newBuilder(f, driver2)

		ginkgo.It("work", func(ctx context.Context) {
			parameters1 := b1.parameters()
			parameters2 := b2.parameters()
			claim1 := b1.externalClaim(resourcev1alpha1.AllocationModeWaitForFirstConsumer)
			claim2 := b2.externalClaim(resourcev1alpha1.AllocationModeWaitForFirstConsumer)
			pod := b1.podExternal()
			pod.Spec.ResourceClaims = append(pod.Spec.ResourceClaims,
				v1.PodResourceClaim{
					Name: "claim2",
					Source: v1.ClaimSource{
						ResourceClaimName: &claim2.Name,
					},
				},
			)
			b1.create(ctx, parameters1, parameters2, claim1, claim2, pod)
			b1.testPod(ctx, f.ClientSet, pod)
		})
	})
})

// builder contains a running counter to make objects unique within thir
// namespace.
type builder struct {
	f      *framework.Framework
	driver *Driver

	podCounter        int
	parametersCounter int
	claimCounter      int

	classParametersName string
}

// className returns the default resource class name.
func (b *builder) className() string {
	return b.f.UniqueName + b.driver.NameSuffix + "-class"
}

// class returns the resource class that the builder's other objects
// reference.
func (b *builder) class() *resourcev1alpha1.ResourceClass {
	class := &resourcev1alpha1.ResourceClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: b.className(),
		},
		DriverName:    b.driver.Name,
		SuitableNodes: b.nodeSelector(),
	}
	if b.classParametersName != "" {
		class.ParametersRef = &resourcev1alpha1.ResourceClassParametersReference{
			Kind:      "ConfigMap",
			Name:      b.classParametersName,
			Namespace: b.f.Namespace.Name,
		}
	}
	return class
}

// nodeSelector returns a node selector that matches all nodes on which the
// kubelet plugin was deployed.
func (b *builder) nodeSelector() *v1.NodeSelector {
	return &v1.NodeSelector{
		NodeSelectorTerms: []v1.NodeSelectorTerm{
			{
				MatchExpressions: []v1.NodeSelectorRequirement{
					{
						Key:      "kubernetes.io/hostname",
						Operator: v1.NodeSelectorOpIn,
						Values:   b.driver.Nodenames(),
					},
				},
			},
		},
	}
}

// externalClaim returns external resource claim
// that test pods can reference
func (b *builder) externalClaim(allocationMode resourcev1alpha1.AllocationMode) *resourcev1alpha1.ResourceClaim {
	b.claimCounter++
	name := "external-claim" + b.driver.NameSuffix // This is what podExternal expects.
	if b.claimCounter > 1 {
		name += fmt.Sprintf("-%d", b.claimCounter)
	}
	return &resourcev1alpha1.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: resourcev1alpha1.ResourceClaimSpec{
			ResourceClassName: b.className(),
			ParametersRef: &resourcev1alpha1.ResourceClaimParametersReference{
				Kind: "ConfigMap",
				Name: b.parametersName(),
			},
			AllocationMode: allocationMode,
		},
	}
}

// parametersName returns the current ConfigMap name for resource
// claim or class parameters.
func (b *builder) parametersName() string {
	return fmt.Sprintf("parameters%s-%d", b.driver.NameSuffix, b.parametersCounter)
}

// parametersEnv returns the default env variables.
func (b *builder) parametersEnv() map[string]string {
	return map[string]string{
		"a": "b",
	}
}

// parameters returns a config map with the default env variables.
func (b *builder) parameters(kv ...string) *v1.ConfigMap {
	b.parametersCounter++
	data := map[string]string{}
	for i := 0; i < len(kv); i += 2 {
		data[kv[i]] = kv[i+1]
	}
	if len(data) == 0 {
		data = b.parametersEnv()
	}
	return &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: b.f.Namespace.Name,
			Name:      b.parametersName(),
		},
		Data: data,
	}
}

// makePod returns a simple pod with no resource claims.
// The pod prints its env and waits.
func (b *builder) pod() *v1.Pod {
	pod := e2epod.MakePod(b.f.Namespace.Name, nil, nil, false, "env && sleep 100000")
	pod.Labels = make(map[string]string)
	pod.Spec.RestartPolicy = v1.RestartPolicyNever
	// Let kubelet kill the pods quickly. Setting
	// TerminationGracePeriodSeconds to zero would bypass kubelet
	// completely because then the apiserver enables a force-delete even
	// when DeleteOptions for the pod don't ask for it (see
	// https://github.com/kubernetes/kubernetes/blob/0f582f7c3f504e807550310d00f130cb5c18c0c3/pkg/registry/core/pod/strategy.go#L151-L171).
	//
	// We don't do that because it breaks tracking of claim usage: the
	// kube-controller-manager assumes that kubelet is done with the pod
	// once it got removed or has a grace period of 0. Setting the grace
	// period to zero directly in DeletionOptions or indirectly through
	// TerminationGracePeriodSeconds causes the controller to remove
	// the pod from ReservedFor before it actually has stopped on
	// the node.
	one := int64(1)
	pod.Spec.TerminationGracePeriodSeconds = &one
	pod.ObjectMeta.GenerateName = ""
	b.podCounter++
	pod.ObjectMeta.Name = fmt.Sprintf("tester%s-%d", b.driver.NameSuffix, b.podCounter)
	return pod
}

// makePodInline adds an inline resource claim with default class name and parameters.
func (b *builder) podInline(allocationMode resourcev1alpha1.AllocationMode) (*v1.Pod, *resourcev1alpha1.ResourceClaimTemplate) {
	pod := b.pod()
	pod.Spec.Containers[0].Name = "with-resource"
	podClaimName := "my-inline-claim"
	pod.Spec.Containers[0].Resources.Claims = []v1.ResourceClaim{{Name: podClaimName}}
	pod.Spec.ResourceClaims = []v1.PodResourceClaim{
		{
			Name: podClaimName,
			Source: v1.ClaimSource{
				ResourceClaimTemplateName: &pod.Name,
			},
		},
	}
	template := &resourcev1alpha1.ResourceClaimTemplate{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pod.Name,
			Namespace: pod.Namespace,
		},
		Spec: resourcev1alpha1.ResourceClaimTemplateSpec{
			Spec: resourcev1alpha1.ResourceClaimSpec{
				ResourceClassName: b.className(),
				ParametersRef: &resourcev1alpha1.ResourceClaimParametersReference{
					Kind: "ConfigMap",
					Name: b.parametersName(),
				},
				AllocationMode: allocationMode,
			},
		},
	}
	return pod, template
}

// podInlineMultiple returns a pod with inline resource claim referenced by 3 containers
func (b *builder) podInlineMultiple(allocationMode resourcev1alpha1.AllocationMode) (*v1.Pod, *resourcev1alpha1.ResourceClaimTemplate) {
	pod, template := b.podInline(allocationMode)
	pod.Spec.Containers = append(pod.Spec.Containers, *pod.Spec.Containers[0].DeepCopy(), *pod.Spec.Containers[0].DeepCopy())
	pod.Spec.Containers[1].Name = pod.Spec.Containers[1].Name + "-1"
	pod.Spec.Containers[2].Name = pod.Spec.Containers[1].Name + "-2"
	return pod, template
}

// podExternal adds a pod that references external resource claim with default class name and parameters.
func (b *builder) podExternal() *v1.Pod {
	pod := b.pod()
	pod.Spec.Containers[0].Name = "with-resource"
	podClaimName := "resource-claim"
	externalClaimName := "external-claim" + b.driver.NameSuffix
	pod.Spec.ResourceClaims = []v1.PodResourceClaim{
		{
			Name: podClaimName,
			Source: v1.ClaimSource{
				ResourceClaimName: &externalClaimName,
			},
		},
	}
	pod.Spec.Containers[0].Resources.Claims = []v1.ResourceClaim{{Name: podClaimName}}
	return pod
}

// podShared returns a pod with 3 containers that reference external resource claim with default class name and parameters.
func (b *builder) podExternalMultiple() *v1.Pod {
	pod := b.podExternal()
	pod.Spec.Containers = append(pod.Spec.Containers, *pod.Spec.Containers[0].DeepCopy(), *pod.Spec.Containers[0].DeepCopy())
	pod.Spec.Containers[1].Name = pod.Spec.Containers[1].Name + "-1"
	pod.Spec.Containers[2].Name = pod.Spec.Containers[1].Name + "-2"
	return pod
}

// create takes a bunch of objects and calls their Create function.
func (b *builder) create(ctx context.Context, objs ...klog.KMetadata) {
	for _, obj := range objs {
		ginkgo.By(fmt.Sprintf("creating %T %s", obj, obj.GetName()), func() {
			var err error
			switch obj := obj.(type) {
			case *resourcev1alpha1.ResourceClass:
				_, err = b.f.ClientSet.ResourceV1alpha1().ResourceClasses().Create(ctx, obj, metav1.CreateOptions{})
			case *v1.Pod:
				_, err = b.f.ClientSet.CoreV1().Pods(b.f.Namespace.Name).Create(ctx, obj, metav1.CreateOptions{})
			case *v1.ConfigMap:
				_, err = b.f.ClientSet.CoreV1().ConfigMaps(b.f.Namespace.Name).Create(ctx, obj, metav1.CreateOptions{})
			case *resourcev1alpha1.ResourceClaim:
				_, err = b.f.ClientSet.ResourceV1alpha1().ResourceClaims(b.f.Namespace.Name).Create(ctx, obj, metav1.CreateOptions{})
			case *resourcev1alpha1.ResourceClaimTemplate:
				_, err = b.f.ClientSet.ResourceV1alpha1().ResourceClaimTemplates(b.f.Namespace.Name).Create(ctx, obj, metav1.CreateOptions{})
			default:
				framework.Fail(fmt.Sprintf("internal error, unsupported type %T", obj), 1)
			}
			framework.ExpectNoErrorWithOffset(1, err, "create %T", obj)
		})
	}
}

// testPod runs pod and checks if container logs contain expected environment variables
func (b *builder) testPod(ctx context.Context, clientSet kubernetes.Interface, pod *v1.Pod, env ...string) {
	err := e2epod.WaitForPodRunningInNamespace(ctx, clientSet, pod)
	framework.ExpectNoError(err, "start pod")

	for _, container := range pod.Spec.Containers {
		log, err := e2epod.GetPodLogs(ctx, clientSet, pod.Namespace, pod.Name, container.Name)
		framework.ExpectNoError(err, "get logs")
		if len(env) == 0 {
			for key, value := range b.parametersEnv() {
				envStr := fmt.Sprintf("\nuser_%s=%s\n", key, value)
				gomega.Expect(log).To(gomega.ContainSubstring(envStr), "container env variables")
			}
		} else {
			for i := 0; i < len(env); i += 2 {
				envStr := fmt.Sprintf("\n%s=%s\n", env[i], env[i+1])
				gomega.Expect(log).To(gomega.ContainSubstring(envStr), "container env variables")
			}
		}
	}
}

func newBuilder(f *framework.Framework, driver *Driver) *builder {
	b := &builder{f: f, driver: driver}

	ginkgo.BeforeEach(b.setUp)

	return b
}

func (b *builder) setUp() {
	b.podCounter = 0
	b.parametersCounter = 0
	b.claimCounter = 0
	b.create(context.Background(), b.class())
	ginkgo.DeferCleanup(b.tearDown)
}

func (b *builder) tearDown(ctx context.Context) {
	err := b.f.ClientSet.ResourceV1alpha1().ResourceClasses().Delete(ctx, b.className(), metav1.DeleteOptions{})
	framework.ExpectNoError(err, "delete resource class")

	// Before we allow the namespace and all objects in it do be deleted by
	// the framework, we must ensure that test pods and the claims that
	// they use are deleted. Otherwise the driver might get deleted first,
	// in which case deleting the claims won't work anymore.
	ginkgo.By("delete pods and claims")
	pods, err := b.listTestPods(ctx)
	framework.ExpectNoError(err, "list pods")
	for _, pod := range pods {
		if pod.DeletionTimestamp != nil {
			continue
		}
		ginkgo.By(fmt.Sprintf("deleting %T %s", &pod, klog.KObj(&pod)))
		err := b.f.ClientSet.CoreV1().Pods(b.f.Namespace.Name).Delete(ctx, pod.Name, metav1.DeleteOptions{})
		if !apierrors.IsNotFound(err) {
			framework.ExpectNoError(err, "delete pod")
		}
	}
	gomega.Eventually(func() ([]v1.Pod, error) {
		return b.listTestPods(ctx)
	}).WithTimeout(time.Minute).Should(gomega.BeEmpty(), "remaining pods despite deletion")

	claims, err := b.f.ClientSet.ResourceV1alpha1().ResourceClaims(b.f.Namespace.Name).List(ctx, metav1.ListOptions{})
	framework.ExpectNoError(err, "get resource claims")
	for _, claim := range claims.Items {
		if claim.DeletionTimestamp != nil {
			continue
		}
		ginkgo.By(fmt.Sprintf("deleting %T %s", &claim, klog.KObj(&claim)))
		err := b.f.ClientSet.ResourceV1alpha1().ResourceClaims(b.f.Namespace.Name).Delete(ctx, claim.Name, metav1.DeleteOptions{})
		if !apierrors.IsNotFound(err) {
			framework.ExpectNoError(err, "delete claim")
		}
	}

	for host, plugin := range b.driver.Nodes {
		ginkgo.By(fmt.Sprintf("waiting for resources on %s to be unprepared", host))
		gomega.Eventually(plugin.GetPreparedResources).WithTimeout(time.Minute).Should(gomega.BeEmpty(), "prepared claims on host %s", host)
	}

	ginkgo.By("waiting for claims to be deallocated and deleted")
	gomega.Eventually(func() ([]resourcev1alpha1.ResourceClaim, error) {
		claims, err := b.f.ClientSet.ResourceV1alpha1().ResourceClaims(b.f.Namespace.Name).List(ctx, metav1.ListOptions{})
		if err != nil {
			return nil, err
		}
		return claims.Items, nil
	}).WithTimeout(time.Minute).Should(gomega.BeEmpty(), "claims in the namespaces")
}

func (b *builder) listTestPods(ctx context.Context) ([]v1.Pod, error) {
	pods, err := b.f.ClientSet.CoreV1().Pods(b.f.Namespace.Name).List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, err
	}

	var testPods []v1.Pod
	for _, pod := range pods.Items {
		if pod.Labels["app.kubernetes.io/part-of"] == "dra-test-driver" {
			continue
		}
		testPods = append(testPods, pod)
	}
	return testPods, nil
}

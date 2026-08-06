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

package podgroupprotection

import (
	"context"
	"fmt"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	fwk "k8s.io/kube-scheduler/framework"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/controller/scheduling/podgroupprotection"
	"k8s.io/kubernetes/pkg/features"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
)

const testNamespace = "default"

func cpgKey(name string) fwk.EntityKey {
	return fwk.CompositePodGroupKey(testNamespace, name)
}

func pgKey(name string) fwk.EntityKey {
	return fwk.PodGroupKey(testNamespace, name)
}

func setup(t *testing.T) (context.Context, kubeapiservertesting.TearDownFunc, clientset.Interface, *podgroupprotection.Controller, informers.SharedInformerFactory) {
	tCtx := ktesting.Init(t)

	// Enable feature gates for CompositePodGroup
	featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.GenericWorkload, true)
	featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.TopologyAwareWorkloadScheduling, true)
	featuregatetesting.SetFeatureGateDuringTest(t, feature.DefaultFeatureGate, features.CompositePodGroup, true)

	flags := append(framework.DefaultTestServerFlags(),
		"--enable-admission-plugins=PodGroupProtection",
		"--feature-gates=GenericWorkload=true,TopologyAwareWorkloadScheduling=true,CompositePodGroup=true",
		fmt.Sprintf("--runtime-config=%s=true,%s=true", schedulingv1alpha3.SchemeGroupVersion, schedulingv1beta1.SchemeGroupVersion),
	)

	// Start test server with admission plugin enabled
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, flags, framework.SharedEtcd())

	config := restclient.CopyConfig(server.ClientConfig)
	clientSet, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatalf("Error in create clientset: %v", err)
	}

	informerFactory := informers.NewSharedInformerFactory(clientSet, 0)

	pgInformer := informerFactory.Scheduling().V1beta1().PodGroups()
	cpgInformer := informerFactory.Scheduling().V1alpha3().CompositePodGroups()
	podInformer := informerFactory.Core().V1().Pods()

	ctrl, err := podgroupprotection.NewPodGroupProtectionController(
		tCtx.Logger(),
		pgInformer,
		cpgInformer,
		podInformer,
		clientSet,
		true,
	)
	if err != nil {
		t.Fatalf("Failed to create PodGroupProtectionController: %v", err)
	}

	tearDown := func() {
		tCtx.Cancel("tearing down")
		server.TearDownFn()
	}

	return tCtx, tearDown, clientSet, ctrl, informerFactory
}

type testAction string

const (
	deleteCPG testAction = "delete-cpg"
	deletePG  testAction = "delete-pg"
	deletePod testAction = "delete-pod"
)

type testStep struct {
	action           testAction
	targetName       string
	expectedExisting sets.Set[fwk.EntityKey]
}

type gcTestCase struct {
	name        string
	initialCPGs []*schedulingv1alpha3.CompositePodGroup
	initialPGs  []*schedulingv1beta1.PodGroup
	initialPods []*v1.Pod
	steps       []testStep
}

func TestCompositePodGroupGarbageCollection(t *testing.T) {
	tests := []gcTestCase{
		{
			name: "CPG with no children gets deleted immediately upon deletion request",
			initialCPGs: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Namespace(testNamespace).Name("standalone-cpg").WorkloadRef("test-wl", "test-tpl").BasicPolicy().Obj(),
			},
			steps: []testStep{
				{
					action:           deleteCPG,
					targetName:       "standalone-cpg",
					expectedExisting: sets.New[fwk.EntityKey](),
				},
			},
		},
		{
			name: "CPG with child CPG and child PG: deletion blocked until all children are deleted",
			initialCPGs: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Namespace(testNamespace).Name("root-cpg").WorkloadRef("test-wl", "test-tpl").BasicPolicy().Obj(),
				st.MakeCompositePodGroup().Namespace(testNamespace).Name("child-cpg").ParentCompositePodGroup("root-cpg").WorkloadRef("test-wl", "test-tpl").BasicPolicy().Obj(),
			},
			initialPGs: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Namespace(testNamespace).Name("child-pg").ParentCompositePodGroup("root-cpg").WorkloadRef("test-tpl", "test-wl").BasicPolicy().Obj(),
			},
			steps: []testStep{
				{
					action:           deleteCPG,
					targetName:       "root-cpg",
					expectedExisting: sets.New(cpgKey("root-cpg"), cpgKey("child-cpg"), pgKey("child-pg")),
				},
				{
					action:           deleteCPG,
					targetName:       "child-cpg",
					expectedExisting: sets.New(cpgKey("root-cpg"), pgKey("child-pg")),
				},
				{
					action:           deletePG,
					targetName:       "child-pg",
					expectedExisting: sets.New[fwk.EntityKey](),
				},
			},
		},
		{
			name: "Multi-level CPG hierarchy: grandparent -> parent -> child PG",
			initialCPGs: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Namespace(testNamespace).Name("grandparent-cpg").WorkloadRef("test-wl", "test-tpl").BasicPolicy().Obj(),
				st.MakeCompositePodGroup().Namespace(testNamespace).Name("parent-cpg").ParentCompositePodGroup("grandparent-cpg").WorkloadRef("test-wl", "test-tpl").BasicPolicy().Obj(),
			},
			initialPGs: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Namespace(testNamespace).Name("child-pg").ParentCompositePodGroup("parent-cpg").WorkloadRef("test-tpl", "test-wl").BasicPolicy().Obj(),
			},
			steps: []testStep{
				{
					action:           deleteCPG,
					targetName:       "grandparent-cpg",
					expectedExisting: sets.New(cpgKey("grandparent-cpg"), cpgKey("parent-cpg"), pgKey("child-pg")),
				},
				{
					action:           deleteCPG,
					targetName:       "parent-cpg",
					expectedExisting: sets.New(cpgKey("grandparent-cpg"), cpgKey("parent-cpg"), pgKey("child-pg")),
				},
				{
					action:           deletePG,
					targetName:       "child-pg",
					expectedExisting: sets.New[fwk.EntityKey](),
				},
			},
		},
		{
			name: "CPG with child PG containing active Pod: Pod protects PG, PG protects CPG",
			initialCPGs: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Namespace(testNamespace).Name("root-cpg").WorkloadRef("test-wl", "test-tpl").BasicPolicy().Obj(),
			},
			initialPGs: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Namespace(testNamespace).Name("child-pg").ParentCompositePodGroup("root-cpg").WorkloadRef("test-tpl", "test-wl").BasicPolicy().Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Namespace(testNamespace).Name("active-pod").PodGroupName("child-pg").Obj(),
			},
			steps: []testStep{
				{
					action:           deleteCPG,
					targetName:       "root-cpg",
					expectedExisting: sets.New(cpgKey("root-cpg"), pgKey("child-pg")),
				},
				{
					action:           deletePG,
					targetName:       "child-pg",
					expectedExisting: sets.New(cpgKey("root-cpg"), pgKey("child-pg")),
				},
				{
					action:           deletePod,
					targetName:       "active-pod",
					expectedExisting: sets.New[fwk.EntityKey](),
				},
			},
		},
		{
			name: "CPG with multiple sibling child PGs: partial child deletion keeps parent protected",
			initialCPGs: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Namespace(testNamespace).Name("root-cpg").WorkloadRef("test-wl", "test-tpl").BasicPolicy().Obj(),
			},
			initialPGs: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Namespace(testNamespace).Name("child-pg-1").ParentCompositePodGroup("root-cpg").WorkloadRef("test-tpl", "test-wl").BasicPolicy().Obj(),
				st.MakePodGroup().Namespace(testNamespace).Name("child-pg-2").ParentCompositePodGroup("root-cpg").WorkloadRef("test-tpl", "test-wl").BasicPolicy().Obj(),
			},
			steps: []testStep{
				{
					action:           deleteCPG,
					targetName:       "root-cpg",
					expectedExisting: sets.New(cpgKey("root-cpg"), pgKey("child-pg-1"), pgKey("child-pg-2")),
				},
				{
					action:           deletePG,
					targetName:       "child-pg-1",
					expectedExisting: sets.New(cpgKey("root-cpg"), pgKey("child-pg-2")),
				},
				{
					action:           deletePG,
					targetName:       "child-pg-2",
					expectedExisting: sets.New[fwk.EntityKey](),
				},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx, tearDown, clientSet, ctrl, informerFactory := setup(t)
			defer tearDown()

			informerFactory.Start(ctx.Done())
			informerFactory.WaitForCacheSync(ctx.Done())

			go ctrl.Run(ctx, 1)

			// Track all initial objects created in this test case
			allObjects := sets.New[fwk.EntityKey]()
			for _, cpg := range tc.initialCPGs {
				allObjects.Insert(cpgKey(cpg.Name))
			}
			for _, pg := range tc.initialPGs {
				allObjects.Insert(pgKey(pg.Name))
			}

			// Create all initial CPGs
			for _, cpg := range tc.initialCPGs {
				if _, err := clientSet.SchedulingV1alpha3().CompositePodGroups(cpg.Namespace).Create(ctx, cpg, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create CPG %s: %v", cpg.Name, err)
				}
			}

			// Create all initial PGs
			for _, pg := range tc.initialPGs {
				if _, err := clientSet.SchedulingV1beta1().PodGroups(pg.Namespace).Create(ctx, pg, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create PG %s: %v", pg.Name, err)
				}
			}

			// Create all initial Pods
			for _, pod := range tc.initialPods {
				podCopy := pod.DeepCopy()
				if len(podCopy.Spec.Containers) == 0 {
					podCopy.Spec.Containers = []v1.Container{{Name: "c1", Image: "pause"}}
				}
				if _, err := clientSet.CoreV1().Pods(podCopy.Namespace).Create(ctx, podCopy, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create Pod %s: %v", podCopy.Name, err)
				}
			}

			// Wait for admission plugin to stamp finalizers on CPGs and PGs
			err := wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, 10*time.Second, true, func(c context.Context) (bool, error) {
				for _, cpg := range tc.initialCPGs {
					obj, err := clientSet.SchedulingV1alpha3().CompositePodGroups(cpg.Namespace).Get(c, cpg.Name, metav1.GetOptions{})
					if err != nil || len(obj.Finalizers) == 0 {
						return false, nil
					}
				}
				for _, pg := range tc.initialPGs {
					obj, err := clientSet.SchedulingV1beta1().PodGroups(pg.Namespace).Get(c, pg.Name, metav1.GetOptions{})
					if err != nil || len(obj.Finalizers) == 0 {
						return false, nil
					}
				}
				return true, nil
			})
			if err != nil {
				t.Fatalf("Timeout waiting for admission finalizers to be stamped: %v", err)
			}

			// Execute steps sequentially
			for i, step := range tc.steps {
				t.Logf("Executing step %d (%s on %s)", i, step.action, step.targetName)
				switch step.action {
				case deleteCPG:
					if err := clientSet.SchedulingV1alpha3().CompositePodGroups(testNamespace).Delete(ctx, step.targetName, metav1.DeleteOptions{}); err != nil {
						t.Fatalf("Step %d: failed to delete CPG %s: %v", i, step.targetName, err)
					}
				case deletePG:
					if err := clientSet.SchedulingV1beta1().PodGroups(testNamespace).Delete(ctx, step.targetName, metav1.DeleteOptions{}); err != nil {
						t.Fatalf("Step %d: failed to delete PG %s: %v", i, step.targetName, err)
					}
				case deletePod:
					if err := clientSet.CoreV1().Pods(testNamespace).Delete(ctx, step.targetName, metav1.DeleteOptions{}); err != nil {
						t.Fatalf("Step %d: failed to delete Pod %s: %v", i, step.targetName, err)
					}
				default:
					t.Fatalf("Step %d: unknown action %s", i, step.action)
				}

				// Check expected state of all tracked objects using EntityKey
				for key := range allObjects {
					shouldExist := step.expectedExisting.Has(key)
					err := wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, 10*time.Second, true, func(c context.Context) (bool, error) {
						switch key.Type {
						case fwk.CompositePodGroupKeyType:
							cpg, err := clientSet.SchedulingV1alpha3().CompositePodGroups(key.Namespace).Get(c, key.Name, metav1.GetOptions{})
							if shouldExist {
								if err == nil && len(cpg.Finalizers) > 0 {
									return true, nil
								}
								return false, nil
							}
							if apierrors.IsNotFound(err) {
								return true, nil
							}
							return false, err
						case fwk.PodGroupKeyType:
							pg, err := clientSet.SchedulingV1beta1().PodGroups(key.Namespace).Get(c, key.Name, metav1.GetOptions{})
							if shouldExist {
								if err == nil && len(pg.Finalizers) > 0 {
									return true, nil
								}
								return false, nil
							}
							if apierrors.IsNotFound(err) {
								return true, nil
							}
							return false, err
						default:
							return false, fmt.Errorf("unsupported EntityKey type: %s", key.Type)
						}
					})
					if err != nil {
						t.Fatalf("Step %d: %s %s failed expected existence status (want exist=%v): %v", i, key.Type, key.Name, shouldExist, err)
					}
				}
			}
		})
	}
}

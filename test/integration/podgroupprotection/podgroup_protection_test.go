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
	"sync"
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
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/controller/scheduling/podgroupprotection"
	"k8s.io/kubernetes/pkg/features"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
)

type objectKind string

const (
	compositePodGroupKind objectKind = "CompositePodGroup"
	podGroupKind          objectKind = "PodGroup"
)

type objectKey struct {
	kind objectKind
	name string
}

func cpgKey(name string) objectKey {
	return objectKey{kind: compositePodGroupKind, name: name}
}

func pgKey(name string) objectKey {
	return objectKey{kind: podGroupKind, name: name}
}

func setup(t *testing.T) (context.Context, kubeapiservertesting.TearDownFunc, clientset.Interface, *podgroupprotection.Controller, informers.SharedInformerFactory, *sync.WaitGroup) {
	tCtx := ktesting.Init(t)

	// Enable feature gates for CompositePodGroup
	featuregatetesting.SetFeatureGatesDuringTest(t, feature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.GenericWorkload:                 true,
		features.TopologyAwareWorkloadScheduling: true,
		features.CompositePodGroup:               true,
	})

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

	var wg sync.WaitGroup
	tearDown := func() {
		tCtx.Cancel("tearing down")
		wg.Wait()
		server.TearDownFn()
	}

	return tCtx, tearDown, clientSet, ctrl, informerFactory, &wg
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
	expectedExisting sets.Set[objectKey]
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
				st.MakeCompositePodGroup().Name("standalone-cpg").WorkloadRef("test-wl", "test-tpl").BasicPolicy().Obj(),
			},
			steps: []testStep{
				{
					action:           deleteCPG,
					targetName:       "standalone-cpg",
					expectedExisting: sets.New[objectKey](),
				},
			},
		},
		{
			name: "CPG with child CPG and child PG: deletion blocked until all children are deleted",
			initialCPGs: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("root-cpg").WorkloadRef("test-wl", "test-tpl").BasicPolicy().Obj(),
				st.MakeCompositePodGroup().Name("child-cpg").ParentCompositePodGroup("root-cpg").WorkloadRef("test-wl", "test-tpl").BasicPolicy().Obj(),
			},
			initialPGs: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("child-pg").ParentCompositePodGroup("root-cpg").WorkloadRef("test-tpl", "test-wl").BasicPolicy().Obj(),
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
					expectedExisting: sets.New[objectKey](),
				},
			},
		},
		{
			name: "Multi-level CPG hierarchy: grandparent -> parent -> child PG",
			initialCPGs: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("grandparent-cpg").WorkloadRef("test-wl", "test-tpl").BasicPolicy().Obj(),
				st.MakeCompositePodGroup().Name("parent-cpg").ParentCompositePodGroup("grandparent-cpg").WorkloadRef("test-wl", "test-tpl").BasicPolicy().Obj(),
			},
			initialPGs: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("child-pg").ParentCompositePodGroup("parent-cpg").WorkloadRef("test-tpl", "test-wl").BasicPolicy().Obj(),
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
					expectedExisting: sets.New[objectKey](),
				},
			},
		},
		{
			name: "CPG with child PG containing active Pod: Pod protects PG, PG protects CPG",
			initialCPGs: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("root-cpg").WorkloadRef("test-wl", "test-tpl").BasicPolicy().Obj(),
			},
			initialPGs: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("child-pg").ParentCompositePodGroup("root-cpg").WorkloadRef("test-tpl", "test-wl").BasicPolicy().Obj(),
			},
			initialPods: []*v1.Pod{
				st.MakePod().Name("active-pod").PodGroupName("child-pg").Obj(),
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
					expectedExisting: sets.New[objectKey](),
				},
			},
		},
		{
			name: "CPG with multiple sibling child PGs: partial child deletion keeps parent protected",
			initialCPGs: []*schedulingv1alpha3.CompositePodGroup{
				st.MakeCompositePodGroup().Name("root-cpg").WorkloadRef("test-wl", "test-tpl").BasicPolicy().Obj(),
			},
			initialPGs: []*schedulingv1beta1.PodGroup{
				st.MakePodGroup().Name("child-pg-1").ParentCompositePodGroup("root-cpg").WorkloadRef("test-tpl", "test-wl").BasicPolicy().Obj(),
				st.MakePodGroup().Name("child-pg-2").ParentCompositePodGroup("root-cpg").WorkloadRef("test-tpl", "test-wl").BasicPolicy().Obj(),
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
					expectedExisting: sets.New[objectKey](),
				},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx, tearDown, clientSet, ctrl, informerFactory, wg := setup(t)
			defer tearDown()

			ns := framework.CreateNamespaceOrDie(clientSet, "cpg-gc", t).Name

			informerFactory.Start(ctx.Done())
			informerFactory.WaitForCacheSync(ctx.Done())

			wg.Go(func() {
				ctrl.Run(ctx, 1)
			})

			// Track all initial objects created in this test case
			allObjects := sets.New[objectKey]()
			for _, cpg := range tc.initialCPGs {
				allObjects.Insert(cpgKey(cpg.Name))
			}
			for _, pg := range tc.initialPGs {
				allObjects.Insert(pgKey(pg.Name))
			}

			// Create all initial CPGs
			for _, cpg := range tc.initialCPGs {
				cpgCopy := cpg.DeepCopy()
				cpgCopy.Namespace = ns
				if _, err := clientSet.SchedulingV1alpha3().CompositePodGroups(ns).Create(ctx, cpgCopy, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create CPG %s: %v", cpgCopy.Name, err)
				}
			}

			// Create all initial PGs
			for _, pg := range tc.initialPGs {
				pgCopy := pg.DeepCopy()
				pgCopy.Namespace = ns
				if _, err := clientSet.SchedulingV1beta1().PodGroups(ns).Create(ctx, pgCopy, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create PG %s: %v", pgCopy.Name, err)
				}
			}

			// Create all initial Pods
			for _, pod := range tc.initialPods {
				podCopy := pod.DeepCopy()
				podCopy.Namespace = ns
				if len(podCopy.Spec.Containers) == 0 {
					podCopy.Spec.Containers = []v1.Container{{Name: "c1", Image: "pause"}}
				}
				if _, err := clientSet.CoreV1().Pods(ns).Create(ctx, podCopy, metav1.CreateOptions{}); err != nil {
					t.Fatalf("Failed to create Pod %s: %v", podCopy.Name, err)
				}
			}

			// Wait for admission plugin to stamp finalizers on CPGs and PGs
			err := wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, 10*time.Second, true, func(c context.Context) (bool, error) {
				for key := range allObjects {
					obj, err := getObject(c, clientSet, ns, key)
					if err != nil || len(obj.GetFinalizers()) == 0 {
						return false, nil
					}
				}
				return true, nil
			})
			if err != nil {
				t.Fatalf("Timeout waiting for admission finalizers to be stamped: %v", err)
			}

			deletedObjects := sets.New[objectKey]()

			// Execute steps sequentially
			for i, step := range tc.steps {
				t.Logf("Executing step %d (%s on %s)", i, step.action, step.targetName)
				switch step.action {
				case deleteCPG:
					deletedObjects.Insert(cpgKey(step.targetName))
					if err := clientSet.SchedulingV1alpha3().CompositePodGroups(ns).Delete(ctx, step.targetName, metav1.DeleteOptions{}); err != nil {
						t.Fatalf("Step %d: failed to delete CPG %s: %v", i, step.targetName, err)
					}
				case deletePG:
					deletedObjects.Insert(pgKey(step.targetName))
					if err := clientSet.SchedulingV1beta1().PodGroups(ns).Delete(ctx, step.targetName, metav1.DeleteOptions{}); err != nil {
						t.Fatalf("Step %d: failed to delete PG %s: %v", i, step.targetName, err)
					}
				case deletePod:
					if err := clientSet.CoreV1().Pods(ns).Delete(ctx, step.targetName, metav1.DeleteOptions{}); err != nil {
						t.Fatalf("Step %d: failed to delete Pod %s: %v", i, step.targetName, err)
					}
				default:
					t.Fatalf("Step %d: unknown action %s", i, step.action)
				}

				// First, wait for all objects that should no longer exist to be removed (NotFound).
				for key := range allObjects {
					if step.expectedExisting.Has(key) {
						continue
					}
					if err := waitForObjectRemoval(ctx, clientSet, ns, key); err != nil {
						t.Fatalf("Step %d: %s %s was expected to be deleted, but still exists or failed: %v", i, key.kind, key.name, err)
					}
				}

				// Next, verify all objects that should still exist.
				// If an object has been deleted, it must be terminating (DeletionTimestamp != nil) with finalizer present.
				for key := range step.expectedExisting {
					if err := waitForObjectProtection(ctx, clientSet, ns, key, deletedObjects.Has(key)); err != nil {
						t.Fatalf("Step %d: %s %s failed expected existence status: %v", i, key.kind, key.name, err)
					}
				}
			}
		})
	}
}

func getObject(ctx context.Context, clientSet clientset.Interface, ns string, key objectKey) (metav1.Object, error) {
	switch key.kind {
	case compositePodGroupKind:
		return clientSet.SchedulingV1alpha3().CompositePodGroups(ns).Get(ctx, key.name, metav1.GetOptions{})
	case podGroupKind:
		return clientSet.SchedulingV1beta1().PodGroups(ns).Get(ctx, key.name, metav1.GetOptions{})
	default:
		return nil, fmt.Errorf("unsupported objectKey kind: %s", key.kind)
	}
}

func waitForObjectRemoval(ctx context.Context, clientSet clientset.Interface, ns string, key objectKey) error {
	return wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, 10*time.Second, true, func(c context.Context) (bool, error) {
		_, err := getObject(c, clientSet, ns, key)
		if apierrors.IsNotFound(err) {
			return true, nil
		}
		if err != nil {
			return false, err
		}
		return false, nil
	})
}

func waitForObjectProtection(ctx context.Context, clientSet clientset.Interface, ns string, key objectKey, isDeleted bool) error {
	return wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, 10*time.Second, true, func(c context.Context) (bool, error) {
		obj, err := getObject(c, clientSet, ns, key)
		if apierrors.IsNotFound(err) {
			return false, nil
		}
		if err != nil {
			return false, err
		}
		if len(obj.GetFinalizers()) == 0 {
			return false, nil
		}
		if isDeleted && obj.GetDeletionTimestamp() == nil {
			return false, nil
		}
		return true, nil
	})
}

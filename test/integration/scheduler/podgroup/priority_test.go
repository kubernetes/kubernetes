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

package podgroup

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	schedulingv1 "k8s.io/api/scheduling/v1"
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	fwk "k8s.io/kube-scheduler/framework"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/features"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/utils/ptr"
)

func TestPodGroupPriorityClassPreemptionPolicy(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.GenericWorkload:                 true,
		features.CompositePodGroup:               true,
		features.PodGroupPreemptionPolicy:        true,
		features.TopologyAwareWorkloadScheduling: true,
	})
	// The scheduler test helpers disable Priority admission, which this regression needs.
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{
		"--runtime-config=scheduling.k8s.io/v1alpha3=true,scheduling.k8s.io/v1beta1=true",
	}, framework.SharedEtcd())
	defer server.TearDownFn()

	ctx := t.Context()
	client := kubernetes.NewForConfigOrDie(server.ClientConfig)
	for _, pc := range []*schedulingv1.PriorityClass{
		{
			ObjectMeta:       metav1.ObjectMeta{Name: "never"},
			Value:            1000,
			PreemptionPolicy: ptr.To(v1.PreemptNever),
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "default-policy"},
			Value:      1000,
		},
	} {
		created, err := client.SchedulingV1().PriorityClasses().Create(ctx, pc, metav1.CreateOptions{})
		if err != nil {
			t.Fatal(err)
		}
		// Probe admission's cache, not API storage. An explicit policy keeps this
		// readiness check independent of the omitted-policy regression below.
		probe := st.MakePodGroup().Name("priority-cache-probe").BasicPolicy().
			PreemptionPolicy(schedulingv1beta1.PreemptionPolicy(*created.PreemptionPolicy)).Obj()
		probe.Spec.PriorityClassName = created.Name
		if err := wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, wait.ForeverTestTimeout, true, func(ctx context.Context) (bool, error) {
			_, err := client.SchedulingV1beta1().PodGroups(metav1.NamespaceDefault).Create(ctx, probe, metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}})
			if apierrors.IsForbidden(err) && strings.Contains(err.Error(), fmt.Sprintf("no PriorityClass with name %s was found", created.Name)) {
				return false, nil
			}
			return err == nil, err
		}); err != nil {
			t.Fatalf("waiting for PriorityClass %q in priority admission cache: %v", created.Name, err)
		}
	}

	for _, resource := range []struct {
		name      string
		create    func(t *testing.T, name, className string, policy *v1.PreemptionPolicy) error
		getPolicy func(t *testing.T, name string) *v1.PreemptionPolicy
	}{
		{
			name: "v1alpha3/PodGroup",
			create: func(t *testing.T, name, className string, policy *v1.PreemptionPolicy) error {
				group := &schedulingv1alpha3.PodGroup{
					ObjectMeta: metav1.ObjectMeta{Name: name},
					Spec: schedulingv1alpha3.PodGroupSpec{
						PriorityClassName: className,
						SchedulingPolicy: schedulingv1alpha3.PodGroupSchedulingPolicy{
							Basic: &schedulingv1alpha3.BasicSchedulingPolicy{},
						},
					},
				}
				if policy != nil {
					group.Spec.PreemptionPolicy = new(schedulingv1alpha3.PreemptionPolicy(*policy))
				}
				_, err := client.SchedulingV1alpha3().PodGroups(metav1.NamespaceDefault).Create(t.Context(), group, metav1.CreateOptions{})
				return err
			},
			getPolicy: func(t *testing.T, name string) *v1.PreemptionPolicy {
				t.Helper()
				group, err := client.SchedulingV1alpha3().PodGroups(metav1.NamespaceDefault).Get(t.Context(), name, metav1.GetOptions{})
				if err != nil {
					t.Fatal(err)
				}
				if group.Spec.PreemptionPolicy == nil {
					return nil
				}
				return new(v1.PreemptionPolicy(*group.Spec.PreemptionPolicy))
			},
		},
		{
			name: "v1beta1/PodGroup",
			create: func(t *testing.T, name, className string, policy *v1.PreemptionPolicy) error {
				group := &schedulingv1beta1.PodGroup{
					ObjectMeta: metav1.ObjectMeta{Name: name},
					Spec: schedulingv1beta1.PodGroupSpec{
						PriorityClassName: className,
						SchedulingPolicy: schedulingv1beta1.PodGroupSchedulingPolicy{
							Basic: &schedulingv1beta1.BasicSchedulingPolicy{},
						},
					},
				}
				if policy != nil {
					group.Spec.PreemptionPolicy = new(schedulingv1beta1.PreemptionPolicy(*policy))
				}
				_, err := client.SchedulingV1beta1().PodGroups(metav1.NamespaceDefault).Create(t.Context(), group, metav1.CreateOptions{})
				return err
			},
			getPolicy: func(t *testing.T, name string) *v1.PreemptionPolicy {
				t.Helper()
				group, err := client.SchedulingV1beta1().PodGroups(metav1.NamespaceDefault).Get(t.Context(), name, metav1.GetOptions{})
				if err != nil {
					t.Fatal(err)
				}
				if group.Spec.PreemptionPolicy == nil {
					return nil
				}
				return new(v1.PreemptionPolicy(*group.Spec.PreemptionPolicy))
			},
		},
		{
			name: "v1alpha3/CompositePodGroup",
			create: func(t *testing.T, name, className string, policy *v1.PreemptionPolicy) error {
				group := &schedulingv1alpha3.CompositePodGroup{
					ObjectMeta: metav1.ObjectMeta{Name: name},
					Spec: schedulingv1alpha3.CompositePodGroupSpec{
						PriorityClassName: className,
						SchedulingPolicy: schedulingv1alpha3.CompositePodGroupSchedulingPolicy{
							Basic: &schedulingv1alpha3.CompositeBasicSchedulingPolicy{},
						},
						WorkloadRef: &schedulingv1alpha3.WorkloadReference{
							WorkloadName: "workload",
							TemplateName: "template",
						},
					},
				}
				if policy != nil {
					group.Spec.PreemptionPolicy = new(schedulingv1alpha3.PreemptionPolicy(*policy))
				}
				_, err := client.SchedulingV1alpha3().CompositePodGroups(metav1.NamespaceDefault).Create(t.Context(), group, metav1.CreateOptions{})
				return err
			},
			getPolicy: func(t *testing.T, name string) *v1.PreemptionPolicy {
				t.Helper()
				group, err := client.SchedulingV1alpha3().CompositePodGroups(metav1.NamespaceDefault).Get(t.Context(), name, metav1.GetOptions{})
				if err != nil {
					t.Fatal(err)
				}
				if group.Spec.PreemptionPolicy == nil {
					return nil
				}
				return new(v1.PreemptionPolicy(*group.Spec.PreemptionPolicy))
			},
		},
		{
			name: "v1beta1/CompositePodGroup",
			create: func(t *testing.T, name, className string, policy *v1.PreemptionPolicy) error {
				group := &schedulingv1beta1.CompositePodGroup{
					ObjectMeta: metav1.ObjectMeta{Name: name},
					Spec: schedulingv1beta1.CompositePodGroupSpec{
						PriorityClassName: className,
						SchedulingPolicy: schedulingv1beta1.CompositePodGroupSchedulingPolicy{
							Basic: &schedulingv1beta1.CompositeBasicSchedulingPolicy{},
						},
						WorkloadRef: &schedulingv1beta1.WorkloadReference{
							WorkloadName: "workload",
							TemplateName: "template",
						},
					},
				}
				if policy != nil {
					group.Spec.PreemptionPolicy = new(schedulingv1beta1.PreemptionPolicy(*policy))
				}
				_, err := client.SchedulingV1beta1().CompositePodGroups(metav1.NamespaceDefault).Create(t.Context(), group, metav1.CreateOptions{})
				return err
			},
			getPolicy: func(t *testing.T, name string) *v1.PreemptionPolicy {
				t.Helper()
				group, err := client.SchedulingV1beta1().CompositePodGroups(metav1.NamespaceDefault).Get(t.Context(), name, metav1.GetOptions{})
				if err != nil {
					t.Fatal(err)
				}
				if group.Spec.PreemptionPolicy == nil {
					return nil
				}
				return new(v1.PreemptionPolicy(*group.Spec.PreemptionPolicy))
			},
		},
	} {
		t.Run(resource.name, func(t *testing.T) {
			for _, tc := range []struct {
				name           string
				className      string
				policy         *v1.PreemptionPolicy
				expectedPolicy v1.PreemptionPolicy
				wantForbidden  bool
			}{
				{name: "omitted-never", className: "never", expectedPolicy: v1.PreemptNever},
				{name: "matching-never", className: "never", policy: ptr.To(v1.PreemptNever), expectedPolicy: v1.PreemptNever},
				{name: "conflicting-preempt", className: "never", policy: ptr.To(v1.PreemptLowerPriority), wantForbidden: true},
				{name: "omitted-default", className: "default-policy", expectedPolicy: v1.PreemptLowerPriority},
				{name: "matching-default", className: "default-policy", policy: ptr.To(v1.PreemptLowerPriority), expectedPolicy: v1.PreemptLowerPriority},
				{name: "conflicting-never", className: "default-policy", policy: ptr.To(v1.PreemptNever), wantForbidden: true},
				{name: "no-class", expectedPolicy: v1.PreemptLowerPriority},
			} {
				t.Run(tc.name, func(t *testing.T) {
					name := tc.name + "-" + strings.ReplaceAll(resource.name, "/", "-")
					name = strings.ToLower(name)
					err := resource.create(t, name, tc.className, tc.policy)
					if tc.wantForbidden {
						if !apierrors.IsForbidden(err) || !strings.Contains(err.Error(), "priority admission controller computed") {
							t.Fatalf("expected priority admission rejection, got %v", err)
						}
						return
					}
					if err != nil {
						t.Fatalf("creating group: %v", err)
					}
					policy := resource.getPolicy(t, name)
					if policy == nil {
						t.Fatalf("stored preemptionPolicy is nil, want %q", tc.expectedPolicy)
					}
					if *policy != tc.expectedPolicy {
						t.Fatalf("stored preemptionPolicy = %q, want %q", *policy, tc.expectedPolicy)
					}
				})
			}
		})
	}
}

func TestPodGroupPreemptionPolicyCompatibility(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.GenericWorkload:                 true,
		features.CompositePodGroup:               true,
		features.TopologyAwareWorkloadScheduling: true,
	})

	tests := []struct {
		name                           string
		enablePodGroupPreemptionPolicy bool
		disablePriorityAdmission       bool
	}{
		{
			name: "created before enabling PodGroupPreemptionPolicy",
		},
		{
			name:                           "created with Priority admission disabled",
			enablePodGroupPreemptionPolicy: true,
			disablePriorityAdmission:       true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Reuse the storage prefix so the restarted server reads the original objects.
			etcdConfig := framework.SharedEtcd()
			flags := []string{"--runtime-config=scheduling.k8s.io/v1alpha3=true,scheduling.k8s.io/v1beta1=true"}
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodGroupPreemptionPolicy, tt.enablePodGroupPreemptionPolicy)
			createFlags := append([]string{}, flags...)
			if tt.disablePriorityAdmission {
				createFlags = append(createFlags, "--disable-admission-plugins=Priority")
			}
			server := kubeapiservertesting.StartTestServerOrDie(t, nil, createFlags, etcdConfig)
			defer server.TearDownFn()
			client := kubernetes.NewForConfigOrDie(server.ClientConfig)

			originalPG, err := client.SchedulingV1beta1().PodGroups(metav1.NamespaceDefault).Create(t.Context(),
				st.MakePodGroup().Name("pg").BasicPolicy().Obj(), metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("creating PodGroup: %v", err)
			}
			originalCPG, err := client.SchedulingV1beta1().CompositePodGroups(metav1.NamespaceDefault).Create(t.Context(),
				st.MakeCompositePodGroup().Name("cpg").WorkloadRef("workload", "template").BasicPolicy().Obj(), metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("creating CompositePodGroup: %v", err)
			}
			if originalPG.Spec.PreemptionPolicy != nil || originalCPG.Spec.PreemptionPolicy != nil {
				t.Fatal("expected both groups to be created without a preemptionPolicy")
			}

			server.TearDownFn()
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodGroupPreemptionPolicy, true)
			server = kubeapiservertesting.StartTestServerOrDie(t, nil, flags, etcdConfig)
			defer server.TearDownFn()
			client = kubernetes.NewForConfigOrDie(server.ClientConfig)

			pg, err := client.SchedulingV1beta1().PodGroups(metav1.NamespaceDefault).Get(t.Context(), originalPG.Name, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("reading PodGroup after restart: %v", err)
			}
			if pg.UID != originalPG.UID || pg.Spec.PreemptionPolicy != nil {
				t.Fatalf("expected original PodGroup with nil preemptionPolicy, got UID %q and policy %v", pg.UID, pg.Spec.PreemptionPolicy)
			}
			alphaPG, err := client.SchedulingV1alpha3().PodGroups(metav1.NamespaceDefault).Get(t.Context(), originalPG.Name, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("reading v1alpha3 PodGroup after restart: %v", err)
			}
			if alphaPG.UID != originalPG.UID || alphaPG.Spec.PreemptionPolicy != nil {
				t.Fatalf("expected original v1alpha3 PodGroup with nil preemptionPolicy, got UID %q and policy %v", alphaPG.UID, alphaPG.Spec.PreemptionPolicy)
			}

			cpg, err := client.SchedulingV1beta1().CompositePodGroups(metav1.NamespaceDefault).Get(t.Context(), originalCPG.Name, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("reading CompositePodGroup after restart: %v", err)
			}
			if cpg.UID != originalCPG.UID || cpg.Spec.PreemptionPolicy != nil {
				t.Fatalf("expected original CompositePodGroup with nil preemptionPolicy, got UID %q and policy %v", cpg.UID, cpg.Spec.PreemptionPolicy)
			}
			alphaCPG, err := client.SchedulingV1alpha3().CompositePodGroups(metav1.NamespaceDefault).Get(t.Context(), originalCPG.Name, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("reading v1alpha3 CompositePodGroup after restart: %v", err)
			}
			if alphaCPG.UID != originalCPG.UID || alphaCPG.Spec.PreemptionPolicy != nil {
				t.Fatalf("expected original v1alpha3 CompositePodGroup with nil preemptionPolicy, got UID %q and policy %v", alphaCPG.UID, alphaCPG.Spec.PreemptionPolicy)
			}

			for _, group := range []*fwk.GenericPodGroup{fwk.NewGenericPodGroup(pg), fwk.NewGenericCompositePodGroup(cpg)} {
				if policy := group.GetPreemptionPolicy(); policy != v1.PreemptLowerPriority {
					t.Errorf("scheduler preemption policy for %q = %q, want %q", group.GetName(), policy, v1.PreemptLowerPriority)
				}
			}
		})
	}
}

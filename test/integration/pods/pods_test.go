/*
Copyright 2015 The Kubernetes Authors.

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

package pods

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	v1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/version"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	typedv1 "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/rest"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	rbachelper "k8s.io/kubernetes/pkg/apis/rbac/v1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration"
	"k8s.io/kubernetes/test/integration/authutil"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestPodTopologyLabels(t *testing.T) {
	tests := []podTopologyTestCase{
		{
			name: "zone and region topology labels copied from assigned Node",
			targetNodeLabels: map[string]string{
				"topology.kubernetes.io/zone":   "zone",
				"topology.kubernetes.io/region": "region",
			},
			expectedPodLabels: map[string]string{
				"topology.kubernetes.io/zone":   "zone",
				"topology.kubernetes.io/region": "region",
			},
		},
		{
			name: "subdomains of topology.kubernetes.io are not copied",
			targetNodeLabels: map[string]string{
				"sub.topology.kubernetes.io/zone": "zone",
				"topology.kubernetes.io/region":   "region",
			},
			expectedPodLabels: map[string]string{
				"topology.kubernetes.io/region": "region",
			},
		},
		{
			name: "custom topology.kubernetes.io labels are not copied",
			targetNodeLabels: map[string]string{
				"topology.kubernetes.io/custom": "thing",
				"topology.kubernetes.io/zone":   "zone",
				"topology.kubernetes.io/region": "region",
			},
			expectedPodLabels: map[string]string{
				"topology.kubernetes.io/zone":   "zone",
				"topology.kubernetes.io/region": "region",
			},
		},
		{
			name: "labels from Bindings overwriting existing labels on Pod",
			existingPodLabels: map[string]string{
				"topology.kubernetes.io/zone":   "bad-zone",
				"topology.kubernetes.io/region": "bad-region",
				"topology.kubernetes.io/abc":    "123",
			},
			targetNodeLabels: map[string]string{
				"topology.kubernetes.io/zone":   "zone",
				"topology.kubernetes.io/region": "region",
				"topology.kubernetes.io/abc":    "456", // this label isn't in (zone, region) so isn't copied
			},
			expectedPodLabels: map[string]string{
				"topology.kubernetes.io/zone":   "zone",
				"topology.kubernetes.io/region": "region",
				"topology.kubernetes.io/abc":    "123",
			},
		},
	}
	// Enable the feature BEFORE starting the test server, as the admission plugin only checks feature gates
	// on start up and not on each invocation at runtime.
	featuregatetesting.SetFeatureGateEmulationVersionDuringTest(t, utilfeature.DefaultFeatureGate, version.MustParse("1.33"))
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodTopologyLabelsAdmission, true)
	testPodTopologyLabels(t, tests)
}

func TestPodTopologyLabels_FeatureDisabled(t *testing.T) {
	tests := []podTopologyTestCase{
		{
			name: "does nothing when the feature is not enabled",
			targetNodeLabels: map[string]string{
				"topology.kubernetes.io/zone":     "zone",
				"topology.kubernetes.io/region":   "region",
				"topology.kubernetes.io/custom":   "thing",
				"sub.topology.kubernetes.io/zone": "zone",
			},
			expectedPodLabels: map[string]string{},
		},
	}
	// Disable the feature BEFORE starting the test server, as the admission plugin only checks feature gates
	// on start up and not on each invocation at runtime.
	featuregatetesting.SetFeatureGateEmulationVersionDuringTest(t, utilfeature.DefaultFeatureGate, version.MustParse("1.33"))
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodTopologyLabelsAdmission, false)
	testPodTopologyLabels(t, tests)
}

// podTopologyTestCase is defined outside of TestPodTopologyLabels to allow us to re-use the test implementation logic
// between the feature enabled and feature disabled tests.
// This will no longer be required once the feature gate graduates to GA/locked to being enabled.
type podTopologyTestCase struct {
	name              string
	targetNodeLabels  map[string]string
	existingPodLabels map[string]string
	expectedPodLabels map[string]string
}

func testPodTopologyLabels(t *testing.T, tests []podTopologyTestCase) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer server.TearDownFn()
	client := clientset.NewForConfigOrDie(server.ClientConfig)
	ns := framework.CreateNamespaceOrDie(client, "pod-topology-labels", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	prototypePod := func() *v1.Pod {
		return &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "pod-topology-test-",
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "fake-name",
						Image: "fakeimage",
					},
				},
			},
		}
	}
	prototypeNode := func() *v1.Node {
		return &v1.Node{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "podtopology-test-node-",
			},
		}
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// Create the Node we are going to bind to.
			node := prototypeNode()
			// Set the labels on the Node we are going to create.
			node.Labels = test.targetNodeLabels
			ctx := context.Background()

			var err error
			if node, err = client.CoreV1().Nodes().Create(ctx, node, metav1.CreateOptions{}); err != nil {
				t.Errorf("Failed to create node: %v", err)
			}

			pod := prototypePod()
			pod.Labels = test.existingPodLabels
			if pod, err = client.CoreV1().Pods(ns.Name).Create(ctx, pod, metav1.CreateOptions{}); err != nil {
				t.Errorf("Failed to create pod: %v", err)
			}

			binding := &v1.Binding{
				ObjectMeta: metav1.ObjectMeta{Name: pod.Name, Namespace: pod.Namespace},
				Target: v1.ObjectReference{
					Kind: "Node",
					Name: node.Name,
				},
			}
			if err := client.CoreV1().Pods(pod.Namespace).Bind(ctx, binding, metav1.CreateOptions{}); err != nil {
				t.Errorf("Failed to bind pod to node: %v", err)
			}

			if pod, err = client.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{}); err != nil {
				t.Errorf("Failed to fetch bound Pod: %v", err)
			}

			if !apiequality.Semantic.DeepEqual(pod.Labels, test.expectedPodLabels) {
				t.Errorf("Unexpected label values: %v", cmp.Diff(pod.Labels, test.expectedPodLabels))
			}
		})
	}
}

func TestPodUpdateActiveDeadlineSeconds(t *testing.T) {
	// Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer server.TearDownFn()

	client := clientset.NewForConfigOrDie(server.ClientConfig)

	ns := framework.CreateNamespaceOrDie(client, "pod-activedeadline-update", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	var (
		iZero = int64(0)
		i30   = int64(30)
		i60   = int64(60)
		iNeg  = int64(-1)
	)

	prototypePod := func() *v1.Pod {
		return &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "xxx",
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "fake-name",
						Image: "fakeimage",
					},
				},
			},
		}
	}

	cases := []struct {
		name     string
		original *int64
		update   *int64
		valid    bool
	}{
		{
			name:     "no change, nil",
			original: nil,
			update:   nil,
			valid:    true,
		},
		{
			name:     "no change, set",
			original: &i30,
			update:   &i30,
			valid:    true,
		},
		{
			name:     "change to positive from nil",
			original: nil,
			update:   &i60,
			valid:    true,
		},
		{
			name:     "change to smaller positive",
			original: &i60,
			update:   &i30,
			valid:    true,
		},
		{
			name:     "change to larger positive",
			original: &i30,
			update:   &i60,
			valid:    false,
		},
		{
			name:     "change to negative from positive",
			original: &i30,
			update:   &iNeg,
			valid:    false,
		},
		{
			name:     "change to negative from nil",
			original: nil,
			update:   &iNeg,
			valid:    false,
		},
		// zero is not allowed, must be a positive integer
		{
			name:     "change to zero from positive",
			original: &i30,
			update:   &iZero,
			valid:    false,
		},
		{
			name:     "change to nil from positive",
			original: &i30,
			update:   nil,
			valid:    false,
		},
	}

	for i, tc := range cases {
		pod := prototypePod()
		pod.Spec.ActiveDeadlineSeconds = tc.original
		pod.ObjectMeta.Name = fmt.Sprintf("activedeadlineseconds-test-%v", i)

		if _, err := client.CoreV1().Pods(ns.Name).Create(context.TODO(), pod, metav1.CreateOptions{}); err != nil {
			t.Errorf("Failed to create pod: %v", err)
		}

		pod.Spec.ActiveDeadlineSeconds = tc.update

		_, err := client.CoreV1().Pods(ns.Name).Update(context.TODO(), pod, metav1.UpdateOptions{})
		if tc.valid && err != nil {
			t.Errorf("%v: failed to update pod: %v", tc.name, err)
		} else if !tc.valid && err == nil {
			t.Errorf("%v: unexpected allowed update to pod", tc.name)
		}

		integration.DeletePodOrErrorf(t, client, ns.Name, pod.Name)
	}
}

func TestPodReadOnlyFilesystem(t *testing.T) {
	// Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer server.TearDownFn()

	client := clientset.NewForConfigOrDie(server.ClientConfig)

	isReadOnly := true
	ns := framework.CreateNamespaceOrDie(client, "pod-readonly-root", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "xxx",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "fake-name",
					Image: "fakeimage",
					SecurityContext: &v1.SecurityContext{
						ReadOnlyRootFilesystem: &isReadOnly,
					},
				},
			},
		},
	}

	if _, err := client.CoreV1().Pods(ns.Name).Create(context.TODO(), pod, metav1.CreateOptions{}); err != nil {
		t.Errorf("Failed to create pod: %v", err)
	}

	integration.DeletePodOrErrorf(t, client, ns.Name, pod.Name)
}

func TestPodCreateEphemeralContainers(t *testing.T) {
	// Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer server.TearDownFn()

	client := clientset.NewForConfigOrDie(server.ClientConfig)

	ns := framework.CreateNamespaceOrDie(client, "pod-create-ephemeral-containers", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "xxx",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:                     "fake-name",
					Image:                    "fakeimage",
					ImagePullPolicy:          "Always",
					TerminationMessagePolicy: "File",
				},
			},
			EphemeralContainers: []v1.EphemeralContainer{
				{
					EphemeralContainerCommon: v1.EphemeralContainerCommon{
						Name:                     "debugger",
						Image:                    "debugimage",
						ImagePullPolicy:          "Always",
						TerminationMessagePolicy: "File",
					},
				},
			},
		},
	}

	if _, err := client.CoreV1().Pods(ns.Name).Create(context.TODO(), pod, metav1.CreateOptions{}); err == nil {
		t.Errorf("Unexpected allowed creation of pod with ephemeral containers")
		integration.DeletePodOrErrorf(t, client, ns.Name, pod.Name)
	} else if !strings.HasSuffix(err.Error(), "spec.ephemeralContainers: Forbidden: cannot be set on create") {
		t.Errorf("Unexpected error when creating pod with ephemeral containers: %v", err)
	}
}

// setUpEphemeralContainers creates a pod that has Ephemeral Containers. This is a two step
// process because Ephemeral Containers are not allowed during pod creation.
func setUpEphemeralContainers(podsClient typedv1.PodInterface, pod *v1.Pod, containers []v1.EphemeralContainer) (*v1.Pod, error) {
	result, err := podsClient.Create(context.TODO(), pod, metav1.CreateOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to create pod: %v", err)
	}

	if len(containers) == 0 {
		return result, nil
	}

	pod.Spec.EphemeralContainers = containers
	if _, err := podsClient.Update(context.TODO(), pod, metav1.UpdateOptions{}); err == nil {
		return nil, fmt.Errorf("unexpected allowed direct update of ephemeral containers during set up: %v", err)
	}

	result, err = podsClient.UpdateEphemeralContainers(context.TODO(), pod.Name, pod, metav1.UpdateOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to update ephemeral containers for test case set up: %v", err)
	}

	return result, nil
}

func TestPodPatchEphemeralContainers(t *testing.T) {
	// Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer server.TearDownFn()

	client := clientset.NewForConfigOrDie(server.ClientConfig)

	ns := framework.CreateNamespaceOrDie(client, "pod-patch-ephemeral-containers", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	testPod := func(name string) *v1.Pod {
		return &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: name,
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:                     "fake-name",
						Image:                    "fakeimage",
						ImagePullPolicy:          "Always",
						TerminationMessagePolicy: "File",
					},
				},
			},
		}
	}

	cases := []struct {
		name      string
		original  []v1.EphemeralContainer
		patchType types.PatchType
		patchBody []byte
		valid     bool
	}{
		{
			name:      "create single container (strategic)",
			original:  nil,
			patchType: types.StrategicMergePatchType,
			patchBody: []byte(`{
				"spec": {
					"ephemeralContainers": [{
						"name": "debugger1",
						"image": "debugimage",
						"imagePullPolicy": "Always",
						"terminationMessagePolicy": "File"
					}]
				}
			}`),
			valid: true,
		},
		{
			name:      "create single container (merge)",
			original:  nil,
			patchType: types.MergePatchType,
			patchBody: []byte(`{
				"spec": {
					"ephemeralContainers":[{
						"name": "debugger1",
						"image": "debugimage",
						"imagePullPolicy": "Always",
						"terminationMessagePolicy": "File"
					}]
				}
			}`),
			valid: true,
		},
		{
			name:      "create single container (JSON)",
			original:  nil,
			patchType: types.JSONPatchType,
			// Because ephemeralContainers is optional, a JSON patch of an empty ephemeralContainers must add the
			// list rather than simply appending to it.
			patchBody: []byte(`[{
				"op":"add",
				"path":"/spec/ephemeralContainers",
				"value":[{
					"name":"debugger1",
					"image":"debugimage",
					"imagePullPolicy": "Always",
					"terminationMessagePolicy": "File"
				}]
			}]`),
			valid: true,
		},
		{
			name: "add single container (strategic)",
			original: []v1.EphemeralContainer{
				{
					EphemeralContainerCommon: v1.EphemeralContainerCommon{
						Name:                     "debugger1",
						Image:                    "debugimage",
						ImagePullPolicy:          "Always",
						TerminationMessagePolicy: "File",
					},
				},
			},
			patchType: types.StrategicMergePatchType,
			patchBody: []byte(`{
				"spec": {
					"ephemeralContainers":[{
						"name": "debugger2",
						"image": "debugimage",
						"imagePullPolicy": "Always",
						"terminationMessagePolicy": "File"
					}]
				}
			}`),
			valid: true,
		},
		{
			name: "add single container (merge)",
			original: []v1.EphemeralContainer{
				{
					EphemeralContainerCommon: v1.EphemeralContainerCommon{
						Name:                     "debugger1",
						Image:                    "debugimage",
						ImagePullPolicy:          "Always",
						TerminationMessagePolicy: "File",
					},
				},
			},
			patchType: types.MergePatchType,
			patchBody: []byte(`{
				"spec": {
					"ephemeralContainers":[{
						"name": "debugger1",
						"image": "debugimage",
						"imagePullPolicy": "Always",
						"terminationMessagePolicy": "File"
					},{
						"name": "debugger2",
						"image": "debugimage",
						"imagePullPolicy": "Always",
						"terminationMessagePolicy": "File"
					}]
				} 
			}`),
			valid: true,
		},
		{
			name: "add single container (JSON)",
			original: []v1.EphemeralContainer{
				{
					EphemeralContainerCommon: v1.EphemeralContainerCommon{
						Name:                     "debugger1",
						Image:                    "debugimage",
						ImagePullPolicy:          "Always",
						TerminationMessagePolicy: "File",
					},
				},
			},
			patchType: types.JSONPatchType,
			patchBody: []byte(`[{
				"op":"add",
				"path":"/spec/ephemeralContainers/-",
				"value":{
					"name":"debugger2",
					"image":"debugimage",
					"imagePullPolicy": "Always",
					"terminationMessagePolicy": "File"
				}
			}]`),
			valid: true,
		},
		{
			name: "remove all containers (merge)",
			original: []v1.EphemeralContainer{
				{
					EphemeralContainerCommon: v1.EphemeralContainerCommon{
						Name:                     "debugger1",
						Image:                    "debugimage",
						ImagePullPolicy:          "Always",
						TerminationMessagePolicy: "File",
					},
				},
			},
			patchType: types.MergePatchType,
			patchBody: []byte(`{"spec": {"ephemeralContainers":[]}}`),
			valid:     false,
		},
		{
			name: "remove the single container (JSON)",
			original: []v1.EphemeralContainer{
				{
					EphemeralContainerCommon: v1.EphemeralContainerCommon{
						Name:                     "debugger1",
						Image:                    "debugimage",
						ImagePullPolicy:          "Always",
						TerminationMessagePolicy: "File",
					},
				},
			},
			patchType: types.JSONPatchType,
			patchBody: []byte(`[{"op":"remove","path":"/spec/ephemeralContainers/0"}]`),
			valid:     false, // disallowed by policy rather than patch semantics
		},
		{
			name: "remove all containers (JSON)",
			original: []v1.EphemeralContainer{
				{
					EphemeralContainerCommon: v1.EphemeralContainerCommon{
						Name:                     "debugger1",
						Image:                    "debugimage",
						ImagePullPolicy:          "Always",
						TerminationMessagePolicy: "File",
					},
				},
			},
			patchType: types.JSONPatchType,
			patchBody: []byte(`[{"op":"remove","path":"/spec/ephemeralContainers"}]`),
			valid:     false, // disallowed by policy rather than patch semantics
		},
	}

	for i, tc := range cases {
		pod := testPod(fmt.Sprintf("ephemeral-container-test-%v", i))
		if _, err := setUpEphemeralContainers(client.CoreV1().Pods(ns.Name), pod, tc.original); err != nil {
			t.Errorf("%v: %v", tc.name, err)
		}

		if _, err := client.CoreV1().Pods(ns.Name).Patch(context.TODO(), pod.Name, tc.patchType, tc.patchBody, metav1.PatchOptions{}, "ephemeralcontainers"); tc.valid && err != nil {
			t.Errorf("%v: failed to update ephemeral containers: %v", tc.name, err)
		} else if !tc.valid && err == nil {
			t.Errorf("%v: unexpected allowed update to ephemeral containers", tc.name)
		}

		integration.DeletePodOrErrorf(t, client, ns.Name, pod.Name)
	}
}

func TestPodUpdateEphemeralContainers(t *testing.T) {
	// Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer server.TearDownFn()

	client := clientset.NewForConfigOrDie(server.ClientConfig)

	ns := framework.CreateNamespaceOrDie(client, "pod-update-ephemeral-containers", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	testPod := func(name string) *v1.Pod {
		return &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: name,
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "fake-name",
						Image: "fakeimage",
					},
				},
			},
		}
	}

	cases := []struct {
		name     string
		original []v1.EphemeralContainer
		update   []v1.EphemeralContainer
		valid    bool
	}{
		{
			name:     "no change, nil",
			original: nil,
			update:   nil,
			valid:    true,
		},
		{
			name: "no change, set",
			original: []v1.EphemeralContainer{
				{
					EphemeralContainerCommon: v1.EphemeralContainerCommon{
						Name:                     "debugger",
						Image:                    "debugimage",
						ImagePullPolicy:          "Always",
						TerminationMessagePolicy: "File",
					},
				},
			},
			update: []v1.EphemeralContainer{
				{
					EphemeralContainerCommon: v1.EphemeralContainerCommon{
						Name:                     "debugger",
						Image:                    "debugimage",
						ImagePullPolicy:          "Always",
						TerminationMessagePolicy: "File",
					},
				},
			},
			valid: true,
		},
		{
			name:     "add single container",
			original: nil,
			update: []v1.EphemeralContainer{
				{
					EphemeralContainerCommon: v1.EphemeralContainerCommon{
						Name:                     "debugger",
						Image:                    "debugimage",
						ImagePullPolicy:          "Always",
						TerminationMessagePolicy: "File",
					},
				},
			},
			valid: true,
		},
		{
			name: "remove all containers, nil",
			original: []v1.EphemeralContainer{
				{
					EphemeralContainerCommon: v1.EphemeralContainerCommon{
						Name:                     "debugger",
						Image:                    "debugimage",
						ImagePullPolicy:          "Always",
						TerminationMessagePolicy: "File",
					},
				},
			},
			update: nil,
			valid:  false,
		},
		{
			name: "remove all containers, empty",
			original: []v1.EphemeralContainer{
				{
					EphemeralContainerCommon: v1.EphemeralContainerCommon{
						Name:                     "debugger",
						Image:                    "debugimage",
						ImagePullPolicy:          "Always",
						TerminationMessagePolicy: "File",
					},
				},
			},
			update: []v1.EphemeralContainer{},
			valid:  false,
		},
		{
			name: "increase number of containers",
			original: []v1.EphemeralContainer{
				{
					EphemeralContainerCommon: v1.EphemeralContainerCommon{
						Name:                     "debugger1",
						Image:                    "debugimage",
						ImagePullPolicy:          "Always",
						TerminationMessagePolicy: "File",
					},
				},
			},
			update: []v1.EphemeralContainer{
				{
					EphemeralContainerCommon: v1.EphemeralContainerCommon{
						Name:                     "debugger1",
						Image:                    "debugimage",
						ImagePullPolicy:          "Always",
						TerminationMessagePolicy: "File",
					},
				},
				{
					EphemeralContainerCommon: v1.EphemeralContainerCommon{
						Name:                     "debugger2",
						Image:                    "debugimage",
						ImagePullPolicy:          "Always",
						TerminationMessagePolicy: "File",
					},
				},
			},
			valid: true,
		},
		{
			name: "decrease number of containers",
			original: []v1.EphemeralContainer{
				{
					EphemeralContainerCommon: v1.EphemeralContainerCommon{
						Name:                     "debugger1",
						Image:                    "debugimage",
						ImagePullPolicy:          "Always",
						TerminationMessagePolicy: "File",
					},
				},
				{
					EphemeralContainerCommon: v1.EphemeralContainerCommon{
						Name:                     "debugger2",
						Image:                    "debugimage",
						ImagePullPolicy:          "Always",
						TerminationMessagePolicy: "File",
					},
				},
			},
			update: []v1.EphemeralContainer{
				{
					EphemeralContainerCommon: v1.EphemeralContainerCommon{
						Name:                     "debugger1",
						Image:                    "debugimage",
						ImagePullPolicy:          "Always",
						TerminationMessagePolicy: "File",
					},
				},
			},
			valid: false,
		},
	}

	for i, tc := range cases {
		pod, err := setUpEphemeralContainers(client.CoreV1().Pods(ns.Name), testPod(fmt.Sprintf("ephemeral-container-test-%v", i)), tc.original)
		if err != nil {
			t.Errorf("%v: %v", tc.name, err)
		}

		pod.Spec.EphemeralContainers = tc.update
		if _, err := client.CoreV1().Pods(ns.Name).UpdateEphemeralContainers(context.TODO(), pod.Name, pod, metav1.UpdateOptions{}); tc.valid && err != nil {
			t.Errorf("%v: failed to update ephemeral containers: %v", tc.name, err)
		} else if !tc.valid && err == nil {
			t.Errorf("%v: unexpected allowed update to ephemeral containers", tc.name)
		}

		integration.DeletePodOrErrorf(t, client, ns.Name, pod.Name)
	}
}

func TestPodResizeRBAC(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodVerticalScaling, true)
	server := kubeapiservertesting.StartTestServerOrDie(t, nil,
		append(framework.DefaultTestServerFlags(), "--authorization-mode=RBAC"), framework.SharedEtcd())
	defer server.TearDownFn()
	adminClient := clientset.NewForConfigOrDie(server.ClientConfig)

	ns := framework.CreateNamespaceOrDie(adminClient, "pod-resize", t)
	defer framework.DeleteNamespaceOrDie(adminClient, ns, t)

	testPod := func(name string) *v1.Pod {
		return &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: name,
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "fake-name",
						Image: "fakeimage",
					},
				},
			},
		}
	}

	testcases := []struct {
		name               string
		serviceAccountFn   func(t *testing.T, adminClient *clientset.Clientset, clientConfig *rest.Config, rules []rbacv1.PolicyRule) *clientset.Clientset
		serviceAccountRBAC rbacv1.PolicyRule
		allowResize        bool
		allowUpdate        bool
	}{
		{
			name:               "pod-mutator",
			serviceAccountFn:   authutil.ServiceAccountClient(ns.Name, "pod-mutator"),
			serviceAccountRBAC: rbachelper.NewRule("get", "update", "patch").Groups("").Resources("pods").RuleOrDie(),
			allowResize:        false,
			allowUpdate:        true,
		},
		{
			name:               "pod-resizer",
			serviceAccountFn:   authutil.ServiceAccountClient(ns.Name, "pod-resizer"),
			serviceAccountRBAC: rbachelper.NewRule("get", "update", "patch").Groups("").Resources("pods/resize").RuleOrDie(),
			allowResize:        true,
			allowUpdate:        false,
		},
	}

	for i, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			// 1. Create a test pod.
			pod := testPod(fmt.Sprintf("resize-%d", i))
			resp, err := adminClient.CoreV1().Pods(ns.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("Unexpected error when creating pod: %v", err)
				integration.DeletePodOrErrorf(t, adminClient, ns.Name, pod.Name)
			}

			// 2. Create a service account and fetch its client.
			saClient := tc.serviceAccountFn(t, adminClient, server.ClientConfig, []rbacv1.PolicyRule{tc.serviceAccountRBAC})

			// 3. Update pod and check whether it should be allowed.
			resp.Spec.Containers[0].Image = "updated-image"
			if _, err := saClient.CoreV1().Pods(ns.Name).Update(context.TODO(), resp, metav1.UpdateOptions{}); err == nil && !tc.allowUpdate {
				t.Fatalf("Unexpected allowed pod update")
				integration.DeletePodOrErrorf(t, adminClient, ns.Name, pod.Name)
			} else if err != nil && tc.allowUpdate {
				t.Fatalf("Unexpected error when updating pod container resources: %v", err)
				integration.DeletePodOrErrorf(t, adminClient, ns.Name, pod.Name)
			}

			// 4. Resize pod container resource and check whether it should be allowed.
			resp, err = adminClient.CoreV1().Pods(ns.Name).Get(context.TODO(), resp.Name, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("Unexpected error when fetching the pod: %v", err)
				integration.DeletePodOrErrorf(t, adminClient, ns.Name, pod.Name)
			}
			resp.Spec.Containers[0].Resources = v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceEphemeralStorage: resource.MustParse("2Gi"),
				},
			}
			_, err = saClient.CoreV1().Pods(ns.Name).UpdateResize(context.TODO(), resp.Name, resp, metav1.UpdateOptions{})
			if tc.allowResize && err != nil {
				t.Fatalf("Unexpected pod resize failure: %v", err)
				integration.DeletePodOrErrorf(t, adminClient, ns.Name, pod.Name)
			}
			if !tc.allowResize && err == nil {
				t.Fatalf("Unexpected pod resize success")
				integration.DeletePodOrErrorf(t, adminClient, ns.Name, pod.Name)
			}

			// 5. Delete the test pod.
			integration.DeletePodOrErrorf(t, adminClient, ns.Name, pod.Name)
		})
	}
}

func TestPodResize(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodVerticalScaling, true)
	// Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer server.TearDownFn()

	client := clientset.NewForConfigOrDie(server.ClientConfig)

	ns := framework.CreateNamespaceOrDie(client, "pod-resize", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	testPod := func(name string) *v1.Pod {
		return &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: name,
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "fake-name",
						Image: "fakeimage",
					},
				},
			},
		}
	}

	resizeCases := []struct {
		name        string
		originalRes v1.ResourceRequirements
		resize      v1.ResourceRequirements
		valid       bool
	}{
		{
			name: "cpu request change",
			originalRes: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceCPU: resource.MustParse("10m"),
				},
			},
			resize: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceCPU: resource.MustParse("20m"),
				},
			},
			valid: true,
		},
		{
			name: "memory request change",
			originalRes: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceMemory: resource.MustParse("1Gi"),
				},
			},
			resize: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceMemory: resource.MustParse("2Gi"),
				},
			},
			valid: true,
		},
		{
			name: "storage request change",
			originalRes: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceEphemeralStorage: resource.MustParse("1Gi"),
				},
			},
			resize: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceEphemeralStorage: resource.MustParse("2Gi"),
				},
			},
			valid: false,
		},
	}

	for _, tc := range resizeCases {
		pod := testPod("resize")
		pod.Spec.Containers[0].Resources = tc.originalRes
		resp, err := client.CoreV1().Pods(ns.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Unexpected error when creating pod: %v", err)
			integration.DeletePodOrErrorf(t, client, ns.Name, pod.Name)
		}

		// Part 1. Resize
		resp.Spec.Containers[0].Resources = tc.resize
		if _, err := client.CoreV1().Pods(ns.Name).Update(context.TODO(), resp, metav1.UpdateOptions{}); err == nil {
			t.Fatalf("Unexpected allowed pod update")
			integration.DeletePodOrErrorf(t, client, ns.Name, pod.Name)
		} else if !strings.Contains(err.Error(), "spec: Forbidden: pod updates may not change fields other than") {
			t.Fatalf("Unexpected error when updating pod container resources: %v", err)
			integration.DeletePodOrErrorf(t, client, ns.Name, pod.Name)
		}

		resp, err = client.CoreV1().Pods(ns.Name).UpdateResize(context.TODO(), resp.Name, resp, metav1.UpdateOptions{})
		if tc.valid && err != nil {
			t.Fatalf("Unexpected pod resize failure: %v", err)
			integration.DeletePodOrErrorf(t, client, ns.Name, pod.Name)
		}
		if !tc.valid && err == nil {
			t.Fatalf("Unexpected pod resize success")
			integration.DeletePodOrErrorf(t, client, ns.Name, pod.Name)
		}

		// Part 2. Rollback
		if !tc.valid {
			integration.DeletePodOrErrorf(t, client, ns.Name, pod.Name)
			continue
		}
		resp.Spec.Containers[0].Resources = tc.originalRes
		_, err = client.CoreV1().Pods(ns.Name).UpdateResize(context.TODO(), resp.Name, resp, metav1.UpdateOptions{})
		if tc.valid && err != nil {
			t.Fatalf("Unexpected pod resize failure: %v", err)
			integration.DeletePodOrErrorf(t, client, ns.Name, pod.Name)
		}
		if !tc.valid && err == nil {
			t.Fatalf("Unexpected pod resize success")
			integration.DeletePodOrErrorf(t, client, ns.Name, pod.Name)
		}

		integration.DeletePodOrErrorf(t, client, ns.Name, pod.Name)
	}

	patchCases := []struct {
		name        string
		originalRes v1.ResourceRequirements
		patchBody   string
		patchType   types.PatchType
		valid       bool
	}{
		{
			name: "cpu request change (strategic)",
			originalRes: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceCPU: resource.MustParse("10m"),
				},
			},
			patchType: types.StrategicMergePatchType,
			patchBody: `{
				"spec":{
					"containers":[
						{
							"name":"fake-name",
							"resources": {
								"requests": {
									"cpu":"20m"
								}
							}
						}
					]
				}
			}`,
			valid: true,
		},
		{
			name: "cpu request change (merge)",
			originalRes: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceCPU: resource.MustParse("10m"),
				},
			},
			patchType: types.MergePatchType,
			patchBody: `{
				"spec":{
					"containers":[
						{
							"name":"fake-name",
							"resources": {
								"requests": {
									"cpu":"20m"
								}
							}
						}
					]
				}
			}`,
			valid: true,
		},
		{
			name: "cpu request change (JSON)",
			originalRes: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceCPU: resource.MustParse("10m"),
				},
			},
			patchType: types.JSONPatchType,
			patchBody: `[{
				"op":"add",
				"path":"/spec/containers",
				"value":[{
					"name":"fake-name",
					"resources": {
						"requests": {
							"cpu":"20m"
						}
					}
				}]
			}]`,
			valid: true,
		},
		{
			name: "storage request change (merge)",
			originalRes: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceCPU: resource.MustParse("10m"),
				},
			},
			patchType: types.MergePatchType,
			patchBody: `{
				"spec":{
					"containers":[
						{
							"name":"fake-name",
							"resources": {
								"requests": {
									"ephemeral-storage":"20m"
								}
							}
						}
					]
				}
			}`,
			valid: false,
		},
	}

	for _, tc := range patchCases {
		pod := testPod("resize")
		pod.Spec.Containers[0].Resources = tc.originalRes
		if _, err := client.CoreV1().Pods(ns.Name).Create(context.TODO(), pod, metav1.CreateOptions{}); err != nil {
			t.Fatalf("Unexpected error when creating pod: %v", err)
			integration.DeletePodOrErrorf(t, client, ns.Name, pod.Name)
		}

		if _, err := client.CoreV1().Pods(ns.Name).Patch(context.TODO(), pod.Name, tc.patchType, []byte(tc.patchBody), metav1.PatchOptions{}, "resize"); tc.valid && err != nil {
			t.Fatalf("Unexpected pod resize failure: %v", err)
			integration.DeletePodOrErrorf(t, client, ns.Name, pod.Name)
		} else if !tc.valid && err == nil {
			t.Fatalf("Unexpected pod resize success")
			integration.DeletePodOrErrorf(t, client, ns.Name, pod.Name)
		}
		integration.DeletePodOrErrorf(t, client, ns.Name, pod.Name)
	}
}

func TestMutablePodSchedulingDirectives(t *testing.T) {
	// Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer server.TearDownFn()

	client := clientset.NewForConfigOrDie(server.ClientConfig)

	ns := framework.CreateNamespaceOrDie(client, "mutable-pod-scheduling-directives", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	cases := []struct {
		name   string
		create *v1.Pod
		update *v1.Pod
		err    string
	}{
		{
			name: "adding node selector is allowed for gated pods",
			create: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-pod",
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "fake-name",
							Image: "fakeimage",
						},
					},
					SchedulingGates: []v1.PodSchedulingGate{{Name: "baz"}},
				},
			},
			update: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-pod",
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "fake-name",
							Image: "fakeimage",
						},
					},
					NodeSelector: map[string]string{
						"foo": "bar",
					},
					SchedulingGates: []v1.PodSchedulingGate{{Name: "baz"}},
				},
			},
		},
		{
			name: "addition to nodeAffinity is allowed for gated pods",
			create: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-pod",
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "fake-name",
							Image: "fakeimage",
						},
					},
					Affinity: &v1.Affinity{
						NodeAffinity: &v1.NodeAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
								NodeSelectorTerms: []v1.NodeSelectorTerm{
									{
										MatchExpressions: []v1.NodeSelectorRequirement{
											{
												Key:      "expr",
												Operator: v1.NodeSelectorOpIn,
												Values:   []string{"foo"},
											},
										},
									},
								},
							},
						},
					},
					SchedulingGates: []v1.PodSchedulingGate{{Name: "baz"}},
				},
			},
			update: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-pod",
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "fake-name",
							Image: "fakeimage",
						},
					},
					Affinity: &v1.Affinity{
						NodeAffinity: &v1.NodeAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
								// Add 1 MatchExpression and 1 MatchField.
								NodeSelectorTerms: []v1.NodeSelectorTerm{
									{
										MatchExpressions: []v1.NodeSelectorRequirement{
											{
												Key:      "expr",
												Operator: v1.NodeSelectorOpIn,
												Values:   []string{"foo"},
											},
											{
												Key:      "expr2",
												Operator: v1.NodeSelectorOpIn,
												Values:   []string{"foo2"},
											},
										},
										MatchFields: []v1.NodeSelectorRequirement{
											{
												Key:      "metadata.name",
												Operator: v1.NodeSelectorOpIn,
												Values:   []string{"foo"},
											},
										},
									},
								},
							},
						},
					},
					SchedulingGates: []v1.PodSchedulingGate{{Name: "baz"}},
				},
			},
		},
		{
			name: "addition to nodeAffinity is allowed for gated pods with nil affinity",
			create: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-pod",
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "fake-name",
							Image: "fakeimage",
						},
					},
					SchedulingGates: []v1.PodSchedulingGate{{Name: "baz"}},
				},
			},
			update: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-pod",
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "fake-name",
							Image: "fakeimage",
						},
					},
					Affinity: &v1.Affinity{
						NodeAffinity: &v1.NodeAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
								// Add 1 MatchExpression and 1 MatchField.
								NodeSelectorTerms: []v1.NodeSelectorTerm{
									{
										MatchExpressions: []v1.NodeSelectorRequirement{
											{
												Key:      "expr",
												Operator: v1.NodeSelectorOpIn,
												Values:   []string{"foo"},
											},
										},
										MatchFields: []v1.NodeSelectorRequirement{
											{
												Key:      "metadata.name",
												Operator: v1.NodeSelectorOpIn,
												Values:   []string{"foo"},
											},
										},
									},
								},
							},
						},
					},
					SchedulingGates: []v1.PodSchedulingGate{{Name: "baz"}},
				},
			},
		},
	}
	for _, tc := range cases {
		if _, err := client.CoreV1().Pods(ns.Name).Create(context.TODO(), tc.create, metav1.CreateOptions{}); err != nil {
			t.Errorf("Failed to create pod: %v", err)
		}

		_, err := client.CoreV1().Pods(ns.Name).Update(context.TODO(), tc.update, metav1.UpdateOptions{})
		if (tc.err == "" && err != nil) || (tc.err != "" && err != nil && !strings.Contains(err.Error(), tc.err)) {
			t.Errorf("Unexpected error: got %q, want %q", err.Error(), err)
		}
		integration.DeletePodOrErrorf(t, client, ns.Name, tc.update.Name)
	}
}

// Test disabling of RelaxedDNSSearchValidation after a Pod has been created
func TestRelaxedDNSSearchValidation(t *testing.T) {
	// Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
	server := kubeapiservertesting.StartTestServerOrDie(t, nil,
		append(framework.DefaultTestServerFlags(), "--emulated-version=1.32"), framework.SharedEtcd())
	defer server.TearDownFn()

	client := clientset.NewForConfigOrDie(server.ClientConfig)

	ns := framework.CreateNamespaceOrDie(client, "pod-update-dns-search", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	testPod := func(name string) *v1.Pod {
		return &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: name,
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "fake-name",
						Image: "fakeimage",
					},
				},
			},
		}
	}

	cases := []struct {
		name               string
		original           *v1.PodDNSConfig
		valid              bool
		featureGateEnabled bool
		update             bool
	}{
		{
			name:               "new pod with underscore - feature gate enabled",
			original:           &v1.PodDNSConfig{Searches: []string{"_sip._tcp.abc_d.example.com"}},
			valid:              true,
			featureGateEnabled: true,
		},
		{
			name:               "new pod with dot - feature gate enabled",
			original:           &v1.PodDNSConfig{Searches: []string{"."}},
			valid:              true,
			featureGateEnabled: true,
		},

		{
			name:               "new pod without underscore - feature gate enabled",
			original:           &v1.PodDNSConfig{Searches: []string{"example.com"}},
			valid:              true,
			featureGateEnabled: true,
		},
		{
			name:               "new pod with underscore - feature gate disabled",
			original:           &v1.PodDNSConfig{Searches: []string{"_sip._tcp.abc_d.example.com"}},
			valid:              false,
			featureGateEnabled: false,
		},
		{
			name:               "new pod with dot - feature gate disabled",
			original:           &v1.PodDNSConfig{Searches: []string{"."}},
			valid:              false,
			featureGateEnabled: false,
		},
		{
			name:               "new pod without underscore - feature gate disabled",
			original:           &v1.PodDNSConfig{Searches: []string{"example.com"}},
			valid:              true,
			featureGateEnabled: false,
		},
	}

	for _, tc := range cases {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.RelaxedDNSSearchValidation, tc.featureGateEnabled)
		pod := testPod("dns")
		pod.Spec.DNSConfig = tc.original
		_, err := client.CoreV1().Pods(ns.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
		if tc.valid && err != nil {
			t.Errorf("%v: %v", tc.name, err)
		} else if !tc.valid && err == nil {
			t.Errorf("%v: unexpected allowed update to ephemeral containers", tc.name)
		}

		// Disable gate and perform update
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.RelaxedDNSSearchValidation, false)
		pod.ObjectMeta.Labels = map[string]string{"label": "value"}
		_, err = client.CoreV1().Pods(ns.Name).Update(context.TODO(), pod, metav1.UpdateOptions{})

		if tc.valid && err != nil {
			t.Errorf("%v: failed to update ephemeral containers: %v", tc.name, err)
		} else if !tc.valid && err == nil {
			t.Errorf("%v: unexpected allowed update to ephemeral containers", tc.name)
		}

		if tc.valid {
			integration.DeletePodOrErrorf(t, client, ns.Name, pod.Name)
		}
	}
}

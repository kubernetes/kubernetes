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

package dynamicresources

import (
	"context"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	corev1 "k8s.io/api/core/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	corev1ac "k8s.io/client-go/applyconfigurations/core/v1"
	metav1ac "k8s.io/client-go/applyconfigurations/meta/v1"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/klog/v2/ktesting"
	_ "k8s.io/klog/v2/ktesting/init"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/runtime"
)

var (
	podName              = "my-pod"
	podUID               = types.UID("1234")
	resourceName         = "my-resource"
	claimName            = podName + "-" + resourceName
	namespace            = "default"
	simplePod            = buildPod(corev1ac.Pod(podName, "default"))
	podWithClaimName     = buildPod(corev1ac.Pod(podName, namespace).WithUID(podUID).WithSpec(corev1ac.PodSpec().WithResourceClaims(corev1ac.PodResourceClaim().WithName(resourceName).WithClaim(corev1ac.ClaimSource().WithResourceClaimName(claimName)))))
	podWithClaimTemplate = buildPod(corev1ac.Pod(podName, namespace).WithUID(podUID).WithSpec(corev1ac.PodSpec().WithResourceClaims(corev1ac.PodResourceClaim().WithName(resourceName).WithClaim(corev1ac.ClaimSource().WithTemplate(corev1ac.ResourceClaimTemplate())))))
	workerNode           = buildNode(corev1ac.Node("worker"))
	allocatedClaim       = buildResourceClaim(corev1ac.ResourceClaim(claimName, namespace).WithOwnerReferences(metav1ac.OwnerReference().WithUID(podUID).WithController(true)).WithStatus(corev1ac.ResourceClaimStatus().WithAllocation(corev1ac.AllocationResult())))
	otherClaim           = buildResourceClaim(corev1ac.ResourceClaim("not-my-claim", namespace))
)

func TestDynamicResources(t *testing.T) {
	testcases := map[string]struct {
		pod    *corev1.Pod
		nodes  []*corev1.Node
		claims []*corev1.ResourceClaim

		wantPreFilterStatus     *framework.Status
		wantPreFilterResult     *framework.PreFilterResult
		wantStateAfterPreFilter stateData

		wantFilterStatus     []*framework.Status
		wantStateAfterFilter stateData

		wantPreScoreStatus *framework.Status
	}{
		"empty": {
			pod: buildPod(corev1ac.Pod("foo", "default")),
		},
		"claim-reference": {
			pod:    podWithClaimName,
			nodes:  []*corev1.Node{workerNode},
			claims: []*corev1.ResourceClaim{allocatedClaim, otherClaim},
			wantStateAfterPreFilter: stateData{
				claims: []*corev1.ResourceClaim{allocatedClaim},
			},
			wantFilterStatus: []*framework.Status{nil},
			wantStateAfterFilter: stateData{
				claims: []*corev1.ResourceClaim{allocatedClaim},
			},
		},
		"claim-template": {
			pod:    podWithClaimTemplate,
			nodes:  []*corev1.Node{workerNode},
			claims: []*corev1.ResourceClaim{allocatedClaim, otherClaim},
			wantStateAfterPreFilter: stateData{
				claims: []*corev1.ResourceClaim{allocatedClaim},
			},
			wantFilterStatus: []*framework.Status{nil},
			wantStateAfterFilter: stateData{
				claims: []*corev1.ResourceClaim{allocatedClaim},
			},
		},
		"missing-claim": {
			pod:                 podWithClaimTemplate,
			nodes:               []*corev1.Node{workerNode},
			wantPreFilterStatus: framework.NewStatus(framework.UnschedulableAndUnresolvable, `waiting for dynamic resource controller to create the resourceclaim "my-pod-my-resource"`),
		},
		// TODO: *lots* of additional test cases
		// TODO: how do we verify Update calls? Mock them? See https://github.com/kubernetes/kubernetes/blob/75e49ec824b183288e1dbaccfd7dbe77d89db381/pkg/controller/endpointslice/endpointslice_controller_test.go#L1061

	}

	for name, testcase := range testcases {
		// We can run in parallel because logging is per-test.
		testcase := testcase
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			tc := setup(t, testcase.nodes, testcase.claims)
			defer tc.cleanup()

			gotPreFilterResult, gotPreFilterStatus := tc.p.PreFilter(tc.ctx, tc.state, testcase.pod)
			require.Equal(t, testcase.wantPreFilterResult, gotPreFilterResult)
			require.Equal(t, testcase.wantPreFilterStatus, gotPreFilterStatus)
			if !gotPreFilterStatus.IsSuccess() {
				// scheduler framework will skip Filter if PreFilter fails
				return
			}
			require.Equal(t, &testcase.wantStateAfterPreFilter, tc.currentState())

			for i, nodeInfo := range tc.nodeInfos {
				gotStatus := tc.p.Filter(tc.ctx, tc.state, testcase.pod, nodeInfo)
				assert.Equal(t, testcase.wantFilterStatus[i], gotStatus, "node %s", nodeInfo.Node().Name)
			}
			require.Equal(t, &testcase.wantStateAfterFilter, tc.currentState())

			gotStatus := tc.p.PreScore(tc.ctx, tc.state, testcase.pod, testcase.nodes)
			// TODO: verify Update failures
			assert.Equal(t, testcase.wantPreScoreStatus, gotStatus)
		})
	}
}

type testContext struct {
	ctx       context.Context
	client    *fake.Clientset
	p         *dynamicResources
	nodeInfos []*framework.NodeInfo
	state     *framework.CycleState
	cleanup   func()
}

// currentState returns stateData without the fields that we don't want to
// compare.
func (tc *testContext) currentState() *stateData {
	state, err := getStateData(tc.state)
	if err != nil {
		panic(err)
	}
	s := *state
	s.clientset = nil
	s.mutex = sync.Mutex{}
	return &s
}

func setup(t *testing.T, nodes []*v1.Node, claims []*corev1.ResourceClaim) (result *testContext) {
	tc := &testContext{}
	_, ctx := ktesting.NewTestContext(t)
	tc.ctx, tc.cleanup = context.WithCancel(ctx)
	defer func() {
		if result == nil {
			// Must call cancel ourselves.
			tc.cleanup()
		}
	}()

	tc.client = fake.NewSimpleClientset()
	informerFactory := informers.NewSharedInformerFactory(tc.client, 0)

	opts := []runtime.Option{
		runtime.WithClientSet(tc.client),
		runtime.WithInformerFactory(informerFactory),
	}
	fh, err := runtime.NewFramework(nil, nil, tc.ctx.Done(), opts...)
	if err != nil {
		t.Fatal(err)
	}

	pl, err := New(nil, fh)
	if err != nil {
		t.Fatal(err)
	}
	tc.p = pl.(*dynamicResources)

	for _, claim := range claims {
		tc.client.CoreV1().ResourceClaims(claim.Namespace).Create(tc.ctx, claim, metav1.CreateOptions{})
	}

	informerFactory.Start(tc.ctx.Done())

	informerFactory.WaitForCacheSync(tc.ctx.Done())

	for _, node := range nodes {
		nodeInfo := framework.NewNodeInfo()
		nodeInfo.SetNode(node)
		tc.nodeInfos = append(tc.nodeInfos, nodeInfo)
	}
	tc.state = framework.NewCycleState()

	return tc
}

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

package flowcontrol

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	flowcontrol "k8s.io/api/flowcontrol/v1beta2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	machinerytypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	fcboot "k8s.io/apiserver/pkg/apis/flowcontrol/bootstrap"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	flowcontrolapply "k8s.io/client-go/applyconfigurations/flowcontrol/v1beta2"
	clientset "k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
)

func TestConditionIsolation(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.APIPriorityAndFairness, true)()
	// NOTE: disabling the feature should fail the test
	kubeConfig, closeFn := setup(t, 10, 10)
	defer closeFn()

	loopbackClient := clientset.NewForConfigOrDie(kubeConfig)

	stopCh := make(chan struct{})
	defer close(stopCh)
	ctx := context.Background()

	fsOrig := fcboot.SuggestedFlowSchemas[0]
	t.Logf("Testing Status Condition isolation in FlowSchema %q", fsOrig.Name)
	fsClient := loopbackClient.FlowcontrolV1beta2().FlowSchemas()
	var dangleOrig *flowcontrol.FlowSchemaCondition

	wait.PollUntil(time.Second, func() (bool, error) {
		fsGot, err := fsClient.Get(ctx, fsOrig.Name, metav1.GetOptions{})
		if err != nil {
			klog.Errorf("Failed to fetch FlowSchema %q: %v", fsOrig.Name, err)
			return false, nil
		}
		dangleOrig = getCondition(fsGot.Status.Conditions, flowcontrol.FlowSchemaConditionDangling)
		return dangleOrig != nil, nil
	}, stopCh)

	ssaType := flowcontrol.FlowSchemaConditionType("test-ssa")
	patchSSA := flowcontrolapply.FlowSchema(fsOrig.Name).
		WithStatus(flowcontrolapply.FlowSchemaStatus().
			WithConditions(flowcontrolapply.FlowSchemaCondition().
				WithType(ssaType).
				WithStatus(flowcontrol.ConditionTrue).
				WithReason("SSA test").
				WithMessage("for testing").
				WithLastTransitionTime(metav1.Now()),
			))

	postSSA, err := fsClient.ApplyStatus(ctx, patchSSA, metav1.ApplyOptions{FieldManager: "ssa-test"})
	if err != nil {
		t.Error(err)
	}
	danglePostSSA := getCondition(postSSA.Status.Conditions, flowcontrol.FlowSchemaConditionDangling)
	if danglePostSSA == nil || danglePostSSA.Status != dangleOrig.Status {
		t.Errorf("Bad dangle condition after SSA, the FS is now %s", fmtFS(t, postSSA))
	}
	ssaPostSSA := getCondition(postSSA.Status.Conditions, ssaType)
	if ssaPostSSA == nil || ssaPostSSA.Status != flowcontrol.ConditionTrue {
		t.Errorf("Bad SSA condition after SSA, the FS is now %s", fmtFS(t, postSSA))
	}

	smpType := flowcontrol.FlowSchemaConditionType("test-smp")
	smpBytes, err := makeFlowSchemaConditionPatch(flowcontrol.FlowSchemaCondition{
		Type:               smpType,
		Status:             flowcontrol.ConditionFalse,
		Reason:             "SMP test",
		Message:            "for testing too",
		LastTransitionTime: metav1.Now(),
	})
	if err != nil {
		t.Error(err)
	}
	postSMP, err := fsClient.Patch(ctx, fsOrig.Name, machinerytypes.StrategicMergePatchType, smpBytes,
		metav1.PatchOptions{FieldManager: "smp-test"}, "status")
	if err != nil {
		t.Error(err)
	}
	if false /* disabled until patch annotations go into the API (see https://github.com/kubernetes/kubernetes/issues/107574) */ {
		danglePostSMP := getCondition(postSMP.Status.Conditions, flowcontrol.FlowSchemaConditionDangling)
		if danglePostSMP == nil || danglePostSMP.Status != dangleOrig.Status {
			t.Errorf("Bad dangle condition after SMP, the FS is now %s", fmtFS(t, postSMP))
		}
		ssaPostSMP := getCondition(postSMP.Status.Conditions, ssaType)
		if ssaPostSMP == nil || ssaPostSMP.Status != flowcontrol.ConditionTrue {
			t.Errorf("Bad SSA condition after SMP, the FS is now %s", fmtFS(t, postSMP))
		}
	}
	smpPostSMP := getCondition(postSMP.Status.Conditions, smpType)
	if smpPostSMP == nil || smpPostSMP.Status != flowcontrol.ConditionFalse {
		t.Errorf("Bad SMP condition after SMP, the FS is now %s", fmtFS(t, postSMP))
	}

}

func getCondition(conds []flowcontrol.FlowSchemaCondition, condType flowcontrol.FlowSchemaConditionType) *flowcontrol.FlowSchemaCondition {
	for _, cond := range conds {
		if cond.Type == condType {
			return &cond
		}
	}
	return nil
}

// makeFlowSchemaConditionPatch takes in a condition and returns the patch status as a json.
func makeFlowSchemaConditionPatch(condition flowcontrol.FlowSchemaCondition) ([]byte, error) {
	o := struct {
		Status flowcontrol.FlowSchemaStatus `json:"status"`
	}{
		Status: flowcontrol.FlowSchemaStatus{
			Conditions: []flowcontrol.FlowSchemaCondition{
				condition,
			},
		},
	}
	return json.Marshal(o)
}

func fmtFS(t *testing.T, fs *flowcontrol.FlowSchema) string {
	asBytes, err := json.Marshal(fs)
	if err != nil {
		t.Error(err)
	}
	return string(asBytes)
}

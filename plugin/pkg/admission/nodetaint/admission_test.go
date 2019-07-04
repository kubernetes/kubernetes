/*
Copyright 2019 The Kubernetes Authors.

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

package nodetaint

import (
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/component-base/featuregate"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
)

var (
	enableTaintNodesByCondition  = featuregate.NewFeatureGate()
	disableTaintNodesByCondition = featuregate.NewFeatureGate()
)

func init() {
	if err := enableTaintNodesByCondition.Add(map[featuregate.Feature]featuregate.FeatureSpec{features.TaintNodesByCondition: {Default: true}}); err != nil {
		panic(err)
	}
	if err := disableTaintNodesByCondition.Add(map[featuregate.Feature]featuregate.FeatureSpec{features.TaintNodesByCondition: {Default: false}}); err != nil {
		panic(err)
	}
}

func Test_nodeTaints(t *testing.T) {
	var (
		mynode            = &user.DefaultInfo{Name: "system:node:mynode", Groups: []string{"system:nodes"}}
		resource          = api.Resource("nodes").WithVersion("v1")
		notReadyTaint     = api.Taint{Key: TaintNodeNotReady, Effect: api.TaintEffectNoSchedule}
		notReadyCondition = api.NodeCondition{Type: api.NodeReady, Status: api.ConditionFalse}
		myNodeObjMeta     = metav1.ObjectMeta{Name: "mynode"}
		myNodeObj         = api.Node{ObjectMeta: myNodeObjMeta}
		myTaintedNodeObj  = api.Node{ObjectMeta: myNodeObjMeta,
			Spec: api.NodeSpec{Taints: []api.Taint{notReadyTaint}}}
		myUnreadyNodeObj = api.Node{ObjectMeta: myNodeObjMeta,
			Status: api.NodeStatus{Conditions: []api.NodeCondition{notReadyCondition}}}
		nodeKind = api.Kind("Node").WithVersion("v1")
	)
	tests := []struct {
		name           string
		node           api.Node
		oldNode        api.Node
		features       featuregate.FeatureGate
		operation      admission.Operation
		options        runtime.Object
		expectedTaints []api.Taint
	}{
		{
			name:           "notReady taint is added on creation",
			node:           myNodeObj,
			features:       enableTaintNodesByCondition,
			operation:      admission.Create,
			options:        &metav1.CreateOptions{},
			expectedTaints: []api.Taint{notReadyTaint},
		},
		{
			name:           "NotReady taint is not added when TaintNodesByCondition is disabled",
			node:           myNodeObj,
			features:       disableTaintNodesByCondition,
			operation:      admission.Create,
			options:        &metav1.CreateOptions{},
			expectedTaints: nil,
		},
		{
			name:           "already tainted node is not tainted again",
			node:           myTaintedNodeObj,
			features:       enableTaintNodesByCondition,
			operation:      admission.Create,
			options:        &metav1.CreateOptions{},
			expectedTaints: []api.Taint{notReadyTaint},
		},
		{
			name:           "NotReady taint is added to an unready node as well",
			node:           myUnreadyNodeObj,
			features:       enableTaintNodesByCondition,
			operation:      admission.Create,
			options:        &metav1.CreateOptions{},
			expectedTaints: []api.Taint{notReadyTaint},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			attributes := admission.NewAttributesRecord(&tt.node, &tt.oldNode, nodeKind, myNodeObj.Namespace, myNodeObj.Name, resource, "", tt.operation, tt.options, false, mynode)
			c := NewPlugin()
			if tt.features != nil {
				c.features = tt.features
			}
			err := c.Admit(attributes, nil)
			if err != nil {
				t.Errorf("nodePlugin.Admit() error = %v", err)
			}
			node, _ := attributes.GetObject().(*api.Node)
			if !reflect.DeepEqual(node.Spec.Taints, tt.expectedTaints) {
				t.Errorf("Unexpected Node taints. Got %v\nExpected: %v", node.Spec.Taints, tt.expectedTaints)
			}
		})
	}
}

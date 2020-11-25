/*
Copyright 2016 The Kubernetes Authors.

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

package cloud

import (
	"context"
	"errors"
	"reflect"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/klog/v2"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/tools/record"
	cloudprovider "k8s.io/cloud-provider"
	cloudproviderapi "k8s.io/cloud-provider/api"
	fakecloud "k8s.io/cloud-provider/fake"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"
)

func TestEnsureNodeExistsByProviderID(t *testing.T) {

	testCases := []struct {
		testName           string
		node               *v1.Node
		expectedCalls      []string
		expectedNodeExists bool
		hasInstanceID      bool
		existsByProviderID bool
		nodeNameErr        error
		providerIDErr      error
	}{
		{
			testName:           "node exists by provider id",
			existsByProviderID: true,
			providerIDErr:      nil,
			hasInstanceID:      true,
			nodeNameErr:        errors.New("unimplemented"),
			expectedCalls:      []string{"instance-exists-by-provider-id"},
			expectedNodeExists: true,
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node0",
				},
				Spec: v1.NodeSpec{
					ProviderID: "node0",
				},
			},
		},
		{
			testName:           "does not exist by provider id",
			existsByProviderID: false,
			providerIDErr:      nil,
			hasInstanceID:      true,
			nodeNameErr:        errors.New("unimplemented"),
			expectedCalls:      []string{"instance-exists-by-provider-id"},
			expectedNodeExists: false,
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node0",
				},
				Spec: v1.NodeSpec{
					ProviderID: "node0",
				},
			},
		},
		{
			testName:           "exists by instance id",
			existsByProviderID: true,
			providerIDErr:      nil,
			hasInstanceID:      true,
			nodeNameErr:        nil,
			expectedCalls:      []string{"instance-id", "instance-exists-by-provider-id"},
			expectedNodeExists: true,
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node0",
				},
			},
		},
		{
			testName:           "does not exist by no instance id",
			existsByProviderID: true,
			providerIDErr:      nil,
			hasInstanceID:      false,
			nodeNameErr:        cloudprovider.InstanceNotFound,
			expectedCalls:      []string{"instance-id"},
			expectedNodeExists: false,
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node0",
				},
			},
		},
		{
			testName:           "provider id returns error",
			existsByProviderID: false,
			providerIDErr:      errors.New("unimplemented"),
			hasInstanceID:      true,
			nodeNameErr:        cloudprovider.InstanceNotFound,
			expectedCalls:      []string{"instance-exists-by-provider-id"},
			expectedNodeExists: false,
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: "node0",
				},
				Spec: v1.NodeSpec{
					ProviderID: "node0",
				},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.testName, func(t *testing.T) {
			fc := &fakecloud.Cloud{
				ExistsByProviderID: tc.existsByProviderID,
				Err:                tc.nodeNameErr,
				ErrByProviderID:    tc.providerIDErr,
			}

			if tc.hasInstanceID {
				fc.ExtID = map[types.NodeName]string{
					types.NodeName(tc.node.Name): "provider-id://a",
				}
			}

			instances, _ := fc.Instances()
			exists, err := ensureNodeExistsByProviderID(context.TODO(), instances, tc.node)
			assert.Equal(t, err, tc.providerIDErr)

			assert.EqualValues(t, tc.expectedCalls, fc.Calls,
				"expected cloud provider methods `%v` to be called but `%v` was called ",
				tc.expectedCalls, fc.Calls)

			assert.Equal(t, tc.expectedNodeExists, exists,
				"expected exists to be `%t` but got `%t`",
				tc.existsByProviderID, exists)
		})
	}
}

func Test_syncNode(t *testing.T) {
	tests := []struct {
		name         string
		fakeCloud    *fakecloud.Cloud
		existingNode *v1.Node
		updatedNode  *v1.Node
	}{
		{
			name: "node initialized with provider ID",
			fakeCloud: &fakecloud.Cloud{
				EnableInstancesV2: false,
				InstanceTypes: map[types.NodeName]string{
					types.NodeName("node0"): "t1.micro",
				},
				ExtID: map[types.NodeName]string{
					types.NodeName("node0"): "12345",
				},
				Addresses: []v1.NodeAddress{
					{
						Type:    v1.NodeHostName,
						Address: "node0.cloud.internal",
					},
					{
						Type:    v1.NodeInternalIP,
						Address: "10.0.0.1",
					},
					{
						Type:    v1.NodeExternalIP,
						Address: "132.143.154.163",
					},
				},
				ErrByProviderID: nil,
				Err:             nil,
			},
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{
						{
							Key:    cloudproviderapi.TaintExternalCloudProvider,
							Value:  "true",
							Effect: v1.TaintEffectNoSchedule,
						},
					},
				},
			},
			updatedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				Spec: v1.NodeSpec{
					ProviderID: "fake://12345",
				},
				Status: v1.NodeStatus{
					Addresses: []v1.NodeAddress{
						{
							Type:    v1.NodeHostName,
							Address: "node0.cloud.internal",
						},
						{
							Type:    v1.NodeInternalIP,
							Address: "10.0.0.1",
						},
						{
							Type:    v1.NodeExternalIP,
							Address: "132.143.154.163",
						},
					},
				},
			},
		},
		{
			name: "node ignored",
			fakeCloud: &fakecloud.Cloud{
				EnableInstancesV2: false,
				InstanceTypes: map[types.NodeName]string{
					types.NodeName("node0"):        "t1.micro",
					types.NodeName("fake://12345"): "t1.micro",
				},
				ExtID: map[types.NodeName]string{
					types.NodeName("node0"): "12345",
				},
				Addresses: []v1.NodeAddress{
					{
						Type:    v1.NodeHostName,
						Address: "node0.cloud.internal",
					},
					{
						Type:    v1.NodeInternalIP,
						Address: "10.0.0.1",
					},
					{
						Type:    v1.NodeExternalIP,
						Address: "132.143.154.163",
					},
				},
				Err: nil,
			},
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				},
			},
			updatedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				},
			},
		},
		{
			name: "zone/region topology labels added",
			fakeCloud: &fakecloud.Cloud{
				EnableInstancesV2: false,
				InstanceTypes: map[types.NodeName]string{
					types.NodeName("node0"): "t1.micro",
				},
				ExtID: map[types.NodeName]string{
					types.NodeName("node0"): "12345",
				},
				Addresses: []v1.NodeAddress{
					{
						Type:    v1.NodeHostName,
						Address: "node0.cloud.internal",
					},
					{
						Type:    v1.NodeInternalIP,
						Address: "10.0.0.1",
					},
					{
						Type:    v1.NodeExternalIP,
						Address: "132.143.154.163",
					},
				},
				Provider: "aws",
				Zone: cloudprovider.Zone{
					FailureDomain: "us-west-1a",
					Region:        "us-west",
				},
				Err: nil,
			},
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
					Labels:            map[string]string{},
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionUnknown,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{
						{
							Key:    cloudproviderapi.TaintExternalCloudProvider,
							Value:  "true",
							Effect: v1.TaintEffectNoSchedule,
						},
					},
				},
			},
			updatedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
					Labels: map[string]string{
						"failure-domain.beta.kubernetes.io/region": "us-west",
						"failure-domain.beta.kubernetes.io/zone":   "us-west-1a",
						"topology.kubernetes.io/region":            "us-west",
						"topology.kubernetes.io/zone":              "us-west-1a",
					},
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionUnknown,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
					Addresses: []v1.NodeAddress{
						{
							Type:    v1.NodeHostName,
							Address: "node0.cloud.internal",
						},
						{
							Type:    v1.NodeInternalIP,
							Address: "10.0.0.1",
						},
						{
							Type:    v1.NodeExternalIP,
							Address: "132.143.154.163",
						},
					},
				},
				Spec: v1.NodeSpec{
					ProviderID: "aws://12345",
				},
			},
		},
		{
			name: "node addresses",
			fakeCloud: &fakecloud.Cloud{
				EnableInstancesV2: false,
				InstanceTypes:     map[types.NodeName]string{},
				ExtID: map[types.NodeName]string{
					types.NodeName("node0"): "12345",
				},
				Addresses: []v1.NodeAddress{
					{
						Type:    v1.NodeHostName,
						Address: "node0.cloud.internal",
					},
					{
						Type:    v1.NodeInternalIP,
						Address: "10.0.0.1",
					},
					{
						Type:    v1.NodeExternalIP,
						Address: "132.143.154.163",
					},
				},
				ExistsByProviderID: true,
				Err:                nil,
			},
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
					Labels:            map[string]string{},
				},
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{
						{
							Key:    "ImproveCoverageTaint",
							Value:  "true",
							Effect: v1.TaintEffectNoSchedule,
						},
						{
							Key:    cloudproviderapi.TaintExternalCloudProvider,
							Value:  "true",
							Effect: v1.TaintEffectNoSchedule,
						},
					},
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionUnknown,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			updatedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				Spec: v1.NodeSpec{
					ProviderID: "fake://12345",
					Taints: []v1.Taint{
						{
							Key:    "ImproveCoverageTaint",
							Value:  "true",
							Effect: v1.TaintEffectNoSchedule,
						},
					},
				},
				Status: v1.NodeStatus{
					Addresses: []v1.NodeAddress{
						{
							Type:    v1.NodeHostName,
							Address: "node0.cloud.internal",
						},
						{
							Type:    v1.NodeInternalIP,
							Address: "10.0.0.1",
						},
						{
							Type:    v1.NodeExternalIP,
							Address: "132.143.154.163",
						},
					},
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionUnknown,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
		},
		{
			name: "provided node IP address",
			fakeCloud: &fakecloud.Cloud{
				EnableInstancesV2: false,
				Addresses: []v1.NodeAddress{
					{
						Type:    v1.NodeInternalIP,
						Address: "10.0.0.1",
					},
					{
						Type:    v1.NodeExternalIP,
						Address: "132.143.154.163",
					},
				},
				ExistsByProviderID: true,
				Err:                nil,
			},
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
					Annotations: map[string]string{
						cloudproviderapi.AnnotationAlphaProvidedIPAddr: "10.0.0.1",
					},
				},
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{
						{
							Key:    "ImproveCoverageTaint",
							Value:  "true",
							Effect: v1.TaintEffectNoSchedule,
						},
						{
							Key:    cloudproviderapi.TaintExternalCloudProvider,
							Value:  "true",
							Effect: v1.TaintEffectNoSchedule,
						},
					},
					ProviderID: "node0.aws.12345",
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionUnknown,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
					Addresses: []v1.NodeAddress{
						{
							Type:    v1.NodeHostName,
							Address: "node0.cloud.internal",
						},
					},
				},
			},
			updatedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
					Annotations: map[string]string{
						cloudproviderapi.AnnotationAlphaProvidedIPAddr: "10.0.0.1",
					},
				},
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{
						{
							Key:    "ImproveCoverageTaint",
							Value:  "true",
							Effect: v1.TaintEffectNoSchedule,
						},
					},
					ProviderID: "node0.aws.12345",
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionUnknown,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
					Addresses: []v1.NodeAddress{
						{
							Type:    v1.NodeInternalIP,
							Address: "10.0.0.1",
						},
						{
							Type:    v1.NodeExternalIP,
							Address: "132.143.154.163",
						},
						{
							Type:    v1.NodeHostName,
							Address: "node0.cloud.internal",
						},
					},
				},
			},
		},
		{
			name: "provider ID already set",
			fakeCloud: &fakecloud.Cloud{
				EnableInstancesV2: false,
				InstanceTypes:     map[types.NodeName]string{},
				Addresses: []v1.NodeAddress{
					{
						Type:    v1.NodeHostName,
						Address: "node0.cloud.internal",
					},
					{
						Type:    v1.NodeInternalIP,
						Address: "10.0.0.1",
					},
					{
						Type:    v1.NodeExternalIP,
						Address: "132.143.154.163",
					},
				},
				ExistsByProviderID: false,
				Err:                nil,
			},
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
					Labels:            map[string]string{},
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionUnknown,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
				Spec: v1.NodeSpec{
					ProviderID: "test-provider-id",
					Taints: []v1.Taint{
						{
							Key:    "ImproveCoverageTaint",
							Value:  "true",
							Effect: v1.TaintEffectNoSchedule,
						},
						{
							Key:    cloudproviderapi.TaintExternalCloudProvider,
							Value:  "true",
							Effect: v1.TaintEffectNoSchedule,
						},
					},
				},
			},
			updatedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionUnknown,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
					Addresses: []v1.NodeAddress{
						{
							Type:    v1.NodeHostName,
							Address: "node0.cloud.internal",
						},
						{
							Type:    v1.NodeInternalIP,
							Address: "10.0.0.1",
						},
						{
							Type:    v1.NodeExternalIP,
							Address: "132.143.154.163",
						},
					},
				},
				Spec: v1.NodeSpec{
					ProviderID: "test-provider-id",
					Taints: []v1.Taint{
						{
							Key:    "ImproveCoverageTaint",
							Value:  "true",
							Effect: v1.TaintEffectNoSchedule,
						},
					},
				},
			},
		},
		{
			name: "provider ID not implemented",
			fakeCloud: &fakecloud.Cloud{
				EnableInstancesV2: false,
				InstanceTypes:     map[types.NodeName]string{},
				Provider:          "test",
				ExtID:             map[types.NodeName]string{},
				ExtIDErr: map[types.NodeName]error{
					types.NodeName("node0"): cloudprovider.NotImplemented,
				},
				Err: nil,
			},
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionUnknown,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{
						{
							Key:    cloudproviderapi.TaintExternalCloudProvider,
							Value:  "true",
							Effect: v1.TaintEffectNoSchedule,
						},
					},
				},
			},
			updatedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionUnknown,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{},
				},
			},
		},
		{
			name: "[instanceV2] node initialized with provider ID",
			fakeCloud: &fakecloud.Cloud{
				EnableInstancesV2: true,
				InstanceTypes: map[types.NodeName]string{
					types.NodeName("node0"): "t1.micro",
				},
				ExtID: map[types.NodeName]string{
					types.NodeName("node0"): "12345",
				},
				Addresses: []v1.NodeAddress{
					{
						Type:    v1.NodeHostName,
						Address: "node0.cloud.internal",
					},
					{
						Type:    v1.NodeInternalIP,
						Address: "10.0.0.1",
					},
					{
						Type:    v1.NodeExternalIP,
						Address: "132.143.154.163",
					},
				},
				ErrByProviderID: nil,
				Err:             nil,
			},
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{
						{
							Key:    cloudproviderapi.TaintExternalCloudProvider,
							Value:  "true",
							Effect: v1.TaintEffectNoSchedule,
						},
					},
					ProviderID: "fake://12345",
				},
			},
			updatedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				Spec: v1.NodeSpec{
					ProviderID: "fake://12345",
				},
				Status: v1.NodeStatus{
					Addresses: []v1.NodeAddress{
						{
							Type:    v1.NodeHostName,
							Address: "node0.cloud.internal",
						},
						{
							Type:    v1.NodeInternalIP,
							Address: "10.0.0.1",
						},
						{
							Type:    v1.NodeExternalIP,
							Address: "132.143.154.163",
						},
					},
				},
			},
		},
		{
			name: "[instanceV2] node ignored",
			fakeCloud: &fakecloud.Cloud{
				EnableInstancesV2: true,
				InstanceTypes: map[types.NodeName]string{
					types.NodeName("node0"):        "t1.micro",
					types.NodeName("fake://12345"): "t1.micro",
				},
				ExtID: map[types.NodeName]string{
					types.NodeName("node0"): "12345",
				},
				Addresses: []v1.NodeAddress{
					{
						Type:    v1.NodeHostName,
						Address: "node0.cloud.internal",
					},
					{
						Type:    v1.NodeInternalIP,
						Address: "10.0.0.1",
					},
					{
						Type:    v1.NodeExternalIP,
						Address: "132.143.154.163",
					},
				},
				Err: nil,
			},
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				},
			},
			updatedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				},
			},
		},
		{
			name: "[instanceV2] zone/region topology labels added",
			fakeCloud: &fakecloud.Cloud{
				EnableInstancesV2: true,
				InstanceTypes: map[types.NodeName]string{
					types.NodeName("node0"): "t1.micro",
				},
				ExtID: map[types.NodeName]string{
					types.NodeName("node0"): "12345",
				},
				ProviderID: map[types.NodeName]string{
					types.NodeName("node0"): "fake://12345",
				},
				Addresses: []v1.NodeAddress{
					{
						Type:    v1.NodeHostName,
						Address: "node0.cloud.internal",
					},
					{
						Type:    v1.NodeInternalIP,
						Address: "10.0.0.1",
					},
					{
						Type:    v1.NodeExternalIP,
						Address: "132.143.154.163",
					},
				},
				Provider: "aws",
				Zone: cloudprovider.Zone{
					FailureDomain: "us-west-1a",
					Region:        "us-west",
				},
				Err: nil,
			},
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
					Labels:            map[string]string{},
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionUnknown,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{
						{
							Key:    cloudproviderapi.TaintExternalCloudProvider,
							Value:  "true",
							Effect: v1.TaintEffectNoSchedule,
						},
					},
				},
			},
			updatedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
					Labels: map[string]string{
						"failure-domain.beta.kubernetes.io/region": "us-west",
						"failure-domain.beta.kubernetes.io/zone":   "us-west-1a",
						"topology.kubernetes.io/region":            "us-west",
						"topology.kubernetes.io/zone":              "us-west-1a",
					},
				},
				Spec: v1.NodeSpec{
					ProviderID: "fake://12345",
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionUnknown,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
					Addresses: []v1.NodeAddress{
						{
							Type:    v1.NodeHostName,
							Address: "node0.cloud.internal",
						},
						{
							Type:    v1.NodeInternalIP,
							Address: "10.0.0.1",
						},
						{
							Type:    v1.NodeExternalIP,
							Address: "132.143.154.163",
						},
					},
				},
			},
		},
		{
			name: "[instanceV2] node addresses",
			fakeCloud: &fakecloud.Cloud{
				EnableInstancesV2: true,
				InstanceTypes:     map[types.NodeName]string{},
				ExtID: map[types.NodeName]string{
					types.NodeName("node0"): "12345",
				},
				ProviderID: map[types.NodeName]string{
					types.NodeName("node0"): "fake://12345",
				},
				Addresses: []v1.NodeAddress{
					{
						Type:    v1.NodeHostName,
						Address: "node0.cloud.internal",
					},
					{
						Type:    v1.NodeInternalIP,
						Address: "10.0.0.1",
					},
					{
						Type:    v1.NodeExternalIP,
						Address: "132.143.154.163",
					},
				},
				ExistsByProviderID: true,
				Err:                nil,
			},
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
					Labels:            map[string]string{},
				},
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{
						{
							Key:    "ImproveCoverageTaint",
							Value:  "true",
							Effect: v1.TaintEffectNoSchedule,
						},
						{
							Key:    cloudproviderapi.TaintExternalCloudProvider,
							Value:  "true",
							Effect: v1.TaintEffectNoSchedule,
						},
					},
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionUnknown,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
			updatedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{
						{
							Key:    "ImproveCoverageTaint",
							Value:  "true",
							Effect: v1.TaintEffectNoSchedule,
						},
					},
					ProviderID: "fake://12345",
				},
				Status: v1.NodeStatus{
					Addresses: []v1.NodeAddress{
						{
							Type:    v1.NodeHostName,
							Address: "node0.cloud.internal",
						},
						{
							Type:    v1.NodeInternalIP,
							Address: "10.0.0.1",
						},
						{
							Type:    v1.NodeExternalIP,
							Address: "132.143.154.163",
						},
					},
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionUnknown,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
			},
		},
		{
			name: "[instanceV2] provided node IP address",
			fakeCloud: &fakecloud.Cloud{
				EnableInstancesV2: true,
				Addresses: []v1.NodeAddress{
					{
						Type:    v1.NodeInternalIP,
						Address: "10.0.0.1",
					},
					{
						Type:    v1.NodeExternalIP,
						Address: "132.143.154.163",
					},
				},
				ExistsByProviderID: true,
				Err:                nil,
			},
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
					Annotations: map[string]string{
						cloudproviderapi.AnnotationAlphaProvidedIPAddr: "10.0.0.1",
					},
				},
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{
						{
							Key:    "ImproveCoverageTaint",
							Value:  "true",
							Effect: v1.TaintEffectNoSchedule,
						},
						{
							Key:    cloudproviderapi.TaintExternalCloudProvider,
							Value:  "true",
							Effect: v1.TaintEffectNoSchedule,
						},
					},
					ProviderID: "node0.aws.12345",
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionUnknown,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
					Addresses: []v1.NodeAddress{
						{
							Type:    v1.NodeHostName,
							Address: "node0.cloud.internal",
						},
					},
				},
			},
			updatedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
					Annotations: map[string]string{
						cloudproviderapi.AnnotationAlphaProvidedIPAddr: "10.0.0.1",
					},
				},
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{
						{
							Key:    "ImproveCoverageTaint",
							Value:  "true",
							Effect: v1.TaintEffectNoSchedule,
						},
					},
					ProviderID: "node0.aws.12345",
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionUnknown,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
					Addresses: []v1.NodeAddress{
						{
							Type:    v1.NodeInternalIP,
							Address: "10.0.0.1",
						},
						{
							Type:    v1.NodeExternalIP,
							Address: "132.143.154.163",
						},
						{
							Type:    v1.NodeHostName,
							Address: "node0.cloud.internal",
						},
					},
				},
			},
		},
		{
			name: "[instanceV2] provider ID already set",
			fakeCloud: &fakecloud.Cloud{
				EnableInstancesV2: true,
				InstanceTypes:     map[types.NodeName]string{},
				Addresses: []v1.NodeAddress{
					{
						Type:    v1.NodeHostName,
						Address: "node0.cloud.internal",
					},
					{
						Type:    v1.NodeInternalIP,
						Address: "10.0.0.1",
					},
					{
						Type:    v1.NodeExternalIP,
						Address: "132.143.154.163",
					},
				},
				ExistsByProviderID: false,
				Err:                nil,
			},
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
					Labels:            map[string]string{},
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionUnknown,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
				Spec: v1.NodeSpec{
					ProviderID: "test-provider-id",
					Taints: []v1.Taint{
						{
							Key:    "ImproveCoverageTaint",
							Value:  "true",
							Effect: v1.TaintEffectNoSchedule,
						},
						{
							Key:    cloudproviderapi.TaintExternalCloudProvider,
							Value:  "true",
							Effect: v1.TaintEffectNoSchedule,
						},
					},
				},
			},
			updatedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionUnknown,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
					Addresses: []v1.NodeAddress{
						{
							Type:    v1.NodeHostName,
							Address: "node0.cloud.internal",
						},
						{
							Type:    v1.NodeInternalIP,
							Address: "10.0.0.1",
						},
						{
							Type:    v1.NodeExternalIP,
							Address: "132.143.154.163",
						},
					},
				},
				Spec: v1.NodeSpec{
					ProviderID: "test-provider-id",
					Taints: []v1.Taint{
						{
							Key:    "ImproveCoverageTaint",
							Value:  "true",
							Effect: v1.TaintEffectNoSchedule,
						},
					},
				},
			},
		},
		{
			name: "[instanceV2] provider ID not implemented",
			fakeCloud: &fakecloud.Cloud{
				EnableInstancesV2: true,
				InstanceTypes:     map[types.NodeName]string{},
				Provider:          "test",
				ExtID:             map[types.NodeName]string{},
				ExtIDErr: map[types.NodeName]error{
					types.NodeName("node0"): cloudprovider.NotImplemented,
				},
				Err: nil,
			},
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionUnknown,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{
						{
							Key:    cloudproviderapi.TaintExternalCloudProvider,
							Value:  "true",
							Effect: v1.TaintEffectNoSchedule,
						},
					},
				},
			},
			updatedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionUnknown,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{},
				},
			},
		},
		{
			name: "[instanceV2] error getting InstanceMetadata",
			fakeCloud: &fakecloud.Cloud{
				EnableInstancesV2: true,
				InstanceTypes:     map[types.NodeName]string{},
				Provider:          "test",
				ExtID:             map[types.NodeName]string{},
				ExtIDErr: map[types.NodeName]error{
					types.NodeName("node0"): cloudprovider.NotImplemented,
				},
				MetadataErr: errors.New("metadata error"),
			},
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionUnknown,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{
						{
							Key:    cloudproviderapi.TaintExternalCloudProvider,
							Value:  "true",
							Effect: v1.TaintEffectNoSchedule,
						},
					},
				},
			},
			updatedNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				Status: v1.NodeStatus{
					Conditions: []v1.NodeCondition{
						{
							Type:               v1.NodeReady,
							Status:             v1.ConditionUnknown,
							LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
							LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
						},
					},
				},
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{
						{
							Key:    cloudproviderapi.TaintExternalCloudProvider,
							Value:  "true",
							Effect: v1.TaintEffectNoSchedule,
						},
					},
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			clientset := fake.NewSimpleClientset(test.existingNode)
			factory := informers.NewSharedInformerFactory(clientset, 0)

			eventBroadcaster := record.NewBroadcaster()
			cloudNodeController := &CloudNodeController{
				kubeClient:                clientset,
				nodeInformer:              factory.Core().V1().Nodes(),
				nodesLister:               factory.Core().V1().Nodes().Lister(),
				cloud:                     test.fakeCloud,
				recorder:                  eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "cloud-node-controller"}),
				nodeStatusUpdateFrequency: 1 * time.Second,
			}

			factory.Start(nil)
			factory.WaitForCacheSync(nil)

			w := eventBroadcaster.StartLogging(klog.Infof)
			defer w.Stop()

			cloudNodeController.syncNode(context.TODO(), test.existingNode.Name)

			updatedNode, err := clientset.CoreV1().Nodes().Get(context.TODO(), test.existingNode.Name, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("error getting updated nodes: %v", err)
			}

			if !cmp.Equal(updatedNode, test.updatedNode) {
				t.Errorf("unexpected node %s", cmp.Diff(updatedNode, test.updatedNode))
			}
		})
	}
}

// test syncNode with instanceV2, same test case with TestGCECondition.
func TestGCEConditionV2(t *testing.T) {
	existingNode := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "node0",
			CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
		},
		Status: v1.NodeStatus{
			Conditions: []v1.NodeCondition{
				{
					Type:               v1.NodeReady,
					Status:             v1.ConditionUnknown,
					LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
					LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
				},
			},
		},
		Spec: v1.NodeSpec{
			Taints: []v1.Taint{
				{
					Key:    cloudproviderapi.TaintExternalCloudProvider,
					Value:  "true",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
		},
	}

	fakeCloud := &fakecloud.Cloud{
		EnableInstancesV2: true,
		InstanceTypes: map[types.NodeName]string{
			types.NodeName("node0"): "t1.micro",
		},
		Addresses: []v1.NodeAddress{
			{
				Type:    v1.NodeHostName,
				Address: "node0.cloud.internal",
			},
			{
				Type:    v1.NodeInternalIP,
				Address: "10.0.0.1",
			},
			{
				Type:    v1.NodeExternalIP,
				Address: "132.143.154.163",
			},
		},
		Provider: "gce",
		Err:      nil,
	}

	clientset := fake.NewSimpleClientset(existingNode)
	factory := informers.NewSharedInformerFactory(clientset, 0)

	eventBroadcaster := record.NewBroadcaster()
	cloudNodeController := &CloudNodeController{
		kubeClient:                clientset,
		nodeInformer:              factory.Core().V1().Nodes(),
		nodesLister:               factory.Core().V1().Nodes().Lister(),
		cloud:                     fakeCloud,
		recorder:                  eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "cloud-node-controller"}),
		nodeStatusUpdateFrequency: 1 * time.Second,
	}

	factory.Start(nil)
	factory.WaitForCacheSync(nil)

	w := eventBroadcaster.StartLogging(klog.Infof)
	defer w.Stop()

	cloudNodeController.syncNode(context.TODO(), existingNode.Name)

	updatedNode, err := clientset.CoreV1().Nodes().Get(context.TODO(), existingNode.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("error getting updated nodes: %v", err)
	}

	conditionAdded := false
	for _, cond := range updatedNode.Status.Conditions {
		if cond.Status == "True" && cond.Type == "NetworkUnavailable" && cond.Reason == "NoRouteCreated" {
			conditionAdded = true
		}
	}

	assert.True(t, conditionAdded, "Network Route Condition for GCE not added by external cloud initializer")
}

// This test checks that a node with the external cloud provider taint is cloudprovider initialized and
// the GCE route condition is added if cloudprovider is GCE
func TestGCECondition(t *testing.T) {
	existingNode := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "node0",
			CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
		},
		Status: v1.NodeStatus{
			Conditions: []v1.NodeCondition{
				{
					Type:               v1.NodeReady,
					Status:             v1.ConditionUnknown,
					LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
					LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
				},
			},
		},
		Spec: v1.NodeSpec{
			Taints: []v1.Taint{
				{
					Key:    cloudproviderapi.TaintExternalCloudProvider,
					Value:  "true",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
		},
	}

	fakeCloud := &fakecloud.Cloud{
		EnableInstancesV2: false,
		InstanceTypes: map[types.NodeName]string{
			types.NodeName("node0"): "t1.micro",
		},
		Addresses: []v1.NodeAddress{
			{
				Type:    v1.NodeHostName,
				Address: "node0.cloud.internal",
			},
			{
				Type:    v1.NodeInternalIP,
				Address: "10.0.0.1",
			},
			{
				Type:    v1.NodeExternalIP,
				Address: "132.143.154.163",
			},
		},
		Provider: "gce",
		Err:      nil,
	}

	clientset := fake.NewSimpleClientset(existingNode)
	factory := informers.NewSharedInformerFactory(clientset, 0)

	eventBroadcaster := record.NewBroadcaster()
	cloudNodeController := &CloudNodeController{
		kubeClient:                clientset,
		nodeInformer:              factory.Core().V1().Nodes(),
		nodesLister:               factory.Core().V1().Nodes().Lister(),
		cloud:                     fakeCloud,
		recorder:                  eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "cloud-node-controller"}),
		nodeStatusUpdateFrequency: 1 * time.Second,
	}

	factory.Start(nil)
	factory.WaitForCacheSync(nil)

	w := eventBroadcaster.StartLogging(klog.Infof)
	defer w.Stop()

	cloudNodeController.syncNode(context.TODO(), existingNode.Name)

	updatedNode, err := clientset.CoreV1().Nodes().Get(context.TODO(), existingNode.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("error getting updated nodes: %v", err)
	}

	conditionAdded := false
	for _, cond := range updatedNode.Status.Conditions {
		if cond.Status == "True" && cond.Type == "NetworkUnavailable" && cond.Reason == "NoRouteCreated" {
			conditionAdded = true
		}
	}

	assert.True(t, conditionAdded, "Network Route Condition for GCE not added by external cloud initializer")
}

func Test_reconcileNodeLabels(t *testing.T) {
	testcases := []struct {
		name           string
		labels         map[string]string
		expectedLabels map[string]string
		expectedErr    error
	}{
		{
			name: "requires reconcile",
			labels: map[string]string{
				v1.LabelFailureDomainBetaZone:   "foo",
				v1.LabelFailureDomainBetaRegion: "bar",
				v1.LabelInstanceType:            "the-best-type",
			},
			expectedLabels: map[string]string{
				v1.LabelFailureDomainBetaZone:   "foo",
				v1.LabelFailureDomainBetaRegion: "bar",
				v1.LabelTopologyZone:            "foo",
				v1.LabelTopologyRegion:          "bar",
				v1.LabelInstanceType:            "the-best-type",
				v1.LabelInstanceTypeStable:      "the-best-type",
			},
			expectedErr: nil,
		},
		{
			name: "doesn't require reconcile",
			labels: map[string]string{
				v1.LabelFailureDomainBetaZone:   "foo",
				v1.LabelFailureDomainBetaRegion: "bar",
				v1.LabelTopologyZone:            "foo",
				v1.LabelTopologyRegion:          "bar",
				v1.LabelInstanceType:            "the-best-type",
				v1.LabelInstanceTypeStable:      "the-best-type",
			},
			expectedLabels: map[string]string{
				v1.LabelFailureDomainBetaZone:   "foo",
				v1.LabelFailureDomainBetaRegion: "bar",
				v1.LabelTopologyZone:            "foo",
				v1.LabelTopologyRegion:          "bar",
				v1.LabelInstanceType:            "the-best-type",
				v1.LabelInstanceTypeStable:      "the-best-type",
			},
			expectedErr: nil,
		},
		{
			name: "require reconcile -- secondary labels are different from primary",
			labels: map[string]string{
				v1.LabelFailureDomainBetaZone:   "foo",
				v1.LabelFailureDomainBetaRegion: "bar",
				v1.LabelTopologyZone:            "wrongfoo",
				v1.LabelTopologyRegion:          "wrongbar",
				v1.LabelInstanceType:            "the-best-type",
				v1.LabelInstanceTypeStable:      "the-wrong-type",
			},
			expectedLabels: map[string]string{
				v1.LabelFailureDomainBetaZone:   "foo",
				v1.LabelFailureDomainBetaRegion: "bar",
				v1.LabelTopologyZone:            "foo",
				v1.LabelTopologyRegion:          "bar",
				v1.LabelInstanceType:            "the-best-type",
				v1.LabelInstanceTypeStable:      "the-best-type",
			},
			expectedErr: nil,
		},
	}

	for _, test := range testcases {
		t.Run(test.name, func(t *testing.T) {
			testNode := &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "node01",
					Labels: test.labels,
				},
			}

			clientset := fake.NewSimpleClientset(testNode)
			factory := informers.NewSharedInformerFactory(clientset, 0)

			cnc := &CloudNodeController{
				kubeClient:   clientset,
				nodeInformer: factory.Core().V1().Nodes(),
			}

			// activate node informer
			factory.Core().V1().Nodes().Informer()
			factory.Start(nil)
			factory.WaitForCacheSync(nil)

			err := cnc.reconcileNodeLabels("node01")
			if err != test.expectedErr {
				t.Logf("actual err: %v", err)
				t.Logf("expected err: %v", test.expectedErr)
				t.Errorf("unexpected error")
			}

			actualNode, err := clientset.CoreV1().Nodes().Get(context.TODO(), "node01", metav1.GetOptions{})
			if err != nil {
				t.Fatalf("error getting updated node: %v", err)
			}

			if !reflect.DeepEqual(actualNode.Labels, test.expectedLabels) {
				t.Logf("actual node labels: %v", actualNode.Labels)
				t.Logf("expected node labels: %v", test.expectedLabels)
				t.Errorf("updated node did not match expected node")
			}
		})
	}

}

// Tests that node address changes are detected correctly
func TestNodeAddressesChangeDetected(t *testing.T) {
	addressSet1 := []v1.NodeAddress{
		{
			Type:    v1.NodeInternalIP,
			Address: "10.0.0.1",
		},
		{
			Type:    v1.NodeExternalIP,
			Address: "132.143.154.163",
		},
	}
	addressSet2 := []v1.NodeAddress{
		{
			Type:    v1.NodeInternalIP,
			Address: "10.0.0.1",
		},
		{
			Type:    v1.NodeExternalIP,
			Address: "132.143.154.163",
		},
	}

	assert.False(t, nodeAddressesChangeDetected(addressSet1, addressSet2),
		"Node address changes are not detected correctly")

	addressSet1 = []v1.NodeAddress{
		{
			Type:    v1.NodeInternalIP,
			Address: "10.0.0.1",
		},
		{
			Type:    v1.NodeExternalIP,
			Address: "132.143.154.164",
		},
	}
	addressSet2 = []v1.NodeAddress{
		{
			Type:    v1.NodeInternalIP,
			Address: "10.0.0.1",
		},
		{
			Type:    v1.NodeExternalIP,
			Address: "132.143.154.163",
		},
	}

	assert.True(t, nodeAddressesChangeDetected(addressSet1, addressSet2),
		"Node address changes are not detected correctly")

	addressSet1 = []v1.NodeAddress{
		{
			Type:    v1.NodeInternalIP,
			Address: "10.0.0.1",
		},
		{
			Type:    v1.NodeExternalIP,
			Address: "132.143.154.164",
		},
		{
			Type:    v1.NodeHostName,
			Address: "hostname.zone.region.aws.test",
		},
	}
	addressSet2 = []v1.NodeAddress{
		{
			Type:    v1.NodeInternalIP,
			Address: "10.0.0.1",
		},
		{
			Type:    v1.NodeExternalIP,
			Address: "132.143.154.164",
		},
	}

	assert.True(t, nodeAddressesChangeDetected(addressSet1, addressSet2),
		"Node address changes are not detected correctly")

	addressSet1 = []v1.NodeAddress{
		{
			Type:    v1.NodeInternalIP,
			Address: "10.0.0.1",
		},
		{
			Type:    v1.NodeExternalIP,
			Address: "132.143.154.164",
		},
	}
	addressSet2 = []v1.NodeAddress{
		{
			Type:    v1.NodeInternalIP,
			Address: "10.0.0.1",
		},
		{
			Type:    v1.NodeExternalIP,
			Address: "132.143.154.164",
		},
		{
			Type:    v1.NodeHostName,
			Address: "hostname.zone.region.aws.test",
		},
	}

	assert.True(t, nodeAddressesChangeDetected(addressSet1, addressSet2),
		"Node address changes are not detected correctly")

	addressSet1 = []v1.NodeAddress{
		{
			Type:    v1.NodeExternalIP,
			Address: "10.0.0.1",
		},
		{
			Type:    v1.NodeInternalIP,
			Address: "132.143.154.163",
		},
	}
	addressSet2 = []v1.NodeAddress{
		{
			Type:    v1.NodeInternalIP,
			Address: "10.0.0.1",
		},
		{
			Type:    v1.NodeExternalIP,
			Address: "132.143.154.163",
		},
	}

	assert.True(t, nodeAddressesChangeDetected(addressSet1, addressSet2),
		"Node address changes are not detected correctly")
}

// Test updateNodeAddress with instanceV2, same test case with TestNodeAddressesNotUpdate.
func TestNodeAddressesNotUpdateV2(t *testing.T) {
	existingNode := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "node0",
			CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
			Labels:            map[string]string{},
		},
		Status: v1.NodeStatus{
			Conditions: []v1.NodeCondition{
				{
					Type:               v1.NodeReady,
					Status:             v1.ConditionUnknown,
					LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
					LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
				},
			},
		},
		Spec: v1.NodeSpec{
			Taints: []v1.Taint{
				{
					Key:    cloudproviderapi.TaintExternalCloudProvider,
					Value:  "true",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
		},
	}

	clientset := fake.NewSimpleClientset(existingNode)
	factory := informers.NewSharedInformerFactory(clientset, 0)

	fakeCloud := &fakecloud.Cloud{
		EnableInstancesV2: true,
		InstanceTypes:     map[types.NodeName]string{},
		Addresses: []v1.NodeAddress{
			{
				Type:    v1.NodeHostName,
				Address: "node0.cloud.internal",
			},
			{
				Type:    v1.NodeInternalIP,
				Address: "10.0.0.1",
			},
			{
				Type:    v1.NodeExternalIP,
				Address: "132.143.154.163",
			},
		},
		ExistsByProviderID: false,
		Err:                nil,
	}

	cloudNodeController := &CloudNodeController{
		kubeClient:   clientset,
		nodeInformer: factory.Core().V1().Nodes(),
		cloud:        fakeCloud,
	}

	instanceMeta, err := cloudNodeController.getInstanceNodeAddresses(context.TODO(), existingNode)
	if err != nil {
		t.Errorf("get instance metadata with error %v", err)
	}
	cloudNodeController.updateNodeAddress(context.TODO(), existingNode, instanceMeta)

	updatedNode, err := clientset.CoreV1().Nodes().Get(context.TODO(), existingNode.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("error getting updated nodes: %v", err)
	}

	if len(updatedNode.Status.Addresses) > 0 {
		t.Errorf("Node addresses should not be updated")
	}
}

// This test checks that a node with the external cloud provider taint is cloudprovider initialized and
// and node addresses will not be updated when node isn't present according to the cloudprovider
func TestNodeAddressesNotUpdate(t *testing.T) {
	existingNode := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "node0",
			CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
			Labels:            map[string]string{},
		},
		Status: v1.NodeStatus{
			Conditions: []v1.NodeCondition{
				{
					Type:               v1.NodeReady,
					Status:             v1.ConditionUnknown,
					LastHeartbeatTime:  metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
					LastTransitionTime: metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC),
				},
			},
		},
		Spec: v1.NodeSpec{
			Taints: []v1.Taint{
				{
					Key:    cloudproviderapi.TaintExternalCloudProvider,
					Value:  "true",
					Effect: v1.TaintEffectNoSchedule,
				},
			},
		},
	}

	clientset := fake.NewSimpleClientset(existingNode)
	factory := informers.NewSharedInformerFactory(clientset, 0)

	fakeCloud := &fakecloud.Cloud{
		EnableInstancesV2: false,
		InstanceTypes:     map[types.NodeName]string{},
		Addresses: []v1.NodeAddress{
			{
				Type:    v1.NodeHostName,
				Address: "node0.cloud.internal",
			},
			{
				Type:    v1.NodeInternalIP,
				Address: "10.0.0.1",
			},
			{
				Type:    v1.NodeExternalIP,
				Address: "132.143.154.163",
			},
		},
		ExistsByProviderID: false,
		Err:                nil,
	}

	cloudNodeController := &CloudNodeController{
		kubeClient:   clientset,
		nodeInformer: factory.Core().V1().Nodes(),
		cloud:        fakeCloud,
	}

	instanceMeta, err := cloudNodeController.getInstanceNodeAddresses(context.TODO(), existingNode)
	if err != nil {
		t.Errorf("get instance metadata with error %v", err)
	}
	cloudNodeController.updateNodeAddress(context.TODO(), existingNode, instanceMeta)

	updatedNode, err := clientset.CoreV1().Nodes().Get(context.TODO(), existingNode.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("error getting updated nodes: %v", err)
	}

	if len(updatedNode.Status.Addresses) > 0 {
		t.Errorf("Node addresses should not be updated")
	}
}

func TestGetProviderID(t *testing.T) {
	tests := []struct {
		name               string
		fakeCloud          *fakecloud.Cloud
		existingNode       *v1.Node
		expectedProviderID string
	}{
		{
			name: "node initialized with provider ID",
			fakeCloud: &fakecloud.Cloud{
				EnableInstancesV2: false,
				InstanceTypes: map[types.NodeName]string{
					types.NodeName("node0"): "t1.micro",
				},
				ExtID: map[types.NodeName]string{
					types.NodeName("node0"): "12345",
				},
				Addresses: []v1.NodeAddress{
					{
						Type:    v1.NodeHostName,
						Address: "node0.cloud.internal",
					},
					{
						Type:    v1.NodeInternalIP,
						Address: "10.0.0.1",
					},
					{
						Type:    v1.NodeExternalIP,
						Address: "132.143.154.163",
					},
				},
				ErrByProviderID: nil,
				Err:             nil,
			},
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{
						{
							Key:    cloudproviderapi.TaintExternalCloudProvider,
							Value:  "true",
							Effect: v1.TaintEffectNoSchedule,
						},
					},
					ProviderID: "fake://12345",
				},
			},
			expectedProviderID: "fake://12345",
		},
		{
			name: "cloud implemented with Instances (without providerID)",
			fakeCloud: &fakecloud.Cloud{
				EnableInstancesV2: false,
				InstanceTypes: map[types.NodeName]string{
					types.NodeName("node0"):        "t1.micro",
					types.NodeName("fake://12345"): "t1.micro",
				},
				ExtID: map[types.NodeName]string{
					types.NodeName("node0"): "12345",
				},
				Addresses: []v1.NodeAddress{
					{
						Type:    v1.NodeHostName,
						Address: "node0.cloud.internal",
					},
					{
						Type:    v1.NodeInternalIP,
						Address: "10.0.0.1",
					},
					{
						Type:    v1.NodeExternalIP,
						Address: "132.143.154.163",
					},
				},
				Err: nil,
			},
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				},
			},
			expectedProviderID: "fake://12345",
		},
		{
			name: "cloud implemented with InstancesV2 (with providerID)",
			fakeCloud: &fakecloud.Cloud{
				EnableInstancesV2: true,
				InstanceTypes: map[types.NodeName]string{
					types.NodeName("node0"):        "t1.micro",
					types.NodeName("fake://12345"): "t1.micro",
				},
				ExtID: map[types.NodeName]string{
					types.NodeName("node0"): "12345",
				},
				Addresses: []v1.NodeAddress{
					{
						Type:    v1.NodeHostName,
						Address: "node0.cloud.internal",
					},
					{
						Type:    v1.NodeInternalIP,
						Address: "10.0.0.1",
					},
					{
						Type:    v1.NodeExternalIP,
						Address: "132.143.154.163",
					},
				},
				Err: nil,
			},
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{
						{
							Key:    cloudproviderapi.TaintExternalCloudProvider,
							Value:  "true",
							Effect: v1.TaintEffectNoSchedule,
						},
					},
					ProviderID: "fake://12345",
				},
			},
			expectedProviderID: "fake://12345",
		},
		{
			name: "cloud implemented with InstancesV2 (without providerID)",
			fakeCloud: &fakecloud.Cloud{
				EnableInstancesV2: true,
				InstanceTypes: map[types.NodeName]string{
					types.NodeName("node0"):        "t1.micro",
					types.NodeName("fake://12345"): "t1.micro",
				},
				ExtID: map[types.NodeName]string{
					types.NodeName("node0"): "12345",
				},
				Addresses: []v1.NodeAddress{
					{
						Type:    v1.NodeHostName,
						Address: "node0.cloud.internal",
					},
					{
						Type:    v1.NodeInternalIP,
						Address: "10.0.0.1",
					},
					{
						Type:    v1.NodeExternalIP,
						Address: "132.143.154.163",
					},
				},
				Err: nil,
			},
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
				},
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{
						{
							Key:    cloudproviderapi.TaintExternalCloudProvider,
							Value:  "true",
							Effect: v1.TaintEffectNoSchedule,
						},
					},
				},
			},
			expectedProviderID: "",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cloudNodeController := &CloudNodeController{
				cloud: test.fakeCloud,
			}

			providerID, err := cloudNodeController.getProviderID(context.TODO(), test.existingNode)
			if err != nil {
				t.Fatalf("error getting provider ID: %v", err)
			}

			if !cmp.Equal(providerID, test.expectedProviderID) {
				t.Errorf("unexpected providerID %s", cmp.Diff(providerID, test.expectedProviderID))
			}
		})
	}
}

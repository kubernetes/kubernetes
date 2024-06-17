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
	"fmt"
	"reflect"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/informers"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/record"
	cloudprovider "k8s.io/cloud-provider"
	cloudproviderapi "k8s.io/cloud-provider/api"
	fakecloud "k8s.io/cloud-provider/fake"
	_ "k8s.io/controller-manager/pkg/features/register"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"
)

func Test_syncNode(t *testing.T) {
	tests := []struct {
		name         string
		fakeCloud    *fakecloud.Cloud
		existingNode *v1.Node
		updatedNode  *v1.Node
		expectedErr  bool
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
			name: "nil instanceMetadata provided by InstanceV2",
			fakeCloud: &fakecloud.Cloud{
				EnableInstancesV2: true,
				Err:               nil,
				OverrideInstanceMetadata: func(ctx context.Context, node *v1.Node) (*cloudprovider.InstanceMetadata, error) {
					return nil, nil
				},
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
			name:        "provided node IP address is not valid",
			expectedErr: true,
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
						cloudproviderapi.AnnotationAlphaProvidedIPAddr: "invalid-ip",
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
						cloudproviderapi.AnnotationAlphaProvidedIPAddr: "invalid-ip",
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
		},
		{
			name:        "provided node IP address is not present",
			expectedErr: true,
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
						cloudproviderapi.AnnotationAlphaProvidedIPAddr: "10.0.0.2",
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
						cloudproviderapi.AnnotationAlphaProvidedIPAddr: "10.0.0.2",
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
		{ // for backward compatibility the cloud providers that does not implement
			// providerID does not block the node initialization
			name:        "provider ID not implemented",
			expectedErr: false,
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
			name: "[instanceV2] provided additional labels",
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
				Zone: cloudprovider.Zone{
					FailureDomain: "us-west-1a",
					Region:        "us-west",
				},
				AdditionalLabels: map[string]string{
					"topology.k8s.cp/zone-id": "az1",
					"my.custom.label/foo":     "bar",
				},
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
					ProviderID: "node0.cp.12345",
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
					Labels: map[string]string{
						"failure-domain.beta.kubernetes.io/region": "us-west",
						"failure-domain.beta.kubernetes.io/zone":   "us-west-1a",
						"topology.kubernetes.io/region":            "us-west",
						"topology.kubernetes.io/zone":              "us-west-1a",
						"topology.k8s.cp/zone-id":                  "az1",
						"my.custom.label/foo":                      "bar",
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
					ProviderID: "node0.cp.12345",
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
			name: "[instanceV2] provided additional labels with labels to discard",
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
				Zone: cloudprovider.Zone{
					FailureDomain: "us-west-1a",
					Region:        "us-west",
				},
				AdditionalLabels: map[string]string{
					// Kubernetes reserves k8s.io and kubernetes.io namespaces
					// and should be discarded
					"topology.kubernetes.io/region": "us-other-west",
					"topology.k8s.io/region":        "us-other-west",
					// Should discard labels that already exist
					"my.custom.label/foo": "bar",
					"my.custom.label/bar": "foo",
				},
			},
			existingNode: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "node0",
					CreationTimestamp: metav1.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
					Annotations: map[string]string{
						cloudproviderapi.AnnotationAlphaProvidedIPAddr: "10.0.0.1",
					},
					Labels: map[string]string{
						"my.custom.label/foo": "fizz",
						"my.custom.label/bar": "foo",
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
					ProviderID: "node0.cp.12345",
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
					Labels: map[string]string{
						"failure-domain.beta.kubernetes.io/region": "us-west",
						"failure-domain.beta.kubernetes.io/zone":   "us-west-1a",
						"topology.kubernetes.io/region":            "us-west",
						"topology.kubernetes.io/zone":              "us-west-1a",
						"my.custom.label/foo":                      "fizz",
						"my.custom.label/bar":                      "foo",
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
					ProviderID: "node0.cp.12345",
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
			name:        "[instanceV2] provider ID not implemented",
			expectedErr: true,
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
		{
			name:        "[instanceV2] error getting InstanceMetadata",
			expectedErr: true,
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
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			clientset := fake.NewSimpleClientset(test.existingNode)
			factory := informers.NewSharedInformerFactory(clientset, 0)

			eventBroadcaster := record.NewBroadcaster(record.WithContext(ctx))
			cloudNodeController := &CloudNodeController{
				kubeClient:                clientset,
				nodeInformer:              factory.Core().V1().Nodes(),
				nodesLister:               factory.Core().V1().Nodes().Lister(),
				cloud:                     test.fakeCloud,
				recorder:                  eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "cloud-node-controller"}),
				nodeStatusUpdateFrequency: 1 * time.Second,
			}

			stopCh := make(chan struct{})
			defer close(stopCh)

			factory.Start(stopCh)
			factory.WaitForCacheSync(stopCh)

			w := eventBroadcaster.StartLogging(klog.Infof)
			defer w.Stop()

			err := cloudNodeController.syncNode(ctx, test.existingNode.Name)
			if (err != nil) != test.expectedErr {
				t.Fatalf("error got: %v expected: %v", err, test.expectedErr)
			}

			updatedNode, err := clientset.CoreV1().Nodes().Get(ctx, test.existingNode.Name, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("error getting updated nodes: %v", err)
			}

			if !cmp.Equal(updatedNode, test.updatedNode) {
				t.Errorf("unexpected node %s", cmp.Diff(updatedNode, test.updatedNode))
			}
		})
	}
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
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			stopCh := ctx.Done()

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
			factory.Start(stopCh)
			factory.WaitForCacheSync(stopCh)

			err := cnc.reconcileNodeLabels("node01")
			if err != test.expectedErr {
				t.Logf("actual err: %v", err)
				t.Logf("expected err: %v", test.expectedErr)
				t.Errorf("unexpected error")
			}

			actualNode, err := clientset.CoreV1().Nodes().Get(ctx, "node01", metav1.GetOptions{})
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
	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

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

	instanceMeta, err := cloudNodeController.getInstanceNodeAddresses(ctx, existingNode)
	if err != nil {
		t.Errorf("get instance metadata with error %v", err)
	}
	cloudNodeController.updateNodeAddress(ctx, existingNode, instanceMeta)

	updatedNode, err := clientset.CoreV1().Nodes().Get(ctx, existingNode.Name, metav1.GetOptions{})
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
	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

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

	instanceMeta, err := cloudNodeController.getInstanceNodeAddresses(ctx, existingNode)
	if err != nil {
		t.Errorf("get instance metadata with error %v", err)
	}
	cloudNodeController.updateNodeAddress(ctx, existingNode, instanceMeta)

	updatedNode, err := clientset.CoreV1().Nodes().Get(ctx, existingNode.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("error getting updated nodes: %v", err)
	}

	if len(updatedNode.Status.Addresses) > 0 {
		t.Errorf("Node addresses should not be updated")
	}
}

func TestGetInstanceMetadata(t *testing.T) {
	tests := []struct {
		name             string
		fakeCloud        *fakecloud.Cloud
		existingNode     *v1.Node
		expectedMetadata *cloudprovider.InstanceMetadata
		expectErr        bool
	}{
		{
			name: "cloud implemented with Instances and provider ID",
			fakeCloud: &fakecloud.Cloud{
				EnableInstancesV2: false,
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
			expectedMetadata: &cloudprovider.InstanceMetadata{
				ProviderID: "fake://12345",
				NodeAddresses: []v1.NodeAddress{
					{Type: "Hostname", Address: "node0.cloud.internal"},
					{Type: "InternalIP", Address: "10.0.0.1"},
					{Type: "ExternalIP", Address: "132.143.154.163"},
				},
			},
		},
		{
			name: "cloud implemented with Instances (providerID not implemented)",
			fakeCloud: &fakecloud.Cloud{
				EnableInstancesV2: false,
				InstanceTypes: map[types.NodeName]string{
					types.NodeName("node0"):        "t1.micro",
					types.NodeName("fake://12345"): "t1.micro",
				},
				ExtIDErr: map[types.NodeName]error{
					types.NodeName("node0"): cloudprovider.NotImplemented,
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
			expectedMetadata: &cloudprovider.InstanceMetadata{
				NodeAddresses: []v1.NodeAddress{
					{Type: "Hostname", Address: "node0.cloud.internal"},
					{Type: "InternalIP", Address: "10.0.0.1"},
					{Type: "ExternalIP", Address: "132.143.154.163"},
				},
			},
		},
		{
			name: "cloud implemented with Instances (providerID not implemented) and node with providerID",
			fakeCloud: &fakecloud.Cloud{
				EnableInstancesV2: false,
				InstanceTypes: map[types.NodeName]string{
					types.NodeName("node0"):        "t1.micro",
					types.NodeName("fake://12345"): "t1.micro",
				},
				ExtIDErr: map[types.NodeName]error{
					types.NodeName("node0"): cloudprovider.NotImplemented,
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
					ProviderID: "fake://asdasd",
				},
			},
			expectedMetadata: &cloudprovider.InstanceMetadata{
				ProviderID: "fake://asdasd",
				NodeAddresses: []v1.NodeAddress{
					{Type: "Hostname", Address: "node0.cloud.internal"},
					{Type: "InternalIP", Address: "10.0.0.1"},
					{Type: "ExternalIP", Address: "132.143.154.163"},
				},
			},
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
			expectedMetadata: &cloudprovider.InstanceMetadata{
				ProviderID: "fake://12345",
				NodeAddresses: []v1.NodeAddress{
					{Type: "Hostname", Address: "node0.cloud.internal"},
					{Type: "InternalIP", Address: "10.0.0.1"},
					{Type: "ExternalIP", Address: "132.143.154.163"},
				},
			},
		},
		{ // it will be requeueud later
			name:      "cloud implemented with InstancesV2 (without providerID)",
			expectErr: true,
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
			expectedMetadata: &cloudprovider.InstanceMetadata{
				NodeAddresses: []v1.NodeAddress{
					{Type: "Hostname", Address: "node0.cloud.internal"},
					{Type: "InternalIP", Address: "10.0.0.1"},
					{Type: "ExternalIP", Address: "132.143.154.163"},
				},
			},
		},
		{
			name: "cloud implemented with InstancesV2 (without providerID) and node with providerID",
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
					ProviderID: "fake://12345",
				},
			},
			expectedMetadata: &cloudprovider.InstanceMetadata{
				ProviderID:   "fake://12345",
				InstanceType: "t1.micro",
				NodeAddresses: []v1.NodeAddress{
					{Type: "Hostname", Address: "node0.cloud.internal"},
					{Type: "InternalIP", Address: "10.0.0.1"},
					{Type: "ExternalIP", Address: "132.143.154.163"},
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)

			cloudNodeController := &CloudNodeController{
				cloud: test.fakeCloud,
			}

			metadata, err := cloudNodeController.getInstanceMetadata(ctx, test.existingNode)
			if (err != nil) != test.expectErr {
				t.Fatalf("error expected %v got: %v", test.expectErr, err)
			}

			if !cmp.Equal(metadata, test.expectedMetadata) {
				t.Errorf("unexpected metadata %s", cmp.Diff(metadata, test.expectedMetadata))
			}
		})
	}
}

func TestUpdateNodeStatus(t *testing.T) {
	// emaulate the latency of the cloud API calls
	const cloudLatency = 10 * time.Millisecond

	generateNodes := func(n int) []runtime.Object {
		result := []runtime.Object{}
		for i := 0; i < n; i++ {
			result = append(result, &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: fmt.Sprintf("node0%d", i),
				},
			})
		}
		return result
	}

	tests := []struct {
		name    string
		workers int32
		nodes   int
	}{
		{
			name:    "single thread",
			workers: 1,
			nodes:   100,
		},
		{
			name:    "5 workers",
			workers: 5,
			nodes:   100,
		},
		{
			name:    "10 workers",
			workers: 10,
			nodes:   100,
		},
		{
			name:    "30 workers",
			workers: 30,
			nodes:   100,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			fakeCloud := &fakecloud.Cloud{
				EnableInstancesV2: false,
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
				RequestDelay: cloudLatency,
				Err:          nil,
			}

			clientset := fake.NewSimpleClientset()
			clientset.PrependReactor("patch", "nodes", func(action clienttesting.Action) (bool, runtime.Object, error) {
				return true, &v1.Node{}, nil
			})

			factory := informers.NewSharedInformerFactory(clientset, 0)
			eventBroadcaster := record.NewBroadcaster(record.WithContext(ctx))
			nodeInformer := factory.Core().V1().Nodes()
			nodeIndexer := nodeInformer.Informer().GetIndexer()
			cloudNodeController := &CloudNodeController{
				kubeClient:                clientset,
				nodeInformer:              nodeInformer,
				nodesLister:               nodeInformer.Lister(),
				nodesSynced:               func() bool { return true },
				cloud:                     fakeCloud,
				recorder:                  eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "cloud-node-controller"}),
				nodeStatusUpdateFrequency: 1 * time.Second,
				workerCount:               test.workers,
			}

			for _, n := range generateNodes(test.nodes) {
				err := nodeIndexer.Add(n)
				if err != nil {
					t.Fatal(err)
				}
			}

			w := eventBroadcaster.StartStructuredLogging(0)
			defer w.Stop()

			start := time.Now()
			if err := cloudNodeController.UpdateNodeStatus(ctx); err != nil {
				t.Fatalf("error updating node status: %v", err)
			}
			t.Logf("%d workers: processed %d nodes int %v ", test.workers, test.nodes, time.Since(start))
			if len(fakeCloud.Calls) != test.nodes {
				t.Errorf("expected %d cloud-provider calls, got %d", test.nodes, len(fakeCloud.Calls))
			}

		})
	}
}

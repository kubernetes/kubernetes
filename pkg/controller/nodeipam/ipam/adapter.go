//go:build !providerless
// +build !providerless

/*
Copyright 2017 The Kubernetes Authors.

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

package ipam

import (
	"context"
	"encoding/json"
	"fmt"
	"net"

	"k8s.io/klog/v2"
	netutils "k8s.io/utils/net"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientset "k8s.io/client-go/kubernetes"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/tools/record"
	nodeutil "k8s.io/component-helpers/node/util"
	"k8s.io/legacy-cloud-providers/gce"
	"k8s.io/metrics/pkg/client/clientset/versioned/scheme"
)

type adapter struct {
	k8s   clientset.Interface
	cloud *gce.Cloud

	broadcaster record.EventBroadcaster
	recorder    record.EventRecorder
}

func newAdapter(ctx context.Context, k8s clientset.Interface, cloud *gce.Cloud) *adapter {
	broadcaster := record.NewBroadcaster(record.WithContext(ctx))

	ret := &adapter{
		k8s:         k8s,
		cloud:       cloud,
		broadcaster: broadcaster,
		recorder:    broadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "cloudCIDRAllocator"}),
	}

	return ret
}

func (a *adapter) Run(ctx context.Context) {
	defer utilruntime.HandleCrash()

	// Start event processing pipeline.
	a.broadcaster.StartStructuredLogging(3)
	a.broadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: a.k8s.CoreV1().Events("")})
	defer a.broadcaster.Shutdown()

	<-ctx.Done()
}

func (a *adapter) Alias(ctx context.Context, node *v1.Node) (*net.IPNet, error) {
	if node.Spec.ProviderID == "" {
		return nil, fmt.Errorf("node %s doesn't have providerID", node.Name)
	}

	cidrs, err := a.cloud.AliasRangesByProviderID(node.Spec.ProviderID)
	if err != nil {
		return nil, err
	}

	switch len(cidrs) {
	case 0:
		return nil, nil
	case 1:
		break
	default:
		klog.FromContext(ctx).Info("Node has more than one alias assigned, defaulting to the first", "node", klog.KObj(node), "CIDRs", cidrs)
	}

	_, cidrRange, err := netutils.ParseCIDRSloppy(cidrs[0])
	if err != nil {
		return nil, err
	}

	return cidrRange, nil
}

func (a *adapter) AddAlias(ctx context.Context, node *v1.Node, cidrRange *net.IPNet) error {
	if node.Spec.ProviderID == "" {
		return fmt.Errorf("node %s doesn't have providerID", node.Name)
	}

	return a.cloud.AddAliasToInstanceByProviderID(node.Spec.ProviderID, cidrRange)
}

func (a *adapter) Node(ctx context.Context, name string) (*v1.Node, error) {
	return a.k8s.CoreV1().Nodes().Get(context.TODO(), name, metav1.GetOptions{})
}

func (a *adapter) UpdateNodePodCIDR(ctx context.Context, node *v1.Node, cidrRange *net.IPNet) error {
	patch := map[string]interface{}{
		"apiVersion": node.APIVersion,
		"kind":       node.Kind,
		"metadata":   map[string]interface{}{"name": node.Name},
		"spec":       map[string]interface{}{"podCIDR": cidrRange.String()},
	}
	bytes, err := json.Marshal(patch)
	if err != nil {
		return err
	}

	_, err = a.k8s.CoreV1().Nodes().Patch(context.TODO(), node.Name, types.StrategicMergePatchType, bytes, metav1.PatchOptions{})
	return err
}

func (a *adapter) UpdateNodeNetworkUnavailable(nodeName string, unavailable bool) error {
	condition := v1.ConditionFalse
	if unavailable {
		condition = v1.ConditionTrue
	}
	return nodeutil.SetNodeCondition(a.k8s, types.NodeName(nodeName), v1.NodeCondition{
		Type:               v1.NodeNetworkUnavailable,
		Status:             condition,
		Reason:             "RouteCreated",
		Message:            "NodeController created an implicit route",
		LastTransitionTime: metav1.Now(),
	})
}

func (a *adapter) EmitNodeWarningEvent(nodeName, reason, fmt string, args ...interface{}) {
	ref := &v1.ObjectReference{Kind: "Node", Name: nodeName}
	a.recorder.Eventf(ref, v1.EventTypeNormal, reason, fmt, args...)
}

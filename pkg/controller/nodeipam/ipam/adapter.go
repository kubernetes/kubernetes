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
	"net"

	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce"
	nodeutil "k8s.io/kubernetes/pkg/util/node"
	"k8s.io/metrics/pkg/client/clientset/versioned/scheme"
)

type adapter struct {
	k8s   clientset.Interface
	cloud *gce.GCECloud

	recorder record.EventRecorder
}

func newAdapter(k8s clientset.Interface, cloud *gce.GCECloud) *adapter {
	ret := &adapter{
		k8s:   k8s,
		cloud: cloud,
	}

	broadcaster := record.NewBroadcaster()
	broadcaster.StartLogging(glog.Infof)
	ret.recorder = broadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "cloudCIDRAllocator"})
	glog.V(0).Infof("Sending events to api server.")
	broadcaster.StartRecordingToSink(&v1core.EventSinkImpl{
		Interface: k8s.CoreV1().Events(""),
	})

	return ret
}

func (a *adapter) Alias(ctx context.Context, nodeName string) (*net.IPNet, error) {
	cidrs, err := a.cloud.AliasRanges(types.NodeName(nodeName))
	if err != nil {
		return nil, err
	}

	switch len(cidrs) {
	case 0:
		return nil, nil
	case 1:
		break
	default:
		glog.Warningf("Node %q has more than one alias assigned (%v), defaulting to the first", nodeName, cidrs)
	}

	_, cidrRange, err := net.ParseCIDR(cidrs[0])
	if err != nil {
		return nil, err
	}

	return cidrRange, nil
}

func (a *adapter) AddAlias(ctx context.Context, nodeName string, cidrRange *net.IPNet) error {
	return a.cloud.AddAliasToInstance(types.NodeName(nodeName), cidrRange)
}

func (a *adapter) Node(ctx context.Context, name string) (*v1.Node, error) {
	return a.k8s.CoreV1().Nodes().Get(name, metav1.GetOptions{})
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

	_, err = a.k8s.CoreV1().Nodes().Patch(node.Name, types.StrategicMergePatchType, bytes)
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

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

package nodeports

import (
	"context"
	"fmt"
	"sort"
	"strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/component-helpers/resource"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	"k8s.io/kubernetes/pkg/scheduler/util"
)

// NodePorts is a plugin that checks if a node has free ports for the requested pod ports.
type NodePorts struct {
	enableInPlacePodVerticalScalingSchedulerPreemption bool
}

var _ fwk.PreFilterPlugin = &NodePorts{}
var _ fwk.FilterPlugin = &NodePorts{}
var _ fwk.EnqueueExtensions = &NodePorts{}
var _ fwk.SignPlugin = &NodePorts{}

const (
	// Name is the name of the plugin used in the plugin registry and configurations.
	Name = names.NodePorts

	// preFilterStateKey is the key in CycleState to NodePorts pre-computed data.
	// Using the name of the plugin will likely help us avoid collisions with other plugins.
	preFilterStateKey = "PreFilter" + Name

	// ErrReason when node ports aren't available.
	//
	// Deprecated: no longer reported as the status reason, which now starts with
	// ErrReasonPortConflict and names the conflicting ports.
	ErrReason = "node(s) didn't have free ports for the requested pod ports"

	// ErrReasonPortConflict is the prefix of the reason reported on a port conflict.
	// The conflicting ports are appended as " (protocol/hostIP:port)".
	ErrReasonPortConflict = "node(s) port conflict for the requested pod ports"
)

type preFilterState []v1.ContainerPort

// Clone the prefilter state.
func (s preFilterState) Clone() fwk.StateData {
	// The state is not impacted by adding/removing existing pods, hence we don't need to make a deep copy.
	return s
}

// Name returns name of the plugin. It is used in logs, etc.
func (pl *NodePorts) Name() string {
	return Name
}

// NodePort feasibility and scheduling is based on the host ports for the containers.
func (pl *NodePorts) SignPod(ctx context.Context, pod *v1.Pod) ([]fwk.SignFragment, *fwk.Status) {
	return []fwk.SignFragment{
		{Key: fwk.HostPortsSignerName, Value: fwk.HostPortsSigner(pod)},
	}, nil
}

// PreFilter invoked at the prefilter extension point.
func (pl *NodePorts) PreFilter(ctx context.Context, cycleState fwk.CycleState, pod *v1.Pod, nodes []fwk.NodeInfo) (*fwk.PreFilterResult, *fwk.Status) {
	if pl.enableInPlacePodVerticalScalingSchedulerPreemption && resource.IsPodResizeDeferred(pod) {
		return nil, fwk.NewStatus(fwk.Skip)
	}
	s := util.GetHostPorts(pod)
	// Skip if a pod has no ports.
	if len(s) == 0 {
		return nil, fwk.NewStatus(fwk.Skip)
	}
	cycleState.Write(preFilterStateKey, preFilterState(s))
	return nil, nil
}

// PreFilterExtensions do not exist for this plugin.
func (pl *NodePorts) PreFilterExtensions() fwk.PreFilterExtensions {
	return nil
}

func getPreFilterState(cycleState fwk.CycleState) (preFilterState, error) {
	c, err := cycleState.Read(preFilterStateKey)
	if err != nil {
		// preFilterState doesn't exist, likely PreFilter wasn't invoked.
		return nil, fmt.Errorf("reading %q from cycleState: %w", preFilterStateKey, err)
	}

	s, ok := c.(preFilterState)
	if !ok {
		return nil, fmt.Errorf("%+v  convert to nodeports.preFilterState error", c)
	}
	return s, nil
}

// EventsToRegister returns the possible events that may make a Pod
// failed by this plugin schedulable.
func (pl *NodePorts) EventsToRegister(_ context.Context) ([]fwk.ClusterEventWithHint, error) {
	return []fwk.ClusterEventWithHint{
		// Due to immutable fields `spec.containers[*].ports`, pod update events are ignored.
		{Event: fwk.ClusterEvent{Resource: fwk.AssignedPod, ActionType: fwk.Delete}, QueueingHintFn: pl.isSchedulableAfterAssignedPodDeleted},
		// We don't need the QueueingHintFn here because the scheduling of Pods will be always retried with backoff when this Event happens.
		// (the same as Queue)
		{Event: fwk.ClusterEvent{Resource: fwk.Node, ActionType: fwk.Add}},
	}, nil
}

// isSchedulableAfterAssignedPodDeleted is invoked whenever an assigned pod is deleted. It checks whether
// that change made a previously unschedulable pod schedulable.
func (pl *NodePorts) isSchedulableAfterAssignedPodDeleted(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
	deletedPod, _, err := util.As[*v1.Pod](oldObj, nil)
	if err != nil {
		return fwk.Queue, err
	}

	// If the deleted pod is unscheduled, it doesn't make the target pod schedulable.
	if deletedPod.Spec.NodeName == "" && deletedPod.Status.NominatedNodeName == "" {
		logger.V(4).Info("the deleted pod is unscheduled and it doesn't make the target pod schedulable", "pod", klog.KObj(pod), "deletedPod", klog.KObj(deletedPod))
		return fwk.QueueSkip, nil
	}

	// If the deleted pod doesn't use any host ports, it doesn't make the target pod schedulable.
	ports := util.GetHostPorts(deletedPod)
	if len(ports) == 0 {
		return fwk.QueueSkip, nil
	}

	// Verify that `pod` and the deleted pod don't have any common port(s).
	// So, deleting that pod couldn't make `pod` schedulable.
	portsInUse := make(fwk.HostPortInfo, len(ports))
	for _, p := range ports {
		portsInUse.Add(p.HostIP, string(p.Protocol), p.HostPort)
	}
	conflict, found := findConflict(util.GetHostPorts(pod), portsInUse)
	if !found {
		logger.V(4).Info("the deleted pod and the target pod don't have any common port(s), returning QueueSkip as deleting this Pod won't make the Pod schedulable", "pod", klog.KObj(pod), "deletedPod", klog.KObj(deletedPod))
		return fwk.QueueSkip, nil
	}

	if loggerV := logger.V(4); loggerV.Enabled() {
		loggerV.Info("the deleted pod and the target pod have common port(s), returning Queue as deleting this Pod may make the Pod schedulable", "conflict", conflictDescription(conflict), "pod", klog.KObj(pod), "deletedPod", klog.KObj(deletedPod))
	}
	return fwk.Queue, nil
}

// Filter invoked at the filter extension point.
func (pl *NodePorts) Filter(ctx context.Context, cycleState fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status {
	if pl.enableInPlacePodVerticalScalingSchedulerPreemption && resource.IsPodResizeDeferred(pod) {
		return nil
	}
	wantPorts, err := getPreFilterState(cycleState)
	if err != nil {
		return fwk.AsStatus(err)
	}

	portsInUse := nodeInfo.GetUsedPorts()
	if conflict, found := findConflict(wantPorts, portsInUse); found {
		if loggerV := klog.FromContext(ctx).V(4); loggerV.Enabled() {
			loggerV.Info("Host port conflict", "pod", klog.KObj(pod), "node", klog.KObj(nodeInfo.Node()), "conflicts", conflictsDescription(findConflicts(wantPorts, portsInUse)))
		}
		return fwk.NewStatus(fwk.Unschedulable, conflictReason(conflict))
	}

	return nil
}

// Fits checks if the pod has any ports conflicting with nodeInfo's ports.
// It returns true if there are no conflicts (which means that pod fits the node), otherwise false.
func Fits(pod *v1.Pod, nodeInfo fwk.NodeInfo) bool {
	_, found := findConflict(util.GetHostPorts(pod), nodeInfo.GetUsedPorts())
	return !found
}

// ConflictReason returns the reason describing the first port of the pod that conflicts with nodeInfo's ports,
// or an empty string if there is no conflict.
func ConflictReason(pod *v1.Pod, nodeInfo fwk.NodeInfo) string {
	conflict, found := findConflict(util.GetHostPorts(pod), nodeInfo.GetUsedPorts())
	if !found {
		return ""
	}
	return conflictReason(conflict)
}

// portConflict is a requested port together with the in-use host IP(s) it collides with.
type portConflict struct {
	protocol string
	port     int32
	wantIP   string
	inUseIPs []string
}

func newPortConflict(cp v1.ContainerPort, portsInUse fwk.HostPortInfo) portConflict {
	protocol, wantIP := string(cp.Protocol), cp.HostIP
	if protocol == "" {
		protocol = string(v1.ProtocolTCP)
	}
	if wantIP == "" {
		wantIP = fwk.DefaultBindAllHostIP
	}
	return portConflict{
		protocol: protocol,
		port:     cp.HostPort,
		wantIP:   wantIP,
		inUseIPs: conflictingInUseIPs(portsInUse, wantIP, fwk.ProtocolPort{Protocol: protocol, Port: cp.HostPort}),
	}
}

func findConflict(wantPorts []v1.ContainerPort, portsInUse fwk.HostPortInfo) (portConflict, bool) {
	for _, cp := range wantPorts {
		if portsInUse.CheckConflict(cp.HostIP, string(cp.Protocol), cp.HostPort) {
			return newPortConflict(cp, portsInUse), true
		}
	}
	return portConflict{}, false
}

func findConflicts(wantPorts []v1.ContainerPort, portsInUse fwk.HostPortInfo) []portConflict {
	var conflicts []portConflict
	for _, cp := range wantPorts {
		if portsInUse.CheckConflict(cp.HostIP, string(cp.Protocol), cp.HostPort) {
			conflicts = append(conflicts, newPortConflict(cp, portsInUse))
		}
	}
	return conflicts
}

// conflictingInUseIPs mirrors HostPortInfo.CheckConflict: a wildcard request collides with any IP,
// a specific request only with the same IP or the wildcard.
func conflictingInUseIPs(portsInUse fwk.HostPortInfo, wantIP string, pp fwk.ProtocolPort) []string {
	var ips []string
	for ip, ports := range portsInUse {
		if _, ok := ports[pp]; !ok {
			continue
		}
		if wantIP == fwk.DefaultBindAllHostIP || ip == fwk.DefaultBindAllHostIP || ip == wantIP {
			ips = append(ips, ip)
		}
	}
	sort.Strings(ips)
	return ips
}

// conflictDescription renders "TCP/0.0.0.0:80", or "TCP/127.0.0.1:80 conflicts with TCP/0.0.0.0:80"
// when the in-use IP differs from the requested one.
func conflictDescription(c portConflict) string {
	d := fmt.Sprintf("%s/%s:%d", c.protocol, c.wantIP, c.port)
	if len(c.inUseIPs) != 1 || c.inUseIPs[0] != c.wantIP {
		inUse := make([]string, 0, len(c.inUseIPs))
		for _, ip := range c.inUseIPs {
			inUse = append(inUse, fmt.Sprintf("%s/%s:%d", c.protocol, ip, c.port))
		}
		d += " conflicts with " + strings.Join(inUse, ", ")
	}
	return d
}

func conflictsDescription(conflicts []portConflict) string {
	descriptions := make([]string, 0, len(conflicts))
	for _, c := range conflicts {
		descriptions = append(descriptions, conflictDescription(c))
	}
	return strings.Join(descriptions, "; ")
}

func conflictReason(c portConflict) string {
	return ErrReasonPortConflict + " (" + conflictDescription(c) + ")"
}

// New initializes a new plugin and returns it.
func New(_ context.Context, _ runtime.Object, _ fwk.Handle, fts feature.Features) (fwk.Plugin, error) {
	return &NodePorts{enableInPlacePodVerticalScalingSchedulerPreemption: fts.EnableInPlacePodVerticalScalingSchedulerPreemption}, nil
}

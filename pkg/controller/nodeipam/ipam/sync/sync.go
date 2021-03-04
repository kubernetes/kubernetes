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

package sync

import (
	"context"
	"fmt"
	"net"
	"time"

	"k8s.io/klog/v2"

	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/controller/nodeipam/ipam/cidrset"
)

const (
	// InvalidPodCIDR is the event recorded when a node is found with an
	// invalid PodCIDR.
	InvalidPodCIDR = "CloudCIDRAllocatorInvalidPodCIDR"
	// InvalidModeEvent is the event recorded when the CIDR range cannot be
	// sync'd due to the cluster running in the wrong mode.
	InvalidModeEvent = "CloudCIDRAllocatorInvalidMode"
	// MismatchEvent is the event recorded when the CIDR range allocated in the
	// node spec does not match what has been allocated in the cloud.
	MismatchEvent = "CloudCIDRAllocatorMismatch"
)

// cloudAlias is the interface to the cloud platform APIs.
type cloudAlias interface {
	// Alias returns the IP alias for the node.
	Alias(ctx context.Context, node *v1.Node) (*net.IPNet, error)
	// AddAlias adds an alias to the node.
	AddAlias(ctx context.Context, node *v1.Node, cidrRange *net.IPNet) error
}

// kubeAPI is the interface to the Kubernetes APIs.
type kubeAPI interface {
	// Node returns the spec for the Node object.
	Node(ctx context.Context, name string) (*v1.Node, error)
	// UpdateNodePodCIDR updates the PodCIDR in the Node spec.
	UpdateNodePodCIDR(ctx context.Context, node *v1.Node, cidrRange *net.IPNet) error
	// UpdateNodeNetworkUnavailable updates the network unavailable status for the node.
	UpdateNodeNetworkUnavailable(nodeName string, unavailable bool) error
	// EmitNodeWarningEvent emits an event for the given node.
	EmitNodeWarningEvent(nodeName, reason, fmt string, args ...interface{})
}

// controller is the interface to the controller.
type controller interface {
	// ReportResult updates the controller with the result of the latest
	// sync operation.
	ReportResult(err error)
	// ResyncTimeout returns the amount of time to wait before retrying
	// a sync with a node.
	ResyncTimeout() time.Duration
}

// NodeSyncMode is the mode the cloud CIDR allocator runs in.
type NodeSyncMode string

var (
	// SyncFromCloud is the mode that synchronizes the IP allocation from the cloud
	// platform to the node.
	SyncFromCloud NodeSyncMode = "SyncFromCloud"
	// SyncFromCluster is the mode that synchronizes the IP allocation determined
	// by the k8s controller to the cloud provider.
	SyncFromCluster NodeSyncMode = "SyncFromCluster"
)

// IsValidMode returns true if the given mode is valid.
func IsValidMode(m NodeSyncMode) bool {
	switch m {
	case SyncFromCloud:
	case SyncFromCluster:
	default:
		return false
	}
	return true
}

// NodeSync synchronizes the state for a single node in the cluster.
type NodeSync struct {
	c          controller
	cloudAlias cloudAlias
	kubeAPI    kubeAPI
	mode       NodeSyncMode
	nodeName   string
	opChan     chan syncOp
	set        *cidrset.CidrSet
}

// New returns a new syncer for a given node.
func New(c controller, cloudAlias cloudAlias, kubeAPI kubeAPI, mode NodeSyncMode, nodeName string, set *cidrset.CidrSet) *NodeSync {
	return &NodeSync{
		c:          c,
		cloudAlias: cloudAlias,
		kubeAPI:    kubeAPI,
		mode:       mode,
		nodeName:   nodeName,
		opChan:     make(chan syncOp, 1),
		set:        set,
	}
}

// Loop runs the sync loop for a given node. done is an optional channel that
// is closed when the Loop() returns.
func (sync *NodeSync) Loop(done chan struct{}) {
	klog.V(2).Infof("Starting sync loop for node %q", sync.nodeName)

	defer func() {
		if done != nil {
			close(done)
		}
	}()

	timeout := sync.c.ResyncTimeout()
	delayTimer := time.NewTimer(timeout)
	klog.V(4).Infof("Resync node %q in %v", sync.nodeName, timeout)

	for {
		select {
		case op, more := <-sync.opChan:
			if !more {
				klog.V(2).Infof("Stopping sync loop")
				return
			}
			sync.c.ReportResult(op.run(sync))
			if !delayTimer.Stop() {
				<-delayTimer.C
			}
		case <-delayTimer.C:
			klog.V(4).Infof("Running resync for node %q", sync.nodeName)
			sync.c.ReportResult((&updateOp{}).run(sync))
		}

		timeout := sync.c.ResyncTimeout()
		delayTimer.Reset(timeout)
		klog.V(4).Infof("Resync node %q in %v", sync.nodeName, timeout)
	}
}

// Update causes an update operation on the given node. If node is nil, then
// the syncer will fetch the node spec from the API server before syncing.
//
// This method is safe to call from multiple goroutines.
func (sync *NodeSync) Update(node *v1.Node) {
	sync.opChan <- &updateOp{node}
}

// Delete performs the sync operations necessary to remove the node from the
// IPAM state.
//
// This method is safe to call from multiple goroutines.
func (sync *NodeSync) Delete(node *v1.Node) {
	sync.opChan <- &deleteOp{node}
	close(sync.opChan)
}

// syncOp is the interface for generic sync operation.
type syncOp interface {
	// run the requested sync operation.
	run(sync *NodeSync) error
}

// updateOp handles creation and updates of a node.
type updateOp struct {
	node *v1.Node
}

func (op *updateOp) String() string {
	if op.node == nil {
		return fmt.Sprintf("updateOp(nil)")
	}
	return fmt.Sprintf("updateOp(%q,%v)", op.node.Name, op.node.Spec.PodCIDR)
}

func (op *updateOp) run(sync *NodeSync) error {
	klog.V(3).Infof("Running updateOp %+v", op)

	ctx := context.Background()

	if op.node == nil {
		klog.V(3).Infof("Getting node spec for %q", sync.nodeName)
		node, err := sync.kubeAPI.Node(ctx, sync.nodeName)
		if err != nil {
			klog.Errorf("Error getting node %q spec: %v", sync.nodeName, err)
			return err
		}
		op.node = node
	}

	aliasRange, err := sync.cloudAlias.Alias(ctx, op.node)
	if err != nil {
		klog.Errorf("Error getting cloud alias for node %q: %v", sync.nodeName, err)
		return err
	}

	switch {
	case op.node.Spec.PodCIDR == "" && aliasRange == nil:
		err = op.allocateRange(ctx, sync, op.node)
	case op.node.Spec.PodCIDR == "" && aliasRange != nil:
		err = op.updateNodeFromAlias(ctx, sync, op.node, aliasRange)
	case op.node.Spec.PodCIDR != "" && aliasRange == nil:
		err = op.updateAliasFromNode(ctx, sync, op.node)
	case op.node.Spec.PodCIDR != "" && aliasRange != nil:
		err = op.validateRange(ctx, sync, op.node, aliasRange)
	}

	return err
}

// validateRange checks that the allocated range and the alias range
// match.
func (op *updateOp) validateRange(ctx context.Context, sync *NodeSync, node *v1.Node, aliasRange *net.IPNet) error {
	if node.Spec.PodCIDR != aliasRange.String() {
		klog.Errorf("Inconsistency detected between node PodCIDR and node alias (%v != %v)",
			node.Spec.PodCIDR, aliasRange)
		sync.kubeAPI.EmitNodeWarningEvent(node.Name, MismatchEvent,
			"Node.Spec.PodCIDR != cloud alias (%v != %v)", node.Spec.PodCIDR, aliasRange)
		// User intervention is required in this case, as this is most likely due
		// to the user mucking around with their VM aliases on the side.
	} else {
		klog.V(4).Infof("Node %q CIDR range %v is matches cloud assignment", node.Name, node.Spec.PodCIDR)
	}
	return nil
}

// updateNodeFromAlias updates the node from the cloud allocated
// alias.
func (op *updateOp) updateNodeFromAlias(ctx context.Context, sync *NodeSync, node *v1.Node, aliasRange *net.IPNet) error {
	if sync.mode != SyncFromCloud {
		sync.kubeAPI.EmitNodeWarningEvent(node.Name, InvalidModeEvent,
			"Cannot sync from cloud in mode %q", sync.mode)
		return fmt.Errorf("cannot sync from cloud in mode %q", sync.mode)
	}

	klog.V(2).Infof("Updating node spec with alias range, node.PodCIDR = %v", aliasRange)

	if err := sync.set.Occupy(aliasRange); err != nil {
		klog.Errorf("Error occupying range %v for node %v", aliasRange, sync.nodeName)
		return err
	}

	if err := sync.kubeAPI.UpdateNodePodCIDR(ctx, node, aliasRange); err != nil {
		klog.Errorf("Could not update node %q PodCIDR to %v: %v", node.Name, aliasRange, err)
		return err
	}

	klog.V(2).Infof("Node %q PodCIDR set to %v", node.Name, aliasRange)

	if err := sync.kubeAPI.UpdateNodeNetworkUnavailable(node.Name, false); err != nil {
		klog.Errorf("Could not update node NetworkUnavailable status to false: %v", err)
		return err
	}

	klog.V(2).Infof("Updated node %q PodCIDR from cloud alias %v", node.Name, aliasRange)

	return nil
}

// updateAliasFromNode updates the cloud alias given the node allocation.
func (op *updateOp) updateAliasFromNode(ctx context.Context, sync *NodeSync, node *v1.Node) error {
	if sync.mode != SyncFromCluster {
		sync.kubeAPI.EmitNodeWarningEvent(
			node.Name, InvalidModeEvent, "Cannot sync to cloud in mode %q", sync.mode)
		return fmt.Errorf("cannot sync to cloud in mode %q", sync.mode)
	}

	_, aliasRange, err := net.ParseCIDR(node.Spec.PodCIDR)
	if err != nil {
		klog.Errorf("Could not parse PodCIDR (%q) for node %q: %v",
			node.Spec.PodCIDR, node.Name, err)
		return err
	}

	if err := sync.set.Occupy(aliasRange); err != nil {
		klog.Errorf("Error occupying range %v for node %v", aliasRange, sync.nodeName)
		return err
	}

	if err := sync.cloudAlias.AddAlias(ctx, node, aliasRange); err != nil {
		klog.Errorf("Could not add alias %v for node %q: %v", aliasRange, node.Name, err)
		return err
	}

	if err := sync.kubeAPI.UpdateNodeNetworkUnavailable(node.Name, false); err != nil {
		klog.Errorf("Could not update node NetworkUnavailable status to false: %v", err)
		return err
	}

	klog.V(2).Infof("Updated node %q cloud alias with node spec, node.PodCIDR = %v",
		node.Name, node.Spec.PodCIDR)

	return nil
}

// allocateRange allocates a new range and updates both the cloud
// platform and the node allocation.
func (op *updateOp) allocateRange(ctx context.Context, sync *NodeSync, node *v1.Node) error {
	if sync.mode != SyncFromCluster {
		sync.kubeAPI.EmitNodeWarningEvent(node.Name, InvalidModeEvent,
			"Cannot allocate CIDRs in mode %q", sync.mode)
		return fmt.Errorf("controller cannot allocate CIDRS in mode %q", sync.mode)
	}

	cidrRange, err := sync.set.AllocateNext()
	if err != nil {
		return err
	}
	// If addAlias returns a hard error, cidrRange will be leaked as there
	// is no durable record of the range. The missing space will be
	// recovered on the next restart of the controller.
	if err := sync.cloudAlias.AddAlias(ctx, node, cidrRange); err != nil {
		klog.Errorf("Could not add alias %v for node %q: %v", cidrRange, node.Name, err)
		return err
	}

	if err := sync.kubeAPI.UpdateNodePodCIDR(ctx, node, cidrRange); err != nil {
		klog.Errorf("Could not update node %q PodCIDR to %v: %v", node.Name, cidrRange, err)
		return err
	}

	if err := sync.kubeAPI.UpdateNodeNetworkUnavailable(node.Name, false); err != nil {
		klog.Errorf("Could not update node NetworkUnavailable status to false: %v", err)
		return err
	}

	klog.V(2).Infof("Allocated PodCIDR %v for node %q", cidrRange, node.Name)

	return nil
}

// deleteOp handles deletion of a node.
type deleteOp struct {
	node *v1.Node
}

func (op *deleteOp) String() string {
	if op.node == nil {
		return fmt.Sprintf("deleteOp(nil)")
	}
	return fmt.Sprintf("deleteOp(%q,%v)", op.node.Name, op.node.Spec.PodCIDR)
}

func (op *deleteOp) run(sync *NodeSync) error {
	klog.V(3).Infof("Running deleteOp %+v", op)
	if op.node.Spec.PodCIDR == "" {
		klog.V(2).Infof("Node %q was deleted, node had no PodCIDR range assigned", op.node.Name)
		return nil
	}

	_, cidrRange, err := net.ParseCIDR(op.node.Spec.PodCIDR)
	if err != nil {
		klog.Errorf("Deleted node %q has an invalid podCIDR %q: %v",
			op.node.Name, op.node.Spec.PodCIDR, err)
		sync.kubeAPI.EmitNodeWarningEvent(op.node.Name, InvalidPodCIDR,
			"Node %q has an invalid PodCIDR: %q", op.node.Name, op.node.Spec.PodCIDR)
		return nil
	}

	sync.set.Release(cidrRange)
	klog.V(2).Infof("Node %q was deleted, releasing CIDR range %v",
		op.node.Name, op.node.Spec.PodCIDR)

	return nil
}

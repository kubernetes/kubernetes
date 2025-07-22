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
	netutils "k8s.io/utils/net"

	v1 "k8s.io/api/core/v1"
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
func (sync *NodeSync) Loop(logger klog.Logger, done chan struct{}) {
	logger.V(2).Info("Starting sync loop", "node", klog.KRef("", sync.nodeName))

	defer func() {
		if done != nil {
			close(done)
		}
	}()

	timeout := sync.c.ResyncTimeout()
	delayTimer := time.NewTimer(timeout)
	logger.V(4).Info("Try to resync node later", "node", klog.KRef("", sync.nodeName), "resyncTime", timeout)

	for {
		select {
		case op, more := <-sync.opChan:
			if !more {
				logger.V(2).Info("Stopping sync loop")
				return
			}
			sync.c.ReportResult(op.run(logger, sync))
			if !delayTimer.Stop() {
				<-delayTimer.C
			}
		case <-delayTimer.C:
			logger.V(4).Info("Running resync", "node", klog.KRef("", sync.nodeName))
			sync.c.ReportResult((&updateOp{}).run(logger, sync))
		}

		timeout := sync.c.ResyncTimeout()
		delayTimer.Reset(timeout)
		logger.V(4).Info("Try to resync node later", "node", klog.KRef("", sync.nodeName), "resyncTime", timeout)
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
	run(logger klog.Logger, sync *NodeSync) error
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

func (op *updateOp) run(logger klog.Logger, sync *NodeSync) error {
	logger.V(3).Info("Running updateOp", "updateOp", op)

	ctx := context.Background()

	if op.node == nil {
		logger.V(3).Info("Getting node spec", "node", klog.KRef("", sync.nodeName))
		node, err := sync.kubeAPI.Node(ctx, sync.nodeName)
		if err != nil {
			logger.Error(err, "Error getting node pec", "node", klog.KRef("", sync.nodeName))
			return err
		}
		op.node = node
	}

	aliasRange, err := sync.cloudAlias.Alias(ctx, op.node)
	if err != nil {
		logger.Error(err, "Error getting cloud alias for node", "node", klog.KRef("", sync.nodeName))
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
		klog.FromContext(ctx).Error(nil, "Inconsistency detected between node PodCIDR and node alias", "podCIDR", node.Spec.PodCIDR, "alias", aliasRange)
		sync.kubeAPI.EmitNodeWarningEvent(node.Name, MismatchEvent,
			"Node.Spec.PodCIDR != cloud alias (%v != %v)", node.Spec.PodCIDR, aliasRange)
		// User intervention is required in this case, as this is most likely due
		// to the user mucking around with their VM aliases on the side.
	} else {
		klog.FromContext(ctx).V(4).Info("Node CIDR range is matches cloud assignment", "node", klog.KObj(node), "podCIDR", node.Spec.PodCIDR)
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
	logger := klog.FromContext(ctx)
	logger.V(2).Info("Updating node spec with alias range", "podCIDR", aliasRange)

	if err := sync.set.Occupy(aliasRange); err != nil {
		logger.Error(nil, "Error occupying range for node", "node", klog.KRef("", sync.nodeName), "alias", aliasRange)
		return err
	}

	if err := sync.kubeAPI.UpdateNodePodCIDR(ctx, node, aliasRange); err != nil {
		logger.Error(err, "Could not update node PodCIDR", "node", klog.KObj(node), "podCIDR", aliasRange)
		return err
	}

	logger.V(2).Info("Node PodCIDR updated", "node", klog.KObj(node), "podCIDR", aliasRange)

	if err := sync.kubeAPI.UpdateNodeNetworkUnavailable(node.Name, false); err != nil {
		logger.Error(err, "Could not update node NetworkUnavailable status to false")
		return err
	}

	logger.V(2).Info("Updated node PodCIDR from cloud alias", "node", klog.KObj(node), "alias", aliasRange)

	return nil
}

// updateAliasFromNode updates the cloud alias given the node allocation.
func (op *updateOp) updateAliasFromNode(ctx context.Context, sync *NodeSync, node *v1.Node) error {
	if sync.mode != SyncFromCluster {
		sync.kubeAPI.EmitNodeWarningEvent(
			node.Name, InvalidModeEvent, "Cannot sync to cloud in mode %q", sync.mode)
		return fmt.Errorf("cannot sync to cloud in mode %q", sync.mode)
	}

	_, aliasRange, err := netutils.ParseCIDRSloppy(node.Spec.PodCIDR)

	logger := klog.FromContext(ctx)
	if err != nil {
		logger.Error(err, "Could not parse PodCIDR for node", "node", klog.KObj(node), "podCIDR", node.Spec.PodCIDR)
		return err
	}

	if err := sync.set.Occupy(aliasRange); err != nil {
		logger.Error(nil, "Error occupying range for node", "node", klog.KRef("", sync.nodeName), "alias", aliasRange)
		return err
	}

	if err := sync.cloudAlias.AddAlias(ctx, node, aliasRange); err != nil {
		logger.Error(err, "Could not add alias for node", "node", klog.KObj(node), "alias", aliasRange)
		return err
	}

	if err := sync.kubeAPI.UpdateNodeNetworkUnavailable(node.Name, false); err != nil {
		logger.Error(err, "Could not update node NetworkUnavailable status to false")
		return err
	}

	logger.V(2).Info("Updated node cloud alias with node spec", "node", klog.KObj(node), "podCIDR", node.Spec.PodCIDR)

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
	logger := klog.FromContext(ctx)
	if err := sync.cloudAlias.AddAlias(ctx, node, cidrRange); err != nil {
		logger.Error(err, "Could not add alias for node", "node", klog.KObj(node), "alias", cidrRange)
		return err
	}

	if err := sync.kubeAPI.UpdateNodePodCIDR(ctx, node, cidrRange); err != nil {
		logger.Error(err, "Could not update node PodCIDR", "node", klog.KObj(node), "podCIDR", cidrRange)
		return err
	}

	if err := sync.kubeAPI.UpdateNodeNetworkUnavailable(node.Name, false); err != nil {
		logger.Error(err, "Could not update node NetworkUnavailable status to false")
		return err
	}

	logger.V(2).Info("Allocated PodCIDR for node", "node", klog.KObj(node), "podCIDR", cidrRange)

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

func (op *deleteOp) run(logger klog.Logger, sync *NodeSync) error {
	logger.V(3).Info("Running deleteOp", "deleteOp", op)
	if op.node.Spec.PodCIDR == "" {
		logger.V(2).Info("Node was deleted, node had no PodCIDR range assigned", "node", klog.KObj(op.node))
		return nil
	}

	_, cidrRange, err := netutils.ParseCIDRSloppy(op.node.Spec.PodCIDR)
	if err != nil {
		logger.Error(err, "Deleted node has an invalid podCIDR", "node", klog.KObj(op.node), "podCIDR", op.node.Spec.PodCIDR)
		sync.kubeAPI.EmitNodeWarningEvent(op.node.Name, InvalidPodCIDR,
			"Node %q has an invalid PodCIDR: %q", op.node.Name, op.node.Spec.PodCIDR)
		return nil
	}

	sync.set.Release(cidrRange)
	logger.V(2).Info("Node was deleted, releasing CIDR range", "node", klog.KObj(op.node), "podCIDR", op.node.Spec.PodCIDR)

	return nil
}

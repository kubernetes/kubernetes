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
	"fmt"
	"net"
	"sync"
	"time"

	"k8s.io/klog/v2"
	netutils "k8s.io/utils/net"

	v1 "k8s.io/api/core/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	informers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	cloudprovider "k8s.io/cloud-provider"
	"k8s.io/kubernetes/pkg/controller/nodeipam/ipam/cidrset"
	nodesync "k8s.io/kubernetes/pkg/controller/nodeipam/ipam/sync"
	controllerutil "k8s.io/kubernetes/pkg/controller/util/node"
	"k8s.io/legacy-cloud-providers/gce"
)

// Config for the IPAM controller.
type Config struct {
	// Resync is the default timeout duration when there are no errors.
	Resync time.Duration
	// MaxBackoff is the maximum timeout when in a error backoff state.
	MaxBackoff time.Duration
	// InitialRetry is the initial retry interval when an error is reported.
	InitialRetry time.Duration
	// Mode to use to synchronize.
	Mode nodesync.NodeSyncMode
}

// Controller is the controller for synchronizing cluster and cloud node
// pod CIDR range assignments.
type Controller struct {
	config  *Config
	adapter *adapter

	lock    sync.Mutex
	syncers map[string]*nodesync.NodeSync

	set *cidrset.CidrSet
}

// NewController returns a new instance of the IPAM controller.
func NewController(
	ctx context.Context,
	config *Config,
	kubeClient clientset.Interface,
	cloud cloudprovider.Interface,
	clusterCIDR, serviceCIDR *net.IPNet,
	nodeCIDRMaskSize int) (*Controller, error) {

	if !nodesync.IsValidMode(config.Mode) {
		return nil, fmt.Errorf("invalid IPAM controller mode %q", config.Mode)
	}

	gceCloud, ok := cloud.(*gce.Cloud)
	if !ok {
		return nil, fmt.Errorf("cloud IPAM controller does not support %q provider", cloud.ProviderName())
	}

	set, err := cidrset.NewCIDRSet(clusterCIDR, nodeCIDRMaskSize)
	if err != nil {
		return nil, err
	}

	c := &Controller{
		config:  config,
		adapter: newAdapter(ctx, kubeClient, gceCloud),
		syncers: make(map[string]*nodesync.NodeSync),
		set:     set,
	}

	if err := occupyServiceCIDR(c.set, clusterCIDR, serviceCIDR); err != nil {
		return nil, err
	}

	//check whether there is a remaining cidr after occupyServiceCIDR
	cidr, err := c.set.AllocateNext()
	switch err {
	case cidrset.ErrCIDRRangeNoCIDRsRemaining:
		return nil, fmt.Errorf("failed after occupy serviceCIDR: %v", err)
	case nil:
		err := c.set.Release(cidr)
		return c, err
	default:
		return nil, fmt.Errorf("unexpected error when check remaining CIDR range: %v", err)
	}
}

// Start initializes the Controller with the existing list of nodes and
// registers the informers for node changes. This will start synchronization
// of the node and cloud CIDR range allocations.
func (c *Controller) Start(logger klog.Logger, nodeInformer informers.NodeInformer) error {
	logger.Info("Starting IPAM controller", "config", c.config)

	nodes, err := listNodes(logger, c.adapter.k8s)
	if err != nil {
		return err
	}
	for _, node := range nodes.Items {
		if node.Spec.PodCIDR != "" {
			_, cidrRange, err := netutils.ParseCIDRSloppy(node.Spec.PodCIDR)
			if err == nil {
				c.set.Occupy(cidrRange)
				logger.V(3).Info("Occupying CIDR for node", "CIDR", node.Spec.PodCIDR, "node", klog.KObj(&node))
			} else {
				logger.Error(err, "Node has an invalid CIDR", "node", klog.KObj(&node), "CIDR", node.Spec.PodCIDR)
			}
		}

		func() {
			c.lock.Lock()
			defer c.lock.Unlock()

			// XXX/bowei -- stagger the start of each sync cycle.
			syncer := c.newSyncer(node.Name)
			c.syncers[node.Name] = syncer
			go syncer.Loop(logger, nil)
		}()
	}

	nodeInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: controllerutil.CreateAddNodeHandler(func(node *v1.Node) error {
			return c.onAdd(logger, node)
		}),
		UpdateFunc: controllerutil.CreateUpdateNodeHandler(func(_, newNode *v1.Node) error {
			return c.onUpdate(logger, newNode)
		}),
		DeleteFunc: controllerutil.CreateDeleteNodeHandler(logger, func(node *v1.Node) error {
			return c.onDelete(logger, node)
		}),
	})

	return nil
}

func (c *Controller) Run(ctx context.Context) {
	defer utilruntime.HandleCrash()

	go c.adapter.Run(ctx)
	<-ctx.Done()
}

type nodeState struct {
	t Timeout
}

func (ns *nodeState) ReportResult(err error) {
	ns.t.Update(err == nil)
}

func (ns *nodeState) ResyncTimeout() time.Duration {
	return ns.t.Next()
}

func (c *Controller) newSyncer(name string) *nodesync.NodeSync {
	ns := &nodeState{
		Timeout{
			Resync:       c.config.Resync,
			MaxBackoff:   c.config.MaxBackoff,
			InitialRetry: c.config.InitialRetry,
		},
	}
	return nodesync.New(ns, c.adapter, c.adapter, c.config.Mode, name, c.set)
}

func (c *Controller) onAdd(logger klog.Logger, node *v1.Node) error {
	c.lock.Lock()
	defer c.lock.Unlock()

	syncer, ok := c.syncers[node.Name]
	if !ok {
		syncer = c.newSyncer(node.Name)
		c.syncers[node.Name] = syncer
		go syncer.Loop(logger, nil)
	} else {
		logger.Info("Add for node that already exists", "node", klog.KObj(node))
	}
	syncer.Update(node)

	return nil
}

func (c *Controller) onUpdate(logger klog.Logger, node *v1.Node) error {
	c.lock.Lock()
	defer c.lock.Unlock()

	if sync, ok := c.syncers[node.Name]; ok {
		sync.Update(node)
	} else {
		logger.Error(nil, "Received update for non-existent node", "node", klog.KObj(node))
		return fmt.Errorf("unknown node %q", node.Name)
	}

	return nil
}

func (c *Controller) onDelete(logger klog.Logger, node *v1.Node) error {
	c.lock.Lock()
	defer c.lock.Unlock()

	if syncer, ok := c.syncers[node.Name]; ok {
		syncer.Delete(node)
		delete(c.syncers, node.Name)
	} else {
		logger.Info("Node was already deleted", "node", klog.KObj(node))
	}

	return nil
}

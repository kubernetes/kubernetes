/*
Copyright 2014 The Kubernetes Authors.

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

package node

import (
	"context"
	"sync"
	"time"

	"github.com/onsi/ginkgo/v2"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"

	"k8s.io/kubernetes/test/e2e/framework"
	e2essh "k8s.io/kubernetes/test/e2e/framework/ssh"
)

// NodeKiller is a utility to simulate node failures.
type NodeKiller struct {
	config   framework.NodeKillerConfig
	client   clientset.Interface
	provider string
}

// NewNodeKiller creates new NodeKiller.
func NewNodeKiller(config framework.NodeKillerConfig, client clientset.Interface, provider string) *NodeKiller {
	config.NodeKillerStopCtx, config.NodeKillerStop = context.WithCancel(context.Background())
	return &NodeKiller{config, client, provider}
}

// Run starts NodeKiller until stopCh is closed.
func (k *NodeKiller) Run(ctx context.Context) {
	// wait.JitterUntil starts work immediately, so wait first.
	time.Sleep(wait.Jitter(k.config.Interval, k.config.JitterFactor))
	wait.JitterUntilWithContext(ctx, func(ctx context.Context) {
		nodes := k.pickNodes(ctx)
		k.kill(ctx, nodes)
	}, k.config.Interval, k.config.JitterFactor, true)
}

func (k *NodeKiller) pickNodes(ctx context.Context) []v1.Node {
	nodes, err := GetReadySchedulableNodes(ctx, k.client)
	framework.ExpectNoError(err)
	numNodes := int(k.config.FailureRatio * float64(len(nodes.Items)))

	nodes, err = GetBoundedReadySchedulableNodes(ctx, k.client, numNodes)
	framework.ExpectNoError(err)
	return nodes.Items
}

func (k *NodeKiller) kill(ctx context.Context, nodes []v1.Node) {
	wg := sync.WaitGroup{}
	wg.Add(len(nodes))
	for _, node := range nodes {
		node := node
		go func() {
			defer ginkgo.GinkgoRecover()
			defer wg.Done()

			framework.Logf("Stopping docker and kubelet on %q to simulate failure", node.Name)
			err := e2essh.IssueSSHCommand(ctx, "sudo systemctl stop docker kubelet", k.provider, &node)
			if err != nil {
				framework.Logf("ERROR while stopping node %q: %v", node.Name, err)
				return
			}

			time.Sleep(k.config.SimulatedDowntime)

			framework.Logf("Rebooting %q to repair the node", node.Name)
			err = e2essh.IssueSSHCommand(ctx, "sudo reboot", k.provider, &node)
			if err != nil {
				framework.Logf("ERROR while rebooting node %q: %v", node.Name, err)
				return
			}
		}()
	}
	wg.Wait()
}

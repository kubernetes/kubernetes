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

package framework

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strings"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	nodectlr "k8s.io/kubernetes/pkg/controller/nodelifecycle"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
	e2essh "k8s.io/kubernetes/test/e2e/framework/ssh"
	testutils "k8s.io/kubernetes/test/utils"
)

const etcdImage = "3.4.3-0"

const sleepTime = 20 * time.Second

// EtcdUpgrade upgrades etcd on GCE.
func EtcdUpgrade(targetStorage, targetVersion string) error {
	switch TestContext.Provider {
	case "gce":
		return etcdUpgradeGCE(targetStorage, targetVersion)
	default:
		return fmt.Errorf("EtcdUpgrade() is not implemented for provider %s", TestContext.Provider)
	}
}

// MasterUpgrade upgrades master node on GCE/GKE.
func MasterUpgrade(v string) error {
	switch TestContext.Provider {
	case "gce":
		return masterUpgradeGCE(v, false)
	case "gke":
		return masterUpgradeGKE(v)
	case "kubernetes-anywhere":
		return masterUpgradeKubernetesAnywhere(v)
	default:
		return fmt.Errorf("MasterUpgrade() is not implemented for provider %s", TestContext.Provider)
	}
}

func etcdUpgradeGCE(targetStorage, targetVersion string) error {
	env := append(
		os.Environ(),
		"TEST_ETCD_VERSION="+targetVersion,
		"STORAGE_BACKEND="+targetStorage,
		"TEST_ETCD_IMAGE="+etcdImage)

	_, _, err := RunCmdEnv(env, gceUpgradeScript(), "-l", "-M")
	return err
}

// MasterUpgradeGCEWithKubeProxyDaemonSet upgrades master node on GCE with enabling/disabling the daemon set of kube-proxy.
// TODO(mrhohn): Remove this function when kube-proxy is run as a DaemonSet by default.
func MasterUpgradeGCEWithKubeProxyDaemonSet(v string, enableKubeProxyDaemonSet bool) error {
	return masterUpgradeGCE(v, enableKubeProxyDaemonSet)
}

// TODO(mrhohn): Remove 'enableKubeProxyDaemonSet' when kube-proxy is run as a DaemonSet by default.
func masterUpgradeGCE(rawV string, enableKubeProxyDaemonSet bool) error {
	env := append(os.Environ(), fmt.Sprintf("KUBE_PROXY_DAEMONSET=%v", enableKubeProxyDaemonSet))
	// TODO: Remove these variables when they're no longer needed for downgrades.
	if TestContext.EtcdUpgradeVersion != "" && TestContext.EtcdUpgradeStorage != "" {
		env = append(env,
			"TEST_ETCD_VERSION="+TestContext.EtcdUpgradeVersion,
			"STORAGE_BACKEND="+TestContext.EtcdUpgradeStorage,
			"TEST_ETCD_IMAGE="+etcdImage)
	} else {
		// In e2e tests, we skip the confirmation prompt about
		// implicit etcd upgrades to simulate the user entering "y".
		env = append(env, "TEST_ALLOW_IMPLICIT_ETCD_UPGRADE=true")
	}

	v := "v" + rawV
	_, _, err := RunCmdEnv(env, gceUpgradeScript(), "-M", v)
	return err
}

func locationParamGKE() string {
	if TestContext.CloudConfig.MultiMaster {
		// GKE Regional Clusters are being tested.
		return fmt.Sprintf("--region=%s", TestContext.CloudConfig.Region)
	}
	return fmt.Sprintf("--zone=%s", TestContext.CloudConfig.Zone)
}

func appendContainerCommandGroupIfNeeded(args []string) []string {
	if TestContext.CloudConfig.Region != "" {
		// TODO(wojtek-t): Get rid of it once Regional Clusters go to GA.
		return append([]string{"beta"}, args...)
	}
	return args
}

func masterUpgradeGKE(v string) error {
	Logf("Upgrading master to %q", v)
	args := []string{
		"container",
		"clusters",
		fmt.Sprintf("--project=%s", TestContext.CloudConfig.ProjectID),
		locationParamGKE(),
		"upgrade",
		TestContext.CloudConfig.Cluster,
		"--master",
		fmt.Sprintf("--cluster-version=%s", v),
		"--quiet",
	}
	_, _, err := RunCmd("gcloud", appendContainerCommandGroupIfNeeded(args)...)
	if err != nil {
		return err
	}

	waitForSSHTunnels()

	return nil
}

func masterUpgradeKubernetesAnywhere(v string) error {
	Logf("Upgrading master to %q", v)

	kaPath := TestContext.KubernetesAnywherePath
	originalConfigPath := filepath.Join(kaPath, ".config")
	backupConfigPath := filepath.Join(kaPath, ".config.bak")
	updatedConfigPath := filepath.Join(kaPath, fmt.Sprintf(".config-%s", v))

	// modify config with specified k8s version
	if _, _, err := RunCmd("sed",
		"-i.bak", // writes original to .config.bak
		fmt.Sprintf(`s/kubernetes_version=.*$/kubernetes_version=%q/`, v),
		originalConfigPath); err != nil {
		return err
	}

	defer func() {
		// revert .config.bak to .config
		if err := os.Rename(backupConfigPath, originalConfigPath); err != nil {
			Logf("Could not rename %s back to %s", backupConfigPath, originalConfigPath)
		}
	}()

	// invoke ka upgrade
	if _, _, err := RunCmd("make", "-C", TestContext.KubernetesAnywherePath,
		"WAIT_FOR_KUBECONFIG=y", "upgrade-master"); err != nil {
		return err
	}

	// move .config to .config.<version>
	if err := os.Rename(originalConfigPath, updatedConfigPath); err != nil {
		return err
	}

	return nil
}

// NodeUpgrade upgrades nodes on GCE/GKE.
func NodeUpgrade(f *Framework, v string, img string) error {
	// Perform the upgrade.
	var err error
	switch TestContext.Provider {
	case "gce":
		err = nodeUpgradeGCE(v, img, false)
	case "gke":
		err = nodeUpgradeGKE(v, img)
	default:
		err = fmt.Errorf("NodeUpgrade() is not implemented for provider %s", TestContext.Provider)
	}
	if err != nil {
		return err
	}
	return waitForNodesReadyAfterUpgrade(f)
}

// NodeUpgradeGCEWithKubeProxyDaemonSet upgrades nodes on GCE with enabling/disabling the daemon set of kube-proxy.
// TODO(mrhohn): Remove this function when kube-proxy is run as a DaemonSet by default.
func NodeUpgradeGCEWithKubeProxyDaemonSet(f *Framework, v string, img string, enableKubeProxyDaemonSet bool) error {
	// Perform the upgrade.
	if err := nodeUpgradeGCE(v, img, enableKubeProxyDaemonSet); err != nil {
		return err
	}
	return waitForNodesReadyAfterUpgrade(f)
}

func waitForNodesReadyAfterUpgrade(f *Framework) error {
	// Wait for it to complete and validate nodes are healthy.
	//
	// TODO(ihmccreery) We shouldn't have to wait for nodes to be ready in
	// GKE; the operation shouldn't return until they all are.
	numNodes, err := TotalRegistered(f.ClientSet)
	if err != nil {
		return fmt.Errorf("couldn't detect number of nodes")
	}
	Logf("Waiting up to %v for all %d nodes to be ready after the upgrade", RestartNodeReadyAgainTimeout, numNodes)
	if _, err := CheckReady(f.ClientSet, numNodes, RestartNodeReadyAgainTimeout); err != nil {
		return err
	}
	return nil
}

// TODO(mrhohn): Remove 'enableKubeProxyDaemonSet' when kube-proxy is run as a DaemonSet by default.
func nodeUpgradeGCE(rawV, img string, enableKubeProxyDaemonSet bool) error {
	v := "v" + rawV
	env := append(os.Environ(), fmt.Sprintf("KUBE_PROXY_DAEMONSET=%v", enableKubeProxyDaemonSet))
	if img != "" {
		env = append(env, "KUBE_NODE_OS_DISTRIBUTION="+img)
		_, _, err := RunCmdEnv(env, gceUpgradeScript(), "-N", "-o", v)
		return err
	}
	_, _, err := RunCmdEnv(env, gceUpgradeScript(), "-N", v)
	return err
}

func nodeUpgradeGKE(v string, img string) error {
	Logf("Upgrading nodes to version %q and image %q", v, img)
	nps, err := nodePoolsGKE()
	if err != nil {
		return err
	}
	Logf("Found node pools %v", nps)
	for _, np := range nps {
		args := []string{
			"container",
			"clusters",
			fmt.Sprintf("--project=%s", TestContext.CloudConfig.ProjectID),
			locationParamGKE(),
			"upgrade",
			TestContext.CloudConfig.Cluster,
			fmt.Sprintf("--node-pool=%s", np),
			fmt.Sprintf("--cluster-version=%s", v),
			"--quiet",
		}
		if len(img) > 0 {
			args = append(args, fmt.Sprintf("--image-type=%s", img))
		}
		_, _, err = RunCmd("gcloud", appendContainerCommandGroupIfNeeded(args)...)

		if err != nil {
			return err
		}

		waitForSSHTunnels()
	}
	return nil
}

func nodePoolsGKE() ([]string, error) {
	args := []string{
		"container",
		"node-pools",
		fmt.Sprintf("--project=%s", TestContext.CloudConfig.ProjectID),
		locationParamGKE(),
		"list",
		fmt.Sprintf("--cluster=%s", TestContext.CloudConfig.Cluster),
		`--format="get(name)"`,
	}
	stdout, _, err := RunCmd("gcloud", appendContainerCommandGroupIfNeeded(args)...)
	if err != nil {
		return nil, err
	}
	if len(strings.TrimSpace(stdout)) == 0 {
		return []string{}, nil
	}
	return strings.Fields(stdout), nil
}

func gceUpgradeScript() string {
	if len(TestContext.GCEUpgradeScript) == 0 {
		return path.Join(TestContext.RepoRoot, "cluster/gce/upgrade.sh")
	}
	return TestContext.GCEUpgradeScript
}

func waitForSSHTunnels() {
	Logf("Waiting for SSH tunnels to establish")
	RunKubectl("run", "ssh-tunnel-test",
		"--image=busybox",
		"--restart=Never",
		"--command", "--",
		"echo", "Hello")
	defer RunKubectl("delete", "pod", "ssh-tunnel-test")

	// allow up to a minute for new ssh tunnels to establish
	wait.PollImmediate(5*time.Second, time.Minute, func() (bool, error) {
		_, err := RunKubectl("logs", "ssh-tunnel-test")
		return err == nil, nil
	})
}

// NodeKiller is a utility to simulate node failures.
type NodeKiller struct {
	config   NodeKillerConfig
	client   clientset.Interface
	provider string
}

// NewNodeKiller creates new NodeKiller.
func NewNodeKiller(config NodeKillerConfig, client clientset.Interface, provider string) *NodeKiller {
	config.NodeKillerStopCh = make(chan struct{})
	return &NodeKiller{config, client, provider}
}

// Run starts NodeKiller until stopCh is closed.
func (k *NodeKiller) Run(stopCh <-chan struct{}) {
	// wait.JitterUntil starts work immediately, so wait first.
	time.Sleep(wait.Jitter(k.config.Interval, k.config.JitterFactor))
	wait.JitterUntil(func() {
		nodes := k.pickNodes()
		k.kill(nodes)
	}, k.config.Interval, k.config.JitterFactor, true, stopCh)
}

func (k *NodeKiller) pickNodes() []v1.Node {
	nodes, err := GetReadySchedulableNodes(k.client)
	ExpectNoError(err)
	numNodes := int(k.config.FailureRatio * float64(len(nodes.Items)))

	nodes, err = GetBoundedReadySchedulableNodes(k.client, numNodes)
	ExpectNoError(err)
	return nodes.Items
}

func (k *NodeKiller) kill(nodes []v1.Node) {
	wg := sync.WaitGroup{}
	wg.Add(len(nodes))
	for _, node := range nodes {
		node := node
		go func() {
			defer wg.Done()

			Logf("Stopping docker and kubelet on %q to simulate failure", node.Name)
			err := e2essh.IssueSSHCommand("sudo systemctl stop docker kubelet", k.provider, &node)
			if err != nil {
				Logf("ERROR while stopping node %q: %v", node.Name, err)
				return
			}

			time.Sleep(k.config.SimulatedDowntime)

			Logf("Rebooting %q to repair the node", node.Name)
			err = e2essh.IssueSSHCommand("sudo reboot", k.provider, &node)
			if err != nil {
				Logf("ERROR while rebooting node %q: %v", node.Name, err)
				return
			}
		}()
	}
	wg.Wait()
}

// DeleteNodeOnCloudProvider deletes the specified node.
func DeleteNodeOnCloudProvider(node *v1.Node) error {
	return TestContext.CloudConfig.Provider.DeleteNode(node)
}

// TotalRegistered returns number of registered Nodes excluding Master Node.
func TotalRegistered(c clientset.Interface) (int, error) {
	nodes, err := waitListSchedulableNodes(c)
	if err != nil {
		Logf("Failed to list nodes: %v", err)
		return 0, err
	}
	return len(nodes.Items), nil
}

// GetReadySchedulableNodes addresses the common use case of getting nodes you can do work on.
// 1) Needs to be schedulable.
// 2) Needs to be ready.
// If EITHER 1 or 2 is not true, most tests will want to ignore the node entirely.
// If there are no nodes that are both ready and schedulable, this will return an error.
func GetReadySchedulableNodes(c clientset.Interface) (nodes *v1.NodeList, err error) {
	nodes, err = waitListSchedulableNodes(c)
	if err != nil {
		return nil, fmt.Errorf("listing schedulable nodes error: %s", err)
	}
	Filter(nodes, func(node v1.Node) bool {
		return IsNodeSchedulable(&node) && IsNodeUntainted(&node)
	})
	if len(nodes.Items) == 0 {
		return nil, fmt.Errorf("there are currently no ready, schedulable nodes in the cluster")
	}
	return nodes, nil
}

// GetBoundedReadySchedulableNodes is like GetReadySchedulableNodes except that it returns
// at most maxNodes nodes. Use this to keep your test case from blowing up when run on a
// large cluster.
func GetBoundedReadySchedulableNodes(c clientset.Interface, maxNodes int) (nodes *v1.NodeList, err error) {
	nodes, err = GetReadySchedulableNodes(c)
	if err != nil {
		return nil, err
	}
	if len(nodes.Items) > maxNodes {
		shuffled := make([]v1.Node, maxNodes)
		perm := rand.Perm(len(nodes.Items))
		for i, j := range perm {
			if j < len(shuffled) {
				shuffled[j] = nodes.Items[i]
			}
		}
		nodes.Items = shuffled
	}
	return nodes, nil
}

// IsNodeSchedulable returns true if:
// 1) doesn't have "unschedulable" field set
// 2) it also returns true from isNodeReady
func IsNodeSchedulable(node *v1.Node) bool {
	if node == nil {
		return false
	}
	return !node.Spec.Unschedulable && isNodeReady(node)
}

// isNodeReady returns true if:
// 1) it's Ready condition is set to true
// 2) doesn't have NetworkUnavailable condition set to true
func isNodeReady(node *v1.Node) bool {
	nodeReady := IsConditionSetAsExpected(node, v1.NodeReady, true)
	networkReady := IsConditionUnset(node, v1.NodeNetworkUnavailable) ||
		IsConditionSetAsExpectedSilent(node, v1.NodeNetworkUnavailable, false)
	return nodeReady && networkReady
}

// IsNodeUntainted tests whether a fake pod can be scheduled on "node", given its current taints.
// TODO: need to discuss wether to return bool and error type
func IsNodeUntainted(node *v1.Node) bool {
	return isNodeUntaintedWithNonblocking(node, "")
}

// isNodeUntaintedWithNonblocking tests whether a fake pod can be scheduled on "node"
// but allows for taints in the list of non-blocking taints.
func isNodeUntaintedWithNonblocking(node *v1.Node, nonblockingTaints string) bool {
	fakePod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "fake-not-scheduled",
			Namespace: "fake-not-scheduled",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "fake-not-scheduled",
					Image: "fake-not-scheduled",
				},
			},
		},
	}

	nodeInfo := schedulernodeinfo.NewNodeInfo()

	// Simple lookup for nonblocking taints based on comma-delimited list.
	nonblockingTaintsMap := map[string]struct{}{}
	for _, t := range strings.Split(nonblockingTaints, ",") {
		if strings.TrimSpace(t) != "" {
			nonblockingTaintsMap[strings.TrimSpace(t)] = struct{}{}
		}
	}

	if len(nonblockingTaintsMap) > 0 {
		nodeCopy := node.DeepCopy()
		nodeCopy.Spec.Taints = []v1.Taint{}
		for _, v := range node.Spec.Taints {
			if _, isNonblockingTaint := nonblockingTaintsMap[v.Key]; !isNonblockingTaint {
				nodeCopy.Spec.Taints = append(nodeCopy.Spec.Taints, v)
			}
		}
		nodeInfo.SetNode(nodeCopy)
	} else {
		nodeInfo.SetNode(node)
	}

	fit, _, err := predicates.PodToleratesNodeTaints(fakePod, nil, nodeInfo)
	if err != nil {
		Failf("Can't test predicates for node %s: %v", node.Name, err)
		return false
	}
	return fit
}

// TODO: better to change to a easy read name
func isNodeConditionSetAsExpected(node *v1.Node, conditionType v1.NodeConditionType, wantTrue, silent bool) bool {
	// Check the node readiness condition (logging all).
	for _, cond := range node.Status.Conditions {
		// Ensure that the condition type and the status matches as desired.
		if cond.Type == conditionType {
			// For NodeReady condition we need to check Taints as well
			if cond.Type == v1.NodeReady {
				hasNodeControllerTaints := false
				// For NodeReady we need to check if Taints are gone as well
				taints := node.Spec.Taints
				for _, taint := range taints {
					if taint.MatchTaint(nodectlr.UnreachableTaintTemplate) || taint.MatchTaint(nodectlr.NotReadyTaintTemplate) {
						hasNodeControllerTaints = true
						break
					}
				}
				if wantTrue {
					if (cond.Status == v1.ConditionTrue) && !hasNodeControllerTaints {
						return true
					}
					msg := ""
					if !hasNodeControllerTaints {
						msg = fmt.Sprintf("Condition %s of node %s is %v instead of %t. Reason: %v, message: %v",
							conditionType, node.Name, cond.Status == v1.ConditionTrue, wantTrue, cond.Reason, cond.Message)
					} else {
						msg = fmt.Sprintf("Condition %s of node %s is %v, but Node is tainted by NodeController with %v. Failure",
							conditionType, node.Name, cond.Status == v1.ConditionTrue, taints)
					}
					if !silent {
						Logf(msg)
					}
					return false
				}
				// TODO: check if the Node is tainted once we enable NC notReady/unreachable taints by default
				if cond.Status != v1.ConditionTrue {
					return true
				}
				if !silent {
					Logf("Condition %s of node %s is %v instead of %t. Reason: %v, message: %v",
						conditionType, node.Name, cond.Status == v1.ConditionTrue, wantTrue, cond.Reason, cond.Message)
				}
				return false
			}
			if (wantTrue && (cond.Status == v1.ConditionTrue)) || (!wantTrue && (cond.Status != v1.ConditionTrue)) {
				return true
			}
			if !silent {
				Logf("Condition %s of node %s is %v instead of %t. Reason: %v, message: %v",
					conditionType, node.Name, cond.Status == v1.ConditionTrue, wantTrue, cond.Reason, cond.Message)
			}
			return false
		}

	}
	if !silent {
		Logf("Couldn't find condition %v on node %v", conditionType, node.Name)
	}
	return false
}

// IsConditionSetAsExpected returns a wantTrue value if the node has a match to the conditionType, otherwise returns an opposite value of the wantTrue with detailed logging.
func IsConditionSetAsExpected(node *v1.Node, conditionType v1.NodeConditionType, wantTrue bool) bool {
	return isNodeConditionSetAsExpected(node, conditionType, wantTrue, false)
}

// IsConditionSetAsExpectedSilent returns a wantTrue value if the node has a match to the conditionType, otherwise returns an opposite value of the wantTrue.
func IsConditionSetAsExpectedSilent(node *v1.Node, conditionType v1.NodeConditionType, wantTrue bool) bool {
	return isNodeConditionSetAsExpected(node, conditionType, wantTrue, true)
}

// IsConditionUnset returns true if conditions of the given node do not have a match to the given conditionType, otherwise false.
func IsConditionUnset(node *v1.Node, conditionType v1.NodeConditionType) bool {
	for _, cond := range node.Status.Conditions {
		if cond.Type == conditionType {
			return false
		}
	}
	return true
}

// waitListSchedulableNodes is a wrapper around listing nodes supporting retries.
func waitListSchedulableNodes(c clientset.Interface) (*v1.NodeList, error) {
	var nodes *v1.NodeList
	var err error
	if wait.PollImmediate(Poll, SingleCallTimeout, func() (bool, error) {
		nodes, err = c.CoreV1().Nodes().List(metav1.ListOptions{FieldSelector: fields.Set{
			"spec.unschedulable": "false",
		}.AsSelector().String()})
		if err != nil {
			if testutils.IsRetryableAPIError(err) {
				return false, nil
			}
			return false, err
		}
		return true, nil
	}) != nil {
		return nodes, err
	}
	return nodes, nil
}

// Filter filters nodes in NodeList in place, removing nodes that do not
// satisfy the given condition
// TODO: consider merging with pkg/client/cache.NodeLister
func Filter(nodeList *v1.NodeList, fn func(node v1.Node) bool) {
	var l []v1.Node

	for _, node := range nodeList.Items {
		if fn(node) {
			l = append(l, node)
		}
	}
	nodeList.Items = l
}

// CheckReady waits up to timeout for cluster to has desired size and
// there is no not-ready nodes in it. By cluster size we mean number of Nodes
// excluding Master Node.
func CheckReady(c clientset.Interface, size int, timeout time.Duration) ([]v1.Node, error) {
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(sleepTime) {
		nodes, err := waitListSchedulableNodes(c)
		if err != nil {
			Logf("Failed to list nodes: %v", err)
			continue
		}
		numNodes := len(nodes.Items)

		// Filter out not-ready nodes.
		Filter(nodes, func(node v1.Node) bool {
			nodeReady := IsConditionSetAsExpected(&node, v1.NodeReady, true)
			networkReady := IsConditionUnset(&node, v1.NodeNetworkUnavailable) || IsConditionSetAsExpected(&node, v1.NodeNetworkUnavailable, false)
			return nodeReady && networkReady
		})
		numReady := len(nodes.Items)

		if numNodes == size && numReady == size {
			Logf("Cluster has reached the desired number of ready nodes %d", size)
			return nodes.Items, nil
		}
		Logf("Waiting for ready nodes %d, current ready %d, not ready nodes %d", size, numReady, numNodes-numReady)
	}
	return nil, fmt.Errorf("timeout waiting %v for number of ready nodes to be %d", timeout, size)
}

// CheckReadyForTests returns a method usable in polling methods which will check that the nodes are
// in a testable state based on schedulability.
func CheckReadyForTests(c clientset.Interface, nonblockingTaints string, allowedNotReadyNodes, largeClusterThreshold int) func() (bool, error) {
	attempt := 0
	var notSchedulable []*v1.Node
	return func() (bool, error) {
		attempt++
		notSchedulable = nil
		opts := metav1.ListOptions{
			ResourceVersion: "0",
			FieldSelector:   fields.Set{"spec.unschedulable": "false"}.AsSelector().String(),
		}
		nodes, err := c.CoreV1().Nodes().List(opts)
		if err != nil {
			Logf("Unexpected error listing nodes: %v", err)
			if testutils.IsRetryableAPIError(err) {
				return false, nil
			}
			return false, err
		}
		for i := range nodes.Items {
			node := &nodes.Items[i]
			if !readyForTests(node, nonblockingTaints) {
				notSchedulable = append(notSchedulable, node)
			}
		}
		// Framework allows for <TestContext.AllowedNotReadyNodes> nodes to be non-ready,
		// to make it possible e.g. for incorrect deployment of some small percentage
		// of nodes (which we allow in cluster validation). Some nodes that are not
		// provisioned correctly at startup will never become ready (e.g. when something
		// won't install correctly), so we can't expect them to be ready at any point.
		//
		// However, we only allow non-ready nodes with some specific reasons.
		if len(notSchedulable) > 0 {
			// In large clusters, log them only every 10th pass.
			if len(nodes.Items) < largeClusterThreshold || attempt%10 == 0 {
				Logf("Unschedulable nodes:")
				for i := range notSchedulable {
					Logf("-> %s Ready=%t Network=%t Taints=%v NonblockingTaints:%v",
						notSchedulable[i].Name,
						IsConditionSetAsExpectedSilent(notSchedulable[i], v1.NodeReady, true),
						IsConditionSetAsExpectedSilent(notSchedulable[i], v1.NodeNetworkUnavailable, false),
						notSchedulable[i].Spec.Taints,
						nonblockingTaints,
					)

				}
				Logf("================================")
			}
		}
		return len(notSchedulable) <= allowedNotReadyNodes, nil
	}
}

// readyForTests determines whether or not we should continue waiting for the nodes
// to enter a testable state. By default this means it is schedulable, NodeReady, and untainted.
// Nodes with taints nonblocking taints are permitted to have that taint and
// also have their node.Spec.Unschedulable field ignored for the purposes of this function.
func readyForTests(node *v1.Node, nonblockingTaints string) bool {
	if hasNonblockingTaint(node, nonblockingTaints) {
		// If the node has one of the nonblockingTaints taints; just check that it is ready
		// and don't require node.Spec.Unschedulable to be set either way.
		if !isNodeReady(node) || !isNodeUntaintedWithNonblocking(node, nonblockingTaints) {
			return false
		}
	} else {
		if !IsNodeSchedulable(node) || !IsNodeUntainted(node) {
			return false
		}
	}
	return true
}

// hasNonblockingTaint returns true if the node contains at least
// one taint with a key matching the regexp.
func hasNonblockingTaint(node *v1.Node, nonblockingTaints string) bool {
	if node == nil {
		return false
	}

	// Simple lookup for nonblocking taints based on comma-delimited list.
	nonblockingTaintsMap := map[string]struct{}{}
	for _, t := range strings.Split(nonblockingTaints, ",") {
		if strings.TrimSpace(t) != "" {
			nonblockingTaintsMap[strings.TrimSpace(t)] = struct{}{}
		}
	}

	for _, taint := range node.Spec.Taints {
		if _, hasNonblockingTaint := nonblockingTaintsMap[taint.Key]; hasNonblockingTaint {
			return true
		}
	}

	return false
}

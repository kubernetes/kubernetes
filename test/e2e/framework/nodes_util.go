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
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"

	// TODO: Remove the following imports (ref: https://github.com/kubernetes/kubernetes/issues/81245)
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2essh "k8s.io/kubernetes/test/e2e/framework/ssh"
)

const etcdImage = "3.4.3-0"

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
func MasterUpgrade(f *Framework, v string) error {
	switch TestContext.Provider {
	case "gce":
		return masterUpgradeGCE(v, false)
	case "gke":
		return masterUpgradeGKE(f.Namespace.Name, v)
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

func masterUpgradeGKE(namespace string, v string) error {
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

	waitForSSHTunnels(namespace)

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
		err = nodeUpgradeGKE(f.Namespace.Name, v, img)
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
	numNodes, err := e2enode.TotalRegistered(f.ClientSet)
	if err != nil {
		return fmt.Errorf("couldn't detect number of nodes")
	}
	Logf("Waiting up to %v for all %d nodes to be ready after the upgrade", RestartNodeReadyAgainTimeout, numNodes)
	if _, err := e2enode.CheckReady(f.ClientSet, numNodes, RestartNodeReadyAgainTimeout); err != nil {
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

func nodeUpgradeGKE(namespace string, v string, img string) error {
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

		waitForSSHTunnels(namespace)
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
		"--format=get(name)",
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

func waitForSSHTunnels(namespace string) {
	Logf("Waiting for SSH tunnels to establish")
	RunKubectl(namespace, "run", "ssh-tunnel-test",
		"--image=busybox",
		"--restart=Never",
		"--command", "--",
		"echo", "Hello")
	defer RunKubectl(namespace, "delete", "pod", "ssh-tunnel-test")

	// allow up to a minute for new ssh tunnels to establish
	wait.PollImmediate(5*time.Second, time.Minute, func() (bool, error) {
		_, err := RunKubectl(namespace, "logs", "ssh-tunnel-test")
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
	nodes, err := e2enode.GetReadySchedulableNodes(k.client)
	ExpectNoError(err)
	numNodes := int(k.config.FailureRatio * float64(len(nodes.Items)))

	nodes, err = e2enode.GetBoundedReadySchedulableNodes(k.client, numNodes)
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

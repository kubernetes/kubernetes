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

package util

import (
	"fmt"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	"os"
	"path"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"

	e2essh "k8s.io/kubernetes/test/e2e/framework/ssh"
)

const etcdImage = "3.4.7-0"

// EtcdUpgrade upgrades etcd on GCE.
func EtcdUpgrade(targetStorage, targetVersion string) error {
	switch framework.TestContext.Provider {
	case "gce":
		return etcdUpgradeGCE(targetStorage, targetVersion)
	default:
		return fmt.Errorf("EtcdUpgrade() is not implemented for provider %s", framework.TestContext.Provider)
	}
}

// MasterUpgrade upgrades master node on GCE/GKE.
func MasterUpgrade(f *framework.Framework, v string) error {
	switch framework.TestContext.Provider {
	case "gce":
		return masterUpgradeGCE(v, false)
	case "gke":
		return MasterUpgradeGKE(f.Namespace.Name, v)
	default:
		return fmt.Errorf("MasterUpgrade() is not implemented for provider %s", framework.TestContext.Provider)
	}
}

func etcdUpgradeGCE(targetStorage, targetVersion string) error {
	env := append(
		os.Environ(),
		"TEST_ETCD_VERSION="+targetVersion,
		"STORAGE_BACKEND="+targetStorage,
		"TEST_ETCD_IMAGE="+etcdImage)

	_, _, err := framework.RunCmdEnv(env, GCEUpgradeScript(), "-l", "-M")
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
	if framework.TestContext.EtcdUpgradeVersion != "" && framework.TestContext.EtcdUpgradeStorage != "" {
		env = append(env,
			"TEST_ETCD_VERSION="+framework.TestContext.EtcdUpgradeVersion,
			"STORAGE_BACKEND="+framework.TestContext.EtcdUpgradeStorage,
			"TEST_ETCD_IMAGE="+etcdImage)
	} else {
		// In e2e tests, we skip the confirmation prompt about
		// implicit etcd upgrades to simulate the user entering "y".
		env = append(env, "TEST_ALLOW_IMPLICIT_ETCD_UPGRADE=true")
	}

	v := "v" + rawV
	_, _, err := framework.RunCmdEnv(env, GCEUpgradeScript(), "-M", v)
	return err
}

// LocationParamGKE returns parameter related to location for gcloud command.
func LocationParamGKE() string {
	if framework.TestContext.CloudConfig.MultiMaster {
		// GKE Regional Clusters are being tested.
		return fmt.Sprintf("--region=%s", framework.TestContext.CloudConfig.Region)
	}
	return fmt.Sprintf("--zone=%s", framework.TestContext.CloudConfig.Zone)
}

// AppendContainerCommandGroupIfNeeded returns container command group parameter if necessary.
func AppendContainerCommandGroupIfNeeded(args []string) []string {
	if framework.TestContext.CloudConfig.Region != "" {
		// TODO(wojtek-t): Get rid of it once Regional Clusters go to GA.
		return append([]string{"beta"}, args...)
	}
	return args
}

// MasterUpgradeGKE upgrades master node to the specified version on GKE.
func MasterUpgradeGKE(namespace string, v string) error {
	framework.Logf("Upgrading master to %q", v)
	args := []string{
		"container",
		"clusters",
		fmt.Sprintf("--project=%s", framework.TestContext.CloudConfig.ProjectID),
		LocationParamGKE(),
		"upgrade",
		framework.TestContext.CloudConfig.Cluster,
		"--master",
		fmt.Sprintf("--cluster-version=%s", v),
		"--quiet",
	}
	_, _, err := framework.RunCmd("gcloud", AppendContainerCommandGroupIfNeeded(args)...)
	if err != nil {
		return err
	}

	WaitForSSHTunnels(namespace)

	return nil
}

// GCEUpgradeScript returns path of script for upgrading on GCE.
func GCEUpgradeScript() string {
	if len(framework.TestContext.GCEUpgradeScript) == 0 {
		return path.Join(framework.TestContext.RepoRoot, "cluster/gce/upgrade.sh")
	}
	return framework.TestContext.GCEUpgradeScript
}

// WaitForSSHTunnels waits for establishing SSH tunnel to busybox pod.
func WaitForSSHTunnels(namespace string) {
	framework.Logf("Waiting for SSH tunnels to establish")
	framework.RunKubectl(namespace, "run", "ssh-tunnel-test",
		"--image=busybox",
		"--restart=Never",
		"--command", "--",
		"echo", "Hello")
	defer framework.RunKubectl(namespace, "delete", "pod", "ssh-tunnel-test")

	// allow up to a minute for new ssh tunnels to establish
	wait.PollImmediate(5*time.Second, time.Minute, func() (bool, error) {
		_, err := framework.RunKubectl(namespace, "logs", "ssh-tunnel-test")
		return err == nil, nil
	})
}

// NodeKiller is a utility to simulate node failures.
type NodeKiller struct {
	config   framework.NodeKillerConfig
	client   clientset.Interface
	provider string
}

// NewNodeKiller creates new NodeKiller.
func NewNodeKiller(config framework.NodeKillerConfig, client clientset.Interface, provider string) *NodeKiller {
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
	framework.ExpectNoError(err)
	numNodes := int(k.config.FailureRatio * float64(len(nodes.Items)))

	nodes, err = e2enode.GetBoundedReadySchedulableNodes(k.client, numNodes)
	framework.ExpectNoError(err)
	return nodes.Items
}

func (k *NodeKiller) kill(nodes []v1.Node) {
	wg := sync.WaitGroup{}
	wg.Add(len(nodes))
	for _, node := range nodes {
		node := node
		go func() {
			defer wg.Done()

			framework.Logf("Stopping docker and kubelet on %q to simulate failure", node.Name)
			err := e2essh.IssueSSHCommand("sudo systemctl stop docker kubelet", k.provider, &node)
			if err != nil {
				framework.Logf("ERROR while stopping node %q: %v", node.Name, err)
				return
			}

			time.Sleep(k.config.SimulatedDowntime)

			framework.Logf("Rebooting %q to repair the node", node.Name)
			err = e2essh.IssueSSHCommand("sudo reboot", k.provider, &node)
			if err != nil {
				framework.Logf("ERROR while rebooting node %q: %v", node.Name, err)
				return
			}
		}()
	}
	wg.Wait()
}

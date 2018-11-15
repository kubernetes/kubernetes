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
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
)

func EtcdUpgrade(target_storage, target_version string) error {
	switch TestContext.Provider {
	case "gce":
		return etcdUpgradeGCE(target_storage, target_version)
	default:
		return fmt.Errorf("EtcdUpgrade() is not implemented for provider %s", TestContext.Provider)
	}
}

func IngressUpgrade(isUpgrade bool) error {
	switch TestContext.Provider {
	case "gce":
		return ingressUpgradeGCE(isUpgrade)
	default:
		return fmt.Errorf("IngressUpgrade() is not implemented for provider %s", TestContext.Provider)
	}
}

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

func etcdUpgradeGCE(target_storage, target_version string) error {
	env := append(
		os.Environ(),
		"TEST_ETCD_VERSION="+target_version,
		"STORAGE_BACKEND="+target_storage,
		"TEST_ETCD_IMAGE=3.2.24-1")

	_, _, err := RunCmdEnv(env, gceUpgradeScript(), "-l", "-M")
	return err
}

func ingressUpgradeGCE(isUpgrade bool) error {
	var command string
	if isUpgrade {
		// User specified image to upgrade to.
		targetImage := TestContext.IngressUpgradeImage
		if targetImage != "" {
			command = fmt.Sprintf("sudo sed -i -re 's|(image:)(.*)|\\1 %s|' /etc/kubernetes/manifests/glbc.manifest", targetImage)
		} else {
			// Upgrade to latest HEAD image.
			command = "sudo sed -i -re 's/(image:)(.*)/\\1 gcr.io\\/k8s-ingress-image-push\\/ingress-gce-e2e-glbc-amd64:master/' /etc/kubernetes/manifests/glbc.manifest"
		}
	} else {
		// Downgrade to latest release image.
		command = "sudo sed -i -re 's/(image:)(.*)/\\1 k8s.gcr.io\\/ingress-gce-glbc-amd64:v1.1.1/' /etc/kubernetes/manifests/glbc.manifest"
	}
	// Kubelet should restart glbc automatically.
	sshResult, err := NodeExec(GetMasterHost(), command)
	LogSSHResult(sshResult)
	return err
}

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
			"TEST_ETCD_IMAGE=3.2.24-1")
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
	numNodes, err := NumberOfRegisteredNodes(f.ClientSet)
	if err != nil {
		return fmt.Errorf("couldn't detect number of nodes")
	}
	Logf("Waiting up to %v for all %d nodes to be ready after the upgrade", RestartNodeReadyAgainTimeout, numNodes)
	if _, err := CheckNodesReady(f.ClientSet, numNodes, RestartNodeReadyAgainTimeout); err != nil {
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
	args := []string{
		"container",
		"clusters",
		fmt.Sprintf("--project=%s", TestContext.CloudConfig.ProjectID),
		locationParamGKE(),
		"upgrade",
		TestContext.CloudConfig.Cluster,
		fmt.Sprintf("--cluster-version=%s", v),
		"--quiet",
	}
	if len(img) > 0 {
		args = append(args, fmt.Sprintf("--image-type=%s", img))
	}
	_, _, err := RunCmd("gcloud", appendContainerCommandGroupIfNeeded(args)...)

	if err != nil {
		return err
	}

	waitForSSHTunnels()

	return nil
}

// MigTemplate (GCE-only) returns the name of the MIG template that the
// nodes of the cluster use.
func MigTemplate() (string, error) {
	var errLast error
	var templ string
	key := "instanceTemplate"
	if wait.Poll(Poll, SingleCallTimeout, func() (bool, error) {
		// TODO(mikedanese): make this hit the compute API directly instead of
		// shelling out to gcloud.
		// An `instance-groups managed describe` call outputs what we want to stdout.
		output, _, err := retryCmd("gcloud", "compute", "instance-groups", "managed",
			fmt.Sprintf("--project=%s", TestContext.CloudConfig.ProjectID),
			"describe",
			fmt.Sprintf("--zone=%s", TestContext.CloudConfig.Zone),
			TestContext.CloudConfig.NodeInstanceGroup)
		if err != nil {
			errLast = fmt.Errorf("gcloud compute instance-groups managed describe call failed with err: %v", err)
			return false, nil
		}

		// The 'describe' call probably succeeded; parse the output and try to
		// find the line that looks like "instanceTemplate: url/to/<templ>" and
		// return <templ>.
		if val := ParseKVLines(output, key); len(val) > 0 {
			url := strings.Split(val, "/")
			templ = url[len(url)-1]
			Logf("MIG group %s using template: %s", TestContext.CloudConfig.NodeInstanceGroup, templ)
			return true, nil
		}
		errLast = fmt.Errorf("couldn't find %s in output to get MIG template. Output: %s", key, output)
		return false, nil
	}) != nil {
		return "", fmt.Errorf("MigTemplate() failed with last error: %v", errLast)
	}
	return templ, nil
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

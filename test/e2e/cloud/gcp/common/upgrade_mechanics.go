/*
Copyright 2021 The Kubernetes Authors.

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

package common

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/version"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2eproviders "k8s.io/kubernetes/test/e2e/framework/providers"
	"k8s.io/kubernetes/test/e2e/upgrades"
	"k8s.io/kubernetes/test/utils/junit"
)

// ControlPlaneUpgradeFunc returns a function that performs control plane upgrade.
func ControlPlaneUpgradeFunc(f *framework.Framework, upgCtx *upgrades.UpgradeContext, testCase *junit.TestCase, controlPlaneExtraEnvs []string) func(ctx context.Context) {
	return func(ctx context.Context) {
		target := upgCtx.Versions[1].Version.String()
		framework.ExpectNoError(controlPlaneUpgrade(ctx, f, target, controlPlaneExtraEnvs))
		framework.ExpectNoError(checkControlPlaneVersion(ctx, f.ClientSet, target))
	}
}

// ClusterUpgradeFunc returns a function that performs full cluster upgrade (both control plane and nodes).
func ClusterUpgradeFunc(f *framework.Framework, upgCtx *upgrades.UpgradeContext, testCase *junit.TestCase, controlPlaneExtraEnvs, nodeExtraEnvs []string) func(ctx context.Context) {
	return func(ctx context.Context) {
		target := upgCtx.Versions[1].Version.String()
		image := upgCtx.Versions[1].NodeImage
		framework.ExpectNoError(controlPlaneUpgrade(ctx, f, target, controlPlaneExtraEnvs))
		framework.ExpectNoError(checkControlPlaneVersion(ctx, f.ClientSet, target))
		framework.ExpectNoError(nodeUpgrade(ctx, f, target, image, nodeExtraEnvs))
		framework.ExpectNoError(checkNodesVersions(ctx, f.ClientSet, target))
	}
}

// ClusterDowngradeFunc returns a function that performs full cluster downgrade (both nodes and control plane).
func ClusterDowngradeFunc(f *framework.Framework, upgCtx *upgrades.UpgradeContext, testCase *junit.TestCase, controlPlaneExtraEnvs, nodeExtraEnvs []string) func(ctx context.Context) {
	return func(ctx context.Context) {
		target := upgCtx.Versions[1].Version.String()
		image := upgCtx.Versions[1].NodeImage
		// Yes this really is a downgrade. And nodes must downgrade first.
		framework.ExpectNoError(nodeUpgrade(ctx, f, target, image, nodeExtraEnvs))
		framework.ExpectNoError(checkNodesVersions(ctx, f.ClientSet, target))
		framework.ExpectNoError(controlPlaneUpgrade(ctx, f, target, controlPlaneExtraEnvs))
		framework.ExpectNoError(checkControlPlaneVersion(ctx, f.ClientSet, target))
	}
}

const etcdImage = "3.4.9-1"

// controlPlaneUpgrade upgrades control plane node on GCE/GKE.
func controlPlaneUpgrade(ctx context.Context, f *framework.Framework, v string, extraEnvs []string) error {
	switch framework.TestContext.Provider {
	case "gce":
		return controlPlaneUpgradeGCE(v, extraEnvs)
	default:
		return fmt.Errorf("controlPlaneUpgrade() is not implemented for provider %s", framework.TestContext.Provider)
	}
}

func controlPlaneUpgradeGCE(rawV string, extraEnvs []string) error {
	env := append(os.Environ(), extraEnvs...)
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
	_, _, err := framework.RunCmdEnv(env, e2eproviders.GCEUpgradeScript(), "-M", v)
	return err
}

func traceRouteToControlPlane() {
	traceroute, err := exec.LookPath("traceroute")
	if err != nil {
		framework.Logf("Could not find traceroute program")
		return
	}
	cmd := exec.Command(traceroute, "-I", framework.APIAddress())
	out, err := cmd.Output()
	if len(out) != 0 {
		framework.Logf("%s", string(out))
	}
	if exiterr, ok := err.(*exec.ExitError); err != nil && ok {
		framework.Logf("Error while running traceroute: %s", exiterr.Stderr)
	}
}

// checkControlPlaneVersion validates the control plane version
func checkControlPlaneVersion(ctx context.Context, c clientset.Interface, want string) error {
	framework.Logf("Checking control plane version")
	var err error
	var v *version.Info
	waitErr := wait.PollUntilContextTimeout(ctx, 5*time.Second, 2*time.Minute, true, func(ctx context.Context) (bool, error) {
		v, err = c.Discovery().ServerVersion()
		if err != nil {
			traceRouteToControlPlane()
			return false, nil
		}
		return true, nil
	})
	if waitErr != nil {
		return fmt.Errorf("CheckControlPlane() couldn't get the control plane version: %w", err)
	}
	// We do prefix trimming and then matching because:
	// want looks like:  0.19.3-815-g50e67d4
	// got  looks like: v0.19.3-815-g50e67d4034e858-dirty
	got := strings.TrimPrefix(v.GitVersion, "v")
	if !strings.HasPrefix(got, want) {
		return fmt.Errorf("control plane had kube-apiserver version %s which does not start with %s", got, want)
	}
	framework.Logf("Control plane is at version %s", want)
	return nil
}

// nodeUpgrade upgrades nodes on GCE/GKE.
func nodeUpgrade(ctx context.Context, f *framework.Framework, v string, img string, extraEnvs []string) error {
	// Perform the upgrade.
	var err error
	switch framework.TestContext.Provider {
	case "gce":
		err = nodeUpgradeGCE(v, img, extraEnvs)
	default:
		err = fmt.Errorf("nodeUpgrade() is not implemented for provider %s", framework.TestContext.Provider)
	}
	if err != nil {
		return err
	}
	return waitForNodesReadyAfterUpgrade(ctx, f)
}

// TODO(mrhohn): Remove 'enableKubeProxyDaemonSet' when kube-proxy is run as a DaemonSet by default.
func nodeUpgradeGCE(rawV, img string, extraEnvs []string) error {
	v := "v" + rawV
	env := append(os.Environ(), extraEnvs...)
	if img != "" {
		env = append(env, "KUBE_NODE_OS_DISTRIBUTION="+img)
		_, _, err := framework.RunCmdEnv(env, e2eproviders.GCEUpgradeScript(), "-N", "-o", v)
		return err
	}
	_, _, err := framework.RunCmdEnv(env, e2eproviders.GCEUpgradeScript(), "-N", v)
	return err
}

func waitForNodesReadyAfterUpgrade(ctx context.Context, f *framework.Framework) error {
	// Wait for it to complete and validate nodes are healthy.
	//
	// TODO(ihmccreery) We shouldn't have to wait for nodes to be ready in
	// GKE; the operation shouldn't return until they all are.
	numNodes, err := e2enode.TotalRegistered(ctx, f.ClientSet)
	if err != nil {
		return fmt.Errorf("couldn't detect number of nodes")
	}
	framework.Logf("Waiting up to %v for all %d nodes to be ready after the upgrade", framework.RestartNodeReadyAgainTimeout, numNodes)
	if _, err := e2enode.CheckReady(ctx, f.ClientSet, numNodes, framework.RestartNodeReadyAgainTimeout); err != nil {
		return err
	}
	return nil
}

// checkNodesVersions validates the nodes versions
func checkNodesVersions(ctx context.Context, cs clientset.Interface, want string) error {
	l, err := e2enode.GetReadySchedulableNodes(ctx, cs)
	if err != nil {
		return err
	}
	for _, n := range l.Items {
		// We do prefix trimming and then matching because:
		// want looks like:  0.19.3-815-g50e67d4
		// kv 	look  like: v0.19.3-815-g50e67d4034e858-dirty
		// kpv 	look  like: v0.19.3-815-g50e67d4034e858-dirty or empty value
		kv, kpv := strings.TrimPrefix(n.Status.NodeInfo.KubeletVersion, "v"),
			strings.TrimPrefix(n.Status.NodeInfo.KubeProxyVersion, "v") //nolint:staticcheck // Keep testing deprecated KubeProxyVersion field until it's being removed
		if !strings.HasPrefix(kv, want) {
			return fmt.Errorf("node %s had kubelet version %s which does not start with %s",
				n.ObjectMeta.Name, kv, want)
		}

		if len(kpv) != 0 || !strings.HasPrefix(kpv, want) {
			return fmt.Errorf("node %s had kube-proxy version %s which does not start with %s or is not empty value",
				n.ObjectMeta.Name, kpv, want)
		}
	}
	return nil
}

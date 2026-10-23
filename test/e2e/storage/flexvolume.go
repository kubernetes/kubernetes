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

package storage

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"path"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2essh "k8s.io/kubernetes/test/e2e/framework/ssh"
	e2etestfiles "k8s.io/kubernetes/test/e2e/framework/testfiles"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

const (
	driverDir = "test/e2e/testing-manifests/flexvolume/"
	// kube-controller-manager's default for --flex-volume-plugin-dir.
	defaultVolumePluginDir  = "/usr/libexec/kubernetes/kubelet-plugins/volume/exec"
	flexVolumePluginDirFlag = "--flex-volume-plugin-dir="
	detachTimeout           = 10 * time.Second
)

// testFlexVolume tests that a client pod using a given flexvolume driver
// successfully mounts it and runs
func testFlexVolume(ctx context.Context, driver string, config e2evolume.TestConfig, f *framework.Framework) {
	tests := []e2evolume.Test{
		{
			Volume: v1.VolumeSource{
				FlexVolume: &v1.FlexVolumeSource{
					Driver: "k8s/" + driver,
				},
			},
			File: "index.html",
			// Must match content of examples/volumes/flexvolume/dummy(-attachable) domount
			ExpectedContent: "Hello from flexvolume!",
		},
	}
	e2evolume.TestVolumeClient(ctx, f, config, nil, "" /* fsType */, tests)
}

// installFlex installs the driver found at filePath on the node, and restarts
// kubelet if 'restart' is true. If node is nil, installs on the master, and restarts
// controller-manager if 'restart' is true.
func installFlex(ctx context.Context, c clientset.Interface, node *v1.Node, vendor, driver, filePath string) {
	flexDir := getFlexDir(ctx, c, node, vendor, driver)
	flexFile := path.Join(flexDir, driver)

	host := ""
	var err error
	if node != nil {
		host, err = e2enode.GetSSHExternalIP(node)
		if err != nil {
			host, err = e2enode.GetSSHInternalIP(node)
		}
	} else {
		instanceWithPort := framework.APIAddress()
		hostName := getHostFromHostPort(instanceWithPort)
		host = net.JoinHostPort(hostName, e2essh.SSHPort)
	}

	framework.ExpectNoError(err)

	cmd := fmt.Sprintf("sudo mkdir -p %s", flexDir)
	sshAndLog(ctx, cmd, host, true /*failOnError*/)

	data, err := e2etestfiles.Read(filePath)
	if err != nil {
		framework.Fail(err.Error())
	}
	cmd = fmt.Sprintf("sudo tee <<'EOF' %s\n%s\nEOF", flexFile, string(data))
	sshAndLog(ctx, cmd, host, true /*failOnError*/)

	cmd = fmt.Sprintf("sudo chmod +x %s", flexFile)
	sshAndLog(ctx, cmd, host, true /*failOnError*/)
}

func uninstallFlex(ctx context.Context, c clientset.Interface, node *v1.Node, vendor, driver string) {
	flexDir := getFlexDir(ctx, c, node, vendor, driver)

	host := ""
	var err error
	if node != nil {
		host, err = e2enode.GetSSHExternalIP(node)
		if err != nil {
			host, err = e2enode.GetSSHInternalIP(node)
		}
	} else {
		instanceWithPort := framework.APIAddress()
		hostName := getHostFromHostPort(instanceWithPort)
		host = net.JoinHostPort(hostName, e2essh.SSHPort)
	}

	if host == "" {
		framework.Failf("Error getting node ip : %v", err)
	}

	cmd := fmt.Sprintf("sudo rm -r %s", flexDir)
	sshAndLog(ctx, cmd, host, false /*failOnError*/)
}

// getFlexDir returns the flexvolume install dir used by the node's kubelet, or by
// kube-controller-manager if node is nil. Skips the test if it cannot be determined.
func getFlexDir(ctx context.Context, c clientset.Interface, node *v1.Node, vendor, driver string) string {
	var volumePluginDir string
	var err error
	if node != nil {
		volumePluginDir, err = kubeletVolumePluginDir(ctx, c, node.Name)
	} else {
		volumePluginDir, err = controllerManagerVolumePluginDir(ctx, c)
	}
	if err != nil {
		e2eskipper.Skipf("Unable to determine flexvolume plugin dir: %v", err)
	}
	return path.Join(volumePluginDir, fmt.Sprintf("/%s~%s/", vendor, driver))
}

func kubeletVolumePluginDir(ctx context.Context, c clientset.Interface, nodeName string) (string, error) {
	raw, err := c.CoreV1().RESTClient().Get().Resource("nodes").Name(nodeName).SubResource("proxy").Suffix("configz").DoRaw(ctx)
	if err != nil {
		return "", fmt.Errorf("reading kubelet configz from node %s: %w", nodeName, err)
	}
	var configz struct {
		KubeletConfig struct {
			VolumePluginDir string `json:"volumePluginDir"`
		} `json:"kubeletconfig"`
	}
	if err := json.Unmarshal(raw, &configz); err != nil {
		return "", fmt.Errorf("decoding kubelet configz from node %s: %w", nodeName, err)
	}
	if configz.KubeletConfig.VolumePluginDir == "" {
		return "", fmt.Errorf("kubelet configz from node %s has no volumePluginDir", nodeName)
	}
	return configz.KubeletConfig.VolumePluginDir, nil
}

// controllerManagerVolumePluginDir reads --flex-volume-plugin-dir from the kube-controller-manager pod spec.
func controllerManagerVolumePluginDir(ctx context.Context, c clientset.Interface) (string, error) {
	pods, err := c.CoreV1().Pods(metav1.NamespaceSystem).List(ctx, metav1.ListOptions{})
	if err != nil {
		return "", fmt.Errorf("listing %s pods: %w", metav1.NamespaceSystem, err)
	}
	for _, pod := range pods.Items {
		if !strings.HasPrefix(pod.Name, "kube-controller-manager") {
			continue
		}
		for _, container := range pod.Spec.Containers {
			// Flags may be embedded in a single shell command string, so scan each field by token.
			for _, arg := range append(container.Command, container.Args...) {
				for token := range strings.FieldsSeq(arg) {
					if dir, ok := strings.CutPrefix(token, flexVolumePluginDirFlag); ok {
						return strings.Trim(dir, `"'`), nil
					}
				}
			}
		}
		return defaultVolumePluginDir, nil
	}
	return "", fmt.Errorf("no kube-controller-manager pod found in %s", metav1.NamespaceSystem)
}

func sshAndLog(ctx context.Context, cmd, host string, failOnError bool) {
	result, err := e2essh.SSH(ctx, cmd, host, framework.TestContext.Provider)
	e2essh.LogResult(result)
	framework.ExpectNoError(err)
	if result.Code != 0 && failOnError {
		framework.Failf("%s returned non-zero, stderr: %s", cmd, result.Stderr)
	}
}

func getHostFromHostPort(hostPort string) string {
	// try to split host and port
	var host string
	var err error
	if host, _, err = net.SplitHostPort(hostPort); err != nil {
		// if SplitHostPort returns an error, the entire hostport is considered as host
		host = hostPort
	}
	return host
}

var _ = utils.SIGDescribe("Flexvolumes", func() {
	f := framework.NewDefaultFramework("flexvolume")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	// note that namespace deletion is handled by delete-namespace flag

	var cs clientset.Interface
	var ns *v1.Namespace
	var node *v1.Node
	var config e2evolume.TestConfig
	var suffix string

	ginkgo.BeforeEach(func(ctx context.Context) {
		e2eskipper.SkipUnlessMasterOSDistroIs("debian", "ubuntu", "gci", "custom")
		e2eskipper.SkipUnlessNodeOSDistroIs("debian", "ubuntu", "gci", "custom")
		e2eskipper.SkipUnlessSSHKeyPresent()

		cs = f.ClientSet
		ns = f.Namespace
		var err error
		node, err = e2enode.GetRandomReadySchedulableNode(ctx, f.ClientSet)
		framework.ExpectNoError(err)
		config = e2evolume.TestConfig{
			Namespace:           ns.Name,
			Prefix:              "flex",
			ClientNodeSelection: e2epod.NodeSelection{Name: node.Name},
		}
		suffix = ns.Name
	})

	ginkgo.It("should be mountable when non-attachable", func(ctx context.Context) {
		driver := "dummy"
		driverInstallAs := driver + "-" + suffix

		ginkgo.By(fmt.Sprintf("installing flexvolume %s on node %s as %s", path.Join(driverDir, driver), node.Name, driverInstallAs))
		installFlex(ctx, cs, node, "k8s", driverInstallAs, path.Join(driverDir, driver))

		testFlexVolume(ctx, driverInstallAs, config, f)

		ginkgo.By(fmt.Sprintf("uninstalling flexvolume %s from node %s", driverInstallAs, node.Name))
		uninstallFlex(ctx, cs, node, "k8s", driverInstallAs)
	})

	f.It("should be mountable when attachable", feature.Flexvolumes, func(ctx context.Context) {
		driver := "dummy-attachable"
		driverInstallAs := driver + "-" + suffix

		ginkgo.By(fmt.Sprintf("installing flexvolume %s on node %s as %s", path.Join(driverDir, driver), node.Name, driverInstallAs))
		installFlex(ctx, cs, node, "k8s", driverInstallAs, path.Join(driverDir, driver))
		ginkgo.By(fmt.Sprintf("installing flexvolume %s on master as %s", path.Join(driverDir, driver), driverInstallAs))
		installFlex(ctx, cs, nil, "k8s", driverInstallAs, path.Join(driverDir, driver))

		testFlexVolume(ctx, driverInstallAs, config, f)

		// Detach might occur after pod deletion. Wait before deleting driver.
		time.Sleep(detachTimeout)

		ginkgo.By(fmt.Sprintf("uninstalling flexvolume %s from node %s", driverInstallAs, node.Name))
		uninstallFlex(ctx, cs, node, "k8s", driverInstallAs)
		ginkgo.By(fmt.Sprintf("uninstalling flexvolume %s from master", driverInstallAs))
		uninstallFlex(ctx, cs, nil, "k8s", driverInstallAs)
	})
})

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
	"fmt"
	"math/rand"
	"net"
	"path"

	. "github.com/onsi/ginkgo"
	"k8s.io/api/core/v1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/generated"
)

const (
	sshPort                = "22"
	driverDir              = "test/e2e/testing-manifests/flexvolume/"
	defaultVolumePluginDir = "/usr/libexec/kubernetes/kubelet-plugins/volume/exec"
	// TODO: change this and config-test.sh when default flex volume install path is changed for GCI
	// On gci, root is read-only and controller-manager containerized. Assume
	// controller-manager has started with --flex-volume-plugin-dir equal to this
	// (see cluster/gce/config-test.sh)
	gciVolumePluginDir = "/etc/srv/kubernetes/kubelet-plugins/volume/exec"
)

// testFlexVolume tests that a client pod using a given flexvolume driver
// successfully mounts it and runs
func testFlexVolume(driver string, cs clientset.Interface, config framework.VolumeTestConfig, f *framework.Framework, clean bool) {
	tests := []framework.VolumeTest{
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
	framework.TestVolumeClient(cs, config, nil, tests)

	if clean {
		framework.VolumeTestCleanup(f, config)
	}
}

// installFlex installs the driver found at filePath on the node and restarts
// kubelet. If node is nil, installs on the master and restarts
// controller-manager.
func installFlex(node *v1.Node, vendor, driver, filePath string) {
	flexDir := getFlexDir(node == nil, vendor, driver)
	flexFile := path.Join(flexDir, driver)

	host := ""
	if node != nil {
		host = framework.GetNodeExternalIP(node)
	} else {
		host = net.JoinHostPort(framework.GetMasterHost(), sshPort)
	}

	cmd := fmt.Sprintf("sudo mkdir -p %s", flexDir)
	sshAndLog(cmd, host)

	data := generated.ReadOrDie(filePath)
	cmd = fmt.Sprintf("sudo tee <<'EOF' %s\n%s\nEOF", flexFile, string(data))
	sshAndLog(cmd, host)

	cmd = fmt.Sprintf("sudo chmod +x %s", flexFile)
	sshAndLog(cmd, host)

	if node != nil {
		err := framework.RestartKubelet(host)
		framework.ExpectNoError(err)
		err = framework.WaitForKubeletUp(host)
		framework.ExpectNoError(err)
	} else {
		err := framework.RestartControllerManager()
		framework.ExpectNoError(err)
		err = framework.WaitForControllerManagerUp()
		framework.ExpectNoError(err)
	}
}

func uninstallFlex(node *v1.Node, vendor, driver string) {
	flexDir := getFlexDir(node == nil, vendor, driver)

	host := ""
	if node != nil {
		host = framework.GetNodeExternalIP(node)
	} else {
		host = net.JoinHostPort(framework.GetMasterHost(), sshPort)
	}

	cmd := fmt.Sprintf("sudo rm -r %s", flexDir)
	sshAndLog(cmd, host)
}

func getFlexDir(master bool, vendor, driver string) string {
	volumePluginDir := defaultVolumePluginDir
	if framework.ProviderIs("gce") {
		if (master && framework.MasterOSDistroIs("gci")) || (!master && framework.NodeOSDistroIs("gci")) {
			volumePluginDir = gciVolumePluginDir
		}
	}
	flexDir := path.Join(volumePluginDir, fmt.Sprintf("/%s~%s/", vendor, driver))
	return flexDir
}

func sshAndLog(cmd, host string) {
	result, err := framework.SSH(cmd, host, framework.TestContext.Provider)
	framework.LogSSHResult(result)
	framework.ExpectNoError(err)
	if result.Code != 0 {
		framework.Failf("%s returned non-zero, stderr: %s", cmd, result.Stderr)
	}
}

var _ = SIGDescribe("Flexvolumes [Disruptive] [Feature:FlexVolume]", func() {
	f := framework.NewDefaultFramework("flexvolume")

	// If 'false', the test won't clear its volumes upon completion. Useful for debugging,
	// note that namespace deletion is handled by delete-namespace flag
	clean := true

	var cs clientset.Interface
	var ns *v1.Namespace
	var node v1.Node
	var config framework.VolumeTestConfig
	var suffix string

	BeforeEach(func() {
		framework.SkipUnlessProviderIs("gce")
		framework.SkipUnlessMasterOSDistroIs("gci")
		framework.SkipUnlessNodeOSDistroIs("debian", "gci")
		framework.SkipUnlessSSHKeyPresent()

		cs = f.ClientSet
		ns = f.Namespace
		nodes := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
		node = nodes.Items[rand.Intn(len(nodes.Items))]
		config = framework.VolumeTestConfig{
			Namespace:      ns.Name,
			Prefix:         "flex",
			ClientNodeName: node.Name,
		}
		suffix = ns.Name
	})

	It("should be mountable when non-attachable", func() {
		driver := "dummy"
		driverInstallAs := driver + "-" + suffix

		By(fmt.Sprintf("installing flexvolume %s on node %s as %s", path.Join(driverDir, driver), node.Name, driverInstallAs))
		installFlex(&node, "k8s", driverInstallAs, path.Join(driverDir, driver))

		testFlexVolume(driverInstallAs, cs, config, f, clean)

		By("waiting for flex client pod to terminate")
		if err := f.WaitForPodTerminated(config.Prefix+"-client", ""); !apierrs.IsNotFound(err) {
			framework.ExpectNoError(err, "Failed to wait client pod terminated: %v", err)
		}

		By(fmt.Sprintf("uninstalling flexvolume %s from node %s", driverInstallAs, node.Name))
		uninstallFlex(&node, "k8s", driverInstallAs)
	})

	It("should be mountable when attachable", func() {
		driver := "dummy-attachable"
		driverInstallAs := driver + "-" + suffix

		By(fmt.Sprintf("installing flexvolume %s on node %s as %s", path.Join(driverDir, driver), node.Name, driverInstallAs))
		installFlex(&node, "k8s", driverInstallAs, path.Join(driverDir, driver))
		By(fmt.Sprintf("installing flexvolume %s on master as %s", path.Join(driverDir, driver), driverInstallAs))
		installFlex(nil, "k8s", driverInstallAs, path.Join(driverDir, driver))

		testFlexVolume(driverInstallAs, cs, config, f, clean)

		By("waiting for flex client pod to terminate")
		if err := f.WaitForPodTerminated(config.Prefix+"-client", ""); !apierrs.IsNotFound(err) {
			framework.ExpectNoError(err, "Failed to wait client pod terminated: %v", err)
		}

		By(fmt.Sprintf("uninstalling flexvolume %s from node %s", driverInstallAs, node.Name))
		uninstallFlex(&node, "k8s", driverInstallAs)
		By(fmt.Sprintf("uninstalling flexvolume %s from master", driverInstallAs))
		uninstallFlex(nil, "k8s", driverInstallAs)
	})
})

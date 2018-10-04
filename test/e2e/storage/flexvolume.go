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

	"time"

	. "github.com/onsi/ginkgo"
	"k8s.io/api/core/v1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	versionutil "k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/version"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/framework/testfiles"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

const (
	sshPort                = "22"
	driverDir              = "test/e2e/testing-manifests/flexvolume/"
	defaultVolumePluginDir = "/usr/libexec/kubernetes/kubelet-plugins/volume/exec"
	// TODO: change this and config-test.sh when default flex volume install path is changed for GCI
	// On gci, root is read-only and controller-manager containerized. Assume
	// controller-manager has started with --flex-volume-plugin-dir equal to this
	// (see cluster/gce/config-test.sh)
	gciVolumePluginDir        = "/home/kubernetes/flexvolume"
	gciVolumePluginDirLegacy  = "/etc/srv/kubernetes/kubelet-plugins/volume/exec"
	gciVolumePluginDirVersion = "1.10.0"
	detachTimeout             = 10 * time.Second
)

// testFlexVolume tests that a client pod using a given flexvolume driver
// successfully mounts it and runs
func testFlexVolume(driver string, cs clientset.Interface, config framework.VolumeTestConfig, f *framework.Framework) {
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

	framework.VolumeTestCleanup(f, config)
}

// installFlex installs the driver found at filePath on the node, and restarts
// kubelet if 'restart' is true. If node is nil, installs on the master, and restarts
// controller-manager if 'restart' is true.
func installFlex(c clientset.Interface, node *v1.Node, vendor, driver, filePath string) {
	flexDir := getFlexDir(c, node, vendor, driver)
	flexFile := path.Join(flexDir, driver)

	host := ""
	var err error
	if node != nil {
		host, err = framework.GetNodeExternalIP(node)
		if err != nil {
			host, err = framework.GetNodeInternalIP(node)
		}
	} else {
		masterHostWithPort := framework.GetMasterHost()
		hostName := getHostFromHostPort(masterHostWithPort)
		host = net.JoinHostPort(hostName, sshPort)
	}

	framework.ExpectNoError(err)

	cmd := fmt.Sprintf("sudo mkdir -p %s", flexDir)
	sshAndLog(cmd, host, true /*failOnError*/)

	data := testfiles.ReadOrDie(filePath, Fail)
	cmd = fmt.Sprintf("sudo tee <<'EOF' %s\n%s\nEOF", flexFile, string(data))
	sshAndLog(cmd, host, true /*failOnError*/)

	cmd = fmt.Sprintf("sudo chmod +x %s", flexFile)
	sshAndLog(cmd, host, true /*failOnError*/)
}

func uninstallFlex(c clientset.Interface, node *v1.Node, vendor, driver string) {
	flexDir := getFlexDir(c, node, vendor, driver)

	host := ""
	var err error
	if node != nil {
		host, err = framework.GetNodeExternalIP(node)
		if err != nil {
			host, err = framework.GetNodeInternalIP(node)
		}
	} else {
		masterHostWithPort := framework.GetMasterHost()
		hostName := getHostFromHostPort(masterHostWithPort)
		host = net.JoinHostPort(hostName, sshPort)
	}

	if host == "" {
		framework.Failf("Error getting node ip : %v", err)
	}

	cmd := fmt.Sprintf("sudo rm -r %s", flexDir)
	sshAndLog(cmd, host, false /*failOnError*/)
}

func getFlexDir(c clientset.Interface, node *v1.Node, vendor, driver string) string {
	volumePluginDir := defaultVolumePluginDir
	if framework.ProviderIs("gce") {
		if node == nil && framework.MasterOSDistroIs("gci", "ubuntu") {
			v, err := getMasterVersion(c)
			if err != nil {
				framework.Failf("Error getting master version: %v", err)
			}

			if v.AtLeast(versionutil.MustParseGeneric(gciVolumePluginDirVersion)) {
				volumePluginDir = gciVolumePluginDir
			} else {
				volumePluginDir = gciVolumePluginDirLegacy
			}
		} else if node != nil && framework.NodeOSDistroIs("gci", "ubuntu") {
			if getNodeVersion(node).AtLeast(versionutil.MustParseGeneric(gciVolumePluginDirVersion)) {
				volumePluginDir = gciVolumePluginDir
			} else {
				volumePluginDir = gciVolumePluginDirLegacy
			}
		}
	}
	flexDir := path.Join(volumePluginDir, fmt.Sprintf("/%s~%s/", vendor, driver))
	return flexDir
}

func sshAndLog(cmd, host string, failOnError bool) {
	result, err := framework.SSH(cmd, host, framework.TestContext.Provider)
	framework.LogSSHResult(result)
	framework.ExpectNoError(err)
	if result.Code != 0 && failOnError {
		framework.Failf("%s returned non-zero, stderr: %s", cmd, result.Stderr)
	}
}

func getMasterVersion(c clientset.Interface) (*versionutil.Version, error) {
	var err error
	var v *version.Info
	waitErr := wait.PollImmediate(5*time.Second, 2*time.Minute, func() (bool, error) {
		v, err = c.Discovery().ServerVersion()
		return err == nil, nil
	})
	if waitErr != nil {
		return nil, fmt.Errorf("Could not get the master version: %v", waitErr)
	}

	return versionutil.MustParseSemantic(v.GitVersion), nil
}

func getNodeVersion(node *v1.Node) *versionutil.Version {
	return versionutil.MustParseSemantic(node.Status.NodeInfo.KubeletVersion)
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

	// note that namespace deletion is handled by delete-namespace flag

	var cs clientset.Interface
	var ns *v1.Namespace
	var node v1.Node
	var config framework.VolumeTestConfig
	var suffix string

	BeforeEach(func() {
		framework.SkipUnlessProviderIs("gce", "local")
		framework.SkipUnlessMasterOSDistroIs("debian", "ubuntu", "gci", "custom")
		framework.SkipUnlessNodeOSDistroIs("debian", "ubuntu", "gci", "custom")
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
		installFlex(cs, &node, "k8s", driverInstallAs, path.Join(driverDir, driver))

		testFlexVolume(driverInstallAs, cs, config, f)

		By("waiting for flex client pod to terminate")
		if err := f.WaitForPodTerminated(config.Prefix+"-client", ""); !apierrs.IsNotFound(err) {
			framework.ExpectNoError(err, "Failed to wait client pod terminated: %v", err)
		}

		By(fmt.Sprintf("uninstalling flexvolume %s from node %s", driverInstallAs, node.Name))
		uninstallFlex(cs, &node, "k8s", driverInstallAs)
	})

	It("should be mountable when attachable", func() {
		driver := "dummy-attachable"
		driverInstallAs := driver + "-" + suffix

		By(fmt.Sprintf("installing flexvolume %s on node %s as %s", path.Join(driverDir, driver), node.Name, driverInstallAs))
		installFlex(cs, &node, "k8s", driverInstallAs, path.Join(driverDir, driver))
		By(fmt.Sprintf("installing flexvolume %s on master as %s", path.Join(driverDir, driver), driverInstallAs))
		installFlex(cs, nil, "k8s", driverInstallAs, path.Join(driverDir, driver))

		testFlexVolume(driverInstallAs, cs, config, f)

		By("waiting for flex client pod to terminate")
		if err := f.WaitForPodTerminated(config.Prefix+"-client", ""); !apierrs.IsNotFound(err) {
			framework.ExpectNoError(err, "Failed to wait client pod terminated: %v", err)
		}

		// Detach might occur after pod deletion. Wait before deleting driver.
		time.Sleep(detachTimeout)

		By(fmt.Sprintf("uninstalling flexvolume %s from node %s", driverInstallAs, node.Name))
		uninstallFlex(cs, &node, "k8s", driverInstallAs)
		By(fmt.Sprintf("uninstalling flexvolume %s from master", driverInstallAs))
		uninstallFlex(cs, nil, "k8s", driverInstallAs)
	})
})

/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package e2e

import (
	"fmt"
	"time"

	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// TODO: it would probably be slightly better to build up the objects
// in the code and then serialize to yaml.
var addon_controller_v1 = `
apiVersion: v1
kind: ReplicationController
metadata:
  name: addon-test-v1
  namespace: %s
  labels:
    k8s-app: addon-test
    version: v1
    kubernetes.io/cluster-service: "true"
spec:
  replicas: 2
  selector:
    k8s-app: addon-test
    version: v1
  template:
    metadata:
      labels:
        k8s-app: addon-test
        version: v1
        kubernetes.io/cluster-service: "true"
    spec:
      containers:
      - image: gcr.io/google_containers/serve_hostname:1.1
        name: addon-test
        ports:
        - containerPort: 9376
          protocol: TCP
`

var addon_controller_v2 = `
apiVersion: v1
kind: ReplicationController
metadata:
  name: addon-test-v2
  namespace: %s
  labels:
    k8s-app: addon-test
    version: v2
    kubernetes.io/cluster-service: "true"
spec:
  replicas: 2
  selector:
    k8s-app: addon-test
    version: v2
  template:
    metadata:
      labels:
        k8s-app: addon-test
        version: v2
        kubernetes.io/cluster-service: "true"
    spec:
      containers:
      - image: gcr.io/google_containers/serve_hostname:1.1
        name: addon-test
        ports:
        - containerPort: 9376
          protocol: TCP
`

var addon_service_v1 = `
apiVersion: v1
kind: Service
metadata:
  name: addon-test
  namespace: %s
  labels:
    k8s-app: addon-test
    kubernetes.io/cluster-service: "true"
    kubernetes.io/name: addon-test
spec:
  ports:
  - port: 9376
    protocol: TCP
    targetPort: 9376
  selector:
    k8s-app: addon-test
`

var addon_service_v2 = `
apiVersion: v1
kind: Service
metadata:
  name: addon-test-updated
  namespace: %s
  labels:
    k8s-app: addon-test
    kubernetes.io/cluster-service: "true"
    kubernetes.io/name: addon-test
    newLabel: newValue
spec:
  ports:
  - port: 9376
    protocol: TCP
    targetPort: 9376
  selector:
    k8s-app: addon-test
`

var invalid_addon_controller_v1 = `
apiVersion: v1
kind: ReplicationController
metadata:
  name: invalid-addon-test-v1
  namespace: %s
  labels:
    k8s-app: invalid-addon-test
    version: v1
spec:
  replicas: 2
  selector:
    k8s-app: invalid-addon-test
    version: v1
  template:
    metadata:
      labels:
        k8s-app: invalid-addon-test
        version: v1
        kubernetes.io/cluster-service: "true"
    spec:
      containers:
      - image: gcr.io/google_containers/serve_hostname:1.1
        name: invalid-addon-test
        ports:
        - containerPort: 9376
          protocol: TCP
`

var invalid_addon_service_v1 = `
apiVersion: v1
kind: Service
metadata:
  name: ivalid-addon-test
  namespace: %s
  labels:
    k8s-app: invalid-addon-test
    kubernetes.io/name: invalid-addon-test
spec:
  ports:
  - port: 9377
    protocol: TCP
    targetPort: 9376
  selector:
    k8s-app: invalid-addon-test
`

var addonTestPollInterval = 3 * time.Second
var addonTestPollTimeout = 5 * time.Minute
var defaultNsName = api.NamespaceDefault

type stringPair struct {
	data, fileName string
}

var _ = Describe("Addon update", func() {

	var dir string
	f := NewFramework("addon-update-test")

	BeforeEach(func() {
		// This test requires:
		// - SSH master access
		// ... so the provider check should be identical to the intersection of
		// providers that provide those capabilities.
		SkipUnlessProviderIs(providersWitMasterSSH...)

		// Reduce the addon update intervals so that we have faster response
		// to changes in the addon directory.
		// do not use "service" command because it clears the environment variables
		sshExecAndVerify("sudo TEST_ADDON_CHECK_INTERVAL_SEC=1 /etc/init.d/kube-addons restart")
	})

	AfterEach(func() {
		// restart addon_update with the default options
		sshExecAndVerify("sudo /etc/init.d/kube-addons restart")
	})

	// WARNING: the test is not parallel-friendly!
	It("should propagate add-on file changes", func() {
		// This test requires:
		// - SSH
		// - master access
		// ... so the provider check should be identical to the intersection of
		// providers that provide those capabilities.
		SkipUnlessProviderIs("gce")

		//these tests are long, so I squeezed several cases in one scenario
		dir = f.Namespace.Name // we use it only to give a unique string for each test execution

		temporaryRemotePathPrefix := "addon-test-dir"
		temporaryRemotePath := temporaryRemotePathPrefix + "/" + dir                                                // in home directory on kubernetes-master
		defer SSH(fmt.Sprintf("rm -rf %s", temporaryRemotePathPrefix), getMasterHost()+":22", testContext.Provider) // ignore the result in cleanup
		sshExecAndVerify(fmt.Sprintf("mkdir -p %s", temporaryRemotePath))

		rcv1 := "addon-controller-v1.yaml"
		rcv2 := "addon-controller-v2.yaml"
		rcInvalid := "invalid-addon-controller-v1.yaml"

		svcv1 := "addon-service-v1.yaml"
		svcv2 := "addon-service-v2.yaml"
		svcInvalid := "invalid-addon-service-v1.yaml"

		var remoteFiles []stringPair = []stringPair{
			{fmt.Sprintf(addon_controller_v1, defaultNsName), rcv1},
			{fmt.Sprintf(addon_controller_v2, f.Namespace.Name), rcv2},
			{fmt.Sprintf(addon_service_v1, f.Namespace.Name), svcv1},
			{fmt.Sprintf(addon_service_v2, f.Namespace.Name), svcv2},
			{fmt.Sprintf(invalid_addon_controller_v1, f.Namespace.Name), rcInvalid},
			{fmt.Sprintf(invalid_addon_service_v1, defaultNsName), svcInvalid},
		}

		for _, p := range remoteFiles {
			err := SCP(getMasterHost()+":22", testContext.Provider, p.data, temporaryRemotePath, p.fileName, 0644)
			Expect(err).NotTo(HaveOccurred())
		}

		// directory on kubernetes-master
		destinationDirPrefix := "/etc/kubernetes/addons/addon-test-dir"
		destinationDir := destinationDirPrefix + "/" + dir

		// cleanup from previous tests
		sshExecAndVerify(fmt.Sprintf("sudo rm -rf %s", destinationDirPrefix))

		defer sshExecAndVerify(fmt.Sprintf("sudo rm -rf %s", destinationDirPrefix)) // ignore result in cleanup
		sshExecAndVerify(fmt.Sprintf("sudo mkdir -p %s", destinationDir))

		By("copy invalid manifests to the destination dir (without kubernetes.io/cluster-service label)")
		sshExecAndVerify(fmt.Sprintf("sudo cp %s/%s %s/%s", temporaryRemotePath, rcInvalid, destinationDir, rcInvalid))
		sshExecAndVerify(fmt.Sprintf("sudo cp %s/%s %s/%s", temporaryRemotePath, svcInvalid, destinationDir, svcInvalid))
		// we will verify at the end of the test that the objects weren't created from the invalid manifests

		By("copy new manifests")
		sshExecAndVerify(fmt.Sprintf("sudo cp %s/%s %s/%s", temporaryRemotePath, rcv1, destinationDir, rcv1))
		sshExecAndVerify(fmt.Sprintf("sudo cp %s/%s %s/%s", temporaryRemotePath, svcv1, destinationDir, svcv1))

		waitForServiceInAddonTest(f.Client, f.Namespace.Name, "addon-test", true)
		waitForReplicationControllerInAddonTest(f.Client, defaultNsName, "addon-test-v1", true)

		By("update manifests")
		sshExecAndVerify(fmt.Sprintf("sudo cp %s/%s %s/%s", temporaryRemotePath, rcv2, destinationDir, rcv2))
		sshExecAndVerify(fmt.Sprintf("sudo cp %s/%s %s/%s", temporaryRemotePath, svcv2, destinationDir, svcv2))
		sshExecAndVerify(fmt.Sprintf("sudo rm %s/%s", destinationDir, rcv1))
		sshExecAndVerify(fmt.Sprintf("sudo rm %s/%s", destinationDir, svcv1))
		/**
		 * Note that we have a small race condition here - the kube-addon-updater
		 * May notice that a new rc/service file appeared, while the old one will still be there.
		 * But it is ok - as long as we don't have rolling update, the result will be the same
		 */

		waitForServiceInAddonTest(f.Client, f.Namespace.Name, "addon-test-updated", true)
		waitForReplicationControllerInAddonTest(f.Client, f.Namespace.Name, "addon-test-v2", true)

		waitForServiceInAddonTest(f.Client, f.Namespace.Name, "addon-test", false)
		waitForReplicationControllerInAddonTest(f.Client, defaultNsName, "addon-test-v1", false)

		By("remove manifests")
		sshExecAndVerify(fmt.Sprintf("sudo rm %s/%s", destinationDir, rcv2))
		sshExecAndVerify(fmt.Sprintf("sudo rm %s/%s", destinationDir, svcv2))

		waitForServiceInAddonTest(f.Client, f.Namespace.Name, "addon-test-updated", false)
		waitForReplicationControllerInAddonTest(f.Client, f.Namespace.Name, "addon-test-v2", false)

		By("verify invalid API addons weren't created")
		_, err := f.Client.ReplicationControllers(f.Namespace.Name).Get("invalid-addon-test-v1")
		Expect(err).To(HaveOccurred())
		_, err = f.Client.ReplicationControllers(defaultNsName).Get("invalid-addon-test-v1")
		Expect(err).To(HaveOccurred())
		_, err = f.Client.Services(f.Namespace.Name).Get("ivalid-addon-test")
		Expect(err).To(HaveOccurred())
		_, err = f.Client.Services(defaultNsName).Get("ivalid-addon-test")
		Expect(err).To(HaveOccurred())

		// invalid addons will be deleted by the deferred function
	})
})

func waitForServiceInAddonTest(c *client.Client, addonNamespace, name string, exist bool) {
	expectNoError(waitForService(c, addonNamespace, name, exist, addonTestPollInterval, addonTestPollTimeout))
}

func waitForReplicationControllerInAddonTest(c *client.Client, addonNamespace, name string, exist bool) {
	expectNoError(waitForReplicationController(c, addonNamespace, name, exist, addonTestPollInterval, addonTestPollTimeout))
}

func sshExecAndVerify(cmd string) {
	_, _, rc, err := SSH(cmd, getMasterHost()+":22", testContext.Provider)
	Expect(err).NotTo(HaveOccurred())
	Expect(rc).To(Equal(0), "error return code from executing command on the cluster: %s", cmd)
}

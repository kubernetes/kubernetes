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
	"bytes"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"golang.org/x/crypto/ssh"
	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/test/e2e/framework"

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
      - image: gcr.io/google_containers/serve_hostname:v1.4
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
      - image: gcr.io/google_containers/serve_hostname:v1.4
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
      - image: gcr.io/google_containers/serve_hostname:v1.4
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

var _ = framework.KubeDescribe("Addon update", func() {

	var dir string
	var sshClient *ssh.Client
	f := framework.NewDefaultFramework("addon-update-test")

	BeforeEach(func() {
		// This test requires:
		// - SSH master access
		// ... so the provider check should be identical to the intersection of
		// providers that provide those capabilities.
		if !framework.ProviderIs("gce") {
			return
		}

		var err error
		sshClient, err = getMasterSSHClient()
		Expect(err).NotTo(HaveOccurred())
	})

	AfterEach(func() {
		if sshClient != nil {
			sshClient.Close()
		}
	})

	// WARNING: the test is not parallel-friendly!
	It("should propagate add-on file changes", func() {
		// This test requires:
		// - SSH
		// - master access
		// ... so the provider check should be identical to the intersection of
		// providers that provide those capabilities.
		framework.SkipUnlessProviderIs("gce")

		//these tests are long, so I squeezed several cases in one scenario
		Expect(sshClient).NotTo(BeNil())
		dir = f.Namespace.Name // we use it only to give a unique string for each test execution

		temporaryRemotePathPrefix := "addon-test-dir"
		temporaryRemotePath := temporaryRemotePathPrefix + "/" + dir                  // in home directory on kubernetes-master
		defer sshExec(sshClient, fmt.Sprintf("rm -rf %s", temporaryRemotePathPrefix)) // ignore the result in cleanup
		sshExecAndVerify(sshClient, fmt.Sprintf("mkdir -p %s", temporaryRemotePath))

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
			err := writeRemoteFile(sshClient, p.data, temporaryRemotePath, p.fileName, 0644)
			Expect(err).NotTo(HaveOccurred())
		}

		// directory on kubernetes-master
		destinationDirPrefix := "/etc/kubernetes/addons/addon-test-dir"
		destinationDir := destinationDirPrefix + "/" + dir

		// cleanup from previous tests
		_, _, _, err := sshExec(sshClient, fmt.Sprintf("sudo rm -rf %s", destinationDirPrefix))
		Expect(err).NotTo(HaveOccurred())

		defer sshExec(sshClient, fmt.Sprintf("sudo rm -rf %s", destinationDirPrefix)) // ignore result in cleanup
		sshExecAndVerify(sshClient, fmt.Sprintf("sudo mkdir -p %s", destinationDir))

		By("copy invalid manifests to the destination dir (without kubernetes.io/cluster-service label)")
		sshExecAndVerify(sshClient, fmt.Sprintf("sudo cp %s/%s %s/%s", temporaryRemotePath, rcInvalid, destinationDir, rcInvalid))
		sshExecAndVerify(sshClient, fmt.Sprintf("sudo cp %s/%s %s/%s", temporaryRemotePath, svcInvalid, destinationDir, svcInvalid))
		// we will verify at the end of the test that the objects weren't created from the invalid manifests

		By("copy new manifests")
		sshExecAndVerify(sshClient, fmt.Sprintf("sudo cp %s/%s %s/%s", temporaryRemotePath, rcv1, destinationDir, rcv1))
		sshExecAndVerify(sshClient, fmt.Sprintf("sudo cp %s/%s %s/%s", temporaryRemotePath, svcv1, destinationDir, svcv1))

		waitForServiceInAddonTest(f.Client, f.Namespace.Name, "addon-test", true)
		waitForReplicationControllerInAddonTest(f.Client, defaultNsName, "addon-test-v1", true)

		By("update manifests")
		sshExecAndVerify(sshClient, fmt.Sprintf("sudo cp %s/%s %s/%s", temporaryRemotePath, rcv2, destinationDir, rcv2))
		sshExecAndVerify(sshClient, fmt.Sprintf("sudo cp %s/%s %s/%s", temporaryRemotePath, svcv2, destinationDir, svcv2))
		sshExecAndVerify(sshClient, fmt.Sprintf("sudo rm %s/%s", destinationDir, rcv1))
		sshExecAndVerify(sshClient, fmt.Sprintf("sudo rm %s/%s", destinationDir, svcv1))
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
		sshExecAndVerify(sshClient, fmt.Sprintf("sudo rm %s/%s", destinationDir, rcv2))
		sshExecAndVerify(sshClient, fmt.Sprintf("sudo rm %s/%s", destinationDir, svcv2))

		waitForServiceInAddonTest(f.Client, f.Namespace.Name, "addon-test-updated", false)
		waitForReplicationControllerInAddonTest(f.Client, f.Namespace.Name, "addon-test-v2", false)

		By("verify invalid API addons weren't created")
		_, err = f.Client.ReplicationControllers(f.Namespace.Name).Get("invalid-addon-test-v1")
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
	framework.ExpectNoError(framework.WaitForService(c, addonNamespace, name, exist, addonTestPollInterval, addonTestPollTimeout))
}

func waitForReplicationControllerInAddonTest(c *client.Client, addonNamespace, name string, exist bool) {
	framework.ExpectNoError(framework.WaitForReplicationController(c, addonNamespace, name, exist, addonTestPollInterval, addonTestPollTimeout))
}

// TODO use the framework.SSH code, either adding an SCP to it or copying files
// differently.
func getMasterSSHClient() (*ssh.Client, error) {
	// Get a signer for the provider.
	signer, err := framework.GetSigner(framework.TestContext.Provider)
	if err != nil {
		return nil, fmt.Errorf("error getting signer for provider %s: '%v'", framework.TestContext.Provider, err)
	}

	sshUser := os.Getenv("KUBE_SSH_USER")
	if sshUser == "" {
		sshUser = os.Getenv("USER")
	}
	config := &ssh.ClientConfig{
		User: sshUser,
		Auth: []ssh.AuthMethod{ssh.PublicKeys(signer)},
	}

	host := framework.GetMasterHost() + ":22"
	client, err := ssh.Dial("tcp", host, config)
	if err != nil {
		return nil, fmt.Errorf("error getting SSH client to host %s: '%v'", host, err)
	}
	return client, err
}

func sshExecAndVerify(client *ssh.Client, cmd string) {
	_, _, rc, err := sshExec(client, cmd)
	Expect(err).NotTo(HaveOccurred())
	Expect(rc).To(Equal(0), "error return code from executing command on the cluster: %s", cmd)
}

func sshExec(client *ssh.Client, cmd string) (string, string, int, error) {
	framework.Logf("Executing '%s' on %v", cmd, client.RemoteAddr())
	session, err := client.NewSession()
	if err != nil {
		return "", "", 0, fmt.Errorf("error creating session to host %s: '%v'", client.RemoteAddr(), err)
	}
	defer session.Close()

	// Run the command.
	code := 0
	var bout, berr bytes.Buffer

	session.Stdout, session.Stderr = &bout, &berr
	err = session.Run(cmd)
	if err != nil {
		// Check whether the command failed to run or didn't complete.
		if exiterr, ok := err.(*ssh.ExitError); ok {
			// If we got an ExitError and the exit code is nonzero, we'll
			// consider the SSH itself successful (just that the command run
			// errored on the host).
			if code = exiterr.ExitStatus(); code != 0 {
				err = nil
			}
		} else {
			// Some other kind of error happened (e.g. an IOError); consider the
			// SSH unsuccessful.
			err = fmt.Errorf("failed running `%s` on %s: '%v'", cmd, client.RemoteAddr(), err)
		}
	}
	return bout.String(), berr.String(), code, err
}

func writeRemoteFile(sshClient *ssh.Client, data, dir, fileName string, mode os.FileMode) error {
	framework.Logf(fmt.Sprintf("Writing remote file '%s/%s' on %v", dir, fileName, sshClient.RemoteAddr()))
	session, err := sshClient.NewSession()
	if err != nil {
		return fmt.Errorf("error creating session to host %s: '%v'", sshClient.RemoteAddr(), err)
	}
	defer session.Close()

	fileSize := len(data)
	pipe, err := session.StdinPipe()
	if err != nil {
		return err
	}
	defer pipe.Close()
	if err := session.Start(fmt.Sprintf("scp -t %s", dir)); err != nil {
		return err
	}
	fmt.Fprintf(pipe, "C%#o %d %s\n", mode, fileSize, fileName)
	io.Copy(pipe, strings.NewReader(data))
	fmt.Fprint(pipe, "\x00")
	pipe.Close()
	return session.Wait()
}

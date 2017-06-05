/*
Copyright 2015 The Kubernetes Authors.

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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// TODO: it would probably be slightly better to build up the objects
// in the code and then serialize to yaml.
var reconcile_addon_controller = `
apiVersion: v1
kind: ReplicationController
metadata:
  name: addon-reconcile-test
  namespace: %s
  labels:
    k8s-app: addon-reconcile-test
    kubernetes.io/cluster-service: "true"
    addonmanager.kubernetes.io/mode: Reconcile
spec:
  replicas: 2
  selector:
    k8s-app: addon-reconcile-test
  template:
    metadata:
      labels:
        k8s-app: addon-reconcile-test
    spec:
      containers:
      - image: gcr.io/google_containers/serve_hostname:v1.4
        name: addon-reconcile-test
        ports:
        - containerPort: 9376
          protocol: TCP
`

// Should update "reconcile" class addon.
var reconcile_addon_controller_updated = `
apiVersion: v1
kind: ReplicationController
metadata:
  name: addon-reconcile-test
  namespace: %s
  labels:
    k8s-app: addon-reconcile-test
    kubernetes.io/cluster-service: "true"
    addonmanager.kubernetes.io/mode: Reconcile
    newLabel: addon-reconcile-test
spec:
  replicas: 2
  selector:
    k8s-app: addon-reconcile-test
  template:
    metadata:
      labels:
        k8s-app: addon-reconcile-test
    spec:
      containers:
      - image: gcr.io/google_containers/serve_hostname:v1.4
        name: addon-reconcile-test
        ports:
        - containerPort: 9376
          protocol: TCP
`

var ensure_exists_addon_service = `
apiVersion: v1
kind: Service
metadata:
  name: addon-ensure-exists-test
  namespace: %s
  labels:
    k8s-app: addon-ensure-exists-test
    addonmanager.kubernetes.io/mode: EnsureExists
spec:
  ports:
  - port: 9376
    protocol: TCP
    targetPort: 9376
  selector:
    k8s-app: addon-ensure-exists-test
`

// Should create but don't update "ensure exist" class addon.
var ensure_exists_addon_service_updated = `
apiVersion: v1
kind: Service
metadata:
  name: addon-ensure-exists-test
  namespace: %s
  labels:
    k8s-app: addon-ensure-exists-test
    addonmanager.kubernetes.io/mode: EnsureExists
    newLabel: addon-ensure-exists-test
spec:
  ports:
  - port: 9376
    protocol: TCP
    targetPort: 9376
  selector:
    k8s-app: addon-ensure-exists-test
`

var deprecated_label_addon_service = `
apiVersion: v1
kind: Service
metadata:
  name: addon-deprecated-label-test
  namespace: %s
  labels:
    k8s-app: addon-deprecated-label-test
    kubernetes.io/cluster-service: "true"
spec:
  ports:
  - port: 9376
    protocol: TCP
    targetPort: 9376
  selector:
    k8s-app: addon-deprecated-label-test
`

// Should update addon with label "kubernetes.io/cluster-service=true".
var deprecated_label_addon_service_updated = `
apiVersion: v1
kind: Service
metadata:
  name: addon-deprecated-label-test
  namespace: %s
  labels:
    k8s-app: addon-deprecated-label-test
    kubernetes.io/cluster-service: "true"
    newLabel: addon-deprecated-label-test
spec:
  ports:
  - port: 9376
    protocol: TCP
    targetPort: 9376
  selector:
    k8s-app: addon-deprecated-label-test
`

// Should not create addon without valid label.
var invalid_addon_controller = `
apiVersion: v1
kind: ReplicationController
metadata:
  name: invalid-addon-test
  namespace: %s
  labels:
    k8s-app: invalid-addon-test
    addonmanager.kubernetes.io/mode: NotMatch
spec:
  replicas: 2
  selector:
    k8s-app: invalid-addon-test
  template:
    metadata:
      labels:
        k8s-app: invalid-addon-test
    spec:
      containers:
      - image: gcr.io/google_containers/serve_hostname:v1.4
        name: invalid-addon-test
        ports:
        - containerPort: 9376
          protocol: TCP
`

const (
	addonTestPollInterval = 3 * time.Second
	addonTestPollTimeout  = 5 * time.Minute
	addonNsName           = metav1.NamespaceSystem
)

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
		Expect(err).NotTo(HaveOccurred(), "Failed to get the master SSH client.")
	})

	AfterEach(func() {
		if sshClient != nil {
			sshClient.Close()
		}
	})

	// WARNING: the test is not parallel-friendly!
	It("should propagate add-on file changes [Slow]", func() {
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

		rcAddonReconcile := "addon-reconcile-controller.yaml"
		rcAddonReconcileUpdated := "addon-reconcile-controller-Updated.yaml"
		rcInvalid := "invalid-addon-controller.yaml"

		svcAddonDeprecatedLabel := "addon-deprecated-label-service.yaml"
		svcAddonDeprecatedLabelUpdated := "addon-deprecated-label-service-updated.yaml"
		svcAddonEnsureExists := "addon-ensure-exists-service.yaml"
		svcAddonEnsureExistsUpdated := "addon-ensure-exists-service-updated.yaml"

		var remoteFiles []stringPair = []stringPair{
			{fmt.Sprintf(reconcile_addon_controller, addonNsName), rcAddonReconcile},
			{fmt.Sprintf(reconcile_addon_controller_updated, addonNsName), rcAddonReconcileUpdated},
			{fmt.Sprintf(deprecated_label_addon_service, addonNsName), svcAddonDeprecatedLabel},
			{fmt.Sprintf(deprecated_label_addon_service_updated, addonNsName), svcAddonDeprecatedLabelUpdated},
			{fmt.Sprintf(ensure_exists_addon_service, addonNsName), svcAddonEnsureExists},
			{fmt.Sprintf(ensure_exists_addon_service_updated, addonNsName), svcAddonEnsureExistsUpdated},
			{fmt.Sprintf(invalid_addon_controller, addonNsName), rcInvalid},
		}

		for _, p := range remoteFiles {
			err := writeRemoteFile(sshClient, p.data, temporaryRemotePath, p.fileName, 0644)
			Expect(err).NotTo(HaveOccurred(), "Failed to write file %q at remote path %q with ssh client %+v", p.fileName, temporaryRemotePath, sshClient)
		}

		// directory on kubernetes-master
		destinationDirPrefix := "/etc/kubernetes/addons/addon-test-dir"
		destinationDir := destinationDirPrefix + "/" + dir

		// cleanup from previous tests
		_, _, _, err := sshExec(sshClient, fmt.Sprintf("sudo rm -rf %s", destinationDirPrefix))
		Expect(err).NotTo(HaveOccurred(), "Failed to remove remote dir %q with ssh client %+v", destinationDirPrefix, sshClient)

		defer sshExec(sshClient, fmt.Sprintf("sudo rm -rf %s", destinationDirPrefix)) // ignore result in cleanup
		sshExecAndVerify(sshClient, fmt.Sprintf("sudo mkdir -p %s", destinationDir))

		By("copy invalid manifests to the destination dir")
		sshExecAndVerify(sshClient, fmt.Sprintf("sudo cp %s/%s %s/%s", temporaryRemotePath, rcInvalid, destinationDir, rcInvalid))
		// we will verify at the end of the test that the objects weren't created from the invalid manifests

		By("copy new manifests")
		sshExecAndVerify(sshClient, fmt.Sprintf("sudo cp %s/%s %s/%s", temporaryRemotePath, rcAddonReconcile, destinationDir, rcAddonReconcile))
		sshExecAndVerify(sshClient, fmt.Sprintf("sudo cp %s/%s %s/%s", temporaryRemotePath, svcAddonDeprecatedLabel, destinationDir, svcAddonDeprecatedLabel))
		sshExecAndVerify(sshClient, fmt.Sprintf("sudo cp %s/%s %s/%s", temporaryRemotePath, svcAddonEnsureExists, destinationDir, svcAddonEnsureExists))
		// Delete the "ensure exist class" addon at the end.
		defer func() {
			framework.Logf("Cleaning up ensure exist class addon.")
			Expect(f.ClientSet.Core().Services(addonNsName).Delete("addon-ensure-exists-test", nil)).NotTo(HaveOccurred())
		}()

		waitForReplicationControllerInAddonTest(f.ClientSet, addonNsName, "addon-reconcile-test", true)
		waitForServiceInAddonTest(f.ClientSet, addonNsName, "addon-deprecated-label-test", true)
		waitForServiceInAddonTest(f.ClientSet, addonNsName, "addon-ensure-exists-test", true)

		// Replace the manifests with new contents.
		By("update manifests")
		sshExecAndVerify(sshClient, fmt.Sprintf("sudo cp %s/%s %s/%s", temporaryRemotePath, rcAddonReconcileUpdated, destinationDir, rcAddonReconcile))
		sshExecAndVerify(sshClient, fmt.Sprintf("sudo cp %s/%s %s/%s", temporaryRemotePath, svcAddonDeprecatedLabelUpdated, destinationDir, svcAddonDeprecatedLabel))
		sshExecAndVerify(sshClient, fmt.Sprintf("sudo cp %s/%s %s/%s", temporaryRemotePath, svcAddonEnsureExistsUpdated, destinationDir, svcAddonEnsureExists))

		// Wait for updated addons to have the new added label.
		reconcileSelector := labels.SelectorFromSet(labels.Set(map[string]string{"newLabel": "addon-reconcile-test"}))
		waitForReplicationControllerwithSelectorInAddonTest(f.ClientSet, addonNsName, true, reconcileSelector)
		deprecatedLabelSelector := labels.SelectorFromSet(labels.Set(map[string]string{"newLabel": "addon-deprecated-label-test"}))
		waitForServicewithSelectorInAddonTest(f.ClientSet, addonNsName, true, deprecatedLabelSelector)
		// "Ensure exist class" addon should not be updated.
		ensureExistSelector := labels.SelectorFromSet(labels.Set(map[string]string{"newLabel": "addon-ensure-exists-test"}))
		waitForServicewithSelectorInAddonTest(f.ClientSet, addonNsName, false, ensureExistSelector)

		By("remove manifests")
		sshExecAndVerify(sshClient, fmt.Sprintf("sudo rm %s/%s", destinationDir, rcAddonReconcile))
		sshExecAndVerify(sshClient, fmt.Sprintf("sudo rm %s/%s", destinationDir, svcAddonDeprecatedLabel))
		sshExecAndVerify(sshClient, fmt.Sprintf("sudo rm %s/%s", destinationDir, svcAddonEnsureExists))

		waitForReplicationControllerInAddonTest(f.ClientSet, addonNsName, "addon-reconcile-test", false)
		waitForServiceInAddonTest(f.ClientSet, addonNsName, "addon-deprecated-label-test", false)
		// "Ensure exist class" addon will not be deleted when manifest is removed.
		waitForServiceInAddonTest(f.ClientSet, addonNsName, "addon-ensure-exists-test", true)

		By("verify invalid addons weren't created")
		_, err = f.ClientSet.Core().ReplicationControllers(addonNsName).Get("invalid-addon-test", metav1.GetOptions{})
		Expect(err).To(HaveOccurred())

		// Invalid addon manifests and the "ensure exist class" addon will be deleted by the deferred function.
	})
})

func waitForServiceInAddonTest(c clientset.Interface, addonNamespace, name string, exist bool) {
	framework.ExpectNoError(framework.WaitForService(c, addonNamespace, name, exist, addonTestPollInterval, addonTestPollTimeout))
}

func waitForReplicationControllerInAddonTest(c clientset.Interface, addonNamespace, name string, exist bool) {
	framework.ExpectNoError(framework.WaitForReplicationController(c, addonNamespace, name, exist, addonTestPollInterval, addonTestPollTimeout))
}

func waitForServicewithSelectorInAddonTest(c clientset.Interface, addonNamespace string, exist bool, selector labels.Selector) {
	framework.ExpectNoError(framework.WaitForServiceWithSelector(c, addonNamespace, selector, exist, addonTestPollInterval, addonTestPollTimeout))
}

func waitForReplicationControllerwithSelectorInAddonTest(c clientset.Interface, addonNamespace string, exist bool, selector labels.Selector) {
	framework.ExpectNoError(framework.WaitForReplicationControllerwithSelector(c, addonNamespace, selector, exist, addonTestPollInterval,
		addonTestPollTimeout))
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
	Expect(err).NotTo(HaveOccurred(), "Failed to execute %q with ssh client %+v", cmd, client)
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

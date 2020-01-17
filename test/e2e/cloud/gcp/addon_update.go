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

package gcp

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
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2essh "k8s.io/kubernetes/test/e2e/framework/ssh"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
)

// TODO: it would probably be slightly better to build up the objects
// in the code and then serialize to yaml.
var reconcileAddonController = `
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
      - image: %s
        name: addon-reconcile-test
        ports:
        - containerPort: 9376
          protocol: TCP
`

// Should update "reconcile" class addon.
var reconcileAddonControllerUpdated = `
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
      - image: %s
        name: addon-reconcile-test
        ports:
        - containerPort: 9376
          protocol: TCP
`

var ensureExistsAddonService = `
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
var ensureExistsAddonServiceUpdated = `
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

var deprecatedLabelAddonService = `
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
var deprecatedLabelAddonServiceUpdated = `
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
var invalidAddonController = `
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
      - image: %s
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

var serveHostnameImage = imageutils.GetE2EImage(imageutils.Agnhost)

type stringPair struct {
	data, fileName string
}

var _ = SIGDescribe("Addon update", func() {

	var dir string
	var sshClient *ssh.Client
	f := framework.NewDefaultFramework("addon-update-test")

	ginkgo.BeforeEach(func() {
		// This test requires:
		// - SSH master access
		// ... so the provider check should be identical to the intersection of
		// providers that provide those capabilities.
		if !framework.ProviderIs("gce") {
			return
		}

		var err error
		sshClient, err = getMasterSSHClient()
		framework.ExpectNoError(err, "Failed to get the master SSH client.")
	})

	ginkgo.AfterEach(func() {
		if sshClient != nil {
			sshClient.Close()
		}
	})

	// WARNING: the test is not parallel-friendly!
	ginkgo.It("should propagate add-on file changes [Slow]", func() {
		// This test requires:
		// - SSH
		// - master access
		// ... so the provider check should be identical to the intersection of
		// providers that provide those capabilities.
		e2eskipper.SkipUnlessProviderIs("gce")

		//these tests are long, so I squeezed several cases in one scenario
		framework.ExpectNotEqual(sshClient, nil)
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
			{fmt.Sprintf(reconcileAddonController, addonNsName, serveHostnameImage), rcAddonReconcile},
			{fmt.Sprintf(reconcileAddonControllerUpdated, addonNsName, serveHostnameImage), rcAddonReconcileUpdated},
			{fmt.Sprintf(deprecatedLabelAddonService, addonNsName), svcAddonDeprecatedLabel},
			{fmt.Sprintf(deprecatedLabelAddonServiceUpdated, addonNsName), svcAddonDeprecatedLabelUpdated},
			{fmt.Sprintf(ensureExistsAddonService, addonNsName), svcAddonEnsureExists},
			{fmt.Sprintf(ensureExistsAddonServiceUpdated, addonNsName), svcAddonEnsureExistsUpdated},
			{fmt.Sprintf(invalidAddonController, addonNsName, serveHostnameImage), rcInvalid},
		}

		for _, p := range remoteFiles {
			err := writeRemoteFile(sshClient, p.data, temporaryRemotePath, p.fileName, 0644)
			framework.ExpectNoError(err, "Failed to write file %q at remote path %q with ssh client %+v", p.fileName, temporaryRemotePath, sshClient)
		}

		// directory on kubernetes-master
		destinationDirPrefix := "/etc/kubernetes/addons/addon-test-dir"
		destinationDir := destinationDirPrefix + "/" + dir

		// cleanup from previous tests
		_, _, _, err := sshExec(sshClient, fmt.Sprintf("sudo rm -rf %s", destinationDirPrefix))
		framework.ExpectNoError(err, "Failed to remove remote dir %q with ssh client %+v", destinationDirPrefix, sshClient)

		defer sshExec(sshClient, fmt.Sprintf("sudo rm -rf %s", destinationDirPrefix)) // ignore result in cleanup
		sshExecAndVerify(sshClient, fmt.Sprintf("sudo mkdir -p %s", destinationDir))

		ginkgo.By("copy invalid manifests to the destination dir")
		sshExecAndVerify(sshClient, fmt.Sprintf("sudo cp %s/%s %s/%s", temporaryRemotePath, rcInvalid, destinationDir, rcInvalid))
		// we will verify at the end of the test that the objects weren't created from the invalid manifests

		ginkgo.By("copy new manifests")
		sshExecAndVerify(sshClient, fmt.Sprintf("sudo cp %s/%s %s/%s", temporaryRemotePath, rcAddonReconcile, destinationDir, rcAddonReconcile))
		sshExecAndVerify(sshClient, fmt.Sprintf("sudo cp %s/%s %s/%s", temporaryRemotePath, svcAddonDeprecatedLabel, destinationDir, svcAddonDeprecatedLabel))
		sshExecAndVerify(sshClient, fmt.Sprintf("sudo cp %s/%s %s/%s", temporaryRemotePath, svcAddonEnsureExists, destinationDir, svcAddonEnsureExists))
		// Delete the "ensure exist class" addon at the end.
		defer func() {
			framework.Logf("Cleaning up ensure exist class addon.")
			err := f.ClientSet.CoreV1().Services(addonNsName).Delete("addon-ensure-exists-test", nil)
			framework.ExpectNoError(err)
		}()

		waitForReplicationControllerInAddonTest(f.ClientSet, addonNsName, "addon-reconcile-test", true)
		waitForServiceInAddonTest(f.ClientSet, addonNsName, "addon-deprecated-label-test", true)
		waitForServiceInAddonTest(f.ClientSet, addonNsName, "addon-ensure-exists-test", true)

		// Replace the manifests with new contents.
		ginkgo.By("update manifests")
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

		ginkgo.By("remove manifests")
		sshExecAndVerify(sshClient, fmt.Sprintf("sudo rm %s/%s", destinationDir, rcAddonReconcile))
		sshExecAndVerify(sshClient, fmt.Sprintf("sudo rm %s/%s", destinationDir, svcAddonDeprecatedLabel))
		sshExecAndVerify(sshClient, fmt.Sprintf("sudo rm %s/%s", destinationDir, svcAddonEnsureExists))

		waitForReplicationControllerInAddonTest(f.ClientSet, addonNsName, "addon-reconcile-test", false)
		waitForServiceInAddonTest(f.ClientSet, addonNsName, "addon-deprecated-label-test", false)
		// "Ensure exist class" addon will not be deleted when manifest is removed.
		waitForServiceInAddonTest(f.ClientSet, addonNsName, "addon-ensure-exists-test", true)

		ginkgo.By("verify invalid addons weren't created")
		_, err = f.ClientSet.CoreV1().ReplicationControllers(addonNsName).Get("invalid-addon-test", metav1.GetOptions{})
		framework.ExpectError(err)

		// Invalid addon manifests and the "ensure exist class" addon will be deleted by the deferred function.
	})
})

func waitForServiceInAddonTest(c clientset.Interface, addonNamespace, name string, exist bool) {
	framework.ExpectNoError(framework.WaitForService(c, addonNamespace, name, exist, addonTestPollInterval, addonTestPollTimeout))
}

func waitForReplicationControllerInAddonTest(c clientset.Interface, addonNamespace, name string, exist bool) {
	framework.ExpectNoError(waitForReplicationController(c, addonNamespace, name, exist, addonTestPollInterval, addonTestPollTimeout))
}

func waitForServicewithSelectorInAddonTest(c clientset.Interface, addonNamespace string, exist bool, selector labels.Selector) {
	framework.ExpectNoError(waitForServiceWithSelector(c, addonNamespace, selector, exist, addonTestPollInterval, addonTestPollTimeout))
}

func waitForReplicationControllerwithSelectorInAddonTest(c clientset.Interface, addonNamespace string, exist bool, selector labels.Selector) {
	framework.ExpectNoError(waitForReplicationControllerWithSelector(c, addonNamespace, selector, exist, addonTestPollInterval,
		addonTestPollTimeout))
}

// waitForReplicationController waits until the RC appears (exist == true), or disappears (exist == false)
func waitForReplicationController(c clientset.Interface, namespace, name string, exist bool, interval, timeout time.Duration) error {
	err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		_, err := c.CoreV1().ReplicationControllers(namespace).Get(name, metav1.GetOptions{})
		if err != nil {
			framework.Logf("Get ReplicationController %s in namespace %s failed (%v).", name, namespace, err)
			return !exist, nil
		}
		framework.Logf("ReplicationController %s in namespace %s found.", name, namespace)
		return exist, nil
	})
	if err != nil {
		stateMsg := map[bool]string{true: "to appear", false: "to disappear"}
		return fmt.Errorf("error waiting for ReplicationController %s/%s %s: %v", namespace, name, stateMsg[exist], err)
	}
	return nil
}

// waitForServiceWithSelector waits until any service with given selector appears (exist == true), or disappears (exist == false)
func waitForServiceWithSelector(c clientset.Interface, namespace string, selector labels.Selector, exist bool, interval,
	timeout time.Duration) error {
	err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		services, err := c.CoreV1().Services(namespace).List(metav1.ListOptions{LabelSelector: selector.String()})
		switch {
		case len(services.Items) != 0:
			framework.Logf("Service with %s in namespace %s found.", selector.String(), namespace)
			return exist, nil
		case len(services.Items) == 0:
			framework.Logf("Service with %s in namespace %s disappeared.", selector.String(), namespace)
			return !exist, nil
		case !testutils.IsRetryableAPIError(err):
			framework.Logf("Non-retryable failure while listing service.")
			return false, err
		default:
			framework.Logf("List service with %s in namespace %s failed: %v", selector.String(), namespace, err)
			return false, nil
		}
	})
	if err != nil {
		stateMsg := map[bool]string{true: "to appear", false: "to disappear"}
		return fmt.Errorf("error waiting for service with %s in namespace %s %s: %v", selector.String(), namespace, stateMsg[exist], err)
	}
	return nil
}

// waitForReplicationControllerWithSelector waits until any RC with given selector appears (exist == true), or disappears (exist == false)
func waitForReplicationControllerWithSelector(c clientset.Interface, namespace string, selector labels.Selector, exist bool, interval,
	timeout time.Duration) error {
	err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		rcs, err := c.CoreV1().ReplicationControllers(namespace).List(metav1.ListOptions{LabelSelector: selector.String()})
		switch {
		case len(rcs.Items) != 0:
			framework.Logf("ReplicationController with %s in namespace %s found.", selector.String(), namespace)
			return exist, nil
		case len(rcs.Items) == 0:
			framework.Logf("ReplicationController with %s in namespace %s disappeared.", selector.String(), namespace)
			return !exist, nil
		default:
			framework.Logf("List ReplicationController with %s in namespace %s failed: %v", selector.String(), namespace, err)
			return false, nil
		}
	})
	if err != nil {
		stateMsg := map[bool]string{true: "to appear", false: "to disappear"}
		return fmt.Errorf("error waiting for ReplicationControllers with %s in namespace %s %s: %v", selector.String(), namespace, stateMsg[exist], err)
	}
	return nil
}

// TODO use the ssh.SSH code, either adding an SCP to it or copying files
// differently.
func getMasterSSHClient() (*ssh.Client, error) {
	// Get a signer for the provider.
	signer, err := e2essh.GetSigner(framework.TestContext.Provider)
	if err != nil {
		return nil, fmt.Errorf("error getting signer for provider %s: '%v'", framework.TestContext.Provider, err)
	}

	sshUser := os.Getenv("KUBE_SSH_USER")
	if sshUser == "" {
		sshUser = os.Getenv("USER")
	}
	config := &ssh.ClientConfig{
		User:            sshUser,
		Auth:            []ssh.AuthMethod{ssh.PublicKeys(signer)},
		HostKeyCallback: ssh.InsecureIgnoreHostKey(),
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
	framework.ExpectNoError(err, "Failed to execute %q with ssh client %+v", cmd, client)
	framework.ExpectEqual(rc, 0, "error return code from executing command on the cluster: %s", cmd)
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

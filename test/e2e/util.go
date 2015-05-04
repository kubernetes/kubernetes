/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"io/ioutil"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"code.google.com/p/go-uuid/uuid"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/clientauth"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/wait"

	"golang.org/x/crypto/ssh"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	// Initial pod start can be delayed O(minutes) by slow docker pulls
	// TODO: Make this 30 seconds once #4566 is resolved.
	podStartTimeout = 5 * time.Minute
)

type TestContextType struct {
	KubeConfig  string
	KubeContext string
	AuthConfig  string
	CertDir     string
	Host        string
	RepoRoot    string
	Provider    string
	CloudConfig CloudConfig
}

var testContext TestContextType

func Logf(format string, a ...interface{}) {
	fmt.Fprintf(GinkgoWriter, "INFO: "+format+"\n", a...)
}

func Failf(format string, a ...interface{}) {
	Fail(fmt.Sprintf(format, a...), 1)
}

func providerIs(providers ...string) bool {
	if testContext.Provider == "" {
		Fail("testContext.Provider is not defined")
	}
	for _, provider := range providers {
		if strings.ToLower(provider) == strings.ToLower(testContext.Provider) {
			return true
		}
	}
	return false
}

type podCondition func(pod *api.Pod) (bool, error)

func waitForPodCondition(c *client.Client, ns, podName, desc string, condition podCondition) error {
	By(fmt.Sprintf("waiting up to %v for pod %s status to be %s", podStartTimeout, podName, desc))
	for start := time.Now(); time.Since(start) < podStartTimeout; time.Sleep(5 * time.Second) {
		pod, err := c.Pods(ns).Get(podName)
		if err != nil {
			Logf("Get pod %v in ns %v failed, ignoring for 5s: %v", podName, ns, err)
			continue
		}
		done, err := condition(pod)
		if done {
			return err
		}
		Logf("Waiting for pod %s in namespace %s status to be %q (found %q) (%v)", podName, ns, desc, pod.Status.Phase, time.Since(start))
	}
	return fmt.Errorf("gave up waiting for pod %s to be %s after %.2f seconds", podName, desc, podStartTimeout.Seconds())
}

// createNS should be used by every test, note that we append a common prefix to the provided test name.
func createTestingNS(baseName string, c *client.Client) (*api.Namespace, error) {
	namespaceObj := &api.Namespace{
		ObjectMeta: api.ObjectMeta{
			Name:      fmt.Sprintf("e2e-tests-%v-%v", baseName, uuid.New()),
			Namespace: "",
		},
		Status: api.NamespaceStatus{},
	}
	_, err := c.Namespaces().Create(namespaceObj)
	return namespaceObj, err
}

func waitForPodRunningInNamespace(c *client.Client, podName string, namespace string) error {
	return waitForPodCondition(c, namespace, podName, "running", func(pod *api.Pod) (bool, error) {
		return (pod.Status.Phase == api.PodRunning), nil
	})
}

func waitForPodRunning(c *client.Client, podName string) error {
	return waitForPodRunningInNamespace(c, podName, api.NamespaceDefault)
}

// waitForPodNotPending returns an error if it took too long for the pod to go out of pending state.
func waitForPodNotPending(c *client.Client, ns, podName string) error {
	return waitForPodCondition(c, ns, podName, "!pending", func(pod *api.Pod) (bool, error) {
		if pod.Status.Phase != api.PodPending {
			Logf("Saw pod %s in namespace %s out of pending state (found %q)", podName, ns, pod.Status.Phase)
			return true, nil
		}
		return false, nil
	})
}

// waitForPodSuccessInNamespace returns nil if the pod reached state success, or an error if it reached failure or ran too long.
func waitForPodSuccessInNamespace(c *client.Client, podName string, contName string, namespace string) error {
	return waitForPodCondition(c, namespace, podName, "success or failure", func(pod *api.Pod) (bool, error) {
		// Cannot use pod.Status.Phase == api.PodSucceeded/api.PodFailed due to #2632
		ci, ok := api.GetContainerStatus(pod.Status.ContainerStatuses, contName)
		if !ok {
			Logf("No Status.Info for container %s in pod %s yet", contName, podName)
		} else {
			if ci.State.Termination != nil {
				if ci.State.Termination.ExitCode == 0 {
					By("Saw pod success")
					return true, nil
				} else {
					return true, fmt.Errorf("pod %s terminated with failure: %+v", podName, ci.State.Termination)
				}
			} else {
				Logf("Nil State.Termination for container %s in pod %s in namespace %s so far", contName, podName, namespace)
			}
		}
		return false, nil
	})
}

// waitForPodSuccess returns nil if the pod reached state success, or an error if it reached failure or ran too long.
// The default namespace is used to identify pods.
func waitForPodSuccess(c *client.Client, podName string, contName string) error {
	return waitForPodSuccessInNamespace(c, podName, contName, api.NamespaceDefault)
}

func loadConfig() (*client.Config, error) {
	switch {
	case testContext.KubeConfig != "":
		fmt.Printf(">>> testContext.KubeConfig: %s\n", testContext.KubeConfig)
		c, err := clientcmd.LoadFromFile(testContext.KubeConfig)
		if err != nil {
			return nil, fmt.Errorf("error loading KubeConfig: %v", err.Error())
		}
		if testContext.KubeContext != "" {
			fmt.Printf(">>> testContext.KubeContext: %s\n", testContext.KubeContext)
			c.CurrentContext = testContext.KubeContext
		}
		return clientcmd.NewDefaultClientConfig(*c, &clientcmd.ConfigOverrides{}).ClientConfig()
	case testContext.AuthConfig != "":
		fmt.Printf(">>> testContext.AuthConfig: %s\n", testContext.AuthConfig)
		config := &client.Config{
			Host: testContext.Host,
		}
		info, err := clientauth.LoadFromFile(testContext.AuthConfig)
		if err != nil {
			return nil, fmt.Errorf("error loading AuthConfig: %v", err.Error())
		}
		// If the certificate directory is provided, set the cert paths to be there.
		if testContext.CertDir != "" {
			Logf("Expecting certs in %v.", testContext.CertDir)
			info.CAFile = filepath.Join(testContext.CertDir, "ca.crt")
			info.CertFile = filepath.Join(testContext.CertDir, "kubecfg.crt")
			info.KeyFile = filepath.Join(testContext.CertDir, "kubecfg.key")
		}
		mergedConfig, err := info.MergeWithConfig(*config)
		return &mergedConfig, err
	default:
		return nil, fmt.Errorf("either KubeConfig or AuthConfig must be specified to load client config")
	}
}

func loadClient() (*client.Client, error) {
	config, err := loadConfig()
	if err != nil {
		return nil, fmt.Errorf("error creating client: %v", err.Error())
	}
	c, err := client.New(config)
	if err != nil {
		return nil, fmt.Errorf("error creating client: %v", err.Error())
	}
	return c, nil
}

// randomSuffix provides a random string to append to pods,services,rcs.
// TODO: Allow service names to have the same form as names
//       for pods and replication controllers so we don't
//       need to use such a function and can instead
//       use the UUID utilty function.
func randomSuffix() string {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	return strconv.Itoa(r.Int() % 10000)
}

func expectNoError(err error, explain ...interface{}) {
	ExpectWithOffset(1, err).NotTo(HaveOccurred(), explain...)
}

func cleanup(filePath string, selectors ...string) {
	By("using stop to clean up resources")
	runKubectl("stop", "-f", filePath)

	for _, selector := range selectors {
		resources := runKubectl("get", "pods,rc,se", "-l", selector, "--no-headers")
		if resources != "" {
			Failf("Resources left running after stop:\n%s", resources)
		}
	}
}

// validatorFn is the function which is individual tests will implement.
// we may want it to return more than just an error, at some point.
type validatorFn func(c *client.Client, podID string) error

// validateController is a generic mechanism for testing RC's that are running.
// It takes a container name, a test name, and a validator function which is plugged in by a specific test.
// "containername": this is grepped for.
// "containerImage" : this is the name of the image we expect to be launched.  Not to confuse w/ images (kitten.jpg)  which are validated.
// "testname":  which gets bubbled up to the logging/failure messages if errors happen.
// "validator" function: This function is given a podID and a client, and it can do some specific validations that way.
func validateController(c *client.Client, containerImage string, replicas int, containername string, testname string, validator validatorFn, ns string) {
	getPodsTemplate := "--template={{range.items}}{{.metadata.name}} {{end}}"
	// NB: kubectl adds the "exists" function to the standard template functions.
	// This lets us check to see if the "running" entry exists for each of the containers
	// we care about. Exists will never return an error and it's safe to check a chain of
	// things, any one of which may not exist. In the below template, all of info,
	// containername, and running might be nil, so the normal index function isn't very
	// helpful.
	// This template is unit-tested in kubectl, so if you change it, update the unit test.
	// You can read about the syntax here: http://golang.org/pkg/text/template/.
	getContainerStateTemplate := fmt.Sprintf(`--template={{if (exists . "status" "containerStatuses")}}{{range .status.containerStatuses}}{{if (and (eq .name "%s") (exists . "state" "running"))}}true{{end}}{{end}}{{end}}`, containername)

	getImageTemplate := fmt.Sprintf(`--template={{if (exists . "status" "containerStatuses")}}{{range .status.containerStatuses}}{{if eq .name "%s"}}{{.image}}{{end}}{{end}}{{end}}`, containername)

	By(fmt.Sprintf("waiting for all containers in %s pods to come up.", testname)) //testname should be selector
	for start := time.Now(); time.Since(start) < podStartTimeout; time.Sleep(5 * time.Second) {
		getPodsOutput := runKubectl("get", "pods", "-o", "template", getPodsTemplate, "--api-version=v1beta3", "-l", testname, fmt.Sprintf("--namespace=%v", ns))
		pods := strings.Fields(getPodsOutput)
		if numPods := len(pods); numPods != replicas {
			By(fmt.Sprintf("Replicas for %s: expected=%d actual=%d", testname, replicas, numPods))
			continue
		}
		var runningPods []string
		for _, podID := range pods {
			running := runKubectl("get", "pods", podID, "-o", "template", getContainerStateTemplate, "--api-version=v1beta3", fmt.Sprintf("--namespace=%v", ns))
			if running != "true" {
				Logf("%s is created but not running", podID)
				continue
			}

			currentImage := runKubectl("get", "pods", podID, "-o", "template", getImageTemplate, "--api-version=v1beta3", fmt.Sprintf("--namespace=%v", ns))
			if currentImage != containerImage {
				Logf("%s is created but running wrong image; expected: %s, actual: %s", podID, containerImage, currentImage)
				continue
			}

			// Call the generic validator function here.
			// This might validate for example, that (1) getting a url works and (2) url is serving correct content.
			if err := validator(c, podID); err != nil {
				Logf("%s is running right image but validator function failed: %v", podID, err)
				continue
			}

			Logf("%s is verified up and running", podID)
			runningPods = append(runningPods, podID)
		}
		// If we reach here, then all our checks passed.
		if len(runningPods) == replicas {
			return
		}
	}
	// Reaching here means that one of more checks failed multiple times.  Assuming its not a race condition, something is broken.
	Failf("Timed out after %v seconds waiting for %s pods to reach valid state", podStartTimeout.Seconds(), testname)
}

// kubectlCmd runs the kubectl executable.
// kubectlCmd runs the kubectl executable.
func kubectlCmd(args ...string) *exec.Cmd {
	defaultArgs := []string{}

	// Reference a --server option so tests can run anywhere.
	if testContext.Host != "" {
		defaultArgs = append(defaultArgs, "--"+clientcmd.FlagAPIServer+"="+testContext.Host)
	}
	if testContext.KubeConfig != "" {
		defaultArgs = append(defaultArgs, "--"+clientcmd.RecommendedConfigPathFlag+"="+testContext.KubeConfig)

		// Reference the KubeContext
		if testContext.KubeContext != "" {
			defaultArgs = append(defaultArgs, "--"+clientcmd.FlagContext+"="+testContext.KubeContext)
		}

	} else {
		defaultArgs = append(defaultArgs, "--"+clientcmd.FlagAuthPath+"="+testContext.AuthConfig)
		if testContext.CertDir != "" {
			defaultArgs = append(defaultArgs,
				fmt.Sprintf("--certificate-authority=%s", filepath.Join(testContext.CertDir, "ca.crt")),
				fmt.Sprintf("--client-certificate=%s", filepath.Join(testContext.CertDir, "kubecfg.crt")),
				fmt.Sprintf("--client-key=%s", filepath.Join(testContext.CertDir, "kubecfg.key")))
		}
	}
	kubectlArgs := append(defaultArgs, args...)

	//TODO: the "kubectl" path string might be worth externalizing into an (optional) ginko arg.
	cmd := exec.Command("kubectl", kubectlArgs...)
	Logf("Running '%s %s'", cmd.Path, strings.Join(cmd.Args, " "))
	return cmd
}

func runKubectl(args ...string) string {
	var stdout, stderr bytes.Buffer
	cmd := kubectlCmd(args...)
	cmd.Stdout, cmd.Stderr = &stdout, &stderr

	if err := cmd.Run(); err != nil {
		Failf("Error running %v:\nCommand stdout:\n%v\nstderr:\n%v\n", cmd, cmd.Stdout, cmd.Stderr)
		return ""
	}
	Logf(stdout.String())
	// TODO: trimspace should be unnecessary after switching to use kubectl binary directly
	return strings.TrimSpace(stdout.String())
}

// testContainerOutput runs testContainerOutputInNamespace with the default namespace.
func testContainerOutput(scenarioName string, c *client.Client, pod *api.Pod, expectedOutput []string) {
	testContainerOutputInNamespace(scenarioName, c, pod, expectedOutput, api.NamespaceDefault)
}

// testContainerOutputInNamespace runs the given pod in the given namespace and waits
// for the first container in the podSpec to move into the 'Success' status.  It retrieves
// the container log and searches for lines of expected output.
func testContainerOutputInNamespace(scenarioName string, c *client.Client, pod *api.Pod, expectedOutput []string, ns string) {
	By(fmt.Sprintf("Creating a pod to test %v", scenarioName))

	defer c.Pods(ns).Delete(pod.Name, nil)
	if _, err := c.Pods(ns).Create(pod); err != nil {
		Failf("Failed to create pod: %v", err)
	}

	containerName := pod.Spec.Containers[0].Name

	// Wait for client pod to complete.
	expectNoError(waitForPodSuccessInNamespace(c, pod.Name, containerName, ns))

	// Grab its logs.  Get host first.
	podStatus, err := c.Pods(ns).Get(pod.Name)
	if err != nil {
		Failf("Failed to get pod status: %v", err)
	}

	By(fmt.Sprintf("Trying to get logs from host %s pod %s container %s: %v",
		podStatus.Spec.Host, podStatus.Name, containerName, err))
	var logs []byte
	start := time.Now()

	// Sometimes the actual containers take a second to get started, try to get logs for 60s
	for time.Now().Sub(start) < (60 * time.Second) {
		logs, err = c.Get().
			Prefix("proxy").
			Resource("nodes").
			Name(podStatus.Spec.Host).
			Suffix("containerLogs", ns, podStatus.Name, containerName).
			Do().
			Raw()
		fmt.Sprintf("pod logs:%v\n", string(logs))
		By(fmt.Sprintf("pod logs:%v\n", string(logs)))
		if strings.Contains(string(logs), "Internal Error") {
			By(fmt.Sprintf("Failed to get logs from host %q pod %q container %q: %v",
				podStatus.Spec.Host, podStatus.Name, containerName, string(logs)))
			time.Sleep(5 * time.Second)
			continue
		}
		break
	}

	for _, m := range expectedOutput {
		Expect(string(logs)).To(ContainSubstring(m), "%q in container output", m)
	}
}

// Delete a Replication Controller and all pods it spawned
func DeleteRC(c *client.Client, ns, name string) error {
	rc, err := c.ReplicationControllers(ns).Get(name)
	if err != nil {
		return fmt.Errorf("Failed to find replication controller %s in namespace %s: %v", name, ns, err)
	}

	rc.Spec.Replicas = 0

	if _, err := c.ReplicationControllers(ns).Update(rc); err != nil {
		return fmt.Errorf("Failed to resize replication controller %s to zero: %v", name, err)
	}

	// Wait up to 20 minutes until all replicas are killed.
	endTime := time.Now().Add(time.Minute * 20)
	for {
		if time.Now().After(endTime) {
			return fmt.Errorf("Timeout while waiting for replication controller %s replicas to 0", name)
		}
		remainingTime := endTime.Sub(time.Now())
		err := wait.Poll(time.Second, remainingTime, client.ControllerHasDesiredReplicas(c, rc))
		if err != nil {
			Logf("Error while waiting for replication controller %s replicas to read 0: %v", name, err)
		} else {
			break
		}
	}

	// Delete the replication controller.
	if err := c.ReplicationControllers(ns).Delete(name); err != nil {
		return fmt.Errorf("Failed to delete replication controller %s: %v", name, err)
	}
	return nil
}

// Launch a Replication Controller and wait for all pods it spawns
// to become running. The controller will need to be cleaned up external
// to this method
func RunRC(c *client.Client, name string, ns, image string, replicas int) error {
	var last int
	current := 0
	same := 0

	By(fmt.Sprintf("Creating replication controller %s", name))
	_, err := c.ReplicationControllers(ns).Create(&api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name: name,
		},
		Spec: api.ReplicationControllerSpec{
			Replicas: replicas,
			Selector: map[string]string{
				"name": name,
			},
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{"name": name},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:  name,
							Image: image,
							Ports: []api.ContainerPort{{ContainerPort: 80}},
						},
					},
				},
			},
		},
	})
	if err != nil {
		return fmt.Errorf("Error creating replication controller: %v", err)
	}

	By(fmt.Sprintf("Making sure all %d replicas exist", replicas))
	label := labels.SelectorFromSet(labels.Set(map[string]string{"name": name}))
	pods, err := listPods(c, ns, label, fields.Everything())
	if err != nil {
		return fmt.Errorf("Error listing pods: %v", err)
	}
	current = len(pods.Items)
	failCount := 5
	for same < failCount && current < replicas {
		Logf("Controller %s: Found %d pods out of %d", name, current, replicas)
		if last < current {
			same = 0
		} else if last == current {
			same++
		} else if current < last {
			return fmt.Errorf("Controller %s: Number of submitted pods dropped from %d to %d", name, last, current)
		}

		if same >= failCount {
			return fmt.Errorf("Controller %s: No pods submitted for the last %d checks", name, failCount)
		}

		last = current
		time.Sleep(5 * time.Second)
		pods, err = listPods(c, ns, label, fields.Everything())
		if err != nil {
			return fmt.Errorf("Error listing pods: %v", err)
		}
		current = len(pods.Items)
	}
	if current != replicas {
		return fmt.Errorf("Controller %s: Only found %d replicas out of %d", name, current, replicas)
	}
	Logf("Controller %s: Found %d pods out of %d", name, current, replicas)

	By("Waiting for each pod to be running")
	same = 0
	last = 0
	failCount = 10
	current = 0
	for same < failCount && current < replicas {
		current = 0
		waiting := 0
		pending := 0
		unknown := 0
		time.Sleep(10 * time.Second)

		currentPods, err := listPods(c, ns, label, fields.Everything())
		if err != nil {
			return fmt.Errorf("Error listing pods: %v", err)
		}
		if len(currentPods.Items) != len(pods.Items) {
			return fmt.Errorf("Number of reported pods changed: %d vs %d", len(currentPods.Items), len(pods.Items))
		}
		for _, p := range currentPods.Items {
			if p.Status.Phase == api.PodRunning {
				current++
				if err := VerifyContainersAreNotFailed(p); err != nil {
					return err
				}
			} else if p.Status.Phase == api.PodPending {
				if p.Spec.Host == "" {
					waiting++
				} else {
					pending++
				}
			} else if p.Status.Phase == api.PodUnknown {
				unknown++
			}
		}
		Logf("Pod States: %d running, %d pending, %d waiting, %d unknown ", current, pending, waiting, unknown)
		if last < current {
			same = 0
		} else if last == current {
			same++
		} else if current < last {
			return fmt.Errorf("Number of running pods dropped from %d to %d", last, current)
		}
		if same >= failCount {
			return fmt.Errorf("No pods started for the last %d checks", failCount)
		}
		last = current
	}
	if current != replicas {
		return fmt.Errorf("Only %d pods started out of %d", current, replicas)
	}
	return nil
}

// Convenient wrapper around listing pods supporting retries.
func listPods(c *client.Client, namespace string, label labels.Selector, field fields.Selector) (*api.PodList, error) {
	maxRetries := 4
	pods, err := c.Pods(namespace).List(label, field)
	for i := 0; i < maxRetries; i++ {
		if err == nil {
			return pods, nil
		}
		pods, err = c.Pods(namespace).List(label, field)
	}
	return pods, err
}

func VerifyContainersAreNotFailed(pod api.Pod) error {
	var errStrings []string

	statuses := pod.Status.ContainerStatuses
	if len(statuses) == 0 {
		return nil
	} else {
		for _, status := range statuses {
			if status.State.Termination != nil || status.LastTerminationState.Termination != nil || status.RestartCount != 0 {
				errStrings = append(errStrings, fmt.Sprintf("Error: Pod %s (host: %s) : Container %s was found to have terminated %d times", pod.Name, pod.Spec.Host, status.Name, status.RestartCount))
			}
		}
	}

	if len(errStrings) > 0 {
		return fmt.Errorf(strings.Join(errStrings, "\n"))
	}
	return nil
}

// Prints the histogram of the events and returns the number of bad events.
func BadEvents(events []*api.Event) int {
	type histogramKey struct {
		reason string
		source string
	}
	histogram := make(map[histogramKey]int)
	for _, e := range events {
		histogram[histogramKey{reason: e.Reason, source: e.Source.Component}]++
	}
	for key, number := range histogram {
		Logf("- reason: %s, source: %s -> %d", key.reason, key.source, number)
	}
	badPatterns := []string{"kill", "fail"}
	badEvents := 0
	for key, number := range histogram {
		for _, s := range badPatterns {
			if strings.Contains(key.reason, s) {
				Logf("WARNING %d events from %s with reason: %s", number, key.source, key.reason)
				badEvents += number
				break
			}
		}
	}
	return badEvents
}

// SSH synchronously SSHs to a node running on provider and runs cmd. If there
// is no error performing the SSH, the stdout, stderr, and exit code are
// returned.
func SSH(cmd, host, provider string) (string, string, int, error) {
	// Get a signer for the provider.
	signer, err := getSigner(provider)
	if err != nil {
		return "", "", 0, fmt.Errorf("error getting signer for provider %s: '%v'", provider, err)
	}

	// Setup the config, dial the server, and open a session.
	config := &ssh.ClientConfig{
		User: os.Getenv("USER"),
		Auth: []ssh.AuthMethod{ssh.PublicKeys(signer)},
	}
	client, err := ssh.Dial("tcp", host, config)
	if err != nil {
		return "", "", 0, fmt.Errorf("error getting SSH client to host %s: '%v'", host, err)
	}
	session, err := client.NewSession()
	if err != nil {
		return "", "", 0, fmt.Errorf("error creating session to host %s: '%v'", host, err)
	}
	defer session.Close()

	// Run the command.
	code := 0
	var bout, berr bytes.Buffer
	session.Stdout, session.Stderr = &bout, &berr
	if err = session.Run(cmd); err != nil {
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
			err = fmt.Errorf("failed running `%s` on %s: '%v'", cmd, host, err)
		}
	}
	return bout.String(), berr.String(), code, err
}

// getSigner returns an ssh.Signer for the provider ("gce", etc.) that can be
// used to SSH to their nodes.
func getSigner(provider string) (ssh.Signer, error) {
	// Get the directory in which SSH keys are located.
	keydir := filepath.Join(os.Getenv("HOME"), ".ssh")

	// Select the key itself to use. When implementing more providers here,
	// please also add them to any SSH tests that are disabled because of signer
	// support.
	keyfile := ""
	switch provider {
	case "gce", "gke":
		keyfile = "google_compute_engine"
	default:
		return nil, fmt.Errorf("getSigner(...) not implemented for %s", provider)
	}
	key := filepath.Join(keydir, keyfile)
	Logf("Using SSH key: %s", key)

	// Create an actual signer.
	file, err := os.Open(key)
	if err != nil {
		return nil, fmt.Errorf("error opening SSH key %s: '%v'", key, err)
	}
	defer file.Close()
	buffer, err := ioutil.ReadAll(file)
	if err != nil {
		return nil, fmt.Errorf("error reading SSH key %s: '%v'", key, err)
	}
	signer, err := ssh.ParsePrivateKey(buffer)
	if err != nil {
		return nil, fmt.Errorf("error parsing SSH key %s: '%v'", key, err)
	}
	return signer, nil
}

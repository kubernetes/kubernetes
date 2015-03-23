/*
Copyright 2014 Google Inc. All rights reserved.

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
	"math/rand"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/clientauth"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	// Initial pod start can be delayed O(minutes) by slow docker pulls
	// TODO: Make this 30 seconds once #4566 is resolved.
	podStartTimeout = 5 * time.Minute
)

type testContextType struct {
	kubeConfig string
	authConfig string
	certDir    string
	host       string
	repoRoot   string
	provider   string
	gceConfig  GCEConfig
}

var testContext testContextType

func Logf(format string, a ...interface{}) {
	fmt.Fprintf(GinkgoWriter, "INFO: "+format+"\n", a...)
}

func Failf(format string, a ...interface{}) {
	Fail(fmt.Sprintf(format, a...), 1)
}

type podCondition func(pod *api.Pod) (bool, error)

func waitForPodCondition(c *client.Client, ns, podName, desc string, condition podCondition) error {
	By(fmt.Sprintf("waiting up to %v for pod %s status to be %s", podStartTimeout, podName, desc))
	for start := time.Now(); time.Since(start) < podStartTimeout; time.Sleep(5 * time.Second) {
		pod, err := c.Pods(ns).Get(podName)
		if err != nil {
			Logf("Get pod failed, ignoring for 5s: %v", err)
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
		ci, ok := pod.Status.Info[contName]
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
				Logf("Waiting for pod %q in namespace %s status to be success or failure", podName, namespace)
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
	case testContext.kubeConfig != "":
		fmt.Printf(">>> testContext.kubeConfig: %s\n", testContext.kubeConfig)
		c, err := clientcmd.LoadFromFile(testContext.kubeConfig)
		if err != nil {
			return nil, fmt.Errorf("error loading kubeConfig: %v", err.Error())
		}
		return clientcmd.NewDefaultClientConfig(*c, &clientcmd.ConfigOverrides{}).ClientConfig()
	case testContext.authConfig != "":
		config := &client.Config{
			Host: testContext.host,
		}
		info, err := clientauth.LoadFromFile(testContext.authConfig)
		if err != nil {
			return nil, fmt.Errorf("error loading authConfig: %v", err.Error())
		}
		// If the certificate directory is provided, set the cert paths to be there.
		if testContext.certDir != "" {
			Logf("Expecting certs in %v.", testContext.certDir)
			info.CAFile = filepath.Join(testContext.certDir, "ca.crt")
			info.CertFile = filepath.Join(testContext.certDir, "kubecfg.crt")
			info.KeyFile = filepath.Join(testContext.certDir, "kubecfg.key")
		}
		mergedConfig, err := info.MergeWithConfig(*config)
		return &mergedConfig, err
	default:
		return nil, fmt.Errorf("either kubeConfig or authConfig must be specified to load client config")
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
// "testname":  which gets bubbled up to the logging/failure messages if errors happen.
// "validator" function: This function is given a podID and a client, and it can do some specific validations that way.
func validateController(c *client.Client, image string, replicas int, containername string, testname string, validator validatorFn) {
	getPodsTemplate := "--template={{range.items}}{{.id}} {{end}}"
	// NB: kubectl adds the "exists" function to the standard template functions.
	// This lets us check to see if the "running" entry exists for each of the containers
	// we care about. Exists will never return an error and it's safe to check a chain of
	// things, any one of which may not exist. In the below template, all of info,
	// containername, and running might be nil, so the normal index function isn't very
	// helpful.
	// This template is unit-tested in kubectl, so if you change it, update the unit test.
	// You can read about the syntax here: http://golang.org/pkg/text/template/.
	getContainerStateTemplate := fmt.Sprintf(`--template={{and (exists . "currentState" "info" "%s" "state" "running")}}`, containername)

	getImageTemplate := fmt.Sprintf(`--template={{(index .currentState.info "%s").image}}`, containername)

	By(fmt.Sprintf("waiting for all containers in %s pods to come up.", testname)) //testname should be selector
	for start := time.Now(); time.Since(start) < podStartTimeout; time.Sleep(5 * time.Second) {
		getPodsOutput := runKubectl("get", "pods", "-o", "template", getPodsTemplate, "-l", testname)
		pods := strings.Fields(getPodsOutput)
		if numPods := len(pods); numPods != replicas {
			By(fmt.Sprintf("Replicas for %s: expected=%d actual=%d", testname, replicas, numPods))
			continue
		}
		var runningPods []string
		for _, podID := range pods {
			running := runKubectl("get", "pods", podID, "-o", "template", getContainerStateTemplate)
			if running == "false" {
				Logf("%s is created but not running", podID)
				continue
			}

			currentImage := runKubectl("get", "pods", podID, "-o", "template", getImageTemplate)
			if currentImage != image {
				Logf("%s is created but running wrong image; expected: %s, actual: %s", podID, image, currentImage)
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

func kubectlCmd(args ...string) *exec.Cmd {
	defaultArgs := []string{}
	if testContext.kubeConfig != "" {
		defaultArgs = append(defaultArgs, "--"+clientcmd.RecommendedConfigPathFlag+"="+testContext.kubeConfig)
	} else {
		defaultArgs = append(defaultArgs, "--"+clientcmd.FlagAuthPath+"="+testContext.authConfig)
		if testContext.certDir != "" {
			defaultArgs = append(defaultArgs,
				fmt.Sprintf("--certificate-authority=%s", filepath.Join(testContext.certDir, "ca.crt")),
				fmt.Sprintf("--client-certificate=%s", filepath.Join(testContext.certDir, "kubecfg.crt")),
				fmt.Sprintf("--client-key=%s", filepath.Join(testContext.certDir, "kubecfg.key")))
		}
	}
	kubectlArgs := append(defaultArgs, args...)
	// TODO: Remove this once gcloud writes a proper entry in the kubeconfig file.
	if testContext.provider == "gke" {
		kubectlArgs = append(kubectlArgs, "--server="+testContext.host)
	}
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

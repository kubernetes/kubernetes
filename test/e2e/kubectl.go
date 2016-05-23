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
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"mime/multipart"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/ghodss/yaml"

	"k8s.io/kubernetes/pkg/api"
	apierrs "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/unversioned"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/version"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	nautilusImage            = "gcr.io/google_containers/update-demo:nautilus"
	kittenImage              = "gcr.io/google_containers/update-demo:kitten"
	updateDemoSelector       = "name=update-demo"
	updateDemoContainer      = "update-demo"
	frontendSelector         = "app=guestbook,tier=frontend"
	redisMasterSelector      = "app=redis,role=master"
	redisSlaveSelector       = "app=redis,role=slave"
	goproxyContainer         = "goproxy"
	goproxyPodSelector       = "name=goproxy"
	netexecContainer         = "netexec"
	netexecPodSelector       = "name=netexec"
	kubectlProxyPort         = 8011
	guestbookStartupTimeout  = 10 * time.Minute
	guestbookResponseTimeout = 3 * time.Minute
	simplePodSelector        = "name=nginx"
	simplePodName            = "nginx"
	nginxDefaultOutput       = "Welcome to nginx!"
	simplePodPort            = 80
	runJobTimeout            = 5 * time.Minute
	nginxImage               = "gcr.io/google_containers/nginx:1.7.9"
)

var (
	proxyRegexp = regexp.MustCompile("Starting to serve on 127.0.0.1:([0-9]+)")

	// Extended pod logging options were introduced in #13780 (v1.1.0) so we don't expect tests
	// that rely on extended pod logging options to work on clusters before that.
	//
	// TODO(ihmccreery): remove once we don't care about v1.0 anymore, (tentatively in v1.3).
	extendedPodLogFilterVersion = version.MustParse("v1.1.0")

	// NodePorts were made optional in #12831 (v1.1.0) so we don't expect tests that used to
	// require NodePorts but no longer include them to work on clusters before that.
	//
	// TODO(ihmccreery): remove once we don't care about v1.0 anymore, (tentatively in v1.3).
	nodePortsOptionalVersion = version.MustParse("v1.1.0")

	// Jobs were introduced in v1.1, so we don't expect tests that rely on jobs to work on
	// clusters before that.
	//
	// TODO(ihmccreery): remove once we don't care about v1.0 anymore, (tentatively in v1.3).
	jobsVersion = version.MustParse("v1.1.0")

	// Deployments were introduced by default in v1.2, so we don't expect tests that rely on
	// deployments to work on clusters before that.
	//
	// TODO(ihmccreery): remove once we don't care about v1.1 anymore, (tentatively in v1.4).
	deploymentsVersion = version.MustParse("v1.2.0-alpha.7.726")

	// Pod probe parameters were introduced in #15967 (v1.2) so we dont expect tests that use
	// these probe parameters to work on clusters before that.
	//
	// TODO(ihmccreery): remove once we don't care about v1.1 anymore, (tentatively in v1.4).
	podProbeParametersVersion = version.MustParse("v1.2.0-alpha.4")
)

var _ = Describe("Kubectl client", func() {
	defer GinkgoRecover()
	framework := NewDefaultFramework("kubectl")
	var c *client.Client
	var ns string
	BeforeEach(func() {
		c = framework.Client
		ns = framework.Namespace.Name
	})

	Describe("Update Demo", func() {
		var updateDemoRoot, nautilusPath, kittenPath string
		BeforeEach(func() {
			updateDemoRoot = filepath.Join(testContext.RepoRoot, "docs/user-guide/update-demo")
			nautilusPath = filepath.Join(updateDemoRoot, "nautilus-rc.yaml")
			kittenPath = filepath.Join(updateDemoRoot, "kitten-rc.yaml")
		})
		It("should create and stop a replication controller [Conformance]", func() {
			defer cleanup(nautilusPath, ns, updateDemoSelector)

			By("creating a replication controller")
			runKubectlOrDie("create", "-f", nautilusPath, fmt.Sprintf("--namespace=%v", ns))
			validateController(c, nautilusImage, 2, "update-demo", updateDemoSelector, getUDData("nautilus.jpg", ns), ns)
		})

		It("should scale a replication controller [Conformance]", func() {
			defer cleanup(nautilusPath, ns, updateDemoSelector)

			By("creating a replication controller")
			runKubectlOrDie("create", "-f", nautilusPath, fmt.Sprintf("--namespace=%v", ns))
			validateController(c, nautilusImage, 2, "update-demo", updateDemoSelector, getUDData("nautilus.jpg", ns), ns)
			By("scaling down the replication controller")
			runKubectlOrDie("scale", "rc", "update-demo-nautilus", "--replicas=1", "--timeout=5m", fmt.Sprintf("--namespace=%v", ns))
			validateController(c, nautilusImage, 1, "update-demo", updateDemoSelector, getUDData("nautilus.jpg", ns), ns)
			By("scaling up the replication controller")
			runKubectlOrDie("scale", "rc", "update-demo-nautilus", "--replicas=2", "--timeout=5m", fmt.Sprintf("--namespace=%v", ns))
			validateController(c, nautilusImage, 2, "update-demo", updateDemoSelector, getUDData("nautilus.jpg", ns), ns)
		})

		It("should do a rolling update of a replication controller [Conformance]", func() {
			By("creating the initial replication controller")
			runKubectlOrDie("create", "-f", nautilusPath, fmt.Sprintf("--namespace=%v", ns))
			validateController(c, nautilusImage, 2, "update-demo", updateDemoSelector, getUDData("nautilus.jpg", ns), ns)
			By("rolling-update to new replication controller")
			runKubectlOrDie("rolling-update", "update-demo-nautilus", "--update-period=1s", "-f", kittenPath, fmt.Sprintf("--namespace=%v", ns))
			validateController(c, kittenImage, 2, "update-demo", updateDemoSelector, getUDData("kitten.jpg", ns), ns)
			// Everything will hopefully be cleaned up when the namespace is deleted.
		})
	})

	Describe("Guestbook application", func() {
		var guestbookPath string

		BeforeEach(func() {
			guestbookPath = filepath.Join(testContext.RepoRoot, "examples/guestbook")
		})

		It("should create and stop a working application [Conformance]", func() {
			SkipUnlessServerVersionGTE(deploymentsVersion, c)

			defer cleanup(guestbookPath, ns, frontendSelector, redisMasterSelector, redisSlaveSelector)

			By("creating all guestbook components")
			runKubectlOrDie("create", "-f", guestbookPath, fmt.Sprintf("--namespace=%v", ns))

			By("validating guestbook app")
			validateGuestbookApp(c, ns)
		})
	})

	Describe("Simple pod", func() {
		var podPath string

		BeforeEach(func() {
			podPath = filepath.Join(testContext.RepoRoot, "test", "e2e", "testing-manifests", "kubectl", "pod-with-readiness-probe.yaml")
			By(fmt.Sprintf("creating the pod from %v", podPath))
			runKubectlOrDie("create", "-f", podPath, fmt.Sprintf("--namespace=%v", ns))
			checkPodsRunningReady(c, ns, []string{simplePodName}, podStartTimeout)
		})
		AfterEach(func() {
			cleanup(podPath, ns, simplePodSelector)
		})

		It("should support exec", func() {
			By("executing a command in the container")
			execOutput := runKubectlOrDie("exec", fmt.Sprintf("--namespace=%v", ns), simplePodName, "echo", "running", "in", "container")
			if e, a := "running in container", strings.TrimSpace(execOutput); e != a {
				Failf("Unexpected kubectl exec output. Wanted %q, got %q", e, a)
			}

			By("executing a command in the container with noninteractive stdin")
			execOutput = newKubectlCommand("exec", fmt.Sprintf("--namespace=%v", ns), "-i", simplePodName, "cat").
				withStdinData("abcd1234").
				execOrDie()
			if e, a := "abcd1234", execOutput; e != a {
				Failf("Unexpected kubectl exec output. Wanted %q, got %q", e, a)
			}

			// pretend that we're a user in an interactive shell
			r, closer, err := newBlockingReader("echo hi\nexit\n")
			if err != nil {
				Failf("Error creating blocking reader: %v", err)
			}
			// NOTE this is solely for test cleanup!
			defer closer.Close()

			By("executing a command in the container with pseudo-interactive stdin")
			execOutput = newKubectlCommand("exec", fmt.Sprintf("--namespace=%v", ns), "-i", simplePodName, "bash").
				withStdinReader(r).
				execOrDie()
			if e, a := "hi", strings.TrimSpace(execOutput); e != a {
				Failf("Unexpected kubectl exec output. Wanted %q, got %q", e, a)
			}
		})

		It("should support exec through an HTTP proxy", func() {
			// Note: We are skipping local since we want to verify an apiserver with HTTPS.
			// At this time local only supports plain HTTP.
			SkipIfProviderIs("local")
			// Fail if the variable isn't set
			if testContext.Host == "" {
				Failf("--host variable must be set to the full URI to the api server on e2e run.")
			}

			// Make sure the apiServer is set to what kubectl requires
			apiServer := testContext.Host
			apiServerUrl, err := url.Parse(apiServer)
			if err != nil {
				Failf("Unable to parse URL %s. Error=%s", apiServer, err)
			}
			apiServerUrl.Scheme = "https"
			if !strings.Contains(apiServer, ":443") {
				apiServerUrl.Host = apiServerUrl.Host + ":443"
			}
			apiServer = apiServerUrl.String()

			// Build the static kubectl
			By("Finding a static kubectl for upload")
			testStaticKubectlPath, err := findBinary("kubectl", "linux/386")
			if err != nil {
				Logf("No kubectl found: %v.\nAttempting a local build...", err)
				// Fall back to trying to build a local static kubectl
				kubectlContainerPath := path.Join(testContext.RepoRoot, "/examples/kubectl-container/")
				if _, err := os.Stat(path.Join(testContext.RepoRoot, "hack/build-go.sh")); err != nil {
					Failf("Can't build static kubectl due to missing hack/build-go.sh. Error=%s", err)
				}
				By("Building a static kubectl for upload")
				staticKubectlBuild := exec.Command("make", "-C", kubectlContainerPath)
				if out, err := staticKubectlBuild.Output(); err != nil {
					Failf("Unable to create static kubectl. Error=%s, Output=%q", err, out)
				}
				// Verify the static kubectl path
				testStaticKubectlPath = path.Join(kubectlContainerPath, "kubectl")
				_, err := os.Stat(testStaticKubectlPath)
				if err != nil {
					Failf("static kubectl path could not be found in %s. Error=%s", testStaticKubectlPath, err)
				}
			}
			By(fmt.Sprintf("Using the kubectl in %s", testStaticKubectlPath))

			// Verify the kubeconfig path
			kubeConfigFilePath := testContext.KubeConfig
			_, err = os.Stat(kubeConfigFilePath)
			if err != nil {
				Failf("kube config path could not be accessed. Error=%s", err)
			}
			// start exec-proxy-tester container
			netexecPodPath := filepath.Join(testContext.RepoRoot, "test/images/netexec/pod.yaml")

			// Add "validate=false" if the server version is less than 1.2.
			// More details: https://github.com/kubernetes/kubernetes/issues/22884.
			validateFlag := "--validate=true"
			gte, err := serverVersionGTE(podProbeParametersVersion, c)
			if err != nil {
				Failf("Failed to get server version: %v", err)
			}
			if !gte {
				validateFlag = "--validate=false"
			}
			runKubectlOrDie("create", "-f", netexecPodPath, fmt.Sprintf("--namespace=%v", ns), validateFlag)
			checkPodsRunningReady(c, ns, []string{netexecContainer}, podStartTimeout)
			// Clean up
			defer cleanup(netexecPodPath, ns, netexecPodSelector)
			// Upload kubeconfig
			type NetexecOutput struct {
				Output string `json:"output"`
				Error  string `json:"error"`
			}

			var uploadConfigOutput NetexecOutput
			// Upload the kubeconfig file
			By("uploading kubeconfig to netexec")
			pipeConfigReader, postConfigBodyWriter, err := newStreamingUpload(kubeConfigFilePath)
			if err != nil {
				Failf("unable to create streaming upload. Error: %s", err)
			}

			subResourceProxyAvailable, err := serverVersionGTE(subResourcePodProxyVersion, c)
			if err != nil {
				Failf("Unable to determine server version.  Error: %s", err)
			}

			var resp []byte
			if subResourceProxyAvailable {
				resp, err = c.Post().
					Namespace(ns).
					Name("netexec").
					Resource("pods").
					SubResource("proxy").
					Suffix("upload").
					SetHeader("Content-Type", postConfigBodyWriter.FormDataContentType()).
					Body(pipeConfigReader).
					Do().Raw()
			} else {
				resp, err = c.Post().
					Prefix("proxy").
					Namespace(ns).
					Name("netexec").
					Resource("pods").
					Suffix("upload").
					SetHeader("Content-Type", postConfigBodyWriter.FormDataContentType()).
					Body(pipeConfigReader).
					Do().Raw()
			}
			if err != nil {
				Failf("Unable to upload kubeconfig to the remote exec server due to error: %s", err)
			}

			if err := json.Unmarshal(resp, &uploadConfigOutput); err != nil {
				Failf("Unable to read the result from the netexec server. Error: %s", err)
			}
			kubecConfigRemotePath := uploadConfigOutput.Output

			// Upload
			pipeReader, postBodyWriter, err := newStreamingUpload(testStaticKubectlPath)
			if err != nil {
				Failf("unable to create streaming upload. Error: %s", err)
			}

			By("uploading kubectl to netexec")
			var uploadOutput NetexecOutput
			// Upload the kubectl binary
			if subResourceProxyAvailable {
				resp, err = c.Post().
					Namespace(ns).
					Name("netexec").
					Resource("pods").
					SubResource("proxy").
					Suffix("upload").
					SetHeader("Content-Type", postBodyWriter.FormDataContentType()).
					Body(pipeReader).
					Do().Raw()
			} else {
				resp, err = c.Post().
					Prefix("proxy").
					Namespace(ns).
					Name("netexec").
					Resource("pods").
					Suffix("upload").
					SetHeader("Content-Type", postBodyWriter.FormDataContentType()).
					Body(pipeReader).
					Do().Raw()
			}
			if err != nil {
				Failf("Unable to upload kubectl binary to the remote exec server due to error: %s", err)
			}

			if err := json.Unmarshal(resp, &uploadOutput); err != nil {
				Failf("Unable to read the result from the netexec server. Error: %s", err)
			}
			uploadBinaryName := uploadOutput.Output
			// Verify that we got the expected response back in the body
			if !strings.HasPrefix(uploadBinaryName, "/uploads/") {
				Failf("Unable to upload kubectl binary to remote exec server. /uploads/ not in response. Response: %s", uploadBinaryName)
			}

			for _, proxyVar := range []string{"https_proxy", "HTTPS_PROXY"} {
				By("Running kubectl in netexec via an HTTP proxy using " + proxyVar)
				// start the proxy container
				goproxyPodPath := filepath.Join(testContext.RepoRoot, "test/images/goproxy/pod.yaml")
				runKubectlOrDie("create", "-f", goproxyPodPath, fmt.Sprintf("--namespace=%v", ns))
				checkPodsRunningReady(c, ns, []string{goproxyContainer}, podStartTimeout)

				// get the proxy address
				goproxyPod, err := c.Pods(ns).Get(goproxyContainer)
				if err != nil {
					Failf("Unable to get the goproxy pod. Error: %s", err)
				}
				proxyAddr := fmt.Sprintf("http://%s:8080", goproxyPod.Status.PodIP)

				shellCommand := fmt.Sprintf("%s=%s .%s --kubeconfig=%s --server=%s --namespace=%s exec nginx echo running in container",
					proxyVar, proxyAddr, uploadBinaryName, kubecConfigRemotePath, apiServer, ns)
				Logf("About to remote exec: %v", shellCommand)
				// Execute kubectl on remote exec server.
				var netexecShellOutput []byte
				if subResourceProxyAvailable {
					netexecShellOutput, err = c.Post().
						Namespace(ns).
						Name("netexec").
						Resource("pods").
						SubResource("proxy").
						Suffix("shell").
						Param("shellCommand", shellCommand).
						Do().Raw()
				} else {
					netexecShellOutput, err = c.Post().
						Prefix("proxy").
						Namespace(ns).
						Name("netexec").
						Resource("pods").
						Suffix("shell").
						Param("shellCommand", shellCommand).
						Do().Raw()
				}
				if err != nil {
					Failf("Unable to execute kubectl binary on the remote exec server due to error: %s", err)
				}

				var netexecOuput NetexecOutput
				if err := json.Unmarshal(netexecShellOutput, &netexecOuput); err != nil {
					Failf("Unable to read the result from the netexec server. Error: %s", err)
				}

				// Get (and print!) the proxy logs here, so
				// they'll be present in case the below check
				// fails the test, to help diagnose #19500 if
				// it recurs.
				proxyLog := runKubectlOrDie("log", "goproxy", fmt.Sprintf("--namespace=%v", ns))

				// Verify we got the normal output captured by the exec server
				expectedExecOutput := "running in container\n"
				if netexecOuput.Output != expectedExecOutput {
					Failf("Unexpected kubectl exec output. Wanted %q, got  %q", expectedExecOutput, netexecOuput.Output)
				}

				// Verify the proxy server logs saw the connection
				expectedProxyLog := fmt.Sprintf("Accepting CONNECT to %s", strings.TrimRight(strings.TrimLeft(testContext.Host, "https://"), "/api"))

				if !strings.Contains(proxyLog, expectedProxyLog) {
					Failf("Missing expected log result on proxy server for %s. Expected: %q, got %q", proxyVar, expectedProxyLog, proxyLog)
				}
				// Clean up the goproxyPod
				cleanup(goproxyPodPath, ns, goproxyPodSelector)
			}
		})

		It("should support inline execution and attach", func() {
			SkipUnlessServerVersionGTE(jobsVersion, c)

			nsFlag := fmt.Sprintf("--namespace=%v", ns)

			By("executing a command with run and attach with stdin")
			runOutput := newKubectlCommand(nsFlag, "run", "run-test", "--image=busybox", "--restart=Never", "--attach=true", "--stdin", "--", "sh", "-c", "cat && echo 'stdin closed'").
				withStdinData("abcd1234").
				execOrDie()
			Expect(runOutput).To(ContainSubstring("abcd1234"))
			Expect(runOutput).To(ContainSubstring("stdin closed"))
			Expect(c.Extensions().Jobs(ns).Delete("run-test", nil)).To(BeNil())

			By("executing a command with run and attach without stdin")
			runOutput = newKubectlCommand(fmt.Sprintf("--namespace=%v", ns), "run", "run-test-2", "--image=busybox", "--restart=Never", "--attach=true", "--leave-stdin-open=true", "--", "sh", "-c", "cat && echo 'stdin closed'").
				withStdinData("abcd1234").
				execOrDie()
			Expect(runOutput).ToNot(ContainSubstring("abcd1234"))
			Expect(runOutput).To(ContainSubstring("stdin closed"))
			Expect(c.Extensions().Jobs(ns).Delete("run-test-2", nil)).To(BeNil())

			By("executing a command with run and attach with stdin with open stdin should remain running")
			runOutput = newKubectlCommand(nsFlag, "run", "run-test-3", "--image=busybox", "--restart=Never", "--attach=true", "--leave-stdin-open=true", "--stdin", "--", "sh", "-c", "cat && echo 'stdin closed'").
				withStdinData("abcd1234\n").
				execOrDie()
			Expect(runOutput).ToNot(ContainSubstring("stdin closed"))
			runTestPod, _, err := util.GetFirstPod(c, ns, labels.SelectorFromSet(map[string]string{"run": "run-test-3"}))
			if err != nil {
				os.Exit(1)
			}
			if !checkPodsRunningReady(c, ns, []string{runTestPod.Name}, time.Minute) {
				Failf("Pod %q of Job %q should still be running", runTestPod.Name, "run-test-3")
			}

			// NOTE: we cannot guarantee our output showed up in the container logs before stdin was closed, so we have
			// to loop test.
			err = wait.PollImmediate(time.Second, time.Minute, func() (bool, error) {
				if !checkPodsRunningReady(c, ns, []string{runTestPod.Name}, 1*time.Second) {
					Failf("Pod %q of Job %q should still be running", runTestPod.Name, "run-test-3")
				}
				logOutput := runKubectlOrDie(nsFlag, "logs", runTestPod.Name)
				Expect(logOutput).ToNot(ContainSubstring("stdin closed"))
				return strings.Contains(logOutput, "abcd1234"), nil
			})
			if err != nil {
				os.Exit(1)
			}
			Expect(err).To(BeNil())

			Expect(c.Extensions().Jobs(ns).Delete("run-test-3", nil)).To(BeNil())
		})

		It("should support port-forward", func() {
			By("forwarding the container port to a local port")
			cmd, listenPort := runPortForward(ns, simplePodName, simplePodPort)
			defer tryKill(cmd)

			By("curling local port output")
			localAddr := fmt.Sprintf("http://localhost:%d", listenPort)
			body, err := curl(localAddr)
			Logf("got: %s", body)
			if err != nil {
				Failf("Failed http.Get of forwarded port (%s): %v", localAddr, err)
			}
			if !strings.Contains(body, nginxDefaultOutput) {
				Failf("Container port output missing expected value. Wanted:'%s', got: %s", nginxDefaultOutput, body)
			}
		})
	})

	Describe("Kubectl api-versions", func() {
		It("should check if v1 is in available api versions [Conformance]", func() {
			By("validating api verions")
			output := runKubectlOrDie("api-versions")
			if !strings.Contains(output, "v1") {
				Failf("No v1 in kubectl api-versions")
			}
		})
	})

	Describe("Kubectl apply", func() {
		It("should apply a new configuration to an existing RC", func() {
			mkpath := func(file string) string {
				return filepath.Join(testContext.RepoRoot, "examples/guestbook-go", file)
			}
			controllerJson := mkpath("redis-master-controller.json")
			nsFlag := fmt.Sprintf("--namespace=%v", ns)
			By("creating Redis RC")
			runKubectlOrDie("create", "-f", controllerJson, nsFlag)
			By("applying a modified configuration")
			stdin := modifyReplicationControllerConfiguration(controllerJson)
			newKubectlCommand("apply", "-f", "-", nsFlag).
				withStdinReader(stdin).
				execOrDie()
			By("checking the result")
			forEachReplicationController(c, ns, "app", "redis", validateReplicationControllerConfiguration)
		})
		It("should reuse nodePort when apply to an existing SVC", func() {
			mkpath := func(file string) string {
				return filepath.Join(testContext.RepoRoot, "examples/guestbook-go", file)
			}
			serviceJson := mkpath("redis-master-service.json")
			nsFlag := fmt.Sprintf("--namespace=%v", ns)
			By("creating Redis SVC")
			runKubectlOrDie("create", "-f", serviceJson, nsFlag)
			By("getting the original nodePort")
			originalNodePort := runKubectlOrDie("get", "service", "redis-master", nsFlag, "-o", "jsonpath={.spec.ports[0].nodePort}")
			By("applying the same configuration")
			runKubectlOrDie("apply", "-f", serviceJson, nsFlag)
			By("getting the nodePort after applying configuration")
			currentNodePort := runKubectlOrDie("get", "service", "redis-master", nsFlag, "-o", "jsonpath={.spec.ports[0].nodePort}")
			By("checking the result")
			if originalNodePort != currentNodePort {
				Failf("nodePort should keep the same")
			}
		})
	})

	Describe("Kubectl cluster-info", func() {
		It("should check if Kubernetes master services is included in cluster-info [Conformance]", func() {
			By("validating cluster-info")
			output := runKubectlOrDie("cluster-info")
			// Can't check exact strings due to terminal control commands (colors)
			requiredItems := []string{"Kubernetes master", "is running at"}
			if providerIs("gce", "gke") {
				requiredItems = append(requiredItems, "KubeDNS", "Heapster")
			}
			for _, item := range requiredItems {
				if !strings.Contains(output, item) {
					Failf("Missing %s in kubectl cluster-info", item)
				}
			}
		})
	})

	Describe("Kubectl describe", func() {
		It("should check if kubectl describe prints relevant information for rc and pods [Conformance]", func() {
			SkipUnlessServerVersionGTE(nodePortsOptionalVersion, c)

			mkpath := func(file string) string {
				return filepath.Join(testContext.RepoRoot, "examples/guestbook-go", file)
			}
			controllerJson := mkpath("redis-master-controller.json")
			serviceJson := mkpath("redis-master-service.json")

			nsFlag := fmt.Sprintf("--namespace=%v", ns)
			runKubectlOrDie("create", "-f", controllerJson, nsFlag)
			runKubectlOrDie("create", "-f", serviceJson, nsFlag)

			// Pod
			forEachPod(c, ns, "app", "redis", func(pod api.Pod) {
				output := runKubectlOrDie("describe", "pod", pod.Name, nsFlag)
				requiredStrings := [][]string{
					{"Name:", "redis-master-"},
					{"Namespace:", ns},
					{"Node:"},
					{"Labels:", "app=redis", "role=master"},
					{"Status:", "Running"},
					{"IP:"},
					{"Controllers:", "ReplicationController/redis-master"},
					{"Image:", "redis"},
					{"cpu:", "BestEffort"},
					{"State:", "Running"},
				}
				checkOutput(output, requiredStrings)
			})

			// Rc
			output := runKubectlOrDie("describe", "rc", "redis-master", nsFlag)
			requiredStrings := [][]string{
				{"Name:", "redis-master"},
				{"Namespace:", ns},
				{"Image(s):", "redis"},
				{"Selector:", "app=redis,role=master"},
				{"Labels:", "app=redis,role=master"},
				{"Replicas:", "1 current", "1 desired"},
				{"Pods Status:", "1 Running", "0 Waiting", "0 Succeeded", "0 Failed"},
				// {"Events:"} would ordinarily go in the list
				// here, but in some rare circumstances the
				// events are delayed, and instead kubectl
				// prints "No events." This string will match
				// either way.
				{"vents"}}
			checkOutput(output, requiredStrings)

			// Service
			output = runKubectlOrDie("describe", "service", "redis-master", nsFlag)
			requiredStrings = [][]string{
				{"Name:", "redis-master"},
				{"Namespace:", ns},
				{"Labels:", "app=redis", "role=master"},
				{"Selector:", "app=redis", "role=master"},
				{"Type:", "ClusterIP"},
				{"IP:"},
				{"Port:", "<unset>", "6379/TCP"},
				{"Endpoints:"},
				{"Session Affinity:", "None"}}
			checkOutput(output, requiredStrings)

			// Node
			// It should be OK to list unschedulable Nodes here.
			nodes, err := c.Nodes().List(api.ListOptions{})
			Expect(err).NotTo(HaveOccurred())
			node := nodes.Items[0]
			output = runKubectlOrDie("describe", "node", node.Name)
			requiredStrings = [][]string{
				{"Name:", node.Name},
				{"Labels:"},
				{"CreationTimestamp:"},
				{"Conditions:"},
				{"Type", "Status", "LastHeartbeatTime", "LastTransitionTime", "Reason", "Message"},
				{"Addresses:"},
				{"Capacity:"},
				{"Version:"},
				{"Kernel Version:"},
				{"OS Image:"},
				{"Container Runtime Version:"},
				{"Kubelet Version:"},
				{"Kube-Proxy Version:"},
				{"Pods:"}}
			checkOutput(output, requiredStrings)

			// Namespace
			output = runKubectlOrDie("describe", "namespace", ns)
			requiredStrings = [][]string{
				{"Name:", ns},
				{"Labels:"},
				{"Status:", "Active"}}
			checkOutput(output, requiredStrings)

			// Quota and limitrange are skipped for now.
		})
	})

	Describe("Kubectl expose", func() {
		It("should create services for rc [Conformance]", func() {
			mkpath := func(file string) string {
				return filepath.Join(testContext.RepoRoot, "examples/guestbook-go", file)
			}
			controllerJson := mkpath("redis-master-controller.json")
			nsFlag := fmt.Sprintf("--namespace=%v", ns)

			redisPort := 6379

			By("creating Redis RC")
			runKubectlOrDie("create", "-f", controllerJson, nsFlag)
			forEachPod(c, ns, "app", "redis", func(pod api.Pod) {
				lookForStringInLog(ns, pod.Name, "redis-master", "The server is now ready to accept connections", podStartTimeout)
			})
			validateService := func(name string, servicePort int, timeout time.Duration) {
				err := wait.Poll(poll, timeout, func() (bool, error) {
					endpoints, err := c.Endpoints(ns).Get(name)
					if err != nil {
						if apierrs.IsNotFound(err) {
							err = nil
						}
						Logf("Get endpoints failed (interval %v): %v", poll, err)
						return false, err
					}

					uidToPort := getContainerPortsByPodUID(endpoints)
					if len(uidToPort) == 0 {
						Logf("No endpoint found, retrying")
						return false, nil
					}
					if len(uidToPort) > 1 {
						Fail("Too many endpoints found")
					}
					for _, port := range uidToPort {
						if port[0] != redisPort {
							Failf("Wrong endpoint port: %d", port[0])
						}
					}
					return true, nil
				})
				Expect(err).NotTo(HaveOccurred())

				service, err := c.Services(ns).Get(name)
				Expect(err).NotTo(HaveOccurred())

				if len(service.Spec.Ports) != 1 {
					Failf("1 port is expected")
				}
				port := service.Spec.Ports[0]
				if port.Port != servicePort {
					Failf("Wrong service port: %d", port.Port)
				}
				if port.TargetPort.IntValue() != redisPort {
					Failf("Wrong target port: %d")
				}
			}

			By("exposing RC")
			runKubectlOrDie("expose", "rc", "redis-master", "--name=rm2", "--port=1234", fmt.Sprintf("--target-port=%d", redisPort), nsFlag)
			waitForService(c, ns, "rm2", true, poll, serviceStartTimeout)
			validateService("rm2", 1234, serviceStartTimeout)

			By("exposing service")
			runKubectlOrDie("expose", "service", "rm2", "--name=rm3", "--port=2345", fmt.Sprintf("--target-port=%d", redisPort), nsFlag)
			waitForService(c, ns, "rm3", true, poll, serviceStartTimeout)
			validateService("rm3", 2345, serviceStartTimeout)
		})
	})

	Describe("Kubectl label", func() {
		var podPath string
		var nsFlag string
		BeforeEach(func() {
			podPath = filepath.Join(testContext.RepoRoot, "docs/user-guide/pod.yaml")
			By("creating the pod")
			nsFlag = fmt.Sprintf("--namespace=%v", ns)
			runKubectlOrDie("create", "-f", podPath, nsFlag)
			checkPodsRunningReady(c, ns, []string{simplePodName}, podStartTimeout)
		})
		AfterEach(func() {
			cleanup(podPath, ns, simplePodSelector)
		})

		It("should update the label on a resource [Conformance]", func() {
			labelName := "testing-label"
			labelValue := "testing-label-value"

			By("adding the label " + labelName + " with value " + labelValue + " to a pod")
			runKubectlOrDie("label", "pods", simplePodName, labelName+"="+labelValue, nsFlag)
			By("verifying the pod has the label " + labelName + " with the value " + labelValue)
			output := runKubectlOrDie("get", "pod", simplePodName, "-L", labelName, nsFlag)
			if !strings.Contains(output, labelValue) {
				Failf("Failed updating label " + labelName + " to the pod " + simplePodName)
			}

			By("removing the label " + labelName + " of a pod")
			runKubectlOrDie("label", "pods", simplePodName, labelName+"-", nsFlag)
			By("verifying the pod doesn't have the label " + labelName)
			output = runKubectlOrDie("get", "pod", simplePodName, "-L", labelName, nsFlag)
			if strings.Contains(output, labelValue) {
				Failf("Failed removing label " + labelName + " of the pod " + simplePodName)
			}
		})
	})

	Describe("Kubectl logs", func() {
		var rcPath string
		var nsFlag string
		containerName := "redis-master"
		BeforeEach(func() {
			mkpath := func(file string) string {
				return filepath.Join(testContext.RepoRoot, "examples/guestbook-go", file)
			}
			rcPath = mkpath("redis-master-controller.json")
			By("creating an rc")
			nsFlag = fmt.Sprintf("--namespace=%v", ns)
			runKubectlOrDie("create", "-f", rcPath, nsFlag)
		})
		AfterEach(func() {
			cleanup(rcPath, ns, simplePodSelector)
		})

		It("should be able to retrieve and filter logs [Conformance]", func() {
			SkipUnlessServerVersionGTE(extendedPodLogFilterVersion, c)
			// Split("something\n", "\n") returns ["something", ""], so
			// strip trailing newline first
			lines := func(out string) []string {
				return strings.Split(strings.TrimRight(out, "\n"), "\n")
			}
			forEachPod(c, ns, "app", "redis", func(pod api.Pod) {
				By("checking for a matching strings")
				_, err := lookForStringInLog(ns, pod.Name, containerName, "The server is now ready to accept connections", podStartTimeout)
				Expect(err).NotTo(HaveOccurred())

				By("limiting log lines")
				out := runKubectlOrDie("log", pod.Name, containerName, nsFlag, "--tail=1")
				Expect(len(out)).NotTo(BeZero())
				Expect(len(lines(out))).To(Equal(1))

				By("limiting log bytes")
				out = runKubectlOrDie("log", pod.Name, containerName, nsFlag, "--limit-bytes=1")
				Expect(len(lines(out))).To(Equal(1))
				Expect(len(out)).To(Equal(1))

				By("exposing timestamps")
				out = runKubectlOrDie("log", pod.Name, containerName, nsFlag, "--tail=1", "--timestamps")
				l := lines(out)
				Expect(len(l)).To(Equal(1))
				words := strings.Split(l[0], " ")
				Expect(len(words)).To(BeNumerically(">", 1))
				if _, err := time.Parse(time.RFC3339Nano, words[0]); err != nil {
					if _, err := time.Parse(time.RFC3339, words[0]); err != nil {
						Failf("expected %q to be RFC3339 or RFC3339Nano", words[0])
					}
				}

				By("restricting to a time range")
				// Note: we must wait at least two seconds,
				// because the granularity is only 1 second and
				// it could end up rounding the wrong way.
				time.Sleep(2500 * time.Millisecond) // ensure that startup logs on the node are seen as older than 1s
				recent_out := runKubectlOrDie("log", pod.Name, containerName, nsFlag, "--since=1s")
				recent := len(strings.Split(recent_out, "\n"))
				older_out := runKubectlOrDie("log", pod.Name, containerName, nsFlag, "--since=24h")
				older := len(strings.Split(older_out, "\n"))
				Expect(recent).To(BeNumerically("<", older), "expected recent(%v) to be less than older(%v)\nrecent lines:\n%v\nolder lines:\n%v\n", recent, older, recent_out, older_out)
			})
		})
	})

	Describe("Kubectl patch", func() {
		It("should add annotations for pods in rc [Conformance]", func() {
			mkpath := func(file string) string {
				return filepath.Join(testContext.RepoRoot, "examples/guestbook-go", file)
			}
			controllerJson := mkpath("redis-master-controller.json")
			nsFlag := fmt.Sprintf("--namespace=%v", ns)
			By("creating Redis RC")
			runKubectlOrDie("create", "-f", controllerJson, nsFlag)
			By("patching all pods")
			forEachPod(c, ns, "app", "redis", func(pod api.Pod) {
				runKubectlOrDie("patch", "pod", pod.Name, nsFlag, "-p", "{\"metadata\":{\"annotations\":{\"x\":\"y\"}}}")
			})

			By("checking annotations")
			forEachPod(c, ns, "app", "redis", func(pod api.Pod) {
				found := false
				for key, val := range pod.Annotations {
					if key == "x" && val == "y" {
						found = true
					}
				}
				if !found {
					Failf("Added annotation not found")
				}
			})
		})
	})

	Describe("Kubectl version", func() {
		It("should check is all data is printed [Conformance]", func() {
			version := runKubectlOrDie("version")
			requiredItems := []string{"Client Version:", "Server Version:", "Major:", "Minor:", "GitCommit:"}
			for _, item := range requiredItems {
				if !strings.Contains(version, item) {
					Failf("Required item %s not found in %s", item, version)
				}
			}
		})
	})

	Describe("Kubectl run default", func() {
		var nsFlag string
		var name string

		var cleanUp func()

		BeforeEach(func() {
			nsFlag = fmt.Sprintf("--namespace=%v", ns)
			gte, err := serverVersionGTE(deploymentsVersion, c)
			if err != nil {
				Failf("Failed to get server version: %v", err)
			}
			if gte {
				name = "e2e-test-nginx-deployment"
				cleanUp = func() { runKubectlOrDie("delete", "deployment", name, nsFlag) }
			} else {
				name = "e2e-test-nginx-rc"
				cleanUp = func() { runKubectlOrDie("delete", "rc", name, nsFlag) }
			}
		})

		AfterEach(func() {
			cleanUp()
		})

		It("should create an rc or deployment from an image [Conformance]", func() {
			By("running the image " + nginxImage)
			runKubectlOrDie("run", name, "--image="+nginxImage, nsFlag)
			By("verifying the pod controlled by " + name + " gets created")
			label := labels.SelectorFromSet(labels.Set(map[string]string{"run": name}))
			podlist, err := waitForPodsWithLabel(c, ns, label)
			if err != nil {
				Failf("Failed getting pod controlled by %s: %v", name, err)
			}
			pods := podlist.Items
			if pods == nil || len(pods) != 1 || len(pods[0].Spec.Containers) != 1 || pods[0].Spec.Containers[0].Image != nginxImage {
				runKubectlOrDie("get", "pods", "-L", "run", nsFlag)
				Failf("Failed creating 1 pod with expected image %s. Number of pods = %v", nginxImage, len(pods))
			}
		})
	})

	Describe("Kubectl run rc", func() {
		var nsFlag string
		var rcName string

		BeforeEach(func() {
			nsFlag = fmt.Sprintf("--namespace=%v", ns)
			rcName = "e2e-test-nginx-rc"
		})

		AfterEach(func() {
			runKubectlOrDie("delete", "rc", rcName, nsFlag)
		})

		It("should create an rc from an image [Conformance]", func() {
			By("running the image " + nginxImage)
			runKubectlOrDie("run", rcName, "--image="+nginxImage, "--generator=run/v1", nsFlag)
			By("verifying the rc " + rcName + " was created")
			rc, err := c.ReplicationControllers(ns).Get(rcName)
			if err != nil {
				Failf("Failed getting rc %s: %v", rcName, err)
			}
			containers := rc.Spec.Template.Spec.Containers
			if containers == nil || len(containers) != 1 || containers[0].Image != nginxImage {
				Failf("Failed creating rc %s for 1 pod with expected image %s", rcName, nginxImage)
			}

			By("verifying the pod controlled by rc " + rcName + " was created")
			label := labels.SelectorFromSet(labels.Set(map[string]string{"run": rcName}))
			podlist, err := waitForPodsWithLabel(c, ns, label)
			if err != nil {
				Failf("Failed getting pod controlled by rc %s: %v", rcName, err)
			}
			pods := podlist.Items
			if pods == nil || len(pods) != 1 || len(pods[0].Spec.Containers) != 1 || pods[0].Spec.Containers[0].Image != nginxImage {
				runKubectlOrDie("get", "pods", "-L", "run", nsFlag)
				Failf("Failed creating 1 pod with expected image %s. Number of pods = %v", nginxImage, len(pods))
			}

			By("confirm that you can get logs from an rc")
			podNames := []string{}
			for _, pod := range pods {
				podNames = append(podNames, pod.Name)
			}
			if !checkPodsRunningReady(c, ns, podNames, podStartTimeout) {
				Failf("Pods for rc %s were not ready", rcName)
			}
			_, err = runKubectl("logs", "rc/"+rcName, nsFlag)
			// a non-nil error is fine as long as we actually found a pod.
			if err != nil && !strings.Contains(err.Error(), " in pod ") {
				Failf("Failed getting logs by rc %s: %v", rcName, err)
			}
		})
	})

	Describe("Kubectl rolling-update", func() {
		var nsFlag string
		var rcName string

		BeforeEach(func() {
			nsFlag = fmt.Sprintf("--namespace=%v", ns)
			rcName = "e2e-test-nginx-rc"
		})

		AfterEach(func() {
			runKubectlOrDie("delete", "rc", rcName, nsFlag)
		})

		It("should support rolling-update to same image [Conformance]", func() {
			By("running the image " + nginxImage)
			runKubectlOrDie("run", rcName, "--image="+nginxImage, "--generator=run/v1", nsFlag)
			By("verifying the rc " + rcName + " was created")
			rc, err := c.ReplicationControllers(ns).Get(rcName)
			if err != nil {
				Failf("Failed getting rc %s: %v", rcName, err)
			}
			containers := rc.Spec.Template.Spec.Containers
			if containers == nil || len(containers) != 1 || containers[0].Image != nginxImage {
				Failf("Failed creating rc %s for 1 pod with expected image %s", rcName, nginxImage)
			}

			By("rolling-update to same image controller")
			runKubectlOrDie("rolling-update", rcName, "--update-period=1s", "--image="+nginxImage, "--image-pull-policy="+string(api.PullIfNotPresent), nsFlag)
			validateController(c, nginxImage, 1, rcName, "run="+rcName, noOpValidatorFn, ns)
		})
	})

	Describe("Kubectl run deployment", func() {
		var nsFlag string
		var dName string

		BeforeEach(func() {
			nsFlag = fmt.Sprintf("--namespace=%v", ns)
			dName = "e2e-test-nginx-deployment"
		})

		AfterEach(func() {
			runKubectlOrDie("delete", "deployment", dName, nsFlag)
		})

		It("should create a deployment from an image [Conformance]", func() {
			SkipUnlessServerVersionGTE(deploymentsVersion, c)

			By("running the image " + nginxImage)
			runKubectlOrDie("run", dName, "--image="+nginxImage, "--generator=deployment/v1beta1", nsFlag)
			By("verifying the deployment " + dName + " was created")
			d, err := c.Extensions().Deployments(ns).Get(dName)
			if err != nil {
				Failf("Failed getting deployment %s: %v", dName, err)
			}
			containers := d.Spec.Template.Spec.Containers
			if containers == nil || len(containers) != 1 || containers[0].Image != nginxImage {
				Failf("Failed creating deployment %s for 1 pod with expected image %s", dName, nginxImage)
			}

			By("verifying the pod controlled by deployment " + dName + " was created")
			label := labels.SelectorFromSet(labels.Set(map[string]string{"run": dName}))
			podlist, err := waitForPodsWithLabel(c, ns, label)
			if err != nil {
				Failf("Failed getting pod controlled by deployment %s: %v", dName, err)
			}
			pods := podlist.Items
			if pods == nil || len(pods) != 1 || len(pods[0].Spec.Containers) != 1 || pods[0].Spec.Containers[0].Image != nginxImage {
				runKubectlOrDie("get", "pods", "-L", "run", nsFlag)
				Failf("Failed creating 1 pod with expected image %s. Number of pods = %v", nginxImage, len(pods))
			}
		})
	})

	Describe("Kubectl run job", func() {
		var nsFlag string
		var jobName string

		BeforeEach(func() {
			nsFlag = fmt.Sprintf("--namespace=%v", ns)
			jobName = "e2e-test-nginx-job"
		})

		AfterEach(func() {
			runKubectlOrDie("delete", "jobs", jobName, nsFlag)
		})

		It("should create a job from an image when restart is OnFailure [Conformance]", func() {
			SkipUnlessServerVersionGTE(jobsVersion, c)

			By("running the image " + nginxImage)
			runKubectlOrDie("run", jobName, "--restart=OnFailure", "--image="+nginxImage, nsFlag)
			By("verifying the job " + jobName + " was created")
			job, err := c.Extensions().Jobs(ns).Get(jobName)
			if err != nil {
				Failf("Failed getting job %s: %v", jobName, err)
			}
			containers := job.Spec.Template.Spec.Containers
			if containers == nil || len(containers) != 1 || containers[0].Image != nginxImage {
				Failf("Failed creating job %s for 1 pod with expected image %s", jobName, nginxImage)
			}
			if job.Spec.Template.Spec.RestartPolicy != api.RestartPolicyOnFailure {
				Failf("Failed creating a job with correct restart policy for --restart=OnFailure")
			}
		})

		It("should create a job from an image when restart is Never [Conformance]", func() {
			SkipUnlessServerVersionGTE(jobsVersion, c)

			By("running the image " + nginxImage)
			runKubectlOrDie("run", jobName, "--restart=Never", "--image="+nginxImage, nsFlag)
			By("verifying the job " + jobName + " was created")
			job, err := c.Extensions().Jobs(ns).Get(jobName)
			if err != nil {
				Failf("Failed getting job %s: %v", jobName, err)
			}
			containers := job.Spec.Template.Spec.Containers
			if containers == nil || len(containers) != 1 || containers[0].Image != nginxImage {
				Failf("Failed creating job %s for 1 pod with expected image %s", jobName, nginxImage)
			}
			if job.Spec.Template.Spec.RestartPolicy != api.RestartPolicyNever {
				Failf("Failed creating a job with correct restart policy for --restart=OnFailure")
			}
		})
	})

	Describe("Kubectl run --rm job", func() {
		nsFlag := fmt.Sprintf("--namespace=%v", ns)
		jobName := "e2e-test-rm-busybox-job"

		It("should create a job from an image, then delete the job [Conformance]", func() {
			SkipUnlessServerVersionGTE(jobsVersion, c)

			By("executing a command with run --rm and attach with stdin")
			t := time.NewTimer(runJobTimeout)
			defer t.Stop()
			runOutput := newKubectlCommand(nsFlag, "run", jobName, "--image=busybox", "--rm=true", "--restart=Never", "--attach=true", "--stdin", "--", "sh", "-c", "cat && echo 'stdin closed'").
				withStdinData("abcd1234").
				withTimeout(t.C).
				execOrDie()
			Expect(runOutput).To(ContainSubstring("abcd1234"))
			Expect(runOutput).To(ContainSubstring("stdin closed"))

			By("verifying the job " + jobName + " was deleted")
			_, err := c.Extensions().Jobs(ns).Get(jobName)
			Expect(err).To(HaveOccurred())
			Expect(apierrs.IsNotFound(err)).To(BeTrue())
		})
	})

	Describe("Proxy server", func() {
		// TODO: test proxy options (static, prefix, etc)
		It("should support proxy with --port 0 [Conformance]", func() {
			By("starting the proxy server")
			port, cmd, err := startProxyServer()
			if cmd != nil {
				defer tryKill(cmd)
			}
			if err != nil {
				Failf("Failed to start proxy server: %v", err)
			}
			By("curling proxy /api/ output")
			localAddr := fmt.Sprintf("http://localhost:%d/api/", port)
			apiVersions, err := getAPIVersions(localAddr)
			if err != nil {
				Failf("Expected at least one supported apiversion, got error %v", err)
			}
			if len(apiVersions.Versions) < 1 {
				Failf("Expected at least one supported apiversion, got %v", apiVersions)
			}
		})

		It("should support --unix-socket=/path [Conformance]", func() {
			By("Starting the proxy")
			tmpdir, err := ioutil.TempDir("", "kubectl-proxy-unix")
			if err != nil {
				Failf("Failed to create temporary directory: %v", err)
			}
			path := filepath.Join(tmpdir, "test")
			defer os.Remove(path)
			defer os.Remove(tmpdir)
			cmd := kubectlCmd("proxy", fmt.Sprintf("--unix-socket=%s", path))
			stdout, stderr, err := startCmdAndStreamOutput(cmd)
			if err != nil {
				Failf("Failed to start kubectl command: %v", err)
			}
			defer stdout.Close()
			defer stderr.Close()
			defer tryKill(cmd)
			buf := make([]byte, 128)
			if _, err = stdout.Read(buf); err != nil {
				Failf("Expected output from kubectl proxy: %v", err)
			}
			By("retrieving proxy /api/ output")
			_, err = curlUnix("http://unused/api", path)
			if err != nil {
				Failf("Failed get of /api at %s: %v", path, err)
			}
		})
	})
})

// Checks whether the output split by line contains the required elements.
func checkOutput(output string, required [][]string) {
	outputLines := strings.Split(output, "\n")
	currentLine := 0
	for _, requirement := range required {
		for currentLine < len(outputLines) && !strings.Contains(outputLines[currentLine], requirement[0]) {
			currentLine++
		}
		if currentLine == len(outputLines) {
			Failf("Failed to find %s in %s", requirement[0], output)
		}
		for _, item := range requirement[1:] {
			if !strings.Contains(outputLines[currentLine], item) {
				Failf("Failed to find %s in %s", item, outputLines[currentLine])
			}
		}
	}
}

func getAPIVersions(apiEndpoint string) (*unversioned.APIVersions, error) {
	body, err := curl(apiEndpoint)
	if err != nil {
		return nil, fmt.Errorf("Failed http.Get of %s: %v", apiEndpoint, err)
	}
	var apiVersions unversioned.APIVersions
	if err := json.Unmarshal([]byte(body), &apiVersions); err != nil {
		return nil, fmt.Errorf("Failed to parse /api output %s: %v", body, err)
	}
	return &apiVersions, nil
}

func startProxyServer() (int, *exec.Cmd, error) {
	// Specifying port 0 indicates we want the os to pick a random port.
	cmd := kubectlCmd("proxy", "-p", "0")
	stdout, stderr, err := startCmdAndStreamOutput(cmd)
	if err != nil {
		return -1, nil, err
	}
	defer stdout.Close()
	defer stderr.Close()
	buf := make([]byte, 128)
	var n int
	if n, err = stdout.Read(buf); err != nil {
		return -1, cmd, fmt.Errorf("Failed to read from kubectl proxy stdout: %v", err)
	}
	output := string(buf[:n])
	match := proxyRegexp.FindStringSubmatch(output)
	if len(match) == 2 {
		if port, err := strconv.Atoi(match[1]); err == nil {
			return port, cmd, nil
		}
	}
	return -1, cmd, fmt.Errorf("Failed to parse port from proxy stdout: %s", output)
}

func curlUnix(url string, path string) (string, error) {
	dial := func(proto, addr string) (net.Conn, error) {
		return net.Dial("unix", path)
	}
	transport := &http.Transport{
		Dial: dial,
	}
	return curlTransport(url, transport)
}

func curlTransport(url string, transport *http.Transport) (string, error) {
	client := &http.Client{Transport: transport}
	resp, err := client.Get(url)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	return string(body[:]), nil
}

func curl(url string) (string, error) {
	return curlTransport(url, &http.Transport{})
}

func validateGuestbookApp(c *client.Client, ns string) {
	Logf("Waiting for frontend to serve content.")
	if !waitForGuestbookResponse(c, "get", "", `{"data": ""}`, guestbookStartupTimeout, ns) {
		Failf("Frontend service did not start serving content in %v seconds.", guestbookStartupTimeout.Seconds())
	}

	Logf("Trying to add a new entry to the guestbook.")
	if !waitForGuestbookResponse(c, "set", "TestEntry", `{"message": "Updated"}`, guestbookResponseTimeout, ns) {
		Failf("Cannot added new entry in %v seconds.", guestbookResponseTimeout.Seconds())
	}

	Logf("Verifying that added entry can be retrieved.")
	if !waitForGuestbookResponse(c, "get", "", `{"data": "TestEntry"}`, guestbookResponseTimeout, ns) {
		Failf("Entry to guestbook wasn't correctly added in %v seconds.", guestbookResponseTimeout.Seconds())
	}
}

// Returns whether received expected response from guestbook on time.
func waitForGuestbookResponse(c *client.Client, cmd, arg, expectedResponse string, timeout time.Duration, ns string) bool {
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(5 * time.Second) {
		res, err := makeRequestToGuestbook(c, cmd, arg, ns)
		if err == nil && res == expectedResponse {
			return true
		}
		Logf("Failed to get response from guestbook. err: %v, response: %s", err, res)
	}
	return false
}

func makeRequestToGuestbook(c *client.Client, cmd, value string, ns string) (string, error) {
	proxyRequest, errProxy := getServicesProxyRequest(c, c.Get())
	if errProxy != nil {
		return "", errProxy
	}
	result, err := proxyRequest.Namespace(ns).
		Name("frontend").
		Suffix("/guestbook.php").
		Param("cmd", cmd).
		Param("key", "messages").
		Param("value", value).
		Do().
		Raw()
	return string(result), err
}

type updateDemoData struct {
	Image string
}

const applyTestLabel = "kubectl.kubernetes.io/apply-test"

func readBytesFromFile(filename string) []byte {
	file, err := os.Open(filename)
	if err != nil {
		Failf(err.Error())
	}

	data, err := ioutil.ReadAll(file)
	if err != nil {
		Failf(err.Error())
	}

	return data
}

func readReplicationControllerFromFile(filename string) *api.ReplicationController {
	data := readBytesFromFile(filename)
	rc := api.ReplicationController{}
	if err := yaml.Unmarshal(data, &rc); err != nil {
		Failf(err.Error())
	}

	return &rc
}

func modifyReplicationControllerConfiguration(filename string) io.Reader {
	rc := readReplicationControllerFromFile(filename)
	rc.Labels[applyTestLabel] = "ADDED"
	rc.Spec.Selector[applyTestLabel] = "ADDED"
	rc.Spec.Template.Labels[applyTestLabel] = "ADDED"
	data, err := json.Marshal(rc)
	if err != nil {
		Failf("json marshal failed: %s\n", err)
	}

	return bytes.NewReader(data)
}

func forEachReplicationController(c *client.Client, ns, selectorKey, selectorValue string, fn func(api.ReplicationController)) {
	var rcs *api.ReplicationControllerList
	var err error
	for t := time.Now(); time.Since(t) < podListTimeout; time.Sleep(poll) {
		label := labels.SelectorFromSet(labels.Set(map[string]string{selectorKey: selectorValue}))
		options := api.ListOptions{LabelSelector: label}
		rcs, err = c.ReplicationControllers(ns).List(options)
		Expect(err).NotTo(HaveOccurred())
		if len(rcs.Items) > 0 {
			break
		}
	}

	if rcs == nil || len(rcs.Items) == 0 {
		Failf("No replication controllers found")
	}

	for _, rc := range rcs.Items {
		fn(rc)
	}
}

func validateReplicationControllerConfiguration(rc api.ReplicationController) {
	if rc.Name == "redis-master" {
		if _, ok := rc.Annotations[kubectl.LastAppliedConfigAnnotation]; !ok {
			Failf("Annotation not found in modified configuration:\n%v\n", rc)
		}

		if value, ok := rc.Labels[applyTestLabel]; !ok || value != "ADDED" {
			Failf("Added label %s not found in modified configuration:\n%v\n", applyTestLabel, rc)
		}
	}
}

// getUDData creates a validator function based on the input string (i.e. kitten.jpg).
// For example, if you send "kitten.jpg", this function verifies that the image jpg = kitten.jpg
// in the container's json field.
func getUDData(jpgExpected string, ns string) func(*client.Client, string) error {

	// getUDData validates data.json in the update-demo (returns nil if data is ok).
	return func(c *client.Client, podID string) error {
		Logf("validating pod %s", podID)
		subResourceProxyAvailable, err := serverVersionGTE(subResourcePodProxyVersion, c)
		if err != nil {
			return err
		}
		var body []byte
		if subResourceProxyAvailable {
			body, err = c.Get().
				Namespace(ns).
				Resource("pods").
				SubResource("proxy").
				Name(podID).
				Suffix("data.json").
				Do().
				Raw()
		} else {
			body, err = c.Get().
				Prefix("proxy").
				Namespace(ns).
				Resource("pods").
				Name(podID).
				Suffix("data.json").
				Do().
				Raw()
		}
		if err != nil {
			return err
		}
		Logf("got data: %s", body)
		var data updateDemoData
		if err := json.Unmarshal(body, &data); err != nil {
			return err
		}
		Logf("Unmarshalled json jpg/img => %s , expecting %s .", data, jpgExpected)
		if strings.Contains(data.Image, jpgExpected) {
			return nil
		} else {
			return errors.New(fmt.Sprintf("data served up in container is inaccurate, %s didn't contain %s", data, jpgExpected))
		}
	}
}

func noOpValidatorFn(c *client.Client, podID string) error { return nil }

// newBlockingReader returns a reader that allows reading the given string,
// then blocks until Close() is called on the returned closer.
//
// We're explicitly returning the reader and closer separately, because
// the closer needs to be the *os.File we get from os.Pipe(). This is required
// so the exec of kubectl can pass the underlying file descriptor to the exec
// syscall, instead of creating another os.Pipe and blocking on the io.Copy
// between the source (e.g. stdin) and the write half of the pipe.
func newBlockingReader(s string) (io.Reader, io.Closer, error) {
	r, w, err := os.Pipe()
	if err != nil {
		return nil, nil, err
	}
	w.Write([]byte(s))
	return r, w, nil
}

// newStreamingUpload creates a new http.Request that will stream POST
// a file to a URI.
func newStreamingUpload(filePath string) (*io.PipeReader, *multipart.Writer, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, nil, err
	}

	r, w := io.Pipe()

	postBodyWriter := multipart.NewWriter(w)

	go streamingUpload(file, filepath.Base(filePath), postBodyWriter, w)
	return r, postBodyWriter, err
}

// streamingUpload streams a file via a pipe through a multipart.Writer.
// Generally one should use newStreamingUpload instead of calling this directly.
func streamingUpload(file *os.File, fileName string, postBodyWriter *multipart.Writer, w *io.PipeWriter) {
	defer GinkgoRecover()
	defer file.Close()
	defer w.Close()

	// Set up the form file
	fileWriter, err := postBodyWriter.CreateFormFile("file", fileName)
	if err != nil {
		Failf("Unable to to write file at %s to buffer. Error: %s", fileName, err)
	}

	// Copy kubectl binary into the file writer
	if _, err := io.Copy(fileWriter, file); err != nil {
		Failf("Unable to to copy file at %s into the file writer. Error: %s", fileName, err)
	}

	// Nothing more should be written to this instance of the postBodyWriter
	if err := postBodyWriter.Close(); err != nil {
		Failf("Unable to close the writer for file upload. Error: %s", err)
	}
}

var binPrefixes = []string{
	"_output/dockerized/bin",
	"_output/local/bin",
	"platforms",
}

// findBinary searches through likely paths to find the specified binary.  It
// takes the one that has been built most recently.  Platform should be
// specified as '<os>/<arch>'.  For example: 'linux/amd64'.
func findBinary(binName string, platform string) (string, error) {
	var binTime time.Time
	var binPath string

	for _, pre := range binPrefixes {
		tryPath := path.Join(testContext.RepoRoot, pre, platform, binName)
		fi, err := os.Stat(tryPath)
		if err != nil {
			continue
		}
		if fi.ModTime().After(binTime) {
			binPath = tryPath
			binTime = fi.ModTime()
		}
	}

	if len(binPath) > 0 {
		return binPath, nil
	}
	return binPath, fmt.Errorf("Could not find %v for %v", binName, platform)
}

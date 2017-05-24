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
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"mime/multipart"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/elazarl/goproxy"
	"github.com/ghodss/yaml"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/annotations"
	apierrs "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/unversioned"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic/registry"
	uexec "k8s.io/kubernetes/pkg/util/exec"
	utilnet "k8s.io/kubernetes/pkg/util/net"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/version"
	"k8s.io/kubernetes/test/e2e/framework"
	testutils "k8s.io/kubernetes/test/utils"

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
	pausePodSelector         = "name=pause"
	pausePodName             = "pause"
	runJobTimeout            = 5 * time.Minute
	busyboxImage             = "gcr.io/google_containers/busybox:1.24"
	nginxImage               = "gcr.io/google_containers/nginx-slim:0.7"
	newNginxImage            = "gcr.io/google_containers/nginx-slim:0.8"
	kubeCtlManifestPath      = "test/e2e/testing-manifests/kubectl"
	redisControllerFilename  = "redis-master-controller.json"
	redisServiceFilename     = "redis-master-service.json"
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

	// Pod probe parameters were introduced in #15967 (v1.2) so we don't expect tests that use
	// these probe parameters to work on clusters before that.
	//
	// TODO(ihmccreery): remove once we don't care about v1.1 anymore, (tentatively in v1.4).
	podProbeParametersVersion = version.MustParse("v1.2.0-alpha.4")

	// 'kubectl create quota' was introduced in #28351 (v1.4) so we don't expect tests that use
	// 'kubectl create quota' to work on kubectl clients before that.
	kubectlCreateQuotaVersion = version.MustParse("v1.4.0-alpha.2")

	// Returning container command exit codes in kubectl run/exec was introduced in #26541 (v1.4)
	// so we don't expect tests that verifies return code to work on kubectl clients before that.
	kubectlContainerExitCodeVersion = version.MustParse("v1.4.0-alpha.3")
)

// Stops everything from filePath from namespace ns and checks if everything matching selectors from the given namespace is correctly stopped.
// Aware of the kubectl example files map.
func cleanupKubectlInputs(fileContents string, ns string, selectors ...string) {
	By("using delete to clean up resources")
	var nsArg string
	if ns != "" {
		nsArg = fmt.Sprintf("--namespace=%s", ns)
	}
	// support backward compatibility : file paths or raw json - since we are removing file path
	// dependencies from this test.
	framework.RunKubectlOrDieInput(fileContents, "delete", "--grace-period=0", "--force", "-f", "-", nsArg)
	framework.AssertCleanup(ns, selectors...)
}

func readTestFileOrDie(file string) []byte {
	return framework.ReadOrDie(path.Join(kubeCtlManifestPath, file))
}

func runKubectlRetryOrDie(args ...string) string {
	var err error
	var output string
	for i := 0; i < 5; i++ {
		output, err = framework.RunKubectl(args...)
		if err == nil || (!strings.Contains(err.Error(), registry.OptimisticLockErrorMsg) && !strings.Contains(err.Error(), "Operation cannot be fulfilled")) {
			break
		}
		time.Sleep(time.Second)
	}
	// Expect no errors to be present after retries are finished
	// Copied from framework #ExecOrDie
	framework.Logf("stdout: %q", output)
	Expect(err).NotTo(HaveOccurred())
	return output
}

// duplicated setup to avoid polluting "normal" clients with alpha features which confuses the generated clients
var _ = framework.KubeDescribe("Kubectl alpha client", func() {
	defer GinkgoRecover()
	f := framework.NewDefaultGroupVersionFramework("kubectl", BatchV2Alpha1GroupVersion)

	var c clientset.Interface
	var ns string
	BeforeEach(func() {
		c = f.ClientSet
		ns = f.Namespace.Name
	})

	// Customized Wait  / ForEach wrapper for this test.  These demonstrate the

	framework.KubeDescribe("Kubectl run ScheduledJob", func() {
		var nsFlag string
		var sjName string

		BeforeEach(func() {
			nsFlag = fmt.Sprintf("--namespace=%v", ns)
			sjName = "e2e-test-echo-scheduledjob"
		})

		AfterEach(func() {
			framework.RunKubectlOrDie("delete", "cronjobs", sjName, nsFlag)
		})

		It("should create a ScheduledJob", func() {
			framework.SkipIfMissingResource(f.ClientPool, ScheduledJobGroupVersionResource, f.Namespace.Name)

			schedule := "*/5 * * * ?"
			framework.RunKubectlOrDie("run", sjName, "--restart=OnFailure", "--generator=scheduledjob/v2alpha1",
				"--schedule="+schedule, "--image="+busyboxImage, nsFlag)
			By("verifying the ScheduledJob " + sjName + " was created")
			sj, err := c.Batch().CronJobs(ns).Get(sjName)
			if err != nil {
				framework.Failf("Failed getting ScheduledJob %s: %v", sjName, err)
			}
			if sj.Spec.Schedule != schedule {
				framework.Failf("Failed creating a ScheduledJob with correct schedule %s", schedule)
			}
			containers := sj.Spec.JobTemplate.Spec.Template.Spec.Containers
			if containers == nil || len(containers) != 1 || containers[0].Image != busyboxImage {
				framework.Failf("Failed creating ScheduledJob %s for 1 pod with expected image %s: %#v", sjName, busyboxImage, containers)
			}
			if sj.Spec.JobTemplate.Spec.Template.Spec.RestartPolicy != api.RestartPolicyOnFailure {
				framework.Failf("Failed creating a ScheduledJob with correct restart policy for --restart=OnFailure")
			}
		})
	})

	framework.KubeDescribe("Kubectl run CronJob", func() {
		var nsFlag string
		var cjName string

		BeforeEach(func() {
			nsFlag = fmt.Sprintf("--namespace=%v", ns)
			cjName = "e2e-test-echo-cronjob"
		})

		AfterEach(func() {
			framework.RunKubectlOrDie("delete", "cronjobs", cjName, nsFlag)
		})

		It("should create a CronJob", func() {
			framework.SkipIfMissingResource(f.ClientPool, CronJobGroupVersionResource, f.Namespace.Name)

			schedule := "*/5 * * * ?"
			framework.RunKubectlOrDie("run", cjName, "--restart=OnFailure", "--generator=cronjob/v2alpha1",
				"--schedule="+schedule, "--image="+busyboxImage, nsFlag)
			By("verifying the CronJob " + cjName + " was created")
			sj, err := c.Batch().CronJobs(ns).Get(cjName)
			if err != nil {
				framework.Failf("Failed getting CronJob %s: %v", cjName, err)
			}
			if sj.Spec.Schedule != schedule {
				framework.Failf("Failed creating a CronJob with correct schedule %s", schedule)
			}
			containers := sj.Spec.JobTemplate.Spec.Template.Spec.Containers
			if containers == nil || len(containers) != 1 || containers[0].Image != busyboxImage {
				framework.Failf("Failed creating CronJob %s for 1 pod with expected image %s: %#v", cjName, busyboxImage, containers)
			}
			if sj.Spec.JobTemplate.Spec.Template.Spec.RestartPolicy != api.RestartPolicyOnFailure {
				framework.Failf("Failed creating a CronJob with correct restart policy for --restart=OnFailure")
			}
		})
	})
})

var _ = framework.KubeDescribe("Kubectl client", func() {
	defer GinkgoRecover()
	f := framework.NewDefaultFramework("kubectl")

	// Reustable cluster state function.  This won't be adversly affected by lazy initialization of framework.
	clusterState := func() *framework.ClusterVerification {
		return f.NewClusterVerification(
			framework.PodStateVerification{
				Selectors:   map[string]string{"app": "redis"},
				ValidPhases: []api.PodPhase{api.PodRunning /*api.PodPending*/},
			})
	}
	forEachPod := func(podFunc func(p api.Pod)) {
		clusterState().ForEach(podFunc)
	}
	var c clientset.Interface
	var ns string
	BeforeEach(func() {
		c = f.ClientSet
		ns = f.Namespace.Name
	})

	// Customized Wait  / ForEach wrapper for this test.  These demonstrate the
	// idiomatic way to wrap the ClusterVerification structs for syntactic sugar in large
	// test files.
	// Print debug info if atLeast Pods are not found before the timeout
	waitForOrFailWithDebug := func(atLeast int) {
		pods, err := clusterState().WaitFor(atLeast, framework.PodStartTimeout)
		if err != nil || len(pods) < atLeast {
			// TODO: Generalize integrating debug info into these tests so we always get debug info when we need it
			framework.DumpAllNamespaceInfo(c, f.ClientSet_1_5, ns)
			framework.Failf("Verified %v of %v pods , error : %v", len(pods), atLeast, err)
		}
	}

	framework.KubeDescribe("Update Demo", func() {
		var nautilus, kitten []byte
		BeforeEach(func() {
			updateDemoRoot := "test/fixtures/doc-yaml/user-guide/update-demo"
			nautilus = framework.ReadOrDie(filepath.Join(updateDemoRoot, "nautilus-rc.yaml"))
			kitten = framework.ReadOrDie(filepath.Join(updateDemoRoot, "kitten-rc.yaml"))
		})
		It("should create and stop a replication controller [Conformance]", func() {
			defer cleanupKubectlInputs(string(nautilus), ns, updateDemoSelector)

			By("creating a replication controller")
			framework.RunKubectlOrDieInput(string(nautilus[:]), "create", "-f", "-", fmt.Sprintf("--namespace=%v", ns))
			framework.ValidateController(c, nautilusImage, 2, "update-demo", updateDemoSelector, getUDData("nautilus.jpg", ns), ns)
		})

		It("should scale a replication controller [Conformance]", func() {
			defer cleanupKubectlInputs(string(nautilus[:]), ns, updateDemoSelector)

			By("creating a replication controller")
			framework.RunKubectlOrDieInput(string(nautilus[:]), "create", "-f", "-", fmt.Sprintf("--namespace=%v", ns))
			framework.ValidateController(c, nautilusImage, 2, "update-demo", updateDemoSelector, getUDData("nautilus.jpg", ns), ns)
			By("scaling down the replication controller")
			framework.RunKubectlOrDie("scale", "rc", "update-demo-nautilus", "--replicas=1", "--timeout=5m", fmt.Sprintf("--namespace=%v", ns))
			framework.ValidateController(c, nautilusImage, 1, "update-demo", updateDemoSelector, getUDData("nautilus.jpg", ns), ns)
			By("scaling up the replication controller")
			framework.RunKubectlOrDie("scale", "rc", "update-demo-nautilus", "--replicas=2", "--timeout=5m", fmt.Sprintf("--namespace=%v", ns))
			framework.ValidateController(c, nautilusImage, 2, "update-demo", updateDemoSelector, getUDData("nautilus.jpg", ns), ns)
		})

		It("should do a rolling update of a replication controller [Conformance]", func() {
			By("creating the initial replication controller")
			framework.RunKubectlOrDieInput(string(nautilus[:]), "create", "-f", "-", fmt.Sprintf("--namespace=%v", ns))
			framework.ValidateController(c, nautilusImage, 2, "update-demo", updateDemoSelector, getUDData("nautilus.jpg", ns), ns)
			By("rolling-update to new replication controller")
			framework.RunKubectlOrDieInput(string(kitten[:]), "rolling-update", "update-demo-nautilus", "--update-period=1s", "-f", "-", fmt.Sprintf("--namespace=%v", ns))
			framework.ValidateController(c, kittenImage, 2, "update-demo", updateDemoSelector, getUDData("kitten.jpg", ns), ns)
			// Everything will hopefully be cleaned up when the namespace is deleted.
		})
	})

	framework.KubeDescribe("Guestbook application", func() {
		forEachGBFile := func(run func(s string)) {
			for _, gbAppFile := range []string{
				"examples/guestbook/frontend-deployment.yaml",
				"examples/guestbook/frontend-service.yaml",
				"examples/guestbook/redis-master-deployment.yaml",
				"examples/guestbook/redis-master-service.yaml",
				"examples/guestbook/redis-slave-deployment.yaml",
				"examples/guestbook/redis-slave-service.yaml",
			} {
				contents := framework.ReadOrDie(gbAppFile)
				run(string(contents))
			}
		}

		It("should create and stop a working application [Conformance]", func() {
			framework.SkipUnlessServerVersionGTE(deploymentsVersion, c.Discovery())

			defer forEachGBFile(func(contents string) {
				cleanupKubectlInputs(contents, ns)
			})
			By("creating all guestbook components")
			forEachGBFile(func(contents string) {
				framework.Logf(contents)
				framework.RunKubectlOrDieInput(contents, "create", "-f", "-", fmt.Sprintf("--namespace=%v", ns))
			})

			By("validating guestbook app")
			validateGuestbookApp(c, ns)
		})
	})

	framework.KubeDescribe("Simple pod", func() {
		var podPath []byte

		BeforeEach(func() {
			podPath = framework.ReadOrDie(path.Join(kubeCtlManifestPath, "pod-with-readiness-probe.yaml"))
			By(fmt.Sprintf("creating the pod from %v", string(podPath)))
			framework.RunKubectlOrDieInput(string(podPath[:]), "create", "-f", "-", fmt.Sprintf("--namespace=%v", ns))
			Expect(framework.CheckPodsRunningReady(c, ns, []string{simplePodName}, framework.PodStartTimeout)).To(BeTrue())
		})
		AfterEach(func() {
			cleanupKubectlInputs(string(podPath[:]), ns, simplePodSelector)
		})

		It("should support exec", func() {
			By("executing a command in the container")
			execOutput := framework.RunKubectlOrDie("exec", fmt.Sprintf("--namespace=%v", ns), simplePodName, "echo", "running", "in", "container")
			if e, a := "running in container", strings.TrimSpace(execOutput); e != a {
				framework.Failf("Unexpected kubectl exec output. Wanted %q, got %q", e, a)
			}

			By("executing a command in the container with noninteractive stdin")
			execOutput = framework.NewKubectlCommand("exec", fmt.Sprintf("--namespace=%v", ns), "-i", simplePodName, "cat").
				WithStdinData("abcd1234").
				ExecOrDie()
			if e, a := "abcd1234", execOutput; e != a {
				framework.Failf("Unexpected kubectl exec output. Wanted %q, got %q", e, a)
			}

			// pretend that we're a user in an interactive shell
			r, closer, err := newBlockingReader("echo hi\nexit\n")
			if err != nil {
				framework.Failf("Error creating blocking reader: %v", err)
			}
			// NOTE this is solely for test cleanup!
			defer closer.Close()

			By("executing a command in the container with pseudo-interactive stdin")
			execOutput = framework.NewKubectlCommand("exec", fmt.Sprintf("--namespace=%v", ns), "-i", simplePodName, "bash").
				WithStdinReader(r).
				ExecOrDie()
			if e, a := "hi", strings.TrimSpace(execOutput); e != a {
				framework.Failf("Unexpected kubectl exec output. Wanted %q, got %q", e, a)
			}
		})

		It("should support exec through an HTTP proxy", func() {
			// Note: We are skipping local since we want to verify an apiserver with HTTPS.
			// At this time local only supports plain HTTP.
			framework.SkipIfProviderIs("local")
			// Fail if the variable isn't set
			if framework.TestContext.Host == "" {
				framework.Failf("--host variable must be set to the full URI to the api server on e2e run.")
			}

			By("Starting goproxy")
			testSrv, proxyLogs := startLocalProxy()
			defer testSrv.Close()
			proxyAddr := testSrv.URL

			for _, proxyVar := range []string{"https_proxy", "HTTPS_PROXY"} {
				proxyLogs.Reset()
				By("Running kubectl via an HTTP proxy using " + proxyVar)
				output := framework.NewKubectlCommand(fmt.Sprintf("--namespace=%s", ns), "exec", "nginx", "echo", "running", "in", "container").
					WithEnv(append(os.Environ(), fmt.Sprintf("%s=%s", proxyVar, proxyAddr))).
					ExecOrDie()

				// Verify we got the normal output captured by the exec server
				expectedExecOutput := "running in container\n"
				if output != expectedExecOutput {
					framework.Failf("Unexpected kubectl exec output. Wanted %q, got  %q", expectedExecOutput, output)
				}

				// Verify the proxy server logs saw the connection
				expectedProxyLog := fmt.Sprintf("Accepting CONNECT to %s", strings.TrimRight(strings.TrimLeft(framework.TestContext.Host, "https://"), "/api"))

				proxyLog := proxyLogs.String()
				if !strings.Contains(proxyLog, expectedProxyLog) {
					framework.Failf("Missing expected log result on proxy server for %s. Expected: %q, got %q", proxyVar, expectedProxyLog, proxyLog)
				}
			}
		})

		It("should return command exit codes", func() {
			framework.SkipUnlessKubectlVersionGTE(kubectlContainerExitCodeVersion)
			nsFlag := fmt.Sprintf("--namespace=%v", ns)

			By("execing into a container with a successful command")
			_, err := framework.NewKubectlCommand(nsFlag, "exec", "nginx", "--", "/bin/sh", "-c", "exit 0").Exec()
			ExpectNoError(err)

			By("execing into a container with a failing command")
			_, err = framework.NewKubectlCommand(nsFlag, "exec", "nginx", "--", "/bin/sh", "-c", "exit 42").Exec()
			ee, ok := err.(uexec.ExitError)
			Expect(ok).To(Equal(true))
			Expect(ee.ExitStatus()).To(Equal(42))

			By("running a successful command")
			_, err = framework.NewKubectlCommand(nsFlag, "run", "-i", "--image="+busyboxImage, "--restart=Never", "success", "--", "/bin/sh", "-c", "exit 0").Exec()
			ExpectNoError(err)

			By("running a failing command")
			_, err = framework.NewKubectlCommand(nsFlag, "run", "-i", "--image="+busyboxImage, "--restart=Never", "failure-1", "--", "/bin/sh", "-c", "exit 42").Exec()
			ee, ok = err.(uexec.ExitError)
			Expect(ok).To(Equal(true))
			Expect(ee.ExitStatus()).To(Equal(42))

			By("running a failing command without --restart=Never")
			_, err = framework.NewKubectlCommand(nsFlag, "run", "-i", "--image="+busyboxImage, "--restart=OnFailure", "failure-2", "--", "/bin/sh", "-c", "cat && exit 42").
				WithStdinData("abcd1234").
				Exec()
			ExpectNoError(err)

			By("running a failing command without --restart=Never, but with --rm")
			_, err = framework.NewKubectlCommand(nsFlag, "run", "-i", "--image="+busyboxImage, "--restart=OnFailure", "--rm", "failure-3", "--", "/bin/sh", "-c", "cat && exit 42").
				WithStdinData("abcd1234").
				Exec()
			ExpectNoError(err)
			framework.WaitForPodToDisappear(f.ClientSet, ns, "failure-3", labels.Everything(), 2*time.Second, wait.ForeverTestTimeout)

			By("running a failing command with --leave-stdin-open")
			_, err = framework.NewKubectlCommand(nsFlag, "run", "-i", "--image="+busyboxImage, "--restart=Never", "failure-4", "--leave-stdin-open", "--", "/bin/sh", "-c", "exit 42").
				WithStdinData("abcd1234").
				Exec()
			ExpectNoError(err)
		})

		It("should support inline execution and attach", func() {
			framework.SkipIfContainerRuntimeIs("rkt") // #23335
			framework.SkipUnlessServerVersionGTE(jobsVersion, c.Discovery())

			nsFlag := fmt.Sprintf("--namespace=%v", ns)

			By("executing a command with run and attach with stdin")
			runOutput := framework.NewKubectlCommand(nsFlag, "run", "run-test", "--image="+busyboxImage, "--restart=OnFailure", "--attach=true", "--stdin", "--", "sh", "-c", "cat && echo 'stdin closed'").
				WithStdinData("abcd1234").
				ExecOrDie()
			Expect(runOutput).To(ContainSubstring("abcd1234"))
			Expect(runOutput).To(ContainSubstring("stdin closed"))
			Expect(c.Batch().Jobs(ns).Delete("run-test", nil)).To(BeNil())

			By("executing a command with run and attach without stdin")
			runOutput = framework.NewKubectlCommand(fmt.Sprintf("--namespace=%v", ns), "run", "run-test-2", "--image="+busyboxImage, "--restart=OnFailure", "--attach=true", "--leave-stdin-open=true", "--", "sh", "-c", "cat && echo 'stdin closed'").
				WithStdinData("abcd1234").
				ExecOrDie()
			Expect(runOutput).ToNot(ContainSubstring("abcd1234"))
			Expect(runOutput).To(ContainSubstring("stdin closed"))
			Expect(c.Batch().Jobs(ns).Delete("run-test-2", nil)).To(BeNil())

			By("executing a command with run and attach with stdin with open stdin should remain running")
			runOutput = framework.NewKubectlCommand(nsFlag, "run", "run-test-3", "--image="+busyboxImage, "--restart=OnFailure", "--attach=true", "--leave-stdin-open=true", "--stdin", "--", "sh", "-c", "cat && echo 'stdin closed'").
				WithStdinData("abcd1234\n").
				ExecOrDie()
			Expect(runOutput).ToNot(ContainSubstring("stdin closed"))
			g := func(pods []*api.Pod) sort.Interface { return sort.Reverse(controller.ActivePods(pods)) }
			runTestPod, _, err := util.GetFirstPod(f.ClientSet.Core(), ns, labels.SelectorFromSet(map[string]string{"run": "run-test-3"}), 1*time.Minute, g)
			if err != nil {
				os.Exit(1)
			}
			if !framework.CheckPodsRunningReady(c, ns, []string{runTestPod.Name}, time.Minute) {
				framework.Failf("Pod %q of Job %q should still be running", runTestPod.Name, "run-test-3")
			}

			// NOTE: we cannot guarantee our output showed up in the container logs before stdin was closed, so we have
			// to loop test.
			err = wait.PollImmediate(time.Second, time.Minute, func() (bool, error) {
				if !framework.CheckPodsRunningReady(c, ns, []string{runTestPod.Name}, 1*time.Second) {
					framework.Failf("Pod %q of Job %q should still be running", runTestPod.Name, "run-test-3")
				}
				logOutput := framework.RunKubectlOrDie(nsFlag, "logs", runTestPod.Name)
				Expect(logOutput).ToNot(ContainSubstring("stdin closed"))
				return strings.Contains(logOutput, "abcd1234"), nil
			})
			if err != nil {
				os.Exit(1)
			}
			Expect(err).To(BeNil())

			Expect(c.Batch().Jobs(ns).Delete("run-test-3", nil)).To(BeNil())
		})

		It("should support port-forward", func() {
			By("forwarding the container port to a local port")
			cmd := runPortForward(ns, simplePodName, simplePodPort)
			defer cmd.Stop()

			By("curling local port output")
			localAddr := fmt.Sprintf("http://localhost:%d", cmd.port)
			body, err := curl(localAddr)
			framework.Logf("got: %s", body)
			if err != nil {
				framework.Failf("Failed http.Get of forwarded port (%s): %v", localAddr, err)
			}
			if !strings.Contains(body, nginxDefaultOutput) {
				framework.Failf("Container port output missing expected value. Wanted:'%s', got: %s", nginxDefaultOutput, body)
			}
		})
	})

	framework.KubeDescribe("Kubectl api-versions", func() {
		It("should check if v1 is in available api versions [Conformance]", func() {
			By("validating api verions")
			output := framework.RunKubectlOrDie("api-versions")
			if !strings.Contains(output, "v1") {
				framework.Failf("No v1 in kubectl api-versions")
			}
		})
	})

	framework.KubeDescribe("Kubectl apply", func() {
		It("should apply a new configuration to an existing RC", func() {
			controllerJson := readTestFileOrDie(redisControllerFilename)

			nsFlag := fmt.Sprintf("--namespace=%v", ns)
			By("creating Redis RC")
			framework.RunKubectlOrDieInput(string(controllerJson), "create", "-f", "-", nsFlag)
			By("applying a modified configuration")
			stdin := modifyReplicationControllerConfiguration(string(controllerJson))
			framework.NewKubectlCommand("apply", "-f", "-", nsFlag).
				WithStdinReader(stdin).
				ExecOrDie()
			By("checking the result")
			forEachReplicationController(c, ns, "app", "redis", validateReplicationControllerConfiguration)
		})
		It("should reuse port when apply to an existing SVC", func() {
			serviceJson := readTestFileOrDie(redisServiceFilename)
			nsFlag := fmt.Sprintf("--namespace=%v", ns)

			By("creating Redis SVC")
			framework.RunKubectlOrDieInput(string(serviceJson[:]), "create", "-f", "-", nsFlag)

			By("getting the original port")
			originalNodePort := framework.RunKubectlOrDie("get", "service", "redis-master", nsFlag, "-o", "jsonpath={.spec.ports[0].port}")

			By("applying the same configuration")
			framework.RunKubectlOrDieInput(string(serviceJson[:]), "apply", "-f", "-", nsFlag)

			By("getting the port after applying configuration")
			currentNodePort := framework.RunKubectlOrDie("get", "service", "redis-master", nsFlag, "-o", "jsonpath={.spec.ports[0].port}")

			By("checking the result")
			if originalNodePort != currentNodePort {
				framework.Failf("port should keep the same")
			}
		})
	})

	framework.KubeDescribe("Kubectl cluster-info", func() {
		It("should check if Kubernetes master services is included in cluster-info [Conformance]", func() {
			By("validating cluster-info")
			output := framework.RunKubectlOrDie("cluster-info")
			// Can't check exact strings due to terminal control commands (colors)
			requiredItems := []string{"Kubernetes master", "is running at"}
			if framework.ProviderIs("gce", "gke") {
				requiredItems = append(requiredItems, "KubeDNS", "Heapster")
			}
			for _, item := range requiredItems {
				if !strings.Contains(output, item) {
					framework.Failf("Missing %s in kubectl cluster-info", item)
				}
			}
		})
	})

	framework.KubeDescribe("Kubectl describe", func() {
		It("should check if kubectl describe prints relevant information for rc and pods [Conformance]", func() {
			framework.SkipUnlessServerVersionGTE(nodePortsOptionalVersion, c.Discovery())
			kv, err := framework.KubectlVersion()
			Expect(err).NotTo(HaveOccurred())
			framework.SkipUnlessServerVersionGTE(kv, c.Discovery())
			controllerJson := readTestFileOrDie(redisControllerFilename)
			serviceJson := readTestFileOrDie(redisServiceFilename)

			nsFlag := fmt.Sprintf("--namespace=%v", ns)
			framework.RunKubectlOrDieInput(string(controllerJson[:]), "create", "-f", "-", nsFlag)
			framework.RunKubectlOrDieInput(string(serviceJson[:]), "create", "-f", "-", nsFlag)

			By("Waiting for Redis master to start.")
			waitForOrFailWithDebug(1)

			// Pod
			forEachPod(func(pod api.Pod) {
				output := framework.RunKubectlOrDie("describe", "pod", pod.Name, nsFlag)
				requiredStrings := [][]string{
					{"Name:", "redis-master-"},
					{"Namespace:", ns},
					{"Node:"},
					{"Labels:", "app=redis"},
					{"role=master"},
					{"Status:", "Running"},
					{"IP:"},
					{"Controllers:", "ReplicationController/redis-master"},
					{"Image:", redisImage},
					{"State:", "Running"},
					{"QoS Class:", "BestEffort"},
				}
				checkOutput(output, requiredStrings)
			})

			// Rc
			output := framework.RunKubectlOrDie("describe", "rc", "redis-master", nsFlag)
			requiredStrings := [][]string{
				{"Name:", "redis-master"},
				{"Namespace:", ns},
				{"Image(s):", redisImage},
				{"Selector:", "app=redis,role=master"},
				{"Labels:", "app=redis"},
				{"role=master"},
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
			output = framework.RunKubectlOrDie("describe", "service", "redis-master", nsFlag)
			requiredStrings = [][]string{
				{"Name:", "redis-master"},
				{"Namespace:", ns},
				{"Labels:", "app=redis"},
				{"role=master"},
				{"Selector:", "app=redis", "role=master"},
				{"Type:", "ClusterIP"},
				{"IP:"},
				{"Port:", "<unset>", "6379/TCP"},
				{"Endpoints:"},
				{"Session Affinity:", "None"}}
			checkOutput(output, requiredStrings)

			// Node
			// It should be OK to list unschedulable Nodes here.
			nodes, err := c.Core().Nodes().List(api.ListOptions{})
			Expect(err).NotTo(HaveOccurred())
			node := nodes.Items[0]
			output = framework.RunKubectlOrDie("describe", "node", node.Name)
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
			output = framework.RunKubectlOrDie("describe", "namespace", ns)
			requiredStrings = [][]string{
				{"Name:", ns},
				{"Labels:"},
				{"Status:", "Active"}}
			checkOutput(output, requiredStrings)

			// Quota and limitrange are skipped for now.
		})
	})

	framework.KubeDescribe("Kubectl expose", func() {
		It("should create services for rc [Conformance]", func() {
			controllerJson := readTestFileOrDie(redisControllerFilename)
			nsFlag := fmt.Sprintf("--namespace=%v", ns)

			redisPort := 6379

			By("creating Redis RC")

			framework.Logf("namespace %v", ns)
			framework.RunKubectlOrDieInput(string(controllerJson[:]), "create", "-f", "-", nsFlag)

			// It may take a while for the pods to get registered in some cases, wait to be sure.
			By("Waiting for Redis master to start.")
			waitForOrFailWithDebug(1)
			forEachPod(func(pod api.Pod) {
				framework.Logf("wait on redis-master startup in %v ", ns)
				framework.LookForStringInLog(ns, pod.Name, "redis-master", "The server is now ready to accept connections", framework.PodStartTimeout)
			})
			validateService := func(name string, servicePort int, timeout time.Duration) {
				err := wait.Poll(framework.Poll, timeout, func() (bool, error) {
					endpoints, err := c.Core().Endpoints(ns).Get(name)
					if err != nil {
						// log the real error
						framework.Logf("Get endpoints failed (interval %v): %v", framework.Poll, err)

						// if the error is API not found or could not find default credentials or TLS handshake timeout, try again
						if apierrs.IsNotFound(err) ||
							apierrs.IsUnauthorized(err) ||
							apierrs.IsServerTimeout(err) {
							err = nil
						}
						return false, err
					}

					uidToPort := getContainerPortsByPodUID(endpoints)
					if len(uidToPort) == 0 {
						framework.Logf("No endpoint found, retrying")
						return false, nil
					}
					if len(uidToPort) > 1 {
						Fail("Too many endpoints found")
					}
					for _, port := range uidToPort {
						if port[0] != redisPort {
							framework.Failf("Wrong endpoint port: %d", port[0])
						}
					}
					return true, nil
				})
				Expect(err).NotTo(HaveOccurred())

				service, err := c.Core().Services(ns).Get(name)
				Expect(err).NotTo(HaveOccurred())

				if len(service.Spec.Ports) != 1 {
					framework.Failf("1 port is expected")
				}
				port := service.Spec.Ports[0]
				if port.Port != int32(servicePort) {
					framework.Failf("Wrong service port: %d", port.Port)
				}
				if port.TargetPort.IntValue() != redisPort {
					framework.Failf("Wrong target port: %d", port.TargetPort.IntValue())
				}
			}

			By("exposing RC")
			framework.RunKubectlOrDie("expose", "rc", "redis-master", "--name=rm2", "--port=1234", fmt.Sprintf("--target-port=%d", redisPort), nsFlag)
			framework.WaitForService(c, ns, "rm2", true, framework.Poll, framework.ServiceStartTimeout)
			validateService("rm2", 1234, framework.ServiceStartTimeout)

			By("exposing service")
			framework.RunKubectlOrDie("expose", "service", "rm2", "--name=rm3", "--port=2345", fmt.Sprintf("--target-port=%d", redisPort), nsFlag)
			framework.WaitForService(c, ns, "rm3", true, framework.Poll, framework.ServiceStartTimeout)
			validateService("rm3", 2345, framework.ServiceStartTimeout)
		})
	})

	framework.KubeDescribe("Kubectl label", func() {
		var pod []byte
		var nsFlag string
		BeforeEach(func() {
			pod = readTestFileOrDie("pause-pod.yaml")
			By("creating the pod")
			nsFlag = fmt.Sprintf("--namespace=%v", ns)
			framework.RunKubectlOrDieInput(string(pod), "create", "-f", "-", nsFlag)
			Expect(framework.CheckPodsRunningReady(c, ns, []string{pausePodName}, framework.PodStartTimeout)).To(BeTrue())
		})
		AfterEach(func() {
			cleanupKubectlInputs(string(pod[:]), ns, pausePodSelector)
		})

		It("should update the label on a resource [Conformance]", func() {
			labelName := "testing-label"
			labelValue := "testing-label-value"

			By("adding the label " + labelName + " with value " + labelValue + " to a pod")
			framework.RunKubectlOrDie("label", "pods", pausePodName, labelName+"="+labelValue, nsFlag)
			By("verifying the pod has the label " + labelName + " with the value " + labelValue)
			output := framework.RunKubectlOrDie("get", "pod", pausePodName, "-L", labelName, nsFlag)
			if !strings.Contains(output, labelValue) {
				framework.Failf("Failed updating label " + labelName + " to the pod " + pausePodName)
			}

			By("removing the label " + labelName + " of a pod")
			framework.RunKubectlOrDie("label", "pods", pausePodName, labelName+"-", nsFlag)
			By("verifying the pod doesn't have the label " + labelName)
			output = framework.RunKubectlOrDie("get", "pod", pausePodName, "-L", labelName, nsFlag)
			if strings.Contains(output, labelValue) {
				framework.Failf("Failed removing label " + labelName + " of the pod " + pausePodName)
			}
		})
	})

	framework.KubeDescribe("Kubectl logs", func() {
		var rc []byte
		var nsFlag string
		containerName := "redis-master"
		BeforeEach(func() {
			rc = readTestFileOrDie(redisControllerFilename)
			By("creating an rc")
			nsFlag = fmt.Sprintf("--namespace=%v", ns)
			framework.RunKubectlOrDieInput(string(rc[:]), "create", "-f", "-", nsFlag)
		})
		AfterEach(func() {
			cleanupKubectlInputs(string(rc[:]), ns, simplePodSelector)
		})

		It("should be able to retrieve and filter logs [Conformance]", func() {
			framework.SkipUnlessServerVersionGTE(extendedPodLogFilterVersion, c.Discovery())

			// Split("something\n", "\n") returns ["something", ""], so
			// strip trailing newline first
			lines := func(out string) []string {
				return strings.Split(strings.TrimRight(out, "\n"), "\n")
			}

			By("Waiting for Redis master to start.")
			waitForOrFailWithDebug(1)
			forEachPod(func(pod api.Pod) {
				By("checking for a matching strings")
				_, err := framework.LookForStringInLog(ns, pod.Name, containerName, "The server is now ready to accept connections", framework.PodStartTimeout)
				Expect(err).NotTo(HaveOccurred())

				By("limiting log lines")
				out := framework.RunKubectlOrDie("log", pod.Name, containerName, nsFlag, "--tail=1")
				Expect(len(out)).NotTo(BeZero())
				Expect(len(lines(out))).To(Equal(1))

				By("limiting log bytes")
				out = framework.RunKubectlOrDie("log", pod.Name, containerName, nsFlag, "--limit-bytes=1")
				Expect(len(lines(out))).To(Equal(1))
				Expect(len(out)).To(Equal(1))

				By("exposing timestamps")
				out = framework.RunKubectlOrDie("log", pod.Name, containerName, nsFlag, "--tail=1", "--timestamps")
				l := lines(out)
				Expect(len(l)).To(Equal(1))
				words := strings.Split(l[0], " ")
				Expect(len(words)).To(BeNumerically(">", 1))
				if _, err := time.Parse(time.RFC3339Nano, words[0]); err != nil {
					if _, err := time.Parse(time.RFC3339, words[0]); err != nil {
						framework.Failf("expected %q to be RFC3339 or RFC3339Nano", words[0])
					}
				}

				By("restricting to a time range")
				// Note: we must wait at least two seconds,
				// because the granularity is only 1 second and
				// it could end up rounding the wrong way.
				time.Sleep(2500 * time.Millisecond) // ensure that startup logs on the node are seen as older than 1s
				recent_out := framework.RunKubectlOrDie("log", pod.Name, containerName, nsFlag, "--since=1s")
				recent := len(strings.Split(recent_out, "\n"))
				older_out := framework.RunKubectlOrDie("log", pod.Name, containerName, nsFlag, "--since=24h")
				older := len(strings.Split(older_out, "\n"))
				Expect(recent).To(BeNumerically("<", older), "expected recent(%v) to be less than older(%v)\nrecent lines:\n%v\nolder lines:\n%v\n", recent, older, recent_out, older_out)
			})
		})
	})

	framework.KubeDescribe("Kubectl patch", func() {
		It("should add annotations for pods in rc [Conformance]", func() {
			controllerJson := readTestFileOrDie(redisControllerFilename)
			nsFlag := fmt.Sprintf("--namespace=%v", ns)
			By("creating Redis RC")
			framework.RunKubectlOrDieInput(string(controllerJson[:]), "create", "-f", "-", nsFlag)
			By("Waiting for Redis master to start.")
			waitForOrFailWithDebug(1)
			By("patching all pods")
			forEachPod(func(pod api.Pod) {
				framework.RunKubectlOrDie("patch", "pod", pod.Name, nsFlag, "-p", "{\"metadata\":{\"annotations\":{\"x\":\"y\"}}}")
			})

			By("checking annotations")
			forEachPod(func(pod api.Pod) {
				found := false
				for key, val := range pod.Annotations {
					if key == "x" && val == "y" {
						found = true
						break
					}
				}
				if !found {
					framework.Failf("Added annotation not found")
				}
			})
		})
	})

	framework.KubeDescribe("Kubectl version", func() {
		It("should check is all data is printed [Conformance]", func() {
			version := framework.RunKubectlOrDie("version")
			requiredItems := []string{"Client Version:", "Server Version:", "Major:", "Minor:", "GitCommit:"}
			for _, item := range requiredItems {
				if !strings.Contains(version, item) {
					framework.Failf("Required item %s not found in %s", item, version)
				}
			}
		})
	})

	framework.KubeDescribe("Kubectl run default", func() {
		var nsFlag string
		var name string

		var cleanUp func()

		BeforeEach(func() {
			nsFlag = fmt.Sprintf("--namespace=%v", ns)
			gte, err := framework.ServerVersionGTE(deploymentsVersion, c.Discovery())
			if err != nil {
				framework.Failf("Failed to get server version: %v", err)
			}
			if gte {
				name = "e2e-test-nginx-deployment"
				cleanUp = func() { framework.RunKubectlOrDie("delete", "deployment", name, nsFlag) }
			} else {
				name = "e2e-test-nginx-rc"
				cleanUp = func() { framework.RunKubectlOrDie("delete", "rc", name, nsFlag) }
			}
		})

		AfterEach(func() {
			cleanUp()
		})

		It("should create an rc or deployment from an image [Conformance]", func() {
			By("running the image " + nginxImage)
			framework.RunKubectlOrDie("run", name, "--image="+nginxImage, nsFlag)
			By("verifying the pod controlled by " + name + " gets created")
			label := labels.SelectorFromSet(labels.Set(map[string]string{"run": name}))
			podlist, err := framework.WaitForPodsWithLabel(c, ns, label)
			if err != nil {
				framework.Failf("Failed getting pod controlled by %s: %v", name, err)
			}
			pods := podlist.Items
			if pods == nil || len(pods) != 1 || len(pods[0].Spec.Containers) != 1 || pods[0].Spec.Containers[0].Image != nginxImage {
				framework.RunKubectlOrDie("get", "pods", "-L", "run", nsFlag)
				framework.Failf("Failed creating 1 pod with expected image %s. Number of pods = %v", nginxImage, len(pods))
			}
		})
	})

	framework.KubeDescribe("Kubectl run rc", func() {
		var nsFlag string
		var rcName string

		BeforeEach(func() {
			nsFlag = fmt.Sprintf("--namespace=%v", ns)
			rcName = "e2e-test-nginx-rc"
		})

		AfterEach(func() {
			framework.RunKubectlOrDie("delete", "rc", rcName, nsFlag)
		})

		It("should create an rc from an image [Conformance]", func() {
			By("running the image " + nginxImage)
			framework.RunKubectlOrDie("run", rcName, "--image="+nginxImage, "--generator=run/v1", nsFlag)
			By("verifying the rc " + rcName + " was created")
			rc, err := c.Core().ReplicationControllers(ns).Get(rcName)
			if err != nil {
				framework.Failf("Failed getting rc %s: %v", rcName, err)
			}
			containers := rc.Spec.Template.Spec.Containers
			if containers == nil || len(containers) != 1 || containers[0].Image != nginxImage {
				framework.Failf("Failed creating rc %s for 1 pod with expected image %s", rcName, nginxImage)
			}

			By("verifying the pod controlled by rc " + rcName + " was created")
			label := labels.SelectorFromSet(labels.Set(map[string]string{"run": rcName}))
			podlist, err := framework.WaitForPodsWithLabel(c, ns, label)
			if err != nil {
				framework.Failf("Failed getting pod controlled by rc %s: %v", rcName, err)
			}
			pods := podlist.Items
			if pods == nil || len(pods) != 1 || len(pods[0].Spec.Containers) != 1 || pods[0].Spec.Containers[0].Image != nginxImage {
				framework.RunKubectlOrDie("get", "pods", "-L", "run", nsFlag)
				framework.Failf("Failed creating 1 pod with expected image %s. Number of pods = %v", nginxImage, len(pods))
			}

			By("confirm that you can get logs from an rc")
			podNames := []string{}
			for _, pod := range pods {
				podNames = append(podNames, pod.Name)
			}
			if !framework.CheckPodsRunningReady(c, ns, podNames, framework.PodStartTimeout) {
				framework.Failf("Pods for rc %s were not ready", rcName)
			}
			_, err = framework.RunKubectl("logs", "rc/"+rcName, nsFlag)
			// a non-nil error is fine as long as we actually found a pod.
			if err != nil && !strings.Contains(err.Error(), " in pod ") {
				framework.Failf("Failed getting logs by rc %s: %v", rcName, err)
			}
		})
	})

	framework.KubeDescribe("Kubectl rolling-update", func() {
		var nsFlag string
		var rcName string
		var c clientset.Interface

		BeforeEach(func() {
			c = f.ClientSet
			nsFlag = fmt.Sprintf("--namespace=%v", ns)
			rcName = "e2e-test-nginx-rc"
		})

		AfterEach(func() {
			framework.RunKubectlOrDie("delete", "rc", rcName, nsFlag)
		})

		It("should support rolling-update to same image [Conformance]", func() {
			By("running the image " + nginxImage)
			framework.RunKubectlOrDie("run", rcName, "--image="+nginxImage, "--generator=run/v1", nsFlag)
			By("verifying the rc " + rcName + " was created")
			rc, err := c.Core().ReplicationControllers(ns).Get(rcName)
			if err != nil {
				framework.Failf("Failed getting rc %s: %v", rcName, err)
			}
			containers := rc.Spec.Template.Spec.Containers
			if containers == nil || len(containers) != 1 || containers[0].Image != nginxImage {
				framework.Failf("Failed creating rc %s for 1 pod with expected image %s", rcName, nginxImage)
			}
			framework.WaitForRCToStabilize(c, ns, rcName, framework.PodStartTimeout)

			By("rolling-update to same image controller")

			runKubectlRetryOrDie("rolling-update", rcName, "--update-period=1s", "--image="+nginxImage, "--image-pull-policy="+string(api.PullIfNotPresent), nsFlag)
			framework.ValidateController(c, nginxImage, 1, rcName, "run="+rcName, noOpValidatorFn, ns)
		})
	})

	framework.KubeDescribe("Kubectl run deployment", func() {
		var nsFlag string
		var dName string

		BeforeEach(func() {
			nsFlag = fmt.Sprintf("--namespace=%v", ns)
			dName = "e2e-test-nginx-deployment"
		})

		AfterEach(func() {
			err := wait.Poll(framework.Poll, 2*time.Minute, func() (bool, error) {
				out, err := framework.RunKubectl("delete", "deployment", dName, nsFlag)
				if err != nil {
					if strings.Contains(err.Error(), "could not find default credentials") {
						err = nil
					}
					return false, fmt.Errorf("kubectl delete failed output: %s, err: %v", out, err)
				}
				return true, nil
			})
			Expect(err).NotTo(HaveOccurred())
		})

		It("should create a deployment from an image [Conformance]", func() {
			framework.SkipUnlessServerVersionGTE(deploymentsVersion, c.Discovery())

			By("running the image " + nginxImage)
			framework.RunKubectlOrDie("run", dName, "--image="+nginxImage, "--generator=deployment/v1beta1", nsFlag)
			By("verifying the deployment " + dName + " was created")
			d, err := c.Extensions().Deployments(ns).Get(dName)
			if err != nil {
				framework.Failf("Failed getting deployment %s: %v", dName, err)
			}
			containers := d.Spec.Template.Spec.Containers
			if containers == nil || len(containers) != 1 || containers[0].Image != nginxImage {
				framework.Failf("Failed creating deployment %s for 1 pod with expected image %s", dName, nginxImage)
			}

			By("verifying the pod controlled by deployment " + dName + " was created")
			label := labels.SelectorFromSet(labels.Set(map[string]string{"run": dName}))
			podlist, err := framework.WaitForPodsWithLabel(c, ns, label)
			if err != nil {
				framework.Failf("Failed getting pod controlled by deployment %s: %v", dName, err)
			}
			pods := podlist.Items
			if pods == nil || len(pods) != 1 || len(pods[0].Spec.Containers) != 1 || pods[0].Spec.Containers[0].Image != nginxImage {
				framework.RunKubectlOrDie("get", "pods", "-L", "run", nsFlag)
				framework.Failf("Failed creating 1 pod with expected image %s. Number of pods = %v", nginxImage, len(pods))
			}
		})
	})

	framework.KubeDescribe("Kubectl run job", func() {
		var nsFlag string
		var jobName string

		BeforeEach(func() {
			nsFlag = fmt.Sprintf("--namespace=%v", ns)
			jobName = "e2e-test-nginx-job"
		})

		AfterEach(func() {
			framework.RunKubectlOrDie("delete", "jobs", jobName, nsFlag)
		})

		It("should create a job from an image when restart is OnFailure [Conformance]", func() {
			framework.SkipUnlessServerVersionGTE(jobsVersion, c.Discovery())

			By("running the image " + nginxImage)
			framework.RunKubectlOrDie("run", jobName, "--restart=OnFailure", "--generator=job/v1", "--image="+nginxImage, nsFlag)
			By("verifying the job " + jobName + " was created")
			job, err := c.Batch().Jobs(ns).Get(jobName)
			if err != nil {
				framework.Failf("Failed getting job %s: %v", jobName, err)
			}
			containers := job.Spec.Template.Spec.Containers
			if containers == nil || len(containers) != 1 || containers[0].Image != nginxImage {
				framework.Failf("Failed creating job %s for 1 pod with expected image %s: %#v", jobName, nginxImage, containers)
			}
			if job.Spec.Template.Spec.RestartPolicy != api.RestartPolicyOnFailure {
				framework.Failf("Failed creating a job with correct restart policy for --restart=OnFailure")
			}
		})
	})

	framework.KubeDescribe("Kubectl run pod", func() {
		var nsFlag string
		var podName string

		BeforeEach(func() {
			nsFlag = fmt.Sprintf("--namespace=%v", ns)
			podName = "e2e-test-nginx-pod"
		})

		AfterEach(func() {
			framework.RunKubectlOrDie("delete", "pods", podName, nsFlag)
		})

		It("should create a pod from an image when restart is Never [Conformance]", func() {
			framework.SkipUnlessServerVersionGTE(jobsVersion, c.Discovery())

			By("running the image " + nginxImage)
			framework.RunKubectlOrDie("run", podName, "--restart=Never", "--generator=run-pod/v1", "--image="+nginxImage, nsFlag)
			By("verifying the pod " + podName + " was created")
			pod, err := c.Core().Pods(ns).Get(podName)
			if err != nil {
				framework.Failf("Failed getting pod %s: %v", podName, err)
			}
			containers := pod.Spec.Containers
			if containers == nil || len(containers) != 1 || containers[0].Image != nginxImage {
				framework.Failf("Failed creating pod %s with expected image %s", podName, nginxImage)
			}
			if pod.Spec.RestartPolicy != api.RestartPolicyNever {
				framework.Failf("Failed creating a pod with correct restart policy for --restart=Never")
			}
		})
	})

	framework.KubeDescribe("Kubectl replace", func() {
		var nsFlag string
		var podName string

		BeforeEach(func() {
			nsFlag = fmt.Sprintf("--namespace=%v", ns)
			podName = "e2e-test-nginx-pod"
		})

		AfterEach(func() {
			framework.RunKubectlOrDie("delete", "pods", podName, nsFlag)
		})

		It("should update a single-container pod's image [Conformance]", func() {
			framework.SkipUnlessServerVersionGTE(jobsVersion, c.Discovery())

			By("running the image " + nginxImage)
			framework.RunKubectlOrDie("run", podName, "--generator=run-pod/v1", "--image="+nginxImage, "--labels=run="+podName, nsFlag)

			By("verifying the pod " + podName + " is running")
			label := labels.SelectorFromSet(labels.Set(map[string]string{"run": podName}))
			err := testutils.WaitForPodsWithLabelRunning(c, ns, label)
			if err != nil {
				framework.Failf("Failed getting pod %s: %v", podName, err)
			}

			By("verifying the pod " + podName + " was created")
			podJson := framework.RunKubectlOrDie("get", "pod", podName, nsFlag, "-o", "json")
			if !strings.Contains(podJson, podName) {
				framework.Failf("Failed to find pod %s in [%s]", podName, podJson)
			}

			By("replace the image in the pod")
			podJson = strings.Replace(podJson, nginxImage, busyboxImage, 1)
			framework.RunKubectlOrDieInput(podJson, "replace", "-f", "-", nsFlag)

			By("verifying the pod " + podName + " has the right image " + busyboxImage)
			pod, err := c.Core().Pods(ns).Get(podName)
			if err != nil {
				framework.Failf("Failed getting deployment %s: %v", podName, err)
			}
			containers := pod.Spec.Containers
			if containers == nil || len(containers) != 1 || containers[0].Image != busyboxImage {
				framework.Failf("Failed creating pod with expected image %s", busyboxImage)
			}
		})
	})

	framework.KubeDescribe("Kubectl run --rm job", func() {
		jobName := "e2e-test-rm-busybox-job"

		It("should create a job from an image, then delete the job [Conformance]", func() {
			nsFlag := fmt.Sprintf("--namespace=%v", ns)

			// The rkt runtime doesn't support attach, see #23335
			framework.SkipIfContainerRuntimeIs("rkt")
			framework.SkipUnlessServerVersionGTE(jobsVersion, c.Discovery())

			By("executing a command with run --rm and attach with stdin")
			t := time.NewTimer(runJobTimeout)
			defer t.Stop()
			runOutput := framework.NewKubectlCommand(nsFlag, "run", jobName, "--image="+busyboxImage, "--rm=true", "--generator=job/v1", "--restart=OnFailure", "--attach=true", "--stdin", "--", "sh", "-c", "cat && echo 'stdin closed'").
				WithStdinData("abcd1234").
				WithTimeout(t.C).
				ExecOrDie()
			Expect(runOutput).To(ContainSubstring("abcd1234"))
			Expect(runOutput).To(ContainSubstring("stdin closed"))

			By("verifying the job " + jobName + " was deleted")
			_, err := c.Batch().Jobs(ns).Get(jobName)
			Expect(err).To(HaveOccurred())
			Expect(apierrs.IsNotFound(err)).To(BeTrue())
		})
	})

	framework.KubeDescribe("Proxy server", func() {
		// TODO: test proxy options (static, prefix, etc)
		It("should support proxy with --port 0 [Conformance]", func() {
			By("starting the proxy server")
			port, cmd, err := startProxyServer()
			if cmd != nil {
				defer framework.TryKill(cmd)
			}
			if err != nil {
				framework.Failf("Failed to start proxy server: %v", err)
			}
			By("curling proxy /api/ output")
			localAddr := fmt.Sprintf("http://localhost:%d/api/", port)
			apiVersions, err := getAPIVersions(localAddr)
			if err != nil {
				framework.Failf("Expected at least one supported apiversion, got error %v", err)
			}
			if len(apiVersions.Versions) < 1 {
				framework.Failf("Expected at least one supported apiversion, got %v", apiVersions)
			}
		})

		It("should support --unix-socket=/path [Conformance]", func() {
			By("Starting the proxy")
			tmpdir, err := ioutil.TempDir("", "kubectl-proxy-unix")
			if err != nil {
				framework.Failf("Failed to create temporary directory: %v", err)
			}
			path := filepath.Join(tmpdir, "test")
			defer os.Remove(path)
			defer os.Remove(tmpdir)
			cmd := framework.KubectlCmd("proxy", fmt.Sprintf("--unix-socket=%s", path))
			stdout, stderr, err := framework.StartCmdAndStreamOutput(cmd)
			if err != nil {
				framework.Failf("Failed to start kubectl command: %v", err)
			}
			defer stdout.Close()
			defer stderr.Close()
			defer framework.TryKill(cmd)
			buf := make([]byte, 128)
			if _, err = stdout.Read(buf); err != nil {
				framework.Failf("Expected output from kubectl proxy: %v", err)
			}
			By("retrieving proxy /api/ output")
			_, err = curlUnix("http://unused/api", path)
			if err != nil {
				framework.Failf("Failed get of /api at %s: %v", path, err)
			}
		})
	})

	framework.KubeDescribe("Kubectl taint", func() {
		It("should update the taint on a node", func() {
			testTaint := api.Taint{
				Key:    fmt.Sprintf("kubernetes.io/e2e-taint-key-001-%s", string(uuid.NewUUID())),
				Value:  "testing-taint-value",
				Effect: api.TaintEffectNoSchedule,
			}

			nodeName := getNodeThatCanRunPod(f)

			By("adding the taint " + testTaint.ToString() + " to a node")
			runKubectlRetryOrDie("taint", "nodes", nodeName, testTaint.ToString())
			defer framework.RemoveTaintOffNode(f.ClientSet, nodeName, testTaint)

			By("verifying the node has the taint " + testTaint.ToString())
			output := runKubectlRetryOrDie("describe", "node", nodeName)
			requiredStrings := [][]string{
				{"Name:", nodeName},
				{"Taints:"},
				{testTaint.ToString()},
			}
			checkOutput(output, requiredStrings)

			By("removing the taint " + testTaint.ToString() + " of a node")
			runKubectlRetryOrDie("taint", "nodes", nodeName, testTaint.Key+":"+string(testTaint.Effect)+"-")
			By("verifying the node doesn't have the taint " + testTaint.Key)
			output = runKubectlRetryOrDie("describe", "node", nodeName)
			if strings.Contains(output, testTaint.Key) {
				framework.Failf("Failed removing taint " + testTaint.Key + " of the node " + nodeName)
			}
		})

		It("should remove all the taints with the same key off a node", func() {
			testTaint := api.Taint{
				Key:    fmt.Sprintf("kubernetes.io/e2e-taint-key-002-%s", string(uuid.NewUUID())),
				Value:  "testing-taint-value",
				Effect: api.TaintEffectNoSchedule,
			}

			nodeName := getNodeThatCanRunPod(f)

			By("adding the taint " + testTaint.ToString() + " to a node")
			runKubectlRetryOrDie("taint", "nodes", nodeName, testTaint.ToString())
			defer framework.RemoveTaintOffNode(f.ClientSet, nodeName, testTaint)

			By("verifying the node has the taint " + testTaint.ToString())
			output := runKubectlRetryOrDie("describe", "node", nodeName)
			requiredStrings := [][]string{
				{"Name:", nodeName},
				{"Taints:"},
				{testTaint.ToString()},
			}
			checkOutput(output, requiredStrings)

			newTestTaint := api.Taint{
				Key:    testTaint.Key,
				Value:  "another-testing-taint-value",
				Effect: api.TaintEffectPreferNoSchedule,
			}
			By("adding another taint " + newTestTaint.ToString() + " to the node")
			runKubectlRetryOrDie("taint", "nodes", nodeName, newTestTaint.ToString())
			defer framework.RemoveTaintOffNode(f.ClientSet, nodeName, newTestTaint)

			By("verifying the node has the taint " + newTestTaint.ToString())
			output = runKubectlRetryOrDie("describe", "node", nodeName)
			requiredStrings = [][]string{
				{"Name:", nodeName},
				{"Taints:"},
				{newTestTaint.ToString()},
			}
			checkOutput(output, requiredStrings)

			By("removing all taints that have the same key " + testTaint.Key + " of the node")
			runKubectlRetryOrDie("taint", "nodes", nodeName, testTaint.Key+"-")
			By("verifying the node doesn't have the taints that have the same key " + testTaint.Key)
			output = runKubectlRetryOrDie("describe", "node", nodeName)
			if strings.Contains(output, testTaint.Key) {
				framework.Failf("Failed removing taints " + testTaint.Key + " of the node " + nodeName)
			}
		})
	})

	framework.KubeDescribe("Kubectl create quota", func() {
		It("should create a quota without scopes", func() {
			framework.SkipUnlessKubectlVersionGTE(kubectlCreateQuotaVersion)
			nsFlag := fmt.Sprintf("--namespace=%v", ns)
			quotaName := "million"

			By("calling kubectl quota")
			framework.RunKubectlOrDie("create", "quota", quotaName, "--hard=pods=1000000,services=1000000", nsFlag)

			By("verifying that the quota was created")
			quota, err := c.Core().ResourceQuotas(ns).Get(quotaName)
			if err != nil {
				framework.Failf("Failed getting quota %s: %v", quotaName, err)
			}

			if len(quota.Spec.Scopes) != 0 {
				framework.Failf("Expected empty scopes, got %v", quota.Spec.Scopes)
			}
			if len(quota.Spec.Hard) != 2 {
				framework.Failf("Expected two resources, got %v", quota.Spec.Hard)
			}
			r, found := quota.Spec.Hard[api.ResourcePods]
			if expected := resource.MustParse("1000000"); !found || (&r).Cmp(expected) != 0 {
				framework.Failf("Expected pods=1000000, got %v", r)
			}
			r, found = quota.Spec.Hard[api.ResourceServices]
			if expected := resource.MustParse("1000000"); !found || (&r).Cmp(expected) != 0 {
				framework.Failf("Expected services=1000000, got %v", r)
			}
		})

		It("should create a quota with scopes", func() {
			framework.SkipUnlessKubectlVersionGTE(kubectlCreateQuotaVersion)
			nsFlag := fmt.Sprintf("--namespace=%v", ns)
			quotaName := "scopes"

			By("calling kubectl quota")
			framework.RunKubectlOrDie("create", "quota", quotaName, "--hard=pods=1000000", "--scopes=BestEffort,NotTerminating", nsFlag)

			By("verifying that the quota was created")
			quota, err := c.Core().ResourceQuotas(ns).Get(quotaName)
			if err != nil {
				framework.Failf("Failed getting quota %s: %v", quotaName, err)
			}

			if len(quota.Spec.Scopes) != 2 {
				framework.Failf("Expected two scopes, got %v", quota.Spec.Scopes)
			}
			scopes := make(map[api.ResourceQuotaScope]struct{})
			for _, scope := range quota.Spec.Scopes {
				scopes[scope] = struct{}{}
			}
			if _, found := scopes[api.ResourceQuotaScopeBestEffort]; !found {
				framework.Failf("Expected BestEffort scope, got %v", quota.Spec.Scopes)
			}
			if _, found := scopes[api.ResourceQuotaScopeNotTerminating]; !found {
				framework.Failf("Expected NotTerminating scope, got %v", quota.Spec.Scopes)
			}
		})

		It("should reject quota with invalid scopes", func() {
			framework.SkipUnlessKubectlVersionGTE(kubectlCreateQuotaVersion)
			nsFlag := fmt.Sprintf("--namespace=%v", ns)
			quotaName := "scopes"

			By("calling kubectl quota")
			out, err := framework.RunKubectl("create", "quota", quotaName, "--hard=hard=pods=1000000", "--scopes=Foo", nsFlag)
			if err == nil {
				framework.Failf("Expected kubectl to fail, but it succeeded: %s", out)
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
			framework.Failf("Failed to find %s in %s", requirement[0], output)
		}
		for _, item := range requirement[1:] {
			if !strings.Contains(outputLines[currentLine], item) {
				framework.Failf("Failed to find %s in %s", item, outputLines[currentLine])
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
	cmd := framework.KubectlCmd("proxy", "-p", "0")
	stdout, stderr, err := framework.StartCmdAndStreamOutput(cmd)
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
	transport := utilnet.SetTransportDefaults(&http.Transport{
		Dial: dial,
	})
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
	return curlTransport(url, utilnet.SetTransportDefaults(&http.Transport{}))
}

func validateGuestbookApp(c clientset.Interface, ns string) {
	framework.Logf("Waiting for all frontend pods to be Running.")
	label := labels.SelectorFromSet(labels.Set(map[string]string{"tier": "frontend", "app": "guestbook"}))
	err := testutils.WaitForPodsWithLabelRunning(c, ns, label)
	Expect(err).NotTo(HaveOccurred())
	framework.Logf("Waiting for frontend to serve content.")
	if !waitForGuestbookResponse(c, "get", "", `{"data": ""}`, guestbookStartupTimeout, ns) {
		framework.Failf("Frontend service did not start serving content in %v seconds.", guestbookStartupTimeout.Seconds())
	}

	framework.Logf("Trying to add a new entry to the guestbook.")
	if !waitForGuestbookResponse(c, "set", "TestEntry", `{"message": "Updated"}`, guestbookResponseTimeout, ns) {
		framework.Failf("Cannot added new entry in %v seconds.", guestbookResponseTimeout.Seconds())
	}

	framework.Logf("Verifying that added entry can be retrieved.")
	if !waitForGuestbookResponse(c, "get", "", `{"data": "TestEntry"}`, guestbookResponseTimeout, ns) {
		framework.Failf("Entry to guestbook wasn't correctly added in %v seconds.", guestbookResponseTimeout.Seconds())
	}
}

// Returns whether received expected response from guestbook on time.
func waitForGuestbookResponse(c clientset.Interface, cmd, arg, expectedResponse string, timeout time.Duration, ns string) bool {
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(5 * time.Second) {
		res, err := makeRequestToGuestbook(c, cmd, arg, ns)
		if err == nil && res == expectedResponse {
			return true
		}
		framework.Logf("Failed to get response from guestbook. err: %v, response: %s", err, res)
	}
	return false
}

func makeRequestToGuestbook(c clientset.Interface, cmd, value string, ns string) (string, error) {
	proxyRequest, errProxy := framework.GetServicesProxyRequest(c, c.Core().RESTClient().Get())
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
		framework.Failf(err.Error())
	}
	defer file.Close()

	data, err := ioutil.ReadAll(file)
	if err != nil {
		framework.Failf(err.Error())
	}

	return data
}

func readReplicationControllerFromString(contents string) *api.ReplicationController {
	rc := api.ReplicationController{}
	if err := yaml.Unmarshal([]byte(contents), &rc); err != nil {
		framework.Failf(err.Error())
	}

	return &rc
}

func modifyReplicationControllerConfiguration(contents string) io.Reader {
	rc := readReplicationControllerFromString(contents)
	rc.Labels[applyTestLabel] = "ADDED"
	rc.Spec.Selector[applyTestLabel] = "ADDED"
	rc.Spec.Template.Labels[applyTestLabel] = "ADDED"
	data, err := json.Marshal(rc)
	if err != nil {
		framework.Failf("json marshal failed: %s\n", err)
	}

	return bytes.NewReader(data)
}

func forEachReplicationController(c clientset.Interface, ns, selectorKey, selectorValue string, fn func(api.ReplicationController)) {
	var rcs *api.ReplicationControllerList
	var err error
	for t := time.Now(); time.Since(t) < framework.PodListTimeout; time.Sleep(framework.Poll) {
		label := labels.SelectorFromSet(labels.Set(map[string]string{selectorKey: selectorValue}))
		options := api.ListOptions{LabelSelector: label}
		rcs, err = c.Core().ReplicationControllers(ns).List(options)
		Expect(err).NotTo(HaveOccurred())
		if len(rcs.Items) > 0 {
			break
		}
	}

	if rcs == nil || len(rcs.Items) == 0 {
		framework.Failf("No replication controllers found")
	}

	for _, rc := range rcs.Items {
		fn(rc)
	}
}

func validateReplicationControllerConfiguration(rc api.ReplicationController) {
	if rc.Name == "redis-master" {
		if _, ok := rc.Annotations[annotations.LastAppliedConfigAnnotation]; !ok {
			framework.Failf("Annotation not found in modified configuration:\n%v\n", rc)
		}

		if value, ok := rc.Labels[applyTestLabel]; !ok || value != "ADDED" {
			framework.Failf("Added label %s not found in modified configuration:\n%v\n", applyTestLabel, rc)
		}
	}
}

// getUDData creates a validator function based on the input string (i.e. kitten.jpg).
// For example, if you send "kitten.jpg", this function verifies that the image jpg = kitten.jpg
// in the container's json field.
func getUDData(jpgExpected string, ns string) func(clientset.Interface, string) error {

	// getUDData validates data.json in the update-demo (returns nil if data is ok).
	return func(c clientset.Interface, podID string) error {
		framework.Logf("validating pod %s", podID)
		subResourceProxyAvailable, err := framework.ServerVersionGTE(framework.SubResourcePodProxyVersion, c.Discovery())
		if err != nil {
			return err
		}
		var body []byte
		if subResourceProxyAvailable {
			body, err = c.Core().RESTClient().Get().
				Namespace(ns).
				Resource("pods").
				SubResource("proxy").
				Name(podID).
				Suffix("data.json").
				Do().
				Raw()
		} else {
			body, err = c.Core().RESTClient().Get().
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
		framework.Logf("got data: %s", body)
		var data updateDemoData
		if err := json.Unmarshal(body, &data); err != nil {
			return err
		}
		framework.Logf("Unmarshalled json jpg/img => %s , expecting %s .", data, jpgExpected)
		if strings.Contains(data.Image, jpgExpected) {
			return nil
		} else {
			return errors.New(fmt.Sprintf("data served up in container is inaccurate, %s didn't contain %s", data, jpgExpected))
		}
	}
}

func noOpValidatorFn(c clientset.Interface, podID string) error { return nil }

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
		framework.Failf("Unable to to write file at %s to buffer. Error: %s", fileName, err)
	}

	// Copy kubectl binary into the file writer
	if _, err := io.Copy(fileWriter, file); err != nil {
		framework.Failf("Unable to to copy file at %s into the file writer. Error: %s", fileName, err)
	}

	// Nothing more should be written to this instance of the postBodyWriter
	if err := postBodyWriter.Close(); err != nil {
		framework.Failf("Unable to close the writer for file upload. Error: %s", err)
	}
}

func startLocalProxy() (srv *httptest.Server, logs *bytes.Buffer) {
	logs = &bytes.Buffer{}
	p := goproxy.NewProxyHttpServer()
	p.Verbose = true
	p.Logger = log.New(logs, "", 0)
	return httptest.NewServer(p), logs
}

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

// OWNER = sig/cli

package kubectl

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"log"
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
	openapi_v2 "github.com/googleapis/gnostic/OpenAPIv2"

	v1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/authentication/serviceaccount"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/controller"
	commonutils "k8s.io/kubernetes/test/e2e/common"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/framework/auth"
	e2eendpoints "k8s.io/kubernetes/test/e2e/framework/endpoints"
	jobutil "k8s.io/kubernetes/test/e2e/framework/job"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eservice "k8s.io/kubernetes/test/e2e/framework/service"
	"k8s.io/kubernetes/test/e2e/framework/testfiles"
	"k8s.io/kubernetes/test/e2e/scheduling"
	testutils "k8s.io/kubernetes/test/utils"
	"k8s.io/kubernetes/test/utils/crd"
	uexec "k8s.io/utils/exec"
	"sigs.k8s.io/yaml"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"

	"k8s.io/kubectl/pkg/polymorphichelpers"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

const (
	updateDemoSelector       = "name=update-demo"
	guestbookStartupTimeout  = 10 * time.Minute
	guestbookResponseTimeout = 3 * time.Minute
	simplePodSelector        = "name=httpd"
	simplePodName            = "httpd"
	simplePodResourceName    = "pod/httpd"
	httpdDefaultOutput       = "It works!"
	simplePodPort            = 80
	pausePodSelector         = "name=pause"
	pausePodName             = "pause"
	busyboxPodSelector       = "app=busybox1"
	busyboxPodName           = "busybox1"
	runJobTimeout            = 5 * time.Minute
	kubeCtlManifestPath      = "test/e2e/testing-manifests/kubectl"
	redisControllerFilename  = "redis-master-controller.json.in"
	redisServiceFilename     = "redis-master-service.json"
	httpdDeployment1Filename = "httpd-deployment1.yaml.in"
	httpdDeployment2Filename = "httpd-deployment2.yaml.in"
	httpdDeployment3Filename = "httpd-deployment3.yaml.in"
	metaPattern              = `"kind":"%s","apiVersion":"%s/%s","metadata":{"name":"%s"}`
)

var (
	nautilusImage = imageutils.GetE2EImage(imageutils.Nautilus)
	kittenImage   = imageutils.GetE2EImage(imageutils.Kitten)
	redisImage    = imageutils.GetE2EImage(imageutils.Redis)
	httpdImage    = imageutils.GetE2EImage(imageutils.Httpd)
	busyboxImage  = imageutils.GetE2EImage(imageutils.BusyBox)
	agnhostImage  = imageutils.GetE2EImage(imageutils.Agnhost)
)

var (
	proxyRegexp = regexp.MustCompile("Starting to serve on 127.0.0.1:([0-9]+)")

	cronJobGroupVersionResourceAlpha = schema.GroupVersionResource{Group: "batch", Version: "v2alpha1", Resource: "cronjobs"}
	cronJobGroupVersionResourceBeta  = schema.GroupVersionResource{Group: "batch", Version: "v1beta1", Resource: "cronjobs"}
)

// Stops everything from filePath from namespace ns and checks if everything matching selectors from the given namespace is correctly stopped.
// Aware of the kubectl example files map.
func cleanupKubectlInputs(fileContents string, ns string, selectors ...string) {
	ginkgo.By("using delete to clean up resources")
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
	return testfiles.ReadOrDie(path.Join(kubeCtlManifestPath, file), ginkgo.Fail)
}

func runKubectlRetryOrDie(args ...string) string {
	var err error
	var output string
	for i := 0; i < 5; i++ {
		output, err = framework.RunKubectl(args...)
		if err == nil || (!strings.Contains(err.Error(), genericregistry.OptimisticLockErrorMsg) && !strings.Contains(err.Error(), "Operation cannot be fulfilled")) {
			break
		}
		time.Sleep(time.Second)
	}
	// Expect no errors to be present after retries are finished
	// Copied from framework #ExecOrDie
	e2elog.Logf("stdout: %q", output)
	framework.ExpectNoError(err)
	return output
}

// duplicated setup to avoid polluting "normal" clients with alpha features which confuses the generated clients
var _ = SIGDescribe("Kubectl alpha client", func() {
	defer ginkgo.GinkgoRecover()
	f := framework.NewDefaultFramework("kubectl")

	var c clientset.Interface
	var ns string
	ginkgo.BeforeEach(func() {
		c = f.ClientSet
		ns = f.Namespace.Name
	})

	framework.KubeDescribe("Kubectl run CronJob", func() {
		var nsFlag string
		var cjName string

		ginkgo.BeforeEach(func() {
			nsFlag = fmt.Sprintf("--namespace=%v", ns)
			cjName = "e2e-test-echo-cronjob-alpha"
		})

		ginkgo.AfterEach(func() {
			framework.RunKubectlOrDie("delete", "cronjobs", cjName, nsFlag)
		})

		ginkgo.It("should create a CronJob", func() {
			framework.SkipIfMissingResource(f.DynamicClient, cronJobGroupVersionResourceAlpha, f.Namespace.Name)

			schedule := "*/5 * * * ?"
			framework.RunKubectlOrDie("run", cjName, "--restart=OnFailure", "--generator=cronjob/v2alpha1",
				"--schedule="+schedule, "--image="+busyboxImage, nsFlag)
			ginkgo.By("verifying the CronJob " + cjName + " was created")
			sj, err := c.BatchV1beta1().CronJobs(ns).Get(cjName, metav1.GetOptions{})
			if err != nil {
				e2elog.Failf("Failed getting CronJob %s: %v", cjName, err)
			}
			if sj.Spec.Schedule != schedule {
				e2elog.Failf("Failed creating a CronJob with correct schedule %s", schedule)
			}
			containers := sj.Spec.JobTemplate.Spec.Template.Spec.Containers
			if checkContainersImage(containers, busyboxImage) {
				e2elog.Failf("Failed creating CronJob %s for 1 pod with expected image %s: %#v", cjName, busyboxImage, containers)
			}
			if sj.Spec.JobTemplate.Spec.Template.Spec.RestartPolicy != v1.RestartPolicyOnFailure {
				e2elog.Failf("Failed creating a CronJob with correct restart policy for --restart=OnFailure")
			}
		})
	})
})

var _ = SIGDescribe("Kubectl client", func() {
	defer ginkgo.GinkgoRecover()
	f := framework.NewDefaultFramework("kubectl")

	// Reusable cluster state function.  This won't be adversely affected by lazy initialization of framework.
	clusterState := func() *framework.ClusterVerification {
		return f.NewClusterVerification(
			f.Namespace,
			framework.PodStateVerification{
				Selectors:   map[string]string{"app": "redis"},
				ValidPhases: []v1.PodPhase{v1.PodRunning /*v1.PodPending*/},
			})
	}
	forEachPod := func(podFunc func(p v1.Pod)) {
		clusterState().ForEach(podFunc)
	}
	var c clientset.Interface
	var ns string
	ginkgo.BeforeEach(func() {
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
			framework.DumpAllNamespaceInfo(f.ClientSet, ns)
			e2elog.Failf("Verified %d of %d pods , error: %v", len(pods), atLeast, err)
		}
	}

	debugDiscovery := func() {
		home := os.Getenv("HOME")
		if len(home) == 0 {
			e2elog.Logf("no $HOME envvar set")
			return
		}

		cacheDir := filepath.Join(home, ".kube", "cache", "discovery")
		err := filepath.Walk(cacheDir, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return err
			}

			// only pay attention to $host_$port/v1/serverresources.json files
			subpath := strings.TrimPrefix(path, cacheDir+string(filepath.Separator))
			parts := filepath.SplitList(subpath)
			if len(parts) != 3 || parts[1] != "v1" || parts[2] != "serverresources.json" {
				return nil
			}
			e2elog.Logf("%s modified at %s (current time: %s)", path, info.ModTime(), time.Now())

			data, readError := ioutil.ReadFile(path)
			if readError != nil {
				e2elog.Logf("%s error: %v", path, readError)
			} else {
				e2elog.Logf("%s content: %s", path, string(data))
			}
			return nil
		})
		e2elog.Logf("scanned %s for discovery docs: %v", home, err)
	}

	framework.KubeDescribe("Update Demo", func() {
		var nautilus, kitten string
		ginkgo.BeforeEach(func() {
			updateDemoRoot := "test/fixtures/doc-yaml/user-guide/update-demo"
			nautilus = commonutils.SubstituteImageName(string(testfiles.ReadOrDie(filepath.Join(updateDemoRoot, "nautilus-rc.yaml.in"), ginkgo.Fail)))
			kitten = commonutils.SubstituteImageName(string(testfiles.ReadOrDie(filepath.Join(updateDemoRoot, "kitten-rc.yaml.in"), ginkgo.Fail)))
		})
		/*
			Release : v1.9
			Testname: Kubectl, replication controller
			Description: Create a Pod and a container with a given image. Configure replication controller to run 2 replicas. The number of running instances of the Pod MUST equal the number of replicas set on the replication controller which is 2.
		*/
		framework.ConformanceIt("should create and stop a replication controller ", func() {
			defer cleanupKubectlInputs(nautilus, ns, updateDemoSelector)

			ginkgo.By("creating a replication controller")
			framework.RunKubectlOrDieInput(nautilus, "create", "-f", "-", fmt.Sprintf("--namespace=%v", ns))
			framework.ValidateController(c, nautilusImage, 2, "update-demo", updateDemoSelector, getUDData("nautilus.jpg", ns), ns)
		})

		/*
			Release : v1.9
			Testname: Kubectl, scale replication controller
			Description: Create a Pod and a container with a given image. Configure replication controller to run 2 replicas. The number of running instances of the Pod MUST equal the number of replicas set on the replication controller which is 2. Update the replicaset to 1. Number of running instances of the Pod MUST be 1. Update the replicaset to 2. Number of running instances of the Pod MUST be 2.
		*/
		framework.ConformanceIt("should scale a replication controller ", func() {
			defer cleanupKubectlInputs(nautilus, ns, updateDemoSelector)

			ginkgo.By("creating a replication controller")
			framework.RunKubectlOrDieInput(nautilus, "create", "-f", "-", fmt.Sprintf("--namespace=%v", ns))
			framework.ValidateController(c, nautilusImage, 2, "update-demo", updateDemoSelector, getUDData("nautilus.jpg", ns), ns)
			ginkgo.By("scaling down the replication controller")
			debugDiscovery()
			framework.RunKubectlOrDie("scale", "rc", "update-demo-nautilus", "--replicas=1", "--timeout=5m", fmt.Sprintf("--namespace=%v", ns))
			framework.ValidateController(c, nautilusImage, 1, "update-demo", updateDemoSelector, getUDData("nautilus.jpg", ns), ns)
			ginkgo.By("scaling up the replication controller")
			debugDiscovery()
			framework.RunKubectlOrDie("scale", "rc", "update-demo-nautilus", "--replicas=2", "--timeout=5m", fmt.Sprintf("--namespace=%v", ns))
			framework.ValidateController(c, nautilusImage, 2, "update-demo", updateDemoSelector, getUDData("nautilus.jpg", ns), ns)
		})

		/*
			Release : v1.9
			Testname: Kubectl, rolling update replication controller
			Description: Create a Pod and a container with a given image. Configure replication controller to run 2 replicas. The number of running instances of the Pod MUST equal the number of replicas set on the replication controller which is 2. Run a rolling update to run a different version of the container. All running instances SHOULD now be running the newer version of the container as part of the rolling update.
		*/
		framework.ConformanceIt("should do a rolling update of a replication controller ", func() {
			ginkgo.By("creating the initial replication controller")
			framework.RunKubectlOrDieInput(string(nautilus[:]), "create", "-f", "-", fmt.Sprintf("--namespace=%v", ns))
			framework.ValidateController(c, nautilusImage, 2, "update-demo", updateDemoSelector, getUDData("nautilus.jpg", ns), ns)
			ginkgo.By("rolling-update to new replication controller")
			debugDiscovery()
			framework.RunKubectlOrDieInput(string(kitten[:]), "rolling-update", "update-demo-nautilus", "--update-period=1s", "-f", "-", fmt.Sprintf("--namespace=%v", ns))
			framework.ValidateController(c, kittenImage, 2, "update-demo", updateDemoSelector, getUDData("kitten.jpg", ns), ns)
			// Everything will hopefully be cleaned up when the namespace is deleted.
		})
	})

	framework.KubeDescribe("Guestbook application", func() {
		forEachGBFile := func(run func(s string)) {
			guestbookRoot := "test/e2e/testing-manifests/guestbook"
			for _, gbAppFile := range []string{
				"redis-slave-service.yaml",
				"redis-master-service.yaml",
				"frontend-service.yaml",
				"frontend-deployment.yaml.in",
				"redis-master-deployment.yaml.in",
				"redis-slave-deployment.yaml.in",
			} {
				contents := commonutils.SubstituteImageName(string(testfiles.ReadOrDie(filepath.Join(guestbookRoot, gbAppFile), ginkgo.Fail)))
				run(contents)
			}
		}

		/*
			Release : v1.9
			Testname: Kubectl, guestbook application
			Description: Create Guestbook application that contains redis server, 2 instances of redis slave, frontend application, frontend service and redis master service and redis slave service. Using frontend service, the test will write an entry into the guestbook application which will store the entry into the backend redis database. Application flow MUST work as expected and the data written MUST be available to read.
		*/
		framework.ConformanceIt("should create and stop a working application ", func() {
			defer forEachGBFile(func(contents string) {
				cleanupKubectlInputs(contents, ns)
			})
			ginkgo.By("creating all guestbook components")
			forEachGBFile(func(contents string) {
				e2elog.Logf(contents)
				framework.RunKubectlOrDieInput(contents, "create", "-f", "-", fmt.Sprintf("--namespace=%v", ns))
			})

			ginkgo.By("validating guestbook app")
			validateGuestbookApp(c, ns)
		})
	})

	framework.KubeDescribe("Simple pod", func() {
		var podYaml string
		ginkgo.BeforeEach(func() {
			ginkgo.By(fmt.Sprintf("creating the pod from %v", podYaml))
			podYaml = commonutils.SubstituteImageName(string(readTestFileOrDie("pod-with-readiness-probe.yaml.in")))
			framework.RunKubectlOrDieInput(podYaml, "create", "-f", "-", fmt.Sprintf("--namespace=%v", ns))
			gomega.Expect(e2epod.CheckPodsRunningReady(c, ns, []string{simplePodName}, framework.PodStartTimeout)).To(gomega.BeTrue())
		})
		ginkgo.AfterEach(func() {
			cleanupKubectlInputs(podYaml, ns, simplePodSelector)
		})

		ginkgo.It("should support exec", func() {
			ginkgo.By("executing a command in the container")
			execOutput := framework.RunKubectlOrDie("exec", fmt.Sprintf("--namespace=%v", ns), simplePodName, "echo", "running", "in", "container")
			if e, a := "running in container", strings.TrimSpace(execOutput); e != a {
				e2elog.Failf("Unexpected kubectl exec output. Wanted %q, got %q", e, a)
			}

			ginkgo.By("executing a very long command in the container")
			veryLongData := make([]rune, 20000)
			for i := 0; i < len(veryLongData); i++ {
				veryLongData[i] = 'a'
			}
			execOutput = framework.RunKubectlOrDie("exec", fmt.Sprintf("--namespace=%v", ns), simplePodName, "echo", string(veryLongData))
			framework.ExpectEqual(string(veryLongData), strings.TrimSpace(execOutput), "Unexpected kubectl exec output")

			ginkgo.By("executing a command in the container with noninteractive stdin")
			execOutput = framework.NewKubectlCommand("exec", fmt.Sprintf("--namespace=%v", ns), "-i", simplePodName, "cat").
				WithStdinData("abcd1234").
				ExecOrDie()
			if e, a := "abcd1234", execOutput; e != a {
				e2elog.Failf("Unexpected kubectl exec output. Wanted %q, got %q", e, a)
			}

			// pretend that we're a user in an interactive shell
			r, closer, err := newBlockingReader("echo hi\nexit\n")
			if err != nil {
				e2elog.Failf("Error creating blocking reader: %v", err)
			}
			// NOTE this is solely for test cleanup!
			defer closer.Close()

			ginkgo.By("executing a command in the container with pseudo-interactive stdin")
			execOutput = framework.NewKubectlCommand("exec", fmt.Sprintf("--namespace=%v", ns), "-i", simplePodName, "sh").
				WithStdinReader(r).
				ExecOrDie()
			if e, a := "hi", strings.TrimSpace(execOutput); e != a {
				e2elog.Failf("Unexpected kubectl exec output. Wanted %q, got %q", e, a)
			}
		})

		ginkgo.It("should support exec using resource/name", func() {
			ginkgo.By("executing a command in the container")
			execOutput := framework.RunKubectlOrDie("exec", fmt.Sprintf("--namespace=%v", ns), simplePodResourceName, "echo", "running", "in", "container")
			if e, a := "running in container", strings.TrimSpace(execOutput); e != a {
				e2elog.Failf("Unexpected kubectl exec output. Wanted %q, got %q", e, a)
			}
		})

		ginkgo.It("should support exec through an HTTP proxy", func() {
			// Fail if the variable isn't set
			if framework.TestContext.Host == "" {
				e2elog.Failf("--host variable must be set to the full URI to the api server on e2e run.")
			}

			ginkgo.By("Starting goproxy")
			testSrv, proxyLogs := startLocalProxy()
			defer testSrv.Close()
			proxyAddr := testSrv.URL

			for _, proxyVar := range []string{"https_proxy", "HTTPS_PROXY"} {
				proxyLogs.Reset()
				ginkgo.By("Running kubectl via an HTTP proxy using " + proxyVar)
				output := framework.NewKubectlCommand(fmt.Sprintf("--namespace=%s", ns), "exec", "httpd", "echo", "running", "in", "container").
					WithEnv(append(os.Environ(), fmt.Sprintf("%s=%s", proxyVar, proxyAddr))).
					ExecOrDie()

				// Verify we got the normal output captured by the exec server
				expectedExecOutput := "running in container\n"
				if output != expectedExecOutput {
					e2elog.Failf("Unexpected kubectl exec output. Wanted %q, got  %q", expectedExecOutput, output)
				}

				// Verify the proxy server logs saw the connection
				expectedProxyLog := fmt.Sprintf("Accepting CONNECT to %s", strings.TrimSuffix(strings.TrimPrefix(framework.TestContext.Host, "https://"), "/api"))

				proxyLog := proxyLogs.String()
				if !strings.Contains(proxyLog, expectedProxyLog) {
					e2elog.Failf("Missing expected log result on proxy server for %s. Expected: %q, got %q", proxyVar, expectedProxyLog, proxyLog)
				}
			}
		})

		ginkgo.It("should support exec through kubectl proxy", func() {
			// Fail if the variable isn't set
			if framework.TestContext.Host == "" {
				e2elog.Failf("--host variable must be set to the full URI to the api server on e2e run.")
			}

			ginkgo.By("Starting kubectl proxy")
			port, proxyCmd, err := startProxyServer()
			framework.ExpectNoError(err)
			defer framework.TryKill(proxyCmd)

			//proxyLogs.Reset()
			host := fmt.Sprintf("--server=http://127.0.0.1:%d", port)
			ginkgo.By("Running kubectl via kubectl proxy using " + host)
			output := framework.NewKubectlCommand(
				host, fmt.Sprintf("--namespace=%s", ns),
				"exec", "httpd", "echo", "running", "in", "container",
			).ExecOrDie()

			// Verify we got the normal output captured by the exec server
			expectedExecOutput := "running in container\n"
			if output != expectedExecOutput {
				e2elog.Failf("Unexpected kubectl exec output. Wanted %q, got  %q", expectedExecOutput, output)
			}
		})

		ginkgo.It("should return command exit codes", func() {
			nsFlag := fmt.Sprintf("--namespace=%v", ns)

			ginkgo.By("execing into a container with a successful command")
			_, err := framework.NewKubectlCommand(nsFlag, "exec", "httpd", "--", "/bin/sh", "-c", "exit 0").Exec()
			framework.ExpectNoError(err)

			ginkgo.By("execing into a container with a failing command")
			_, err = framework.NewKubectlCommand(nsFlag, "exec", "httpd", "--", "/bin/sh", "-c", "exit 42").Exec()
			ee, ok := err.(uexec.ExitError)
			framework.ExpectEqual(ok, true)
			framework.ExpectEqual(ee.ExitStatus(), 42)

			ginkgo.By("running a successful command")
			_, err = framework.NewKubectlCommand(nsFlag, "run", "-i", "--image="+busyboxImage, "--restart=Never", "success", "--", "/bin/sh", "-c", "exit 0").Exec()
			framework.ExpectNoError(err)

			ginkgo.By("running a failing command")
			_, err = framework.NewKubectlCommand(nsFlag, "run", "-i", "--image="+busyboxImage, "--restart=Never", "failure-1", "--", "/bin/sh", "-c", "exit 42").Exec()
			ee, ok = err.(uexec.ExitError)
			framework.ExpectEqual(ok, true)
			framework.ExpectEqual(ee.ExitStatus(), 42)

			ginkgo.By("running a failing command without --restart=Never")
			_, err = framework.NewKubectlCommand(nsFlag, "run", "-i", "--image="+busyboxImage, "--restart=OnFailure", "failure-2", "--", "/bin/sh", "-c", "cat && exit 42").
				WithStdinData("abcd1234").
				Exec()
			framework.ExpectNoError(err)

			ginkgo.By("running a failing command without --restart=Never, but with --rm")
			_, err = framework.NewKubectlCommand(nsFlag, "run", "-i", "--image="+busyboxImage, "--restart=OnFailure", "--rm", "failure-3", "--", "/bin/sh", "-c", "cat && exit 42").
				WithStdinData("abcd1234").
				Exec()
			framework.ExpectNoError(err)
			e2epod.WaitForPodToDisappear(f.ClientSet, ns, "failure-3", labels.Everything(), 2*time.Second, wait.ForeverTestTimeout)

			ginkgo.By("running a failing command with --leave-stdin-open")
			_, err = framework.NewKubectlCommand(nsFlag, "run", "-i", "--image="+busyboxImage, "--restart=Never", "failure-4", "--leave-stdin-open", "--", "/bin/sh", "-c", "exit 42").
				WithStdinData("abcd1234").
				Exec()
			framework.ExpectNoError(err)
		})

		ginkgo.It("should support inline execution and attach", func() {
			nsFlag := fmt.Sprintf("--namespace=%v", ns)

			ginkgo.By("executing a command with run and attach with stdin")
			runOutput := framework.NewKubectlCommand(nsFlag, "run", "run-test", "--image="+busyboxImage, "--restart=OnFailure", "--attach=true", "--stdin", "--", "sh", "-c", "cat && echo 'stdin closed'").
				WithStdinData("abcd1234").
				ExecOrDie()

			g := func(pods []*v1.Pod) sort.Interface { return sort.Reverse(controller.ActivePods(pods)) }
			runTestPod, _, err := polymorphichelpers.GetFirstPod(f.ClientSet.CoreV1(), ns, "run=run-test", 1*time.Minute, g)
			gomega.Expect(err).To(gomega.BeNil())
			// NOTE: we cannot guarantee our output showed up in the container logs before stdin was closed, so we have
			// to loop test.
			err = wait.PollImmediate(time.Second, time.Minute, func() (bool, error) {
				if !e2epod.CheckPodsRunningReady(c, ns, []string{runTestPod.Name}, 1*time.Second) {
					e2elog.Failf("Pod %q of Job %q should still be running", runTestPod.Name, "run-test")
				}
				logOutput := framework.RunKubectlOrDie(nsFlag, "logs", runTestPod.Name)
				gomega.Expect(runOutput).To(gomega.ContainSubstring("abcd1234"))
				gomega.Expect(runOutput).To(gomega.ContainSubstring("stdin closed"))
				return strings.Contains(logOutput, "abcd1234"), nil
			})
			gomega.Expect(err).To(gomega.BeNil())

			gomega.Expect(c.BatchV1().Jobs(ns).Delete("run-test", nil)).To(gomega.BeNil())

			ginkgo.By("executing a command with run and attach without stdin")
			runOutput = framework.NewKubectlCommand(fmt.Sprintf("--namespace=%v", ns), "run", "run-test-2", "--image="+busyboxImage, "--restart=OnFailure", "--attach=true", "--leave-stdin-open=true", "--", "sh", "-c", "cat && echo 'stdin closed'").
				WithStdinData("abcd1234").
				ExecOrDie()
			gomega.Expect(runOutput).ToNot(gomega.ContainSubstring("abcd1234"))
			gomega.Expect(runOutput).To(gomega.ContainSubstring("stdin closed"))
			gomega.Expect(c.BatchV1().Jobs(ns).Delete("run-test-2", nil)).To(gomega.BeNil())

			ginkgo.By("executing a command with run and attach with stdin with open stdin should remain running")
			runOutput = framework.NewKubectlCommand(nsFlag, "run", "run-test-3", "--image="+busyboxImage, "--restart=OnFailure", "--attach=true", "--leave-stdin-open=true", "--stdin", "--", "sh", "-c", "cat && echo 'stdin closed'").
				WithStdinData("abcd1234\n").
				ExecOrDie()
			gomega.Expect(runOutput).ToNot(gomega.ContainSubstring("stdin closed"))
			g = func(pods []*v1.Pod) sort.Interface { return sort.Reverse(controller.ActivePods(pods)) }
			runTestPod, _, err = polymorphichelpers.GetFirstPod(f.ClientSet.CoreV1(), ns, "run=run-test-3", 1*time.Minute, g)
			gomega.Expect(err).To(gomega.BeNil())
			if !e2epod.CheckPodsRunningReady(c, ns, []string{runTestPod.Name}, time.Minute) {
				e2elog.Failf("Pod %q of Job %q should still be running", runTestPod.Name, "run-test-3")
			}

			// NOTE: we cannot guarantee our output showed up in the container logs before stdin was closed, so we have
			// to loop test.
			err = wait.PollImmediate(time.Second, time.Minute, func() (bool, error) {
				if !e2epod.CheckPodsRunningReady(c, ns, []string{runTestPod.Name}, 1*time.Second) {
					e2elog.Failf("Pod %q of Job %q should still be running", runTestPod.Name, "run-test-3")
				}
				logOutput := framework.RunKubectlOrDie(nsFlag, "logs", runTestPod.Name)
				gomega.Expect(logOutput).ToNot(gomega.ContainSubstring("stdin closed"))
				return strings.Contains(logOutput, "abcd1234"), nil
			})
			gomega.Expect(err).To(gomega.BeNil())

			gomega.Expect(c.BatchV1().Jobs(ns).Delete("run-test-3", nil)).To(gomega.BeNil())
		})

		ginkgo.It("should contain last line of the log", func() {
			nsFlag := fmt.Sprintf("--namespace=%v", ns)
			podName := "run-log-test"

			ginkgo.By("executing a command with run")
			framework.RunKubectlOrDie("run", podName, "--generator=run-pod/v1", "--image="+busyboxImage, "--restart=OnFailure", nsFlag, "--", "sh", "-c", "sleep 10; seq 100 | while read i; do echo $i; sleep 0.01; done; echo EOF")

			if !e2epod.CheckPodsRunningReadyOrSucceeded(c, ns, []string{podName}, framework.PodStartTimeout) {
				e2elog.Failf("Pod for run-log-test was not ready")
			}

			logOutput := framework.RunKubectlOrDie(nsFlag, "logs", "-f", "run-log-test")
			gomega.Expect(logOutput).To(gomega.ContainSubstring("EOF"))
		})

		ginkgo.It("should support port-forward", func() {
			ginkgo.By("forwarding the container port to a local port")
			cmd := runPortForward(ns, simplePodName, simplePodPort)
			defer cmd.Stop()

			ginkgo.By("curling local port output")
			localAddr := fmt.Sprintf("http://localhost:%d", cmd.port)
			body, err := curl(localAddr)
			e2elog.Logf("got: %s", body)
			if err != nil {
				e2elog.Failf("Failed http.Get of forwarded port (%s): %v", localAddr, err)
			}
			if !strings.Contains(body, httpdDefaultOutput) {
				e2elog.Failf("Container port output missing expected value. Wanted:'%s', got: %s", httpdDefaultOutput, body)
			}
		})

		ginkgo.It("should handle in-cluster config", func() {
			ginkgo.By("adding rbac permissions")
			// grant the view permission widely to allow inspection of the `invalid` namespace and the default namespace
			err := auth.BindClusterRole(f.ClientSet.RbacV1(), "view", f.Namespace.Name,
				rbacv1.Subject{Kind: rbacv1.ServiceAccountKind, Namespace: f.Namespace.Name, Name: "default"})
			framework.ExpectNoError(err)

			err = auth.WaitForAuthorizationUpdate(f.ClientSet.AuthorizationV1(),
				serviceaccount.MakeUsername(f.Namespace.Name, "default"),
				f.Namespace.Name, "list", schema.GroupResource{Resource: "pods"}, true)
			framework.ExpectNoError(err)

			ginkgo.By("overriding icc with values provided by flags")
			kubectlPath := framework.TestContext.KubectlPath
			// we need the actual kubectl binary, not the script wrapper
			kubectlPathNormalizer := exec.Command("which", kubectlPath)
			if strings.HasSuffix(kubectlPath, "kubectl.sh") {
				kubectlPathNormalizer = exec.Command(kubectlPath, "path")
			}
			kubectlPathNormalized, err := kubectlPathNormalizer.Output()
			framework.ExpectNoError(err)
			kubectlPath = strings.TrimSpace(string(kubectlPathNormalized))

			inClusterHost := strings.TrimSpace(framework.RunHostCmdOrDie(ns, simplePodName, "printenv KUBERNETES_SERVICE_HOST"))
			inClusterPort := strings.TrimSpace(framework.RunHostCmdOrDie(ns, simplePodName, "printenv KUBERNETES_SERVICE_PORT"))
			inClusterURL := net.JoinHostPort(inClusterHost, inClusterPort)
			e2elog.Logf("copying %s to the %s pod", kubectlPath, simplePodName)
			framework.RunKubectlOrDie("cp", kubectlPath, ns+"/"+simplePodName+":/tmp/")

			// Build a kubeconfig file that will make use of the injected ca and token,
			// but point at the DNS host and the default namespace
			tmpDir, err := ioutil.TempDir("", "icc-override")
			overrideKubeconfigName := "icc-override.kubeconfig"
			framework.ExpectNoError(err)
			defer func() { os.Remove(tmpDir) }()
			framework.ExpectNoError(ioutil.WriteFile(filepath.Join(tmpDir, overrideKubeconfigName), []byte(`
kind: Config
apiVersion: v1
clusters:
- cluster:
    api-version: v1
    server: https://kubernetes.default.svc:443
    certificate-authority: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
  name: kubeconfig-cluster
contexts:
- context:
    cluster: kubeconfig-cluster
    namespace: default
    user: kubeconfig-user
  name: kubeconfig-context
current-context: kubeconfig-context
users:
- name: kubeconfig-user
  user:
    tokenFile: /var/run/secrets/kubernetes.io/serviceaccount/token
`), os.FileMode(0755)))
			e2elog.Logf("copying override kubeconfig to the %s pod", simplePodName)
			framework.RunKubectlOrDie("cp", filepath.Join(tmpDir, overrideKubeconfigName), ns+"/"+simplePodName+":/tmp/")

			framework.ExpectNoError(ioutil.WriteFile(filepath.Join(tmpDir, "invalid-configmap-with-namespace.yaml"), []byte(`
kind: ConfigMap
apiVersion: v1
metadata:
  name: "configmap with namespace and invalid name"
  namespace: configmap-namespace
`), os.FileMode(0755)))
			framework.ExpectNoError(ioutil.WriteFile(filepath.Join(tmpDir, "invalid-configmap-without-namespace.yaml"), []byte(`
kind: ConfigMap
apiVersion: v1
metadata:
  name: "configmap without namespace and invalid name"
`), os.FileMode(0755)))
			e2elog.Logf("copying configmap manifests to the %s pod", simplePodName)
			framework.RunKubectlOrDie("cp", filepath.Join(tmpDir, "invalid-configmap-with-namespace.yaml"), ns+"/"+simplePodName+":/tmp/")
			framework.RunKubectlOrDie("cp", filepath.Join(tmpDir, "invalid-configmap-without-namespace.yaml"), ns+"/"+simplePodName+":/tmp/")

			ginkgo.By("getting pods with in-cluster configs")
			execOutput := framework.RunHostCmdOrDie(ns, simplePodName, "/tmp/kubectl get pods --v=6 2>&1")
			gomega.Expect(execOutput).To(gomega.MatchRegexp("httpd +1/1 +Running"))
			gomega.Expect(execOutput).To(gomega.ContainSubstring("Using in-cluster namespace"))
			gomega.Expect(execOutput).To(gomega.ContainSubstring("Using in-cluster configuration"))

			ginkgo.By("creating an object containing a namespace with in-cluster config")
			_, err = framework.RunHostCmd(ns, simplePodName, "/tmp/kubectl create -f /tmp/invalid-configmap-with-namespace.yaml --v=6 2>&1")
			gomega.Expect(err).To(gomega.ContainSubstring("Using in-cluster namespace"))
			gomega.Expect(err).To(gomega.ContainSubstring("Using in-cluster configuration"))

			gomega.Expect(err).To(gomega.ContainSubstring(fmt.Sprintf("POST https://%s/api/v1/namespaces/configmap-namespace/configmaps", inClusterURL)))

			ginkgo.By("creating an object not containing a namespace with in-cluster config")
			_, err = framework.RunHostCmd(ns, simplePodName, "/tmp/kubectl create -f /tmp/invalid-configmap-without-namespace.yaml --v=6 2>&1")
			gomega.Expect(err).To(gomega.ContainSubstring("Using in-cluster namespace"))
			gomega.Expect(err).To(gomega.ContainSubstring("Using in-cluster configuration"))
			gomega.Expect(err).To(gomega.ContainSubstring(fmt.Sprintf("POST https://%s/api/v1/namespaces/%s/configmaps", inClusterURL, f.Namespace.Name)))

			ginkgo.By("trying to use kubectl with invalid token")
			_, err = framework.RunHostCmd(ns, simplePodName, "/tmp/kubectl get pods --token=invalid --v=7 2>&1")
			e2elog.Logf("got err %v", err)
			framework.ExpectError(err)
			gomega.Expect(err).To(gomega.ContainSubstring("Using in-cluster namespace"))
			gomega.Expect(err).To(gomega.ContainSubstring("Using in-cluster configuration"))
			gomega.Expect(err).To(gomega.ContainSubstring("Response Status: 401 Unauthorized"))

			ginkgo.By("trying to use kubectl with invalid server")
			_, err = framework.RunHostCmd(ns, simplePodName, "/tmp/kubectl get pods --server=invalid --v=6 2>&1")
			e2elog.Logf("got err %v", err)
			framework.ExpectError(err)
			gomega.Expect(err).To(gomega.ContainSubstring("Unable to connect to the server"))
			gomega.Expect(err).To(gomega.ContainSubstring("GET http://invalid/api"))

			ginkgo.By("trying to use kubectl with invalid namespace")
			execOutput = framework.RunHostCmdOrDie(ns, simplePodName, "/tmp/kubectl get pods --namespace=invalid --v=6 2>&1")
			gomega.Expect(execOutput).To(gomega.ContainSubstring("No resources found"))
			gomega.Expect(execOutput).ToNot(gomega.ContainSubstring("Using in-cluster namespace"))
			gomega.Expect(execOutput).To(gomega.ContainSubstring("Using in-cluster configuration"))
			gomega.Expect(execOutput).To(gomega.MatchRegexp(fmt.Sprintf("GET http[s]?://[\\[]?%s[\\]]?:%s/api/v1/namespaces/invalid/pods", inClusterHost, inClusterPort)))

			ginkgo.By("trying to use kubectl with kubeconfig")
			execOutput = framework.RunHostCmdOrDie(ns, simplePodName, "/tmp/kubectl get pods --kubeconfig=/tmp/"+overrideKubeconfigName+" --v=6 2>&1")
			gomega.Expect(execOutput).ToNot(gomega.ContainSubstring("Using in-cluster namespace"))
			gomega.Expect(execOutput).ToNot(gomega.ContainSubstring("Using in-cluster configuration"))
			gomega.Expect(execOutput).To(gomega.ContainSubstring("GET https://kubernetes.default.svc:443/api/v1/namespaces/default/pods"))
		})
	})

	framework.KubeDescribe("Kubectl api-versions", func() {
		/*
			Release : v1.9
			Testname: Kubectl, check version v1
			Description: Run kubectl to get api versions, output MUST contain returned versions with ‘v1’ listed.
		*/
		framework.ConformanceIt("should check if v1 is in available api versions ", func() {
			ginkgo.By("validating api versions")
			output := framework.RunKubectlOrDie("api-versions")
			if !strings.Contains(output, "v1") {
				e2elog.Failf("No v1 in kubectl api-versions")
			}
		})
	})

	framework.KubeDescribe("Kubectl get componentstatuses", func() {
		ginkgo.It("should get componentstatuses", func() {
			ginkgo.By("getting list of componentstatuses")
			output := framework.RunKubectlOrDie("get", "componentstatuses", "-o", "jsonpath={.items[*].metadata.name}")
			components := strings.Split(output, " ")
			ginkgo.By("getting details of componentstatuses")
			for _, component := range components {
				ginkgo.By("getting status of " + component)
				framework.RunKubectlOrDie("get", "componentstatuses", component)
			}
		})
	})

	framework.KubeDescribe("Kubectl apply", func() {
		ginkgo.It("should apply a new configuration to an existing RC", func() {
			controllerJSON := commonutils.SubstituteImageName(string(readTestFileOrDie(redisControllerFilename)))

			nsFlag := fmt.Sprintf("--namespace=%v", ns)
			ginkgo.By("creating Redis RC")
			framework.RunKubectlOrDieInput(controllerJSON, "create", "-f", "-", nsFlag)
			ginkgo.By("applying a modified configuration")
			stdin := modifyReplicationControllerConfiguration(controllerJSON)
			framework.NewKubectlCommand("apply", "-f", "-", nsFlag).
				WithStdinReader(stdin).
				ExecOrDie()
			ginkgo.By("checking the result")
			forEachReplicationController(c, ns, "app", "redis", validateReplicationControllerConfiguration)
		})
		ginkgo.It("should reuse port when apply to an existing SVC", func() {
			serviceJSON := readTestFileOrDie(redisServiceFilename)
			nsFlag := fmt.Sprintf("--namespace=%v", ns)

			ginkgo.By("creating Redis SVC")
			framework.RunKubectlOrDieInput(string(serviceJSON[:]), "create", "-f", "-", nsFlag)

			ginkgo.By("getting the original port")
			originalNodePort := framework.RunKubectlOrDie("get", "service", "redis-master", nsFlag, "-o", "jsonpath={.spec.ports[0].port}")

			ginkgo.By("applying the same configuration")
			framework.RunKubectlOrDieInput(string(serviceJSON[:]), "apply", "-f", "-", nsFlag)

			ginkgo.By("getting the port after applying configuration")
			currentNodePort := framework.RunKubectlOrDie("get", "service", "redis-master", nsFlag, "-o", "jsonpath={.spec.ports[0].port}")

			ginkgo.By("checking the result")
			if originalNodePort != currentNodePort {
				e2elog.Failf("port should keep the same")
			}
		})

		ginkgo.It("apply set/view last-applied", func() {
			deployment1Yaml := commonutils.SubstituteImageName(string(readTestFileOrDie(httpdDeployment1Filename)))
			deployment2Yaml := commonutils.SubstituteImageName(string(readTestFileOrDie(httpdDeployment2Filename)))
			deployment3Yaml := commonutils.SubstituteImageName(string(readTestFileOrDie(httpdDeployment3Filename)))
			nsFlag := fmt.Sprintf("--namespace=%v", ns)

			ginkgo.By("deployment replicas number is 2")
			framework.RunKubectlOrDieInput(deployment1Yaml, "apply", "-f", "-", nsFlag)

			ginkgo.By("check the last-applied matches expectations annotations")
			output := framework.RunKubectlOrDieInput(deployment1Yaml, "apply", "view-last-applied", "-f", "-", nsFlag, "-o", "json")
			requiredString := "\"replicas\": 2"
			if !strings.Contains(output, requiredString) {
				e2elog.Failf("Missing %s in kubectl view-last-applied", requiredString)
			}

			ginkgo.By("apply file doesn't have replicas")
			framework.RunKubectlOrDieInput(deployment2Yaml, "apply", "set-last-applied", "-f", "-", nsFlag)

			ginkgo.By("check last-applied has been updated, annotations doesn't have replicas")
			output = framework.RunKubectlOrDieInput(deployment1Yaml, "apply", "view-last-applied", "-f", "-", nsFlag, "-o", "json")
			requiredString = "\"replicas\": 2"
			if strings.Contains(output, requiredString) {
				e2elog.Failf("Presenting %s in kubectl view-last-applied", requiredString)
			}

			ginkgo.By("scale set replicas to 3")
			httpdDeploy := "httpd-deployment"
			debugDiscovery()
			framework.RunKubectlOrDie("scale", "deployment", httpdDeploy, "--replicas=3", nsFlag)

			ginkgo.By("apply file doesn't have replicas but image changed")
			framework.RunKubectlOrDieInput(deployment3Yaml, "apply", "-f", "-", nsFlag)

			ginkgo.By("verify replicas still is 3 and image has been updated")
			output = framework.RunKubectlOrDieInput(deployment3Yaml, "get", "-f", "-", nsFlag, "-o", "json")
			requiredItems := []string{"\"replicas\": 3", imageutils.GetE2EImage(imageutils.Httpd)}
			for _, item := range requiredItems {
				if !strings.Contains(output, item) {
					e2elog.Failf("Missing %s in kubectl apply", item)
				}
			}
		})
	})

	// definitionMatchesGVK returns true if the specified GVK is listed as an x-kubernetes-group-version-kind extension
	definitionMatchesGVK := func(extensions []*openapi_v2.NamedAny, desiredGVK schema.GroupVersionKind) bool {
		for _, extension := range extensions {
			if extension.GetValue().GetYaml() == "" ||
				extension.GetName() != "x-kubernetes-group-version-kind" {
				continue
			}
			var values []map[string]string
			err := yaml.Unmarshal([]byte(extension.GetValue().GetYaml()), &values)
			if err != nil {
				e2elog.Logf("%v\n%s", err, string(extension.GetValue().GetYaml()))
				continue
			}
			for _, value := range values {
				if value["group"] != desiredGVK.Group {
					continue
				}
				if value["version"] != desiredGVK.Version {
					continue
				}
				if value["kind"] != desiredGVK.Kind {
					continue
				}
				return true
			}
		}
		return false
	}

	// schemaForGVK returns a schema (if defined) for the specified GVK
	schemaForGVK := func(desiredGVK schema.GroupVersionKind) *openapi_v2.Schema {
		d, err := f.ClientSet.Discovery().OpenAPISchema()
		if err != nil {
			e2elog.Failf("%v", err)
		}
		if d == nil || d.Definitions == nil {
			return nil
		}
		for _, p := range d.Definitions.AdditionalProperties {
			if p == nil || p.Value == nil {
				continue
			}
			if !definitionMatchesGVK(p.Value.VendorExtension, desiredGVK) {
				continue
			}
			return p.Value
		}
		return nil
	}

	framework.KubeDescribe("Kubectl client-side validation", func() {
		ginkgo.It("should create/apply a CR with unknown fields for CRD with no validation schema", func() {
			ginkgo.By("create CRD with no validation schema")
			crd, err := crd.CreateTestCRD(f)
			if err != nil {
				e2elog.Failf("failed to create test CRD: %v", err)
			}
			defer crd.CleanUp()

			ginkgo.By("sleep for 10s to wait for potential crd openapi publishing alpha feature")
			time.Sleep(10 * time.Second)

			meta := fmt.Sprintf(metaPattern, crd.Crd.Spec.Names.Kind, crd.Crd.Spec.Group, crd.Crd.Spec.Versions[0].Name, "test-cr")
			randomCR := fmt.Sprintf(`{%s,"a":{"b":[{"c":"d"}]}}`, meta)
			if err := createApplyCustomResource(randomCR, f.Namespace.Name, "test-cr", crd); err != nil {
				e2elog.Failf("%v", err)
			}
		})

		ginkgo.It("should create/apply a valid CR for CRD with validation schema", func() {
			ginkgo.By("prepare CRD with validation schema")
			crd, err := crd.CreateTestCRD(f, func(crd *v1beta1.CustomResourceDefinition) {
				props := &v1beta1.JSONSchemaProps{}
				if err := yaml.Unmarshal(schemaFoo, props); err != nil {
					e2elog.Failf("failed to unmarshal schema: %v", err)
				}
				crd.Spec.Validation = &v1beta1.CustomResourceValidation{OpenAPIV3Schema: props}
			})
			if err != nil {
				e2elog.Failf("failed to create test CRD: %v", err)
			}
			defer crd.CleanUp()

			ginkgo.By("sleep for 10s to wait for potential crd openapi publishing alpha feature")
			time.Sleep(10 * time.Second)

			meta := fmt.Sprintf(metaPattern, crd.Crd.Spec.Names.Kind, crd.Crd.Spec.Group, crd.Crd.Spec.Versions[0].Name, "test-cr")
			validCR := fmt.Sprintf(`{%s,"spec":{"bars":[{"name":"test-bar"}]}}`, meta)
			if err := createApplyCustomResource(validCR, f.Namespace.Name, "test-cr", crd); err != nil {
				e2elog.Failf("%v", err)
			}
		})

		ginkgo.It("should create/apply a valid CR with arbitrary-extra properties for CRD with partially-specified validation schema", func() {
			ginkgo.By("prepare CRD with partially-specified validation schema")
			crd, err := crd.CreateTestCRD(f, func(crd *v1beta1.CustomResourceDefinition) {
				props := &v1beta1.JSONSchemaProps{}
				if err := yaml.Unmarshal(schemaFoo, props); err != nil {
					e2elog.Failf("failed to unmarshal schema: %v", err)
				}
				crd.Spec.Validation = &v1beta1.CustomResourceValidation{OpenAPIV3Schema: props}
			})
			if err != nil {
				e2elog.Failf("failed to create test CRD: %v", err)
			}
			defer crd.CleanUp()

			ginkgo.By("sleep for 10s to wait for potential crd openapi publishing alpha feature")
			time.Sleep(10 * time.Second)

			publishedSchema := schemaForGVK(schema.GroupVersionKind{Group: crd.Crd.Spec.Group, Version: crd.Crd.Spec.Version, Kind: crd.Crd.Spec.Names.Kind})
			expectSuccess := false
			if publishedSchema == nil || publishedSchema.Properties == nil || publishedSchema.Properties.AdditionalProperties == nil || len(publishedSchema.Properties.AdditionalProperties) == 0 {
				// expect success in the following cases:
				// - no schema was published
				// - a schema was published with no properties
				expectSuccess = true
				e2elog.Logf("no schema with properties found, expect apply with extra properties to succeed")
			} else {
				e2elog.Logf("schema with properties found, expect apply with extra properties to fail")
			}

			meta := fmt.Sprintf(metaPattern, crd.Crd.Spec.Names.Kind, crd.Crd.Spec.Group, crd.Crd.Spec.Versions[0].Name, "test-cr")
			validArbitraryCR := fmt.Sprintf(`{%s,"spec":{"bars":[{"name":"test-bar"}],"extraProperty":"arbitrary-value"}}`, meta)
			if err := createApplyCustomResource(validArbitraryCR, f.Namespace.Name, "test-cr", crd); err != nil {
				if expectSuccess {
					e2elog.Failf("%v", err)
				}
			} else {
				if !expectSuccess {
					e2elog.Failf("expected error, got none")
				}
			}
		})

	})

	framework.KubeDescribe("Kubectl cluster-info", func() {
		/*
			Release : v1.9
			Testname: Kubectl, cluster info
			Description: Call kubectl to get cluster-info, output MUST contain cluster-info returned and Kubernetes Master SHOULD be running.
		*/
		framework.ConformanceIt("should check if Kubernetes master services is included in cluster-info ", func() {
			ginkgo.By("validating cluster-info")
			output := framework.RunKubectlOrDie("cluster-info")
			// Can't check exact strings due to terminal control commands (colors)
			requiredItems := []string{"Kubernetes master", "is running at"}
			for _, item := range requiredItems {
				if !strings.Contains(output, item) {
					e2elog.Failf("Missing %s in kubectl cluster-info", item)
				}
			}
		})
	})

	framework.KubeDescribe("Kubectl cluster-info dump", func() {
		ginkgo.It("should check if cluster-info dump succeeds", func() {
			ginkgo.By("running cluster-info dump")
			framework.RunKubectlOrDie("cluster-info", "dump")
		})
	})

	framework.KubeDescribe("Kubectl describe", func() {
		/*
			Release : v1.9
			Testname: Kubectl, describe pod or rc
			Description: Deploy a redis controller and a redis service. Kubectl describe pods SHOULD return the name, namespace, labels, state and other information as expected. Kubectl describe on rc, service, node and namespace SHOULD also return proper information.
		*/
		framework.ConformanceIt("should check if kubectl describe prints relevant information for rc and pods ", func() {
			controllerJSON := commonutils.SubstituteImageName(string(readTestFileOrDie(redisControllerFilename)))
			serviceJSON := readTestFileOrDie(redisServiceFilename)

			nsFlag := fmt.Sprintf("--namespace=%v", ns)
			framework.RunKubectlOrDieInput(controllerJSON, "create", "-f", "-", nsFlag)
			framework.RunKubectlOrDieInput(string(serviceJSON[:]), "create", "-f", "-", nsFlag)

			ginkgo.By("Waiting for Redis master to start.")
			waitForOrFailWithDebug(1)

			// Pod
			forEachPod(func(pod v1.Pod) {
				output := framework.RunKubectlOrDie("describe", "pod", pod.Name, nsFlag)
				requiredStrings := [][]string{
					{"Name:", "redis-master-"},
					{"Namespace:", ns},
					{"Node:"},
					{"Labels:", "app=redis"},
					{"role=master"},
					{"Annotations:"},
					{"Status:", "Running"},
					{"IP:"},
					{"Controlled By:", "ReplicationController/redis-master"},
					{"Image:", redisImage},
					{"State:", "Running"},
					{"QoS Class:", "BestEffort"},
				}
				checkOutput(output, requiredStrings)
			})

			// Rc
			requiredStrings := [][]string{
				{"Name:", "redis-master"},
				{"Namespace:", ns},
				{"Selector:", "app=redis,role=master"},
				{"Labels:", "app=redis"},
				{"role=master"},
				{"Annotations:"},
				{"Replicas:", "1 current", "1 desired"},
				{"Pods Status:", "1 Running", "0 Waiting", "0 Succeeded", "0 Failed"},
				{"Pod Template:"},
				{"Image:", redisImage},
				{"Events:"}}
			checkKubectlOutputWithRetry(requiredStrings, "describe", "rc", "redis-master", nsFlag)

			// Service
			output := framework.RunKubectlOrDie("describe", "service", "redis-master", nsFlag)
			requiredStrings = [][]string{
				{"Name:", "redis-master"},
				{"Namespace:", ns},
				{"Labels:", "app=redis"},
				{"role=master"},
				{"Annotations:"},
				{"Selector:", "app=redis", "role=master"},
				{"Type:", "ClusterIP"},
				{"IP:"},
				{"Port:", "<unset>", "6379/TCP"},
				{"Endpoints:"},
				{"Session Affinity:", "None"}}
			checkOutput(output, requiredStrings)

			// Node
			// It should be OK to list unschedulable Nodes here.
			nodes, err := c.CoreV1().Nodes().List(metav1.ListOptions{})
			framework.ExpectNoError(err)
			node := nodes.Items[0]
			output = framework.RunKubectlOrDie("describe", "node", node.Name)
			requiredStrings = [][]string{
				{"Name:", node.Name},
				{"Labels:"},
				{"Annotations:"},
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
				{"Annotations:"},
				{"Status:", "Active"}}
			checkOutput(output, requiredStrings)

			// Quota and limitrange are skipped for now.
		})
	})

	framework.KubeDescribe("Kubectl expose", func() {
		/*
			Release : v1.9
			Testname: Kubectl, create service, replication controller
			Description: Create a Pod running redis master listening to port 6379. Using kubectl expose the redis master  replication controllers at port 1234. Validate that the replication controller is listening on port 1234 and the target port is set to 6379, port that redis master is listening. Using kubectl expose the redis master as a service at port 2345. The service MUST be listening on port 2345 and the target port is set to 6379, port that redis master is listening.
		*/
		framework.ConformanceIt("should create services for rc ", func() {
			controllerJSON := commonutils.SubstituteImageName(string(readTestFileOrDie(redisControllerFilename)))
			nsFlag := fmt.Sprintf("--namespace=%v", ns)

			redisPort := 6379

			ginkgo.By("creating Redis RC")

			e2elog.Logf("namespace %v", ns)
			framework.RunKubectlOrDieInput(controllerJSON, "create", "-f", "-", nsFlag)

			// It may take a while for the pods to get registered in some cases, wait to be sure.
			ginkgo.By("Waiting for Redis master to start.")
			waitForOrFailWithDebug(1)
			forEachPod(func(pod v1.Pod) {
				e2elog.Logf("wait on redis-master startup in %v ", ns)
				framework.LookForStringInLog(ns, pod.Name, "redis-master", "Ready to accept connections", framework.PodStartTimeout)
			})
			validateService := func(name string, servicePort int, timeout time.Duration) {
				err := wait.Poll(framework.Poll, timeout, func() (bool, error) {
					ep, err := c.CoreV1().Endpoints(ns).Get(name, metav1.GetOptions{})
					if err != nil {
						// log the real error
						e2elog.Logf("Get endpoints failed (interval %v): %v", framework.Poll, err)

						// if the error is API not found or could not find default credentials or TLS handshake timeout, try again
						if apierrs.IsNotFound(err) ||
							apierrs.IsUnauthorized(err) ||
							apierrs.IsServerTimeout(err) {
							err = nil
						}
						return false, err
					}

					uidToPort := e2eendpoints.GetContainerPortsByPodUID(ep)
					if len(uidToPort) == 0 {
						e2elog.Logf("No endpoint found, retrying")
						return false, nil
					}
					if len(uidToPort) > 1 {
						e2elog.Failf("Too many endpoints found")
					}
					for _, port := range uidToPort {
						if port[0] != redisPort {
							e2elog.Failf("Wrong endpoint port: %d", port[0])
						}
					}
					return true, nil
				})
				framework.ExpectNoError(err)

				e2eservice, err := c.CoreV1().Services(ns).Get(name, metav1.GetOptions{})
				framework.ExpectNoError(err)

				if len(e2eservice.Spec.Ports) != 1 {
					e2elog.Failf("1 port is expected")
				}
				port := e2eservice.Spec.Ports[0]
				if port.Port != int32(servicePort) {
					e2elog.Failf("Wrong service port: %d", port.Port)
				}
				if port.TargetPort.IntValue() != redisPort {
					e2elog.Failf("Wrong target port: %d", port.TargetPort.IntValue())
				}
			}

			ginkgo.By("exposing RC")
			framework.RunKubectlOrDie("expose", "rc", "redis-master", "--name=rm2", "--port=1234", fmt.Sprintf("--target-port=%d", redisPort), nsFlag)
			framework.WaitForService(c, ns, "rm2", true, framework.Poll, framework.ServiceStartTimeout)
			validateService("rm2", 1234, framework.ServiceStartTimeout)

			ginkgo.By("exposing service")
			framework.RunKubectlOrDie("expose", "service", "rm2", "--name=rm3", "--port=2345", fmt.Sprintf("--target-port=%d", redisPort), nsFlag)
			framework.WaitForService(c, ns, "rm3", true, framework.Poll, framework.ServiceStartTimeout)
			validateService("rm3", 2345, framework.ServiceStartTimeout)
		})
	})

	framework.KubeDescribe("Kubectl label", func() {
		var podYaml string
		var nsFlag string
		ginkgo.BeforeEach(func() {
			ginkgo.By("creating the pod")
			podYaml = commonutils.SubstituteImageName(string(readTestFileOrDie("pause-pod.yaml.in")))
			nsFlag = fmt.Sprintf("--namespace=%v", ns)
			framework.RunKubectlOrDieInput(podYaml, "create", "-f", "-", nsFlag)
			gomega.Expect(e2epod.CheckPodsRunningReady(c, ns, []string{pausePodName}, framework.PodStartTimeout)).To(gomega.BeTrue())
		})
		ginkgo.AfterEach(func() {
			cleanupKubectlInputs(podYaml, ns, pausePodSelector)
		})

		/*
			Release : v1.9
			Testname: Kubectl, label update
			Description: When a Pod is running, update a Label using ‘kubectl label’ command. The label MUST be created in the Pod. A ‘kubectl get pod’ with -l option on the container MUST verify that the label can be read back. Use ‘kubectl label label-’ to remove the label. ‘kubectl get pod’ with -l option SHOULD not list the deleted label as the label is removed.
		*/
		framework.ConformanceIt("should update the label on a resource ", func() {
			labelName := "testing-label"
			labelValue := "testing-label-value"

			ginkgo.By("adding the label " + labelName + " with value " + labelValue + " to a pod")
			framework.RunKubectlOrDie("label", "pods", pausePodName, labelName+"="+labelValue, nsFlag)
			ginkgo.By("verifying the pod has the label " + labelName + " with the value " + labelValue)
			output := framework.RunKubectlOrDie("get", "pod", pausePodName, "-L", labelName, nsFlag)
			if !strings.Contains(output, labelValue) {
				e2elog.Failf("Failed updating label " + labelName + " to the pod " + pausePodName)
			}

			ginkgo.By("removing the label " + labelName + " of a pod")
			framework.RunKubectlOrDie("label", "pods", pausePodName, labelName+"-", nsFlag)
			ginkgo.By("verifying the pod doesn't have the label " + labelName)
			output = framework.RunKubectlOrDie("get", "pod", pausePodName, "-L", labelName, nsFlag)
			if strings.Contains(output, labelValue) {
				e2elog.Failf("Failed removing label " + labelName + " of the pod " + pausePodName)
			}
		})
	})

	framework.KubeDescribe("Kubectl copy", func() {
		var podYaml string
		var nsFlag string
		ginkgo.BeforeEach(func() {
			ginkgo.By("creating the pod")
			nsFlag = fmt.Sprintf("--namespace=%v", ns)
			podYaml = commonutils.SubstituteImageName(string(readTestFileOrDie("busybox-pod.yaml")))
			framework.RunKubectlOrDieInput(podYaml, "create", "-f", "-", nsFlag)
			gomega.Expect(e2epod.CheckPodsRunningReady(c, ns, []string{busyboxPodName}, framework.PodStartTimeout)).To(gomega.BeTrue())
		})
		ginkgo.AfterEach(func() {
			cleanupKubectlInputs(podYaml, ns, busyboxPodSelector)
		})

		/*
			Release : v1.12
			Testname: Kubectl, copy
			Description: When a Pod is running, copy a known file from it to a temporary local destination.
		*/
		ginkgo.It("should copy a file from a running Pod", func() {
			remoteContents := "foobar\n"
			podSource := fmt.Sprintf("%s:/root/foo/bar/foo.bar", busyboxPodName)
			tempDestination, err := ioutil.TempFile(os.TempDir(), "copy-foobar")
			if err != nil {
				e2elog.Failf("Failed creating temporary destination file: %v", err)
			}

			ginkgo.By("specifying a remote filepath " + podSource + " on the pod")
			framework.RunKubectlOrDie("cp", podSource, tempDestination.Name(), nsFlag)
			ginkgo.By("verifying that the contents of the remote file " + podSource + " have been copied to a local file " + tempDestination.Name())
			localData, err := ioutil.ReadAll(tempDestination)
			if err != nil {
				e2elog.Failf("Failed reading temporary local file: %v", err)
			}
			if string(localData) != remoteContents {
				e2elog.Failf("Failed copying remote file contents. Expected %s but got %s", remoteContents, string(localData))
			}
		})
	})

	framework.KubeDescribe("Kubectl logs", func() {
		var nsFlag string
		podName := "logs-generator"
		containerName := "logs-generator"
		ginkgo.BeforeEach(func() {
			ginkgo.By("creating an pod")
			nsFlag = fmt.Sprintf("--namespace=%v", ns)
			// Agnhost image generates logs for a total of 100 lines over 20s.
			framework.RunKubectlOrDie("run", podName, "--generator=run-pod/v1", "--image="+agnhostImage, nsFlag, "--", "logs-generator", "--log-lines-total", "100", "--run-duration", "20s")
		})
		ginkgo.AfterEach(func() {
			framework.RunKubectlOrDie("delete", "pod", podName, nsFlag)
		})

		/*
			Release : v1.9
			Testname: Kubectl, logs
			Description: When a Pod is running then it MUST generate logs.
			Starting a Pod should have a expected log line. Also log command options MUST work as expected and described below.
				‘kubectl log -tail=1’ should generate a output of one line, the last line in the log.
				‘kubectl --limit-bytes=1’ should generate a single byte output.
				‘kubectl --tail=1 --timestamp should generate one line with timestamp in RFC3339 format
				‘kubectl --since=1s’ should output logs that are only 1 second older from now
				‘kubectl --since=24h’ should output logs that are only 1 day older from now
		*/
		framework.ConformanceIt("should be able to retrieve and filter logs ", func() {
			// Split("something\n", "\n") returns ["something", ""], so
			// strip trailing newline first
			lines := func(out string) []string {
				return strings.Split(strings.TrimRight(out, "\n"), "\n")
			}

			ginkgo.By("Waiting for log generator to start.")
			if !e2epod.CheckPodsRunningReadyOrSucceeded(c, ns, []string{podName}, framework.PodStartTimeout) {
				e2elog.Failf("Pod %s was not ready", podName)
			}

			ginkgo.By("checking for a matching strings")
			_, err := framework.LookForStringInLog(ns, podName, containerName, "/api/v1/namespaces/kube-system", framework.PodStartTimeout)
			framework.ExpectNoError(err)

			ginkgo.By("limiting log lines")
			out := framework.RunKubectlOrDie("log", podName, containerName, nsFlag, "--tail=1")
			gomega.Expect(len(out)).NotTo(gomega.BeZero())
			framework.ExpectEqual(len(lines(out)), 1)

			ginkgo.By("limiting log bytes")
			out = framework.RunKubectlOrDie("log", podName, containerName, nsFlag, "--limit-bytes=1")
			framework.ExpectEqual(len(lines(out)), 1)
			framework.ExpectEqual(len(out), 1)

			ginkgo.By("exposing timestamps")
			out = framework.RunKubectlOrDie("log", podName, containerName, nsFlag, "--tail=1", "--timestamps")
			l := lines(out)
			framework.ExpectEqual(len(l), 1)
			words := strings.Split(l[0], " ")
			gomega.Expect(len(words)).To(gomega.BeNumerically(">", 1))
			if _, err := time.Parse(time.RFC3339Nano, words[0]); err != nil {
				if _, err := time.Parse(time.RFC3339, words[0]); err != nil {
					e2elog.Failf("expected %q to be RFC3339 or RFC3339Nano", words[0])
				}
			}

			ginkgo.By("restricting to a time range")
			// Note: we must wait at least two seconds,
			// because the granularity is only 1 second and
			// it could end up rounding the wrong way.
			time.Sleep(2500 * time.Millisecond) // ensure that startup logs on the node are seen as older than 1s
			recentOut := framework.RunKubectlOrDie("log", podName, containerName, nsFlag, "--since=1s")
			recent := len(strings.Split(recentOut, "\n"))
			olderOut := framework.RunKubectlOrDie("log", podName, containerName, nsFlag, "--since=24h")
			older := len(strings.Split(olderOut, "\n"))
			gomega.Expect(recent).To(gomega.BeNumerically("<", older), "expected recent(%v) to be less than older(%v)\nrecent lines:\n%v\nolder lines:\n%v\n", recent, older, recentOut, olderOut)
		})
	})

	framework.KubeDescribe("Kubectl patch", func() {
		/*
			Release : v1.9
			Testname: Kubectl, patch to annotate
			Description: Start running a redis master and a replication controller. When the pod is running, using ‘kubectl patch’ command add annotations. The annotation MUST be added to running pods and SHOULD be able to read added annotations from each of the Pods running under the replication controller.
		*/
		framework.ConformanceIt("should add annotations for pods in rc ", func() {
			controllerJSON := commonutils.SubstituteImageName(string(readTestFileOrDie(redisControllerFilename)))
			nsFlag := fmt.Sprintf("--namespace=%v", ns)
			ginkgo.By("creating Redis RC")
			framework.RunKubectlOrDieInput(controllerJSON, "create", "-f", "-", nsFlag)
			ginkgo.By("Waiting for Redis master to start.")
			waitForOrFailWithDebug(1)
			ginkgo.By("patching all pods")
			forEachPod(func(pod v1.Pod) {
				framework.RunKubectlOrDie("patch", "pod", pod.Name, nsFlag, "-p", "{\"metadata\":{\"annotations\":{\"x\":\"y\"}}}")
			})

			ginkgo.By("checking annotations")
			forEachPod(func(pod v1.Pod) {
				found := false
				for key, val := range pod.Annotations {
					if key == "x" && val == "y" {
						found = true
						break
					}
				}
				if !found {
					e2elog.Failf("Added annotation not found")
				}
			})
		})
	})

	framework.KubeDescribe("Kubectl version", func() {
		/*
			Release : v1.9
			Testname: Kubectl, version
			Description: The command ‘kubectl version’ MUST return the major, minor versions,  GitCommit, etc of the Client and the Server that the kubectl is configured to connect to.
		*/
		framework.ConformanceIt("should check is all data is printed ", func() {
			version := framework.RunKubectlOrDie("version")
			requiredItems := []string{"Client Version:", "Server Version:", "Major:", "Minor:", "GitCommit:"}
			for _, item := range requiredItems {
				if !strings.Contains(version, item) {
					e2elog.Failf("Required item %s not found in %s", item, version)
				}
			}
		})
	})

	framework.KubeDescribe("Kubectl run default", func() {
		var nsFlag string
		var name string

		var cleanUp func()

		ginkgo.BeforeEach(func() {
			nsFlag = fmt.Sprintf("--namespace=%v", ns)
			name = "e2e-test-httpd-deployment"
			cleanUp = func() { framework.RunKubectlOrDie("delete", "deployment", name, nsFlag) }
		})

		ginkgo.AfterEach(func() {
			cleanUp()
		})

		/*
			Release : v1.9
			Testname: Kubectl, run default
			Description: Command ‘kubectl run’ MUST create a running pod with possible replicas given a image using the option --image=’httpd’. The running Pod SHOULD have one container and the container SHOULD be running the image specified in the ‘run’ command.
		*/
		framework.ConformanceIt("should create an rc or deployment from an image ", func() {
			ginkgo.By("running the image " + httpdImage)
			framework.RunKubectlOrDie("run", name, "--image="+httpdImage, nsFlag)
			ginkgo.By("verifying the pod controlled by " + name + " gets created")
			label := labels.SelectorFromSet(labels.Set(map[string]string{"run": name}))
			podlist, err := e2epod.WaitForPodsWithLabel(c, ns, label)
			if err != nil {
				e2elog.Failf("Failed getting pod controlled by %s: %v", name, err)
			}
			pods := podlist.Items
			if pods == nil || len(pods) != 1 || len(pods[0].Spec.Containers) != 1 || pods[0].Spec.Containers[0].Image != httpdImage {
				framework.RunKubectlOrDie("get", "pods", "-L", "run", nsFlag)
				e2elog.Failf("Failed creating 1 pod with expected image %s. Number of pods = %v", httpdImage, len(pods))
			}
		})
	})

	framework.KubeDescribe("Kubectl run rc", func() {
		var nsFlag string
		var rcName string

		ginkgo.BeforeEach(func() {
			nsFlag = fmt.Sprintf("--namespace=%v", ns)
			rcName = "e2e-test-httpd-rc"
		})

		ginkgo.AfterEach(func() {
			framework.RunKubectlOrDie("delete", "rc", rcName, nsFlag)
		})

		/*
			Release : v1.9
			Testname: Kubectl, run rc
			Description: Command ‘kubectl run’ MUST create a running rc with default one replicas given a image using the option --image=’httpd’. The running replication controller SHOULD have one container and the container SHOULD be running the image specified in the ‘run’ command. Also there MUST be 1 pod controlled by this replica set running 1 container with the image specified. A ‘kubetctl logs’ command MUST return the logs from the container in the replication controller.
		*/
		framework.ConformanceIt("should create an rc from an image ", func() {
			ginkgo.By("running the image " + httpdImage)
			framework.RunKubectlOrDie("run", rcName, "--image="+httpdImage, "--generator=run/v1", nsFlag)
			ginkgo.By("verifying the rc " + rcName + " was created")
			rc, err := c.CoreV1().ReplicationControllers(ns).Get(rcName, metav1.GetOptions{})
			if err != nil {
				e2elog.Failf("Failed getting rc %s: %v", rcName, err)
			}
			containers := rc.Spec.Template.Spec.Containers
			if checkContainersImage(containers, httpdImage) {
				e2elog.Failf("Failed creating rc %s for 1 pod with expected image %s", rcName, httpdImage)
			}

			ginkgo.By("verifying the pod controlled by rc " + rcName + " was created")
			label := labels.SelectorFromSet(labels.Set(map[string]string{"run": rcName}))
			podlist, err := e2epod.WaitForPodsWithLabel(c, ns, label)
			if err != nil {
				e2elog.Failf("Failed getting pod controlled by rc %s: %v", rcName, err)
			}
			pods := podlist.Items
			if pods == nil || len(pods) != 1 || len(pods[0].Spec.Containers) != 1 || pods[0].Spec.Containers[0].Image != httpdImage {
				framework.RunKubectlOrDie("get", "pods", "-L", "run", nsFlag)
				e2elog.Failf("Failed creating 1 pod with expected image %s. Number of pods = %v", httpdImage, len(pods))
			}

			ginkgo.By("confirm that you can get logs from an rc")
			podNames := []string{}
			for _, pod := range pods {
				podNames = append(podNames, pod.Name)
			}
			if !e2epod.CheckPodsRunningReady(c, ns, podNames, framework.PodStartTimeout) {
				e2elog.Failf("Pods for rc %s were not ready", rcName)
			}
			_, err = framework.RunKubectl("logs", "rc/"+rcName, nsFlag)
			// a non-nil error is fine as long as we actually found a pod.
			if err != nil && !strings.Contains(err.Error(), " in pod ") {
				e2elog.Failf("Failed getting logs by rc %s: %v", rcName, err)
			}
		})
	})

	framework.KubeDescribe("Kubectl rolling-update", func() {
		var nsFlag string
		var rcName string
		var c clientset.Interface

		ginkgo.BeforeEach(func() {
			c = f.ClientSet
			nsFlag = fmt.Sprintf("--namespace=%v", ns)
			rcName = "e2e-test-httpd-rc"
		})

		ginkgo.AfterEach(func() {
			framework.RunKubectlOrDie("delete", "rc", rcName, nsFlag)
		})

		/*
			Release : v1.9
			Testname: Kubectl, rolling update
			Description: Command ‘kubectl rolling-update’ MUST replace the specified replication controller with a new replication controller by updating one pod at a time to use the new Pod spec.
		*/
		framework.ConformanceIt("should support rolling-update to same image ", func() {
			ginkgo.By("running the image " + httpdImage)
			framework.RunKubectlOrDie("run", rcName, "--image="+httpdImage, "--generator=run/v1", nsFlag)
			ginkgo.By("verifying the rc " + rcName + " was created")
			rc, err := c.CoreV1().ReplicationControllers(ns).Get(rcName, metav1.GetOptions{})
			if err != nil {
				e2elog.Failf("Failed getting rc %s: %v", rcName, err)
			}
			containers := rc.Spec.Template.Spec.Containers
			if checkContainersImage(containers, httpdImage) {
				e2elog.Failf("Failed creating rc %s for 1 pod with expected image %s", rcName, httpdImage)
			}
			framework.WaitForRCToStabilize(c, ns, rcName, framework.PodStartTimeout)

			ginkgo.By("rolling-update to same image controller")

			debugDiscovery()
			runKubectlRetryOrDie("rolling-update", rcName, "--update-period=1s", "--image="+httpdImage, "--image-pull-policy="+string(v1.PullIfNotPresent), nsFlag)
			framework.ValidateController(c, httpdImage, 1, rcName, "run="+rcName, noOpValidatorFn, ns)
		})
	})

	framework.KubeDescribe("Kubectl run deployment", func() {
		var nsFlag string
		var dName string

		ginkgo.BeforeEach(func() {
			nsFlag = fmt.Sprintf("--namespace=%v", ns)
			dName = "e2e-test-httpd-deployment"
		})

		ginkgo.AfterEach(func() {
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
			framework.ExpectNoError(err)
		})

		/*
			Release : v1.9
			Testname: Kubectl, run deployment
			Description: Command ‘kubectl run’ MUST create a deployment, with --generator=deployment, when a image name is specified in the run command. After the run command there SHOULD be a deployment that should exist with one container running the specified image. Also there SHOULD be a Pod that is controlled by this deployment, with a container running the specified image.
		*/
		framework.ConformanceIt("should create a deployment from an image ", func() {
			ginkgo.By("running the image " + httpdImage)
			framework.RunKubectlOrDie("run", dName, "--image="+httpdImage, "--generator=deployment/apps.v1", nsFlag)
			ginkgo.By("verifying the deployment " + dName + " was created")
			d, err := c.AppsV1().Deployments(ns).Get(dName, metav1.GetOptions{})
			if err != nil {
				e2elog.Failf("Failed getting deployment %s: %v", dName, err)
			}
			containers := d.Spec.Template.Spec.Containers
			if checkContainersImage(containers, httpdImage) {
				e2elog.Failf("Failed creating deployment %s for 1 pod with expected image %s", dName, httpdImage)
			}

			ginkgo.By("verifying the pod controlled by deployment " + dName + " was created")
			label := labels.SelectorFromSet(labels.Set(map[string]string{"run": dName}))
			podlist, err := e2epod.WaitForPodsWithLabel(c, ns, label)
			if err != nil {
				e2elog.Failf("Failed getting pod controlled by deployment %s: %v", dName, err)
			}
			pods := podlist.Items
			if pods == nil || len(pods) != 1 || len(pods[0].Spec.Containers) != 1 || pods[0].Spec.Containers[0].Image != httpdImage {
				framework.RunKubectlOrDie("get", "pods", "-L", "run", nsFlag)
				e2elog.Failf("Failed creating 1 pod with expected image %s. Number of pods = %v", httpdImage, len(pods))
			}
		})
	})

	framework.KubeDescribe("Kubectl run job", func() {
		var nsFlag string
		var jobName string

		ginkgo.BeforeEach(func() {
			nsFlag = fmt.Sprintf("--namespace=%v", ns)
			jobName = "e2e-test-httpd-job"
		})

		ginkgo.AfterEach(func() {
			framework.RunKubectlOrDie("delete", "jobs", jobName, nsFlag)
		})

		/*
			Release : v1.9
			Testname: Kubectl, run job
			Description: Command ‘kubectl run’ MUST create a job, with --generator=job, when a image name is specified in the run command. After the run command there SHOULD be a job that should exist with one container running the specified image. Also there SHOULD be a restart policy on the job spec that SHOULD match the command line.
		*/
		framework.ConformanceIt("should create a job from an image when restart is OnFailure ", func() {
			ginkgo.By("running the image " + httpdImage)
			framework.RunKubectlOrDie("run", jobName, "--restart=OnFailure", "--generator=job/v1", "--image="+httpdImage, nsFlag)
			ginkgo.By("verifying the job " + jobName + " was created")
			job, err := c.BatchV1().Jobs(ns).Get(jobName, metav1.GetOptions{})
			if err != nil {
				e2elog.Failf("Failed getting job %s: %v", jobName, err)
			}
			containers := job.Spec.Template.Spec.Containers
			if checkContainersImage(containers, httpdImage) {
				e2elog.Failf("Failed creating job %s for 1 pod with expected image %s: %#v", jobName, httpdImage, containers)
			}
			if job.Spec.Template.Spec.RestartPolicy != v1.RestartPolicyOnFailure {
				e2elog.Failf("Failed creating a job with correct restart policy for --restart=OnFailure")
			}
		})
	})

	framework.KubeDescribe("Kubectl run CronJob", func() {
		var nsFlag string
		var cjName string

		ginkgo.BeforeEach(func() {
			nsFlag = fmt.Sprintf("--namespace=%v", ns)
			cjName = "e2e-test-echo-cronjob-beta"
		})

		ginkgo.AfterEach(func() {
			framework.RunKubectlOrDie("delete", "cronjobs", cjName, nsFlag)
		})

		ginkgo.It("should create a CronJob", func() {
			framework.SkipIfMissingResource(f.DynamicClient, cronJobGroupVersionResourceBeta, f.Namespace.Name)

			schedule := "*/5 * * * ?"
			framework.RunKubectlOrDie("run", cjName, "--restart=OnFailure", "--generator=cronjob/v1beta1",
				"--schedule="+schedule, "--image="+busyboxImage, nsFlag)
			ginkgo.By("verifying the CronJob " + cjName + " was created")
			cj, err := c.BatchV1beta1().CronJobs(ns).Get(cjName, metav1.GetOptions{})
			if err != nil {
				e2elog.Failf("Failed getting CronJob %s: %v", cjName, err)
			}
			if cj.Spec.Schedule != schedule {
				e2elog.Failf("Failed creating a CronJob with correct schedule %s", schedule)
			}
			containers := cj.Spec.JobTemplate.Spec.Template.Spec.Containers
			if checkContainersImage(containers, busyboxImage) {
				e2elog.Failf("Failed creating CronJob %s for 1 pod with expected image %s: %#v", cjName, busyboxImage, containers)
			}
			if cj.Spec.JobTemplate.Spec.Template.Spec.RestartPolicy != v1.RestartPolicyOnFailure {
				e2elog.Failf("Failed creating a CronJob with correct restart policy for --restart=OnFailure")
			}
		})
	})

	framework.KubeDescribe("Kubectl run pod", func() {
		var nsFlag string
		var podName string

		ginkgo.BeforeEach(func() {
			nsFlag = fmt.Sprintf("--namespace=%v", ns)
			podName = "e2e-test-httpd-pod"
		})

		ginkgo.AfterEach(func() {
			framework.RunKubectlOrDie("delete", "pods", podName, nsFlag)
		})

		/*
			Release : v1.9
			Testname: Kubectl, run pod
			Description: Command ‘kubectl run’ MUST create a pod, with --generator=run-pod, when a image name is specified in the run command. After the run command there SHOULD be a pod that should exist with one container running the specified image.
		*/
		framework.ConformanceIt("should create a pod from an image when restart is Never ", func() {
			ginkgo.By("running the image " + httpdImage)
			framework.RunKubectlOrDie("run", podName, "--restart=Never", "--generator=run-pod/v1", "--image="+httpdImage, nsFlag)
			ginkgo.By("verifying the pod " + podName + " was created")
			pod, err := c.CoreV1().Pods(ns).Get(podName, metav1.GetOptions{})
			if err != nil {
				e2elog.Failf("Failed getting pod %s: %v", podName, err)
			}
			containers := pod.Spec.Containers
			if checkContainersImage(containers, httpdImage) {
				e2elog.Failf("Failed creating pod %s with expected image %s", podName, httpdImage)
			}
			if pod.Spec.RestartPolicy != v1.RestartPolicyNever {
				e2elog.Failf("Failed creating a pod with correct restart policy for --restart=Never")
			}
		})
	})

	framework.KubeDescribe("Kubectl replace", func() {
		var nsFlag string
		var podName string

		ginkgo.BeforeEach(func() {
			nsFlag = fmt.Sprintf("--namespace=%v", ns)
			podName = "e2e-test-httpd-pod"
		})

		ginkgo.AfterEach(func() {
			framework.RunKubectlOrDie("delete", "pods", podName, nsFlag)
		})

		/*
			Release : v1.9
			Testname: Kubectl, replace
			Description: Command ‘kubectl replace’ on a existing Pod with a new spec MUST update the image of the container running in the Pod. A -f option to ‘kubectl replace’ SHOULD force to re-create the resource. The new Pod SHOULD have the container with new change to the image.
		*/
		framework.ConformanceIt("should update a single-container pod's image ", func() {
			ginkgo.By("running the image " + httpdImage)
			framework.RunKubectlOrDie("run", podName, "--generator=run-pod/v1", "--image="+httpdImage, "--labels=run="+podName, nsFlag)

			ginkgo.By("verifying the pod " + podName + " is running")
			label := labels.SelectorFromSet(labels.Set(map[string]string{"run": podName}))
			err := testutils.WaitForPodsWithLabelRunning(c, ns, label)
			if err != nil {
				e2elog.Failf("Failed getting pod %s: %v", podName, err)
			}

			ginkgo.By("verifying the pod " + podName + " was created")
			podJSON := framework.RunKubectlOrDie("get", "pod", podName, nsFlag, "-o", "json")
			if !strings.Contains(podJSON, podName) {
				e2elog.Failf("Failed to find pod %s in [%s]", podName, podJSON)
			}

			ginkgo.By("replace the image in the pod")
			podJSON = strings.Replace(podJSON, httpdImage, busyboxImage, 1)
			framework.RunKubectlOrDieInput(podJSON, "replace", "-f", "-", nsFlag)

			ginkgo.By("verifying the pod " + podName + " has the right image " + busyboxImage)
			pod, err := c.CoreV1().Pods(ns).Get(podName, metav1.GetOptions{})
			if err != nil {
				e2elog.Failf("Failed getting deployment %s: %v", podName, err)
			}
			containers := pod.Spec.Containers
			if checkContainersImage(containers, busyboxImage) {
				e2elog.Failf("Failed creating pod with expected image %s", busyboxImage)
			}
		})
	})

	framework.KubeDescribe("Kubectl run --rm job", func() {
		jobName := "e2e-test-rm-busybox-job"

		/*
			Release : v1.9
			Testname: Kubectl, run job with --rm
			Description: Start a job with a Pod using ‘kubectl run’ but specify --rm=true. Wait for the Pod to start running by verifying that there is output as expected. Now verify that the job has exited and cannot be found. With --rm=true option the job MUST start by running the image specified and then get deleted itself.
		*/
		framework.ConformanceIt("should create a job from an image, then delete the job ", func() {
			nsFlag := fmt.Sprintf("--namespace=%v", ns)

			ginkgo.By("executing a command with run --rm and attach with stdin")
			t := time.NewTimer(runJobTimeout)
			defer t.Stop()
			runOutput := framework.NewKubectlCommand(nsFlag, "run", jobName, "--image="+busyboxImage, "--rm=true", "--generator=job/v1", "--restart=OnFailure", "--attach=true", "--stdin", "--", "sh", "-c", "cat && echo 'stdin closed'").
				WithStdinData("abcd1234").
				WithTimeout(t.C).
				ExecOrDie()
			gomega.Expect(runOutput).To(gomega.ContainSubstring("abcd1234"))
			gomega.Expect(runOutput).To(gomega.ContainSubstring("stdin closed"))

			err := jobutil.WaitForJobGone(c, ns, jobName, wait.ForeverTestTimeout)
			framework.ExpectNoError(err)

			ginkgo.By("verifying the job " + jobName + " was deleted")
			_, err = c.BatchV1().Jobs(ns).Get(jobName, metav1.GetOptions{})
			framework.ExpectError(err)
			gomega.Expect(apierrs.IsNotFound(err)).To(gomega.BeTrue())
		})
	})

	framework.KubeDescribe("Proxy server", func() {
		// TODO: test proxy options (static, prefix, etc)
		/*
			Release : v1.9
			Testname: Kubectl, proxy port zero
			Description: Start a proxy server on port zero by running ‘kubectl proxy’ with --port=0. Call the proxy server by requesting api versions from unix socket. The proxy server MUST provide at least one version string.
		*/
		framework.ConformanceIt("should support proxy with --port 0 ", func() {
			ginkgo.By("starting the proxy server")
			port, cmd, err := startProxyServer()
			if cmd != nil {
				defer framework.TryKill(cmd)
			}
			if err != nil {
				e2elog.Failf("Failed to start proxy server: %v", err)
			}
			ginkgo.By("curling proxy /api/ output")
			localAddr := fmt.Sprintf("http://localhost:%d/api/", port)
			apiVersions, err := getAPIVersions(localAddr)
			if err != nil {
				e2elog.Failf("Expected at least one supported apiversion, got error %v", err)
			}
			if len(apiVersions.Versions) < 1 {
				e2elog.Failf("Expected at least one supported apiversion, got %v", apiVersions)
			}
		})

		/*
			Release : v1.9
			Testname: Kubectl, proxy socket
			Description: Start a proxy server on by running ‘kubectl proxy’ with --unix-socket=<some path>. Call the proxy server by requesting api versions from  http://locahost:0/api. The proxy server MUST provide at least one version string
		*/
		framework.ConformanceIt("should support --unix-socket=/path ", func() {
			ginkgo.By("Starting the proxy")
			tmpdir, err := ioutil.TempDir("", "kubectl-proxy-unix")
			if err != nil {
				e2elog.Failf("Failed to create temporary directory: %v", err)
			}
			path := filepath.Join(tmpdir, "test")
			defer os.Remove(path)
			defer os.Remove(tmpdir)
			cmd := framework.KubectlCmd("proxy", fmt.Sprintf("--unix-socket=%s", path))
			stdout, stderr, err := framework.StartCmdAndStreamOutput(cmd)
			if err != nil {
				e2elog.Failf("Failed to start kubectl command: %v", err)
			}
			defer stdout.Close()
			defer stderr.Close()
			defer framework.TryKill(cmd)
			buf := make([]byte, 128)
			if _, err = stdout.Read(buf); err != nil {
				e2elog.Failf("Expected output from kubectl proxy: %v", err)
			}
			ginkgo.By("retrieving proxy /api/ output")
			_, err = curlUnix("http://unused/api", path)
			if err != nil {
				e2elog.Failf("Failed get of /api at %s: %v", path, err)
			}
		})
	})

	// This test must run [Serial] because it modifies the node so it doesn't allow pods to execute on
	// it, which will affect anything else running in parallel.
	framework.KubeDescribe("Kubectl taint [Serial]", func() {
		ginkgo.It("should update the taint on a node", func() {
			testTaint := v1.Taint{
				Key:    fmt.Sprintf("kubernetes.io/e2e-taint-key-001-%s", string(uuid.NewUUID())),
				Value:  "testing-taint-value",
				Effect: v1.TaintEffectNoSchedule,
			}

			nodeName := scheduling.GetNodeThatCanRunPod(f)

			ginkgo.By("adding the taint " + testTaint.ToString() + " to a node")
			runKubectlRetryOrDie("taint", "nodes", nodeName, testTaint.ToString())
			defer framework.RemoveTaintOffNode(f.ClientSet, nodeName, testTaint)

			ginkgo.By("verifying the node has the taint " + testTaint.ToString())
			output := runKubectlRetryOrDie("describe", "node", nodeName)
			requiredStrings := [][]string{
				{"Name:", nodeName},
				{"Taints:"},
				{testTaint.ToString()},
			}
			checkOutput(output, requiredStrings)

			ginkgo.By("removing the taint " + testTaint.ToString() + " of a node")
			runKubectlRetryOrDie("taint", "nodes", nodeName, testTaint.Key+":"+string(testTaint.Effect)+"-")
			ginkgo.By("verifying the node doesn't have the taint " + testTaint.Key)
			output = runKubectlRetryOrDie("describe", "node", nodeName)
			if strings.Contains(output, testTaint.Key) {
				e2elog.Failf("Failed removing taint " + testTaint.Key + " of the node " + nodeName)
			}
		})

		ginkgo.It("should remove all the taints with the same key off a node", func() {
			testTaint := v1.Taint{
				Key:    fmt.Sprintf("kubernetes.io/e2e-taint-key-002-%s", string(uuid.NewUUID())),
				Value:  "testing-taint-value",
				Effect: v1.TaintEffectNoSchedule,
			}

			nodeName := scheduling.GetNodeThatCanRunPod(f)

			ginkgo.By("adding the taint " + testTaint.ToString() + " to a node")
			runKubectlRetryOrDie("taint", "nodes", nodeName, testTaint.ToString())
			defer framework.RemoveTaintOffNode(f.ClientSet, nodeName, testTaint)

			ginkgo.By("verifying the node has the taint " + testTaint.ToString())
			output := runKubectlRetryOrDie("describe", "node", nodeName)
			requiredStrings := [][]string{
				{"Name:", nodeName},
				{"Taints:"},
				{testTaint.ToString()},
			}
			checkOutput(output, requiredStrings)

			newTestTaint := v1.Taint{
				Key:    testTaint.Key,
				Value:  "another-testing-taint-value",
				Effect: v1.TaintEffectPreferNoSchedule,
			}
			ginkgo.By("adding another taint " + newTestTaint.ToString() + " to the node")
			runKubectlRetryOrDie("taint", "nodes", nodeName, newTestTaint.ToString())
			defer framework.RemoveTaintOffNode(f.ClientSet, nodeName, newTestTaint)

			ginkgo.By("verifying the node has the taint " + newTestTaint.ToString())
			output = runKubectlRetryOrDie("describe", "node", nodeName)
			requiredStrings = [][]string{
				{"Name:", nodeName},
				{"Taints:"},
				{newTestTaint.ToString()},
			}
			checkOutput(output, requiredStrings)

			noExecuteTaint := v1.Taint{
				Key:    testTaint.Key,
				Value:  "testing-taint-value-no-execute",
				Effect: v1.TaintEffectNoExecute,
			}
			ginkgo.By("adding NoExecute taint " + noExecuteTaint.ToString() + " to the node")
			runKubectlRetryOrDie("taint", "nodes", nodeName, noExecuteTaint.ToString())
			defer framework.RemoveTaintOffNode(f.ClientSet, nodeName, noExecuteTaint)

			ginkgo.By("verifying the node has the taint " + noExecuteTaint.ToString())
			output = runKubectlRetryOrDie("describe", "node", nodeName)
			requiredStrings = [][]string{
				{"Name:", nodeName},
				{"Taints:"},
				{noExecuteTaint.ToString()},
			}
			checkOutput(output, requiredStrings)

			ginkgo.By("removing all taints that have the same key " + testTaint.Key + " of the node")
			runKubectlRetryOrDie("taint", "nodes", nodeName, testTaint.Key+"-")
			ginkgo.By("verifying the node doesn't have the taints that have the same key " + testTaint.Key)
			output = runKubectlRetryOrDie("describe", "node", nodeName)
			if strings.Contains(output, testTaint.Key) {
				e2elog.Failf("Failed removing taints " + testTaint.Key + " of the node " + nodeName)
			}
		})
	})

	framework.KubeDescribe("Kubectl create quota", func() {
		ginkgo.It("should create a quota without scopes", func() {
			nsFlag := fmt.Sprintf("--namespace=%v", ns)
			quotaName := "million"

			ginkgo.By("calling kubectl quota")
			framework.RunKubectlOrDie("create", "quota", quotaName, "--hard=pods=1000000,services=1000000", nsFlag)

			ginkgo.By("verifying that the quota was created")
			quota, err := c.CoreV1().ResourceQuotas(ns).Get(quotaName, metav1.GetOptions{})
			if err != nil {
				e2elog.Failf("Failed getting quota %s: %v", quotaName, err)
			}

			if len(quota.Spec.Scopes) != 0 {
				e2elog.Failf("Expected empty scopes, got %v", quota.Spec.Scopes)
			}
			if len(quota.Spec.Hard) != 2 {
				e2elog.Failf("Expected two resources, got %v", quota.Spec.Hard)
			}
			r, found := quota.Spec.Hard[v1.ResourcePods]
			if expected := resource.MustParse("1000000"); !found || (&r).Cmp(expected) != 0 {
				e2elog.Failf("Expected pods=1000000, got %v", r)
			}
			r, found = quota.Spec.Hard[v1.ResourceServices]
			if expected := resource.MustParse("1000000"); !found || (&r).Cmp(expected) != 0 {
				e2elog.Failf("Expected services=1000000, got %v", r)
			}
		})

		ginkgo.It("should create a quota with scopes", func() {
			nsFlag := fmt.Sprintf("--namespace=%v", ns)
			quotaName := "scopes"

			ginkgo.By("calling kubectl quota")
			framework.RunKubectlOrDie("create", "quota", quotaName, "--hard=pods=1000000", "--scopes=BestEffort,NotTerminating", nsFlag)

			ginkgo.By("verifying that the quota was created")
			quota, err := c.CoreV1().ResourceQuotas(ns).Get(quotaName, metav1.GetOptions{})
			if err != nil {
				e2elog.Failf("Failed getting quota %s: %v", quotaName, err)
			}

			if len(quota.Spec.Scopes) != 2 {
				e2elog.Failf("Expected two scopes, got %v", quota.Spec.Scopes)
			}
			scopes := make(map[v1.ResourceQuotaScope]struct{})
			for _, scope := range quota.Spec.Scopes {
				scopes[scope] = struct{}{}
			}
			if _, found := scopes[v1.ResourceQuotaScopeBestEffort]; !found {
				e2elog.Failf("Expected BestEffort scope, got %v", quota.Spec.Scopes)
			}
			if _, found := scopes[v1.ResourceQuotaScopeNotTerminating]; !found {
				e2elog.Failf("Expected NotTerminating scope, got %v", quota.Spec.Scopes)
			}
		})

		ginkgo.It("should reject quota with invalid scopes", func() {
			nsFlag := fmt.Sprintf("--namespace=%v", ns)
			quotaName := "scopes"

			ginkgo.By("calling kubectl quota")
			out, err := framework.RunKubectl("create", "quota", quotaName, "--hard=hard=pods=1000000", "--scopes=Foo", nsFlag)
			if err == nil {
				e2elog.Failf("Expected kubectl to fail, but it succeeded: %s", out)
			}
		})
	})
})

// Checks whether the output split by line contains the required elements.
func checkOutputReturnError(output string, required [][]string) error {
	outputLines := strings.Split(output, "\n")
	currentLine := 0
	for _, requirement := range required {
		for currentLine < len(outputLines) && !strings.Contains(outputLines[currentLine], requirement[0]) {
			currentLine++
		}
		if currentLine == len(outputLines) {
			return fmt.Errorf("failed to find %s in %s", requirement[0], output)
		}
		for _, item := range requirement[1:] {
			if !strings.Contains(outputLines[currentLine], item) {
				return fmt.Errorf("failed to find %s in %s", item, outputLines[currentLine])
			}
		}
	}
	return nil
}

func checkOutput(output string, required [][]string) {
	err := checkOutputReturnError(output, required)
	if err != nil {
		e2elog.Failf("%v", err)
	}
}

func checkKubectlOutputWithRetry(required [][]string, args ...string) {
	var pollErr error
	wait.PollImmediate(time.Second, time.Minute, func() (bool, error) {
		output := framework.RunKubectlOrDie(args...)
		err := checkOutputReturnError(output, required)
		if err != nil {
			pollErr = err
			return false, nil
		}
		pollErr = nil
		return true, nil
	})
	if pollErr != nil {
		e2elog.Failf("%v", pollErr)
	}
	return
}

func checkContainersImage(containers []v1.Container, expectImage string) bool {
	return containers == nil || len(containers) != 1 || containers[0].Image != expectImage
}

func getAPIVersions(apiEndpoint string) (*metav1.APIVersions, error) {
	body, err := curl(apiEndpoint)
	if err != nil {
		return nil, fmt.Errorf("Failed http.Get of %s: %v", apiEndpoint, err)
	}
	var apiVersions metav1.APIVersions
	if err := json.Unmarshal([]byte(body), &apiVersions); err != nil {
		return nil, fmt.Errorf("Failed to parse /api output %s: %v", body, err)
	}
	return &apiVersions, nil
}

func startProxyServer() (int, *exec.Cmd, error) {
	// Specifying port 0 indicates we want the os to pick a random port.
	cmd := framework.KubectlCmd("proxy", "-p", "0", "--disable-filter")
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
	dial := func(ctx context.Context, proto, addr string) (net.Conn, error) {
		var d net.Dialer
		return d.DialContext(ctx, "unix", path)
	}
	transport := utilnet.SetTransportDefaults(&http.Transport{
		DialContext: dial,
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
	e2elog.Logf("Waiting for all frontend pods to be Running.")
	label := labels.SelectorFromSet(labels.Set(map[string]string{"tier": "frontend", "app": "guestbook"}))
	err := testutils.WaitForPodsWithLabelRunning(c, ns, label)
	framework.ExpectNoError(err)
	e2elog.Logf("Waiting for frontend to serve content.")
	if !waitForGuestbookResponse(c, "get", "", `{"data": ""}`, guestbookStartupTimeout, ns) {
		e2elog.Failf("Frontend service did not start serving content in %v seconds.", guestbookStartupTimeout.Seconds())
	}

	e2elog.Logf("Trying to add a new entry to the guestbook.")
	if !waitForGuestbookResponse(c, "set", "TestEntry", `{"message": "Updated"}`, guestbookResponseTimeout, ns) {
		e2elog.Failf("Cannot added new entry in %v seconds.", guestbookResponseTimeout.Seconds())
	}

	e2elog.Logf("Verifying that added entry can be retrieved.")
	if !waitForGuestbookResponse(c, "get", "", `{"data": "TestEntry"}`, guestbookResponseTimeout, ns) {
		e2elog.Failf("Entry to guestbook wasn't correctly added in %v seconds.", guestbookResponseTimeout.Seconds())
	}
}

// Returns whether received expected response from guestbook on time.
func waitForGuestbookResponse(c clientset.Interface, cmd, arg, expectedResponse string, timeout time.Duration, ns string) bool {
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(5 * time.Second) {
		res, err := makeRequestToGuestbook(c, cmd, arg, ns)
		if err == nil && res == expectedResponse {
			return true
		}
		e2elog.Logf("Failed to get response from guestbook. err: %v, response: %s", err, res)
	}
	return false
}

func makeRequestToGuestbook(c clientset.Interface, cmd, value string, ns string) (string, error) {
	proxyRequest, errProxy := e2eservice.GetServicesProxyRequest(c, c.CoreV1().RESTClient().Get())
	if errProxy != nil {
		return "", errProxy
	}

	ctx, cancel := context.WithTimeout(context.Background(), framework.SingleCallTimeout)
	defer cancel()

	result, err := proxyRequest.Namespace(ns).
		Context(ctx).
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

func readReplicationControllerFromString(contents string) *v1.ReplicationController {
	rc := v1.ReplicationController{}
	if err := yaml.Unmarshal([]byte(contents), &rc); err != nil {
		e2elog.Failf(err.Error())
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
		e2elog.Failf("json marshal failed: %s\n", err)
	}

	return bytes.NewReader(data)
}

func forEachReplicationController(c clientset.Interface, ns, selectorKey, selectorValue string, fn func(v1.ReplicationController)) {
	var rcs *v1.ReplicationControllerList
	var err error
	for t := time.Now(); time.Since(t) < framework.PodListTimeout; time.Sleep(framework.Poll) {
		label := labels.SelectorFromSet(labels.Set(map[string]string{selectorKey: selectorValue}))
		options := metav1.ListOptions{LabelSelector: label.String()}
		rcs, err = c.CoreV1().ReplicationControllers(ns).List(options)
		framework.ExpectNoError(err)
		if len(rcs.Items) > 0 {
			break
		}
	}

	if rcs == nil || len(rcs.Items) == 0 {
		e2elog.Failf("No replication controllers found")
	}

	for _, rc := range rcs.Items {
		fn(rc)
	}
}

func validateReplicationControllerConfiguration(rc v1.ReplicationController) {
	if rc.Name == "redis-master" {
		if _, ok := rc.Annotations[v1.LastAppliedConfigAnnotation]; !ok {
			e2elog.Failf("Annotation not found in modified configuration:\n%v\n", rc)
		}

		if value, ok := rc.Labels[applyTestLabel]; !ok || value != "ADDED" {
			e2elog.Failf("Added label %s not found in modified configuration:\n%v\n", applyTestLabel, rc)
		}
	}
}

// getUDData creates a validator function based on the input string (i.e. kitten.jpg).
// For example, if you send "kitten.jpg", this function verifies that the image jpg = kitten.jpg
// in the container's json field.
func getUDData(jpgExpected string, ns string) func(clientset.Interface, string) error {

	// getUDData validates data.json in the update-demo (returns nil if data is ok).
	return func(c clientset.Interface, podID string) error {
		e2elog.Logf("validating pod %s", podID)

		ctx, cancel := context.WithTimeout(context.Background(), framework.SingleCallTimeout)
		defer cancel()

		body, err := c.CoreV1().RESTClient().Get().
			Namespace(ns).
			Resource("pods").
			SubResource("proxy").
			Name(podID).
			Suffix("data.json").
			Do().
			Raw()

		if err != nil {
			if ctx.Err() != nil {
				e2elog.Failf("Failed to retrieve data from container: %v", err)
			}
			return err
		}
		e2elog.Logf("got data: %s", body)
		var data updateDemoData
		if err := json.Unmarshal(body, &data); err != nil {
			return err
		}
		e2elog.Logf("Unmarshalled json jpg/img => %s , expecting %s .", data, jpgExpected)
		if strings.Contains(data.Image, jpgExpected) {
			return nil
		}
		return fmt.Errorf("data served up in container is inaccurate, %s didn't contain %s", data, jpgExpected)
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

func startLocalProxy() (srv *httptest.Server, logs *bytes.Buffer) {
	logs = &bytes.Buffer{}
	p := goproxy.NewProxyHttpServer()
	p.Verbose = true
	p.Logger = log.New(logs, "", 0)
	return httptest.NewServer(p), logs
}

// createApplyCustomResource asserts that given CustomResource be created and applied
// without being rejected by client-side validation
func createApplyCustomResource(resource, namespace, name string, crd *crd.TestCrd) error {
	ns := fmt.Sprintf("--namespace=%v", namespace)
	ginkgo.By("successfully create CR")
	if _, err := framework.RunKubectlInput(resource, ns, "create", "--validate=true", "-f", "-"); err != nil {
		return fmt.Errorf("failed to create CR %s in namespace %s: %v", resource, ns, err)
	}
	if _, err := framework.RunKubectl(ns, "delete", crd.Crd.Spec.Names.Plural, name); err != nil {
		return fmt.Errorf("failed to delete CR %s: %v", name, err)
	}
	ginkgo.By("successfully apply CR")
	if _, err := framework.RunKubectlInput(resource, ns, "apply", "--validate=true", "-f", "-"); err != nil {
		return fmt.Errorf("failed to apply CR %s in namespace %s: %v", resource, ns, err)
	}
	if _, err := framework.RunKubectl(ns, "delete", crd.Crd.Spec.Names.Plural, name); err != nil {
		return fmt.Errorf("failed to delete CR %s: %v", name, err)
	}
	return nil
}

var schemaFoo = []byte(`description: Foo CRD for Testing
type: object
properties:
  spec:
    type: object
    description: Specification of Foo
    properties:
      bars:
        description: List of Bars and their specs.
        type: array
        items:
          type: object
          required:
          - name
          properties:
            name:
              description: Name of Bar.
              type: string
            age:
              description: Age of Bar.
              type: string
            bazs:
              description: List of Bazs.
              items:
                type: string
              type: array
  status:
    description: Status of Foo
    type: object
    properties:
      bars:
        description: List of Bars and their statuses.
        type: array
        items:
          type: object
          properties:
            name:
              description: Name of Bar.
              type: string
            available:
              description: Whether the Bar is installed.
              type: boolean
            quxType:
              description: Indicates to external qux type.
              pattern: in-tree|out-of-tree
              type: string`)

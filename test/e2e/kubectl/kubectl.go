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

	openapi_v2 "github.com/google/gnostic-models/openapiv2"
	"github.com/google/go-cmp/cmp"

	"sigs.k8s.io/yaml"

	v1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	utilnettesting "k8s.io/apimachinery/pkg/util/net/testing"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/authentication/serviceaccount"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/kubectl/pkg/polymorphichelpers"
	"k8s.io/kubernetes/pkg/controller"
	commonutils "k8s.io/kubernetes/test/e2e/common"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eauth "k8s.io/kubernetes/test/e2e/framework/auth"
	e2edebug "k8s.io/kubernetes/test/e2e/framework/debug"
	e2eendpointslice "k8s.io/kubernetes/test/e2e/framework/endpointslice"
	e2ekubectl "k8s.io/kubernetes/test/e2e/framework/kubectl"
	e2enetwork "k8s.io/kubernetes/test/e2e/framework/network"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eoutput "k8s.io/kubernetes/test/e2e/framework/pod/output"
	e2eservice "k8s.io/kubernetes/test/e2e/framework/service"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2etestfiles "k8s.io/kubernetes/test/e2e/framework/testfiles"
	"k8s.io/kubernetes/test/e2e/scheduling"
	testutils "k8s.io/kubernetes/test/utils"
	"k8s.io/kubernetes/test/utils/crd"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
	uexec "k8s.io/utils/exec"
	"k8s.io/utils/pointer"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

const (
	updateDemoSelector        = "name=update-demo"
	guestbookStartupTimeout   = 10 * time.Minute
	guestbookResponseTimeout  = 3 * time.Minute
	simplePodSelector         = "name=httpd"
	simplePodName             = "httpd"
	simplePodResourceName     = "pod/httpd"
	httpdDefaultOutput        = "It works!"
	simplePodPort             = 80
	pausePodSelector          = "name=pause"
	pausePodName              = "pause"
	busyboxPodSelector        = "app=busybox1"
	busyboxPodName            = "busybox1"
	kubeCtlManifestPath       = "test/e2e/testing-manifests/kubectl"
	agnhostControllerFilename = "agnhost-primary-controller.json.in"
	agnhostServiceFilename    = "agnhost-primary-service.json"
	httpdDeployment1Filename  = "httpd-deployment1.yaml.in"
	httpdDeployment2Filename  = "httpd-deployment2.yaml.in"
	httpdDeployment3Filename  = "httpd-deployment3.yaml.in"
	metaPattern               = `"kind":"%s","apiVersion":"%s/%s","metadata":{"name":"%s"}`
)

func unknownFieldMetadataJSON(gvk schema.GroupVersionKind, name string) string {
	return fmt.Sprintf(`"kind":"%s","apiVersion":"%s/%s","metadata":{"unknownMeta": "foo", "name":"%s"}`, gvk.Kind, gvk.Group, gvk.Version, name)
}

var (
	// If this suite still flakes due to timeouts we should change this to framework.PodStartTimeout
	podRunningTimeoutArg = fmt.Sprintf("--pod-running-timeout=%s", framework.PodStartShortTimeout.String())
)

var proxyRegexp = regexp.MustCompile("Starting to serve on 127.0.0.1:([0-9]+)")

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

var schemaFooEmbedded = []byte(`description: Foo CRD with an embedded resource
type: object
properties:
  spec:
    type: object
    properties:
      template:
        type: object
        x-kubernetes-embedded-resource: true
        properties:
          metadata:
            type: object
            properties:
              name:
                type: string
          spec:
            type: object
  metadata:
    type: object
    properties:
      name:
        type: string`)

// Stops everything from filePath from namespace ns and checks if everything matching selectors from the given namespace is correctly stopped.
// Aware of the kubectl example files map.
func cleanupKubectlInputs(fileContents string, ns string, selectors ...string) {
	ginkgo.By("using delete to clean up resources")
	// support backward compatibility : file paths or raw json - since we are removing file path
	// dependencies from this test.
	e2ekubectl.RunKubectlOrDieInput(ns, fileContents, "delete", "--grace-period=0", "--force", "-f", "-")
	assertCleanup(ns, selectors...)
}

// assertCleanup asserts that cleanup of a namespace wrt selectors occurred.
func assertCleanup(ns string, selectors ...string) {
	var e error
	verifyCleanupFunc := func() (bool, error) {
		e = nil
		for _, selector := range selectors {
			resources := e2ekubectl.RunKubectlOrDie(ns, "get", "rc,svc", "-l", selector, "--no-headers")
			if resources != "" {
				e = fmt.Errorf("Resources left running after stop:\n%s", resources)
				return false, nil
			}
			pods := e2ekubectl.RunKubectlOrDie(ns, "get", "pods", "-l", selector, "-o", "go-template={{ range .items }}{{ if not .metadata.deletionTimestamp }}{{ .metadata.name }}{{ \"\\n\" }}{{ end }}{{ end }}")
			if pods != "" {
				e = fmt.Errorf("Pods left unterminated after stop:\n%s", pods)
				return false, nil
			}
		}
		return true, nil
	}
	err := wait.PollImmediate(500*time.Millisecond, 1*time.Minute, verifyCleanupFunc)
	if err != nil {
		framework.Fail(e.Error())
	}
}

func readTestFileOrDie(file string) []byte {
	data, err := e2etestfiles.Read(path.Join(kubeCtlManifestPath, file))
	if err != nil {
		framework.Fail(err.Error(), 1)
	}
	return data
}

func runKubectlRetryOrDie(ns string, args ...string) string {
	var err error
	var output string
	for i := 0; i < 5; i++ {
		output, err = e2ekubectl.RunKubectl(ns, args...)
		if err == nil || (!strings.Contains(err.Error(), genericregistry.OptimisticLockErrorMsg) && !strings.Contains(err.Error(), "Operation cannot be fulfilled")) {
			break
		}
		time.Sleep(time.Second)
	}
	// Expect no errors to be present after retries are finished
	// Copied from framework #ExecOrDie
	framework.Logf("stdout: %q", output)
	framework.ExpectNoError(err)
	return output
}

var _ = SIGDescribe("Kubectl client", func() {
	defer ginkgo.GinkgoRecover()
	f := framework.NewDefaultFramework("kubectl")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	// Reusable cluster state function.  This won't be adversely affected by lazy initialization of framework.
	clusterState := func() *framework.ClusterVerification {
		return f.NewClusterVerification(
			f.Namespace,
			framework.PodStateVerification{
				Selectors:   map[string]string{"app": "agnhost"},
				ValidPhases: []v1.PodPhase{v1.PodRunning /*v1.PodPending*/},
			})
	}
	forEachPod := func(ctx context.Context, podFunc func(p v1.Pod)) {
		_ = clusterState().ForEach(ctx, podFunc)
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
	waitForOrFailWithDebug := func(ctx context.Context, atLeast int) {
		pods, err := clusterState().WaitFor(ctx, atLeast, framework.PodStartTimeout)
		if err != nil || len(pods) < atLeast {
			// TODO: Generalize integrating debug info into these tests so we always get debug info when we need it
			e2edebug.DumpAllNamespaceInfo(ctx, f.ClientSet, ns)
			framework.Failf("Verified %d of %d pods , error: %v", len(pods), atLeast, err)
		}
	}

	debugDiscovery := func() {
		home := os.Getenv("HOME")
		if len(home) == 0 {
			framework.Logf("no $HOME envvar set")
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
			framework.Logf("%s modified at %s (current time: %s)", path, info.ModTime(), time.Now())

			data, readError := os.ReadFile(path)
			if readError != nil {
				framework.Logf("%s error: %v", path, readError)
			} else {
				framework.Logf("%s content: %s", path, string(data))
			}
			return nil
		})
		framework.Logf("scanned %s for discovery docs: %v", home, err)
	}

	ginkgo.Describe("Update Demo", func() {
		var nautilus string
		ginkgo.BeforeEach(func() {
			updateDemoRoot := "test/fixtures/doc-yaml/user-guide/update-demo"
			data, err := e2etestfiles.Read(filepath.Join(updateDemoRoot, "nautilus-rc.yaml.in"))
			if err != nil {
				framework.Fail(err.Error())
			}
			nautilus = commonutils.SubstituteImageName(string(data))
		})
		/*
			Release: v1.9
			Testname: Kubectl, replication controller
			Description: Create a Pod and a container with a given image. Configure replication controller to run 2 replicas. The number of running instances of the Pod MUST equal the number of replicas set on the replication controller which is 2.
		*/
		framework.ConformanceIt("should create and stop a replication controller", func(ctx context.Context) {
			defer cleanupKubectlInputs(nautilus, ns, updateDemoSelector)

			ginkgo.By("creating a replication controller")
			e2ekubectl.RunKubectlOrDieInput(ns, nautilus, "create", "-f", "-")
			validateController(ctx, c, imageutils.GetE2EImage(imageutils.Nautilus), 2, "update-demo", updateDemoSelector, getUDData("nautilus.jpg", ns), ns)
		})

		/*
			Release: v1.9
			Testname: Kubectl, scale replication controller
			Description: Create a Pod and a container with a given image. Configure replication controller to run 2 replicas. The number of running instances of the Pod MUST equal the number of replicas set on the replication controller which is 2. Update the replicaset to 1. Number of running instances of the Pod MUST be 1. Update the replicaset to 2. Number of running instances of the Pod MUST be 2.
		*/
		framework.ConformanceIt("should scale a replication controller", func(ctx context.Context) {
			defer cleanupKubectlInputs(nautilus, ns, updateDemoSelector)
			nautilusImage := imageutils.GetE2EImage(imageutils.Nautilus)

			ginkgo.By("creating a replication controller")
			e2ekubectl.RunKubectlOrDieInput(ns, nautilus, "create", "-f", "-")
			validateController(ctx, c, nautilusImage, 2, "update-demo", updateDemoSelector, getUDData("nautilus.jpg", ns), ns)
			ginkgo.By("scaling down the replication controller")
			debugDiscovery()
			e2ekubectl.RunKubectlOrDie(ns, "scale", "rc", "update-demo-nautilus", "--replicas=1", "--timeout=5m")
			validateController(ctx, c, nautilusImage, 1, "update-demo", updateDemoSelector, getUDData("nautilus.jpg", ns), ns)
			ginkgo.By("scaling up the replication controller")
			debugDiscovery()
			e2ekubectl.RunKubectlOrDie(ns, "scale", "rc", "update-demo-nautilus", "--replicas=2", "--timeout=5m")
			validateController(ctx, c, nautilusImage, 2, "update-demo", updateDemoSelector, getUDData("nautilus.jpg", ns), ns)
		})
	})

	ginkgo.Describe("Guestbook application", func() {
		forEachGBFile := func(run func(s string)) {
			guestbookRoot := "test/e2e/testing-manifests/guestbook"
			for _, gbAppFile := range []string{
				"agnhost-replica-service.yaml",
				"agnhost-primary-service.yaml",
				"frontend-service.yaml",
				"frontend-deployment.yaml.in",
				"agnhost-primary-deployment.yaml.in",
				"agnhost-replica-deployment.yaml.in",
			} {
				data, err := e2etestfiles.Read(filepath.Join(guestbookRoot, gbAppFile))
				if err != nil {
					framework.Fail(err.Error())
				}
				contents := commonutils.SubstituteImageName(string(data))
				run(contents)
			}
		}

		/*
			Release: v1.9
			Testname: Kubectl, guestbook application
			Description: Create Guestbook application that contains an agnhost primary server, 2 agnhost replicas, frontend application, frontend service and agnhost primary service and agnhost replica service. Using frontend service, the test will write an entry into the guestbook application which will store the entry into the backend agnhost store. Application flow MUST work as expected and the data written MUST be available to read.
		*/
		framework.ConformanceIt("should create and stop a working application", func(ctx context.Context) {
			defer forEachGBFile(func(contents string) {
				cleanupKubectlInputs(contents, ns)
			})
			ginkgo.By("creating all guestbook components")
			forEachGBFile(func(contents string) {
				framework.Logf("%s", contents)
				e2ekubectl.RunKubectlOrDieInput(ns, contents, "create", "-f", "-")
			})

			ginkgo.By("validating guestbook app")
			validateGuestbookApp(ctx, c, ns)
		})
	})

	ginkgo.Describe("Simple pod", func() {
		var podYaml string
		ginkgo.BeforeEach(func(ctx context.Context) {
			ginkgo.By(fmt.Sprintf("creating the pod from %v", podYaml))
			podYaml = commonutils.SubstituteImageName(string(readTestFileOrDie("pod-with-readiness-probe.yaml.in")))
			e2ekubectl.RunKubectlOrDieInput(ns, podYaml, "create", "-f", "-")
			framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(ctx, c, simplePodName, ns, framework.PodStartTimeout))
		})
		ginkgo.AfterEach(func() {
			cleanupKubectlInputs(podYaml, ns, simplePodSelector)
		})

		ginkgo.It("should support exec", func(ctx context.Context) {
			ginkgo.By("executing a command in the container")
			execOutput := e2ekubectl.RunKubectlOrDie(ns, "exec", podRunningTimeoutArg, simplePodName, "--", "echo", "running", "in", "container")
			if e, a := "running in container", strings.TrimSpace(execOutput); e != a {
				framework.Failf("Unexpected kubectl exec output. Wanted %q, got %q", e, a)
			}

			ginkgo.By("executing a very long command in the container")
			veryLongData := make([]rune, 20000)
			for i := 0; i < len(veryLongData); i++ {
				veryLongData[i] = 'a'
			}
			execOutput = e2ekubectl.RunKubectlOrDie(ns, "exec", podRunningTimeoutArg, simplePodName, "--", "echo", string(veryLongData))
			gomega.Expect(string(veryLongData)).To(gomega.Equal(strings.TrimSpace(execOutput)), "Unexpected kubectl exec output")

			ginkgo.By("executing a command in the container with noninteractive stdin")
			execOutput = e2ekubectl.NewKubectlCommand(ns, "exec", "-i", podRunningTimeoutArg, simplePodName, "--", "cat").
				WithStdinData("abcd1234").
				ExecOrDie(ns)
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

			ginkgo.By("executing a command in the container with pseudo-interactive stdin")
			execOutput = e2ekubectl.NewKubectlCommand(ns, "exec", "-i", podRunningTimeoutArg, simplePodName, "--", "sh").
				WithStdinReader(r).
				ExecOrDie(ns)
			if e, a := "hi", strings.TrimSpace(execOutput); e != a {
				framework.Failf("Unexpected kubectl exec output. Wanted %q, got %q", e, a)
			}
		})

		ginkgo.It("should support exec using resource/name", func(ctx context.Context) {
			ginkgo.By("executing a command in the container")
			execOutput := e2ekubectl.RunKubectlOrDie(ns, "exec", podRunningTimeoutArg, simplePodResourceName, "--", "echo", "running", "in", "container")
			if e, a := "running in container", strings.TrimSpace(execOutput); e != a {
				framework.Failf("Unexpected kubectl exec output. Wanted %q, got %q", e, a)
			}
		})

		ginkgo.It("should support exec through an HTTP proxy", func(ctx context.Context) {
			testContextHost := getTestContextHost()

			ginkgo.By("Starting http_proxy")
			var proxyLogs bytes.Buffer
			testSrv := httptest.NewServer(utilnettesting.NewHTTPProxyHandler(ginkgo.GinkgoTB(), func(req *http.Request) bool {
				fmt.Fprintf(&proxyLogs, "Accepting %s to %s\n", req.Method, req.Host)
				return true
			}))
			defer testSrv.Close()
			proxyAddr := testSrv.URL

			for _, proxyVar := range []string{"https_proxy", "HTTPS_PROXY"} {
				proxyLogs.Reset()
				ginkgo.By("Running kubectl via an HTTP proxy using " + proxyVar)
				output := e2ekubectl.NewKubectlCommand(ns, "exec", podRunningTimeoutArg, simplePodName, "--", "echo", "running", "in", "container").
					AppendEnv(append(os.Environ(), fmt.Sprintf("%s=%s", proxyVar, proxyAddr))).
					ExecOrDie(ns)

				// Verify we got the normal output captured by the exec server
				expectedExecOutput := "running in container\n"
				if output != expectedExecOutput {
					framework.Failf("Unexpected kubectl exec output. Wanted %q, got  %q", expectedExecOutput, output)
				}

				// Verify the proxy server logs saw the connection
				expectedProxyLog := fmt.Sprintf("Accepting CONNECT to %s", strings.TrimSuffix(strings.TrimPrefix(testContextHost, "https://"), "/api"))

				proxyLog := proxyLogs.String()
				if !strings.Contains(proxyLog, expectedProxyLog) {
					framework.Failf("Missing expected log result on proxy server for %s. Expected: %q, got %q", proxyVar, expectedProxyLog, proxyLog)
				}
			}
		})

		ginkgo.It("should support exec through kubectl proxy", func(ctx context.Context) {
			_ = getTestContextHost()

			ginkgo.By("Starting kubectl proxy")
			port, proxyCmd, err := startProxyServer(ns)
			framework.ExpectNoError(err)
			defer framework.TryKill(proxyCmd)

			//proxyLogs.Reset()
			host := fmt.Sprintf("--server=http://127.0.0.1:%d", port)
			ginkgo.By("Running kubectl via kubectl proxy using " + host)
			output := e2ekubectl.NewKubectlCommand(
				ns, host,
				"exec", podRunningTimeoutArg, simplePodName, "--", "echo", "running", "in", "container",
			).ExecOrDie(ns)

			// Verify we got the normal output captured by the exec server
			expectedExecOutput := "running in container\n"
			if output != expectedExecOutput {
				framework.Failf("Unexpected kubectl exec output. Wanted %q, got  %q", expectedExecOutput, output)
			}
		})

		ginkgo.Context("should return command exit codes", func() {
			ginkgo.It("execing into a container with a successful command", func(ctx context.Context) {
				_, err := e2ekubectl.NewKubectlCommand(ns, "exec", simplePodName, podRunningTimeoutArg, "--", "/bin/sh", "-c", "exit 0").Exec()
				framework.ExpectNoError(err)
			})

			ginkgo.It("execing into a container with a failing command", func(ctx context.Context) {
				_, err := e2ekubectl.NewKubectlCommand(ns, "exec", simplePodName, podRunningTimeoutArg, "--", "/bin/sh", "-c", "exit 42").Exec()
				ee, ok := err.(uexec.ExitError)
				if !ok {
					framework.Failf("Got unexpected error type, expected uexec.ExitError, got %T: %v", err, err)
				}
				gomega.Expect(ee.ExitStatus()).To(gomega.Equal(42))
			})

			ginkgo.It("should support port-forward", func(ctx context.Context) {
				ginkgo.By("forwarding the container port to a local port")
				cmd := runPortForward(ns, simplePodName, simplePodPort)
				defer cmd.Stop()

				ginkgo.By("curling local port output")
				localAddr := fmt.Sprintf("http://localhost:%d", cmd.port)
				body, err := curl(localAddr)
				framework.Logf("got: %s", body)
				if err != nil {
					framework.Failf("Failed http.Get of forwarded port (%s): %v", localAddr, err)
				}
				if !strings.Contains(body, httpdDefaultOutput) {
					framework.Failf("Container port output missing expected value. Wanted:'%s', got: %s", httpdDefaultOutput, body)
				}
			})

			ginkgo.It("should handle in-cluster config", func(ctx context.Context) {
				// TODO: Find a way to download and copy the appropriate kubectl binary, or maybe a multi-arch kubectl image
				// for now this only works on amd64
				e2eskipper.SkipUnlessNodeOSArchIs("amd64")

				ginkgo.By("adding rbac permissions")
				// grant the view permission widely to allow inspection of the `invalid` namespace and the default namespace
				err := e2eauth.BindClusterRole(ctx, f.ClientSet.RbacV1(), "view", f.Namespace.Name,
					rbacv1.Subject{Kind: rbacv1.ServiceAccountKind, Namespace: f.Namespace.Name, Name: "default"})
				framework.ExpectNoError(err)

				err = e2eauth.WaitForAuthorizationUpdate(ctx, f.ClientSet.AuthorizationV1(),
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

				inClusterHost := strings.TrimSpace(e2eoutput.RunHostCmdOrDie(ns, simplePodName, "printenv KUBERNETES_SERVICE_HOST"))
				inClusterPort := strings.TrimSpace(e2eoutput.RunHostCmdOrDie(ns, simplePodName, "printenv KUBERNETES_SERVICE_PORT"))
				inClusterURL := net.JoinHostPort(inClusterHost, inClusterPort)
				framework.Logf("copying %s to the %s pod", kubectlPath, simplePodName)
				e2ekubectl.RunKubectlOrDie(ns, "cp", kubectlPath, ns+"/"+simplePodName+":/tmp/")

				// Build a kubeconfig file that will make use of the injected ca and token,
				// but point at the DNS host and the default namespace
				tmpDir, err := os.MkdirTemp("", "icc-override")
				overrideKubeconfigName := "icc-override.kubeconfig"
				framework.ExpectNoError(err)
				defer func() { os.Remove(tmpDir) }()
				framework.ExpectNoError(os.WriteFile(filepath.Join(tmpDir, overrideKubeconfigName), []byte(`
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
				framework.Logf("copying override kubeconfig to the %s pod", simplePodName)
				e2ekubectl.RunKubectlOrDie(ns, "cp", filepath.Join(tmpDir, overrideKubeconfigName), ns+"/"+simplePodName+":/tmp/")

				framework.ExpectNoError(os.WriteFile(filepath.Join(tmpDir, "invalid-configmap-with-namespace.yaml"), []byte(`
kind: ConfigMap
apiVersion: v1
metadata:
  name: "configmap with namespace and invalid name"
  namespace: configmap-namespace
`), os.FileMode(0755)))
				framework.ExpectNoError(os.WriteFile(filepath.Join(tmpDir, "invalid-configmap-without-namespace.yaml"), []byte(`
kind: ConfigMap
apiVersion: v1
metadata:
  name: "configmap without namespace and invalid name"
`), os.FileMode(0755)))
				framework.Logf("copying configmap manifests to the %s pod", simplePodName)
				e2ekubectl.RunKubectlOrDie(ns, "cp", filepath.Join(tmpDir, "invalid-configmap-with-namespace.yaml"), ns+"/"+simplePodName+":/tmp/")
				e2ekubectl.RunKubectlOrDie(ns, "cp", filepath.Join(tmpDir, "invalid-configmap-without-namespace.yaml"), ns+"/"+simplePodName+":/tmp/")

				ginkgo.By("getting pods with in-cluster configs")
				execOutput := e2eoutput.RunHostCmdOrDie(ns, simplePodName, "/tmp/kubectl get pods --v=6 2>&1")
				gomega.Expect(execOutput).To(gomega.MatchRegexp("httpd +1/1 +Running"))
				gomega.Expect(execOutput).To(gomega.ContainSubstring("Using in-cluster namespace"))
				gomega.Expect(execOutput).To(gomega.ContainSubstring("Using in-cluster configuration"))

				ginkgo.By("creating an object containing a namespace with in-cluster config")
				_, err = e2eoutput.RunHostCmd(ns, simplePodName, "/tmp/kubectl create -f /tmp/invalid-configmap-with-namespace.yaml --v=6 2>&1")
				gomega.Expect(err).To(gomega.ContainSubstring("Using in-cluster namespace"))
				gomega.Expect(err).To(gomega.ContainSubstring("Using in-cluster configuration"))

				gomega.Expect(err).To(gomega.ContainSubstring(fmt.Sprintf(`verb="POST" url="https://%s/api/v1/namespaces/configmap-namespace/configmaps`, inClusterURL)))

				ginkgo.By("creating an object not containing a namespace with in-cluster config")
				_, err = e2eoutput.RunHostCmd(ns, simplePodName, "/tmp/kubectl create -f /tmp/invalid-configmap-without-namespace.yaml --v=6 2>&1")
				gomega.Expect(err).To(gomega.ContainSubstring("Using in-cluster namespace"))
				gomega.Expect(err).To(gomega.ContainSubstring("Using in-cluster configuration"))
				gomega.Expect(err).To(gomega.ContainSubstring(fmt.Sprintf(`verb="POST" url="https://%s/api/v1/namespaces/%s/configmaps`, inClusterURL, f.Namespace.Name)))

				ginkgo.By("trying to use kubectl with invalid token")
				_, err = e2eoutput.RunHostCmd(ns, simplePodName, "/tmp/kubectl get pods --token=invalid --v=7 2>&1")
				framework.Logf("got err %v", err)
				gomega.Expect(err).To(gomega.HaveOccurred())
				gomega.Expect(err).To(gomega.ContainSubstring("Using in-cluster namespace"))
				gomega.Expect(err).To(gomega.ContainSubstring("Using in-cluster configuration"))
				gomega.Expect(err).To(gomega.ContainSubstring(`"Response" status="401 Unauthorized"`))

				ginkgo.By("trying to use kubectl with invalid server")
				_, err = e2eoutput.RunHostCmd(ns, simplePodName, "/tmp/kubectl get pods --server=invalid --v=6 2>&1")
				framework.Logf("got err %v", err)
				gomega.Expect(err).To(gomega.HaveOccurred())
				gomega.Expect(err).To(gomega.ContainSubstring("Unable to connect to the server"))
				gomega.Expect(err).To(gomega.ContainSubstring(`verb="GET" url="http://invalid/api`))

				ginkgo.By("trying to use kubectl with invalid namespace")
				execOutput = e2eoutput.RunHostCmdOrDie(ns, simplePodName, "/tmp/kubectl get pods --namespace=invalid --v=6 2>&1")
				gomega.Expect(execOutput).To(gomega.ContainSubstring("No resources found"))
				gomega.Expect(execOutput).ToNot(gomega.ContainSubstring("Using in-cluster namespace"))
				gomega.Expect(execOutput).To(gomega.ContainSubstring("Using in-cluster configuration"))
				gomega.Expect(execOutput).To(gomega.MatchRegexp(fmt.Sprintf(`verb="GET" url="http[s]?://[\[]?%s[\]]?:%s/api/v1/namespaces/invalid/pods`, inClusterHost, inClusterPort)))

				ginkgo.By("trying to use kubectl with kubeconfig")
				execOutput = e2eoutput.RunHostCmdOrDie(ns, simplePodName, "/tmp/kubectl get pods --kubeconfig=/tmp/"+overrideKubeconfigName+" --v=6 2>&1")
				gomega.Expect(execOutput).ToNot(gomega.ContainSubstring("Using in-cluster namespace"))
				gomega.Expect(execOutput).ToNot(gomega.ContainSubstring("Using in-cluster configuration"))
				gomega.Expect(execOutput).To(gomega.ContainSubstring(`verb="GET" url="https://kubernetes.default.svc:443/api/v1/namespaces/default/pods`))
			})
		})

		ginkgo.Describe("Kubectl run", func() {
			ginkgo.It("running a successful command", func(ctx context.Context) {
				_, err := e2ekubectl.NewKubectlCommand(ns, "run", "-i", "--image="+imageutils.GetE2EImage(imageutils.BusyBox), "--restart=Never", podRunningTimeoutArg, "success", "--", "/bin/sh", "-c", "exit 0").Exec()
				framework.ExpectNoError(err)
			})

			ginkgo.It("running a failing command", func(ctx context.Context) {
				_, err := e2ekubectl.NewKubectlCommand(ns, "run", "-i", "--image="+imageutils.GetE2EImage(imageutils.BusyBox), "--restart=Never", podRunningTimeoutArg, "failure-1", "--", "/bin/sh", "-c", "exit 42").Exec()
				ee, ok := err.(uexec.ExitError)
				if !ok {
					framework.Failf("Got unexpected error type, expected uexec.ExitError, got %T: %v", err, err)
				}
				gomega.Expect(ee.ExitStatus()).To(gomega.Equal(42))
			})

			f.It(f.WithSlow(), "running a failing command without --restart=Never", func(ctx context.Context) {
				_, err := e2ekubectl.NewKubectlCommand(ns, "run", "-i", "--image="+imageutils.GetE2EImage(imageutils.BusyBox), "--restart=OnFailure", podRunningTimeoutArg, "failure-2", "--", "/bin/sh", "-c", "cat && exit 42").
					WithStdinData("abcd1234").
					Exec()
				ee, ok := err.(uexec.ExitError)
				if !ok {
					framework.Failf("Got unexpected error type, expected uexec.ExitError, got %T: %v", err, err)
				}
				if !strings.Contains(ee.String(), "timed out") {
					framework.Failf("Missing expected 'timed out' error, got: %#v", ee)
				}
			})

			f.It(f.WithSlow(), "running a failing command without --restart=Never, but with --rm", func(ctx context.Context) {
				_, err := e2ekubectl.NewKubectlCommand(ns, "run", "-i", "--image="+imageutils.GetE2EImage(imageutils.BusyBox), "--restart=OnFailure", "--rm", podRunningTimeoutArg, "failure-3", "--", "/bin/sh", "-c", "cat && exit 42").
					WithStdinData("abcd1234").
					Exec()
				ee, ok := err.(uexec.ExitError)
				if !ok {
					framework.Failf("Got unexpected error type, expected uexec.ExitError, got %T: %v", err, err)
				}
				if !strings.Contains(ee.String(), "timed out") {
					framework.Failf("Missing expected 'timed out' error, got: %#v", ee)
				}
				framework.ExpectNoError(e2epod.WaitForPodNotFoundInNamespace(ctx, f.ClientSet, "failure-3", ns, 2*v1.DefaultTerminationGracePeriodSeconds*time.Second))
			})

			f.It(f.WithSlow(), "running a failing command with --leave-stdin-open", func(ctx context.Context) {
				_, err := e2ekubectl.NewKubectlCommand(ns, "run", "-i", "--image="+imageutils.GetE2EImage(imageutils.BusyBox), "--restart=Never", podRunningTimeoutArg, "failure-4", "--leave-stdin-open", "--", "/bin/sh", "-c", "exit 42").
					WithStdinData("abcd1234").
					Exec()
				framework.ExpectNoError(err)
			})
		})

		ginkgo.It("should support inline execution and attach", func(ctx context.Context) {
			waitForStdinContent := func(pod, content string) string {
				var logOutput string
				err := wait.Poll(10*time.Second, 5*time.Minute, func() (bool, error) {
					logOutput = e2ekubectl.RunKubectlOrDie(ns, "logs", pod)
					return strings.Contains(logOutput, content), nil
				})

				framework.ExpectNoError(err, "waiting for '%v' output", content)
				return logOutput
			}

			ginkgo.By("executing a command with run and attach with stdin")
			// We wait for a non-empty line so we know kubectl has attached
			e2ekubectl.NewKubectlCommand(ns, "run", "run-test", "--image="+imageutils.GetE2EImage(imageutils.BusyBox), "--restart=OnFailure", podRunningTimeoutArg, "--attach=true", "--stdin", "--", "sh", "-c", "echo -n read: && cat && echo 'stdin closed'").
				WithStdinData("value\nabcd1234").
				ExecOrDie(ns)

			runOutput := waitForStdinContent("run-test", "stdin closed")
			gomega.Expect(runOutput).To(gomega.ContainSubstring("read:value"))
			gomega.Expect(runOutput).To(gomega.ContainSubstring("abcd1234"))
			gomega.Expect(runOutput).To(gomega.ContainSubstring("stdin closed"))

			framework.ExpectNoError(c.CoreV1().Pods(ns).Delete(ctx, "run-test", metav1.DeleteOptions{}))

			ginkgo.By("executing a command with run and attach without stdin")
			// There is a race on this scenario described in #73099
			// It fails if we are not able to attach before the container prints
			// "stdin closed", but hasn't exited yet.
			// We wait 10 seconds before printing to give time to kubectl to attach
			// to the container, this does not solve the race though.
			e2ekubectl.NewKubectlCommand(ns, "run", "run-test-2", "--image="+imageutils.GetE2EImage(imageutils.BusyBox), "--restart=OnFailure", podRunningTimeoutArg, "--attach=true", "--leave-stdin-open=true", "--", "sh", "-c", "cat && echo 'stdin closed'").
				WithStdinData("abcd1234").
				ExecOrDie(ns)

			runOutput = waitForStdinContent("run-test-2", "stdin closed")
			gomega.Expect(runOutput).ToNot(gomega.ContainSubstring("abcd1234"))
			gomega.Expect(runOutput).To(gomega.ContainSubstring("stdin closed"))

			framework.ExpectNoError(c.CoreV1().Pods(ns).Delete(ctx, "run-test-2", metav1.DeleteOptions{}))

			ginkgo.By("executing a command with run and attach with stdin with open stdin should remain running")
			e2ekubectl.NewKubectlCommand(ns, "run", "run-test-3", "--image="+imageutils.GetE2EImage(imageutils.BusyBox), "--restart=OnFailure", podRunningTimeoutArg, "--attach=true", "--leave-stdin-open=true", "--stdin", "--", "sh", "-c", "cat && echo 'stdin closed'").
				WithStdinData("abcd1234\n").
				ExecOrDie(ns)

			runOutput = waitForStdinContent("run-test-3", "abcd1234")
			gomega.Expect(runOutput).To(gomega.ContainSubstring("abcd1234"))
			gomega.Expect(runOutput).ToNot(gomega.ContainSubstring("stdin closed"))

			g := func(pods []*v1.Pod) sort.Interface { return sort.Reverse(controller.ActivePods(pods)) }
			runTestPod, _, err := polymorphichelpers.GetFirstPod(f.ClientSet.CoreV1(), ns, "run=run-test-3", 1*time.Minute, g)
			framework.ExpectNoError(err)
			framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(ctx, c, runTestPod.Name, ns, time.Minute))

			framework.ExpectNoError(c.CoreV1().Pods(ns).Delete(ctx, "run-test-3", metav1.DeleteOptions{}))
		})

		ginkgo.It("should support inline execution and attach with websockets or fallback to spdy", func(ctx context.Context) {
			waitForStdinContent := func(pod, content string) string {
				var logOutput string
				err := wait.PollUntilContextTimeout(ctx, 10*time.Second, 5*time.Minute, false, func(ctx context.Context) (bool, error) {
					logOutput = e2ekubectl.RunKubectlOrDie(ns, "logs", pod)
					return strings.Contains(logOutput, content), nil
				})
				framework.ExpectNoError(err, "waiting for '%v' output", content)
				return logOutput
			}

			ginkgo.By("executing a command with run and attach with stdin")
			// We wait for a non-empty line so we know kubectl has attached
			e2ekubectl.NewKubectlCommand(ns, "run", "run-test", "--image="+imageutils.GetE2EImage(imageutils.BusyBox), "--restart=OnFailure", podRunningTimeoutArg, "--attach=true", "--stdin", "--", "sh", "-c", "echo -n read: && cat && echo 'stdin closed'").
				WithStdinData("value\nabcd1234").
				ExecOrDie(ns)

			runOutput := waitForStdinContent("run-test", "stdin closed")
			gomega.Expect(runOutput).To(gomega.ContainSubstring("read:value"))
			gomega.Expect(runOutput).To(gomega.ContainSubstring("abcd1234"))
			gomega.Expect(runOutput).To(gomega.ContainSubstring("stdin closed"))

			framework.ExpectNoError(c.CoreV1().Pods(ns).Delete(ctx, "run-test", metav1.DeleteOptions{}))

			ginkgo.By("executing a command with run and attach without stdin")
			// There is a race on this scenario described in #73099
			// It fails if we are not able to attach before the container prints
			// "stdin closed", but hasn't exited yet.
			// We wait 10 seconds before printing to give time to kubectl to attach
			// to the container, this does not solve the race though.
			e2ekubectl.NewKubectlCommand(ns, "run", "run-test-2", "--image="+imageutils.GetE2EImage(imageutils.BusyBox), "--restart=OnFailure", podRunningTimeoutArg, "--attach=true", "--leave-stdin-open=true", "--", "sh", "-c", "cat && echo 'stdin closed'").
				WithStdinData("abcd1234").
				ExecOrDie(ns)

			runOutput = waitForStdinContent("run-test-2", "stdin closed")
			gomega.Expect(runOutput).ToNot(gomega.ContainSubstring("abcd1234"))
			gomega.Expect(runOutput).To(gomega.ContainSubstring("stdin closed"))

			framework.ExpectNoError(c.CoreV1().Pods(ns).Delete(ctx, "run-test-2", metav1.DeleteOptions{}))

			ginkgo.By("executing a command with run and attach with stdin with open stdin should remain running")
			e2ekubectl.NewKubectlCommand(ns, "run", "run-test-3", "--image="+imageutils.GetE2EImage(imageutils.BusyBox), "--restart=OnFailure", podRunningTimeoutArg, "--attach=true", "--leave-stdin-open=true", "--stdin", "--", "sh", "-c", "cat && echo 'stdin closed'").
				WithStdinData("abcd1234\n").
				ExecOrDie(ns)

			runOutput = waitForStdinContent("run-test-3", "abcd1234")
			gomega.Expect(runOutput).To(gomega.ContainSubstring("abcd1234"))
			gomega.Expect(runOutput).ToNot(gomega.ContainSubstring("stdin closed"))

			g := func(pods []*v1.Pod) sort.Interface { return sort.Reverse(controller.ActivePods(pods)) }
			runTestPod, _, err := polymorphichelpers.GetFirstPod(f.ClientSet.CoreV1(), ns, "run=run-test-3", 1*time.Minute, g)
			framework.ExpectNoError(err)
			framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(ctx, c, runTestPod.Name, ns, time.Minute))

			framework.ExpectNoError(c.CoreV1().Pods(ns).Delete(ctx, "run-test-3", metav1.DeleteOptions{}))
		})

		ginkgo.It("should contain last line of the log", func(ctx context.Context) {
			podName := "run-log-test"

			ginkgo.By("executing a command with run")
			e2ekubectl.RunKubectlOrDie(ns, "run", podName, "--image="+imageutils.GetE2EImage(imageutils.BusyBox), "--restart=OnFailure", podRunningTimeoutArg, "--", "sh", "-c", "sleep 10; for i in {1..100}; do echo $i; sleep 0.01; done; echo EOF")

			if !e2epod.CheckPodsRunningReadyOrSucceeded(ctx, c, ns, []string{podName}, framework.PodStartTimeout) {
				framework.Failf("Pod for run-log-test was not ready")
			}

			logOutput := e2ekubectl.RunKubectlOrDie(ns, "logs", "-f", "run-log-test")
			gomega.Expect(logOutput).To(gomega.ContainSubstring("EOF"))
		})
	})

	ginkgo.Describe("Kubectl api-versions", func() {
		/*
			Release: v1.9
			Testname: Kubectl, check version v1
			Description: Run kubectl to get api versions, output MUST contain returned versions with 'v1' listed.
		*/
		framework.ConformanceIt("should check if v1 is in available api versions", func(ctx context.Context) {
			ginkgo.By("validating api versions")
			output := e2ekubectl.RunKubectlOrDie(ns, "api-versions")
			if !strings.Contains(output, "v1") {
				framework.Failf("No v1 in kubectl api-versions")
			}
		})
	})

	ginkgo.Describe("Kubectl get componentstatuses", func() {
		ginkgo.It("should get componentstatuses", func(ctx context.Context) {
			ginkgo.By("getting list of componentstatuses")
			output := e2ekubectl.RunKubectlOrDie(ns, "get", "componentstatuses", "-o", "jsonpath={.items[*].metadata.name}")
			components := strings.Split(output, " ")
			ginkgo.By("getting details of componentstatuses")
			for _, component := range components {
				ginkgo.By("getting status of " + component)
				e2ekubectl.RunKubectlOrDie(ns, "get", "componentstatuses", component)
			}
		})
	})

	ginkgo.Describe("Kubectl prune with applyset", func() {
		ginkgo.It("should apply and prune objects", func(ctx context.Context) {
			framework.Logf("applying manifest1")
			manifest1 := `
apiVersion: v1
kind: ConfigMap
metadata:
  name: cm1
  namespace: {{ns}}
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: cm2
  namespace: {{ns}}
`

			manifest1 = strings.ReplaceAll(manifest1, "{{ns}}", ns)
			args := []string{"apply", "--prune", "--applyset=applyset1", "-f", "-"}
			e2ekubectl.NewKubectlCommand(ns, args...).AppendEnv([]string{"KUBECTL_APPLYSET=true"}).WithStdinData(manifest1).ExecOrDie(ns)

			framework.Logf("checking which objects exist")
			objects := mustListObjectsInNamespace(ctx, c, ns)
			names := mustGetNames(objects)
			if diff := cmp.Diff(names, []string{"cm1", "cm2"}); diff != "" {
				framework.Failf("unexpected configmap names (-want +got):\n%s", diff)
			}

			framework.Logf("applying manifest2")
			manifest2 := `
apiVersion: v1
kind: ConfigMap
metadata:
  name: cm1
  namespace: {{ns}}
`
			manifest2 = strings.ReplaceAll(manifest2, "{{ns}}", ns)

			e2ekubectl.NewKubectlCommand(ns, args...).AppendEnv([]string{"KUBECTL_APPLYSET=true"}).WithStdinData(manifest2).ExecOrDie(ns)

			framework.Logf("checking which objects exist")
			objects = mustListObjectsInNamespace(ctx, c, ns)
			names = mustGetNames(objects)
			if diff := cmp.Diff(names, []string{"cm1"}); diff != "" {
				framework.Failf("unexpected configmap names (-want +got):\n%s", diff)
			}

			framework.Logf("applying manifest2 (again)")
			e2ekubectl.NewKubectlCommand(ns, args...).AppendEnv([]string{"KUBECTL_APPLYSET=true"}).WithStdinData(manifest2).ExecOrDie(ns)

			framework.Logf("checking which objects exist")
			objects = mustListObjectsInNamespace(ctx, c, ns)
			names = mustGetNames(objects)
			if diff := cmp.Diff(names, []string{"cm1"}); diff != "" {
				framework.Failf("unexpected configmap names (-want +got):\n%s", diff)
			}
		})
	})

	ginkgo.Describe("Kubectl apply", func() {
		ginkgo.It("should apply a new configuration to an existing RC", func(ctx context.Context) {
			controllerJSON := commonutils.SubstituteImageName(string(readTestFileOrDie(agnhostControllerFilename)))

			ginkgo.By("creating Agnhost RC")
			e2ekubectl.RunKubectlOrDieInput(ns, controllerJSON, "create", "-f", "-")
			ginkgo.By("applying a modified configuration")
			stdin := modifyReplicationControllerConfiguration(controllerJSON)
			e2ekubectl.NewKubectlCommand(ns, "apply", "-f", "-").
				WithStdinReader(stdin).
				ExecOrDie(ns)
			ginkgo.By("checking the result")
			forEachReplicationController(ctx, c, ns, "app", "agnhost", validateReplicationControllerConfiguration)
		})
		ginkgo.It("should reuse port when apply to an existing SVC", func(ctx context.Context) {
			serviceJSON := readTestFileOrDie(agnhostServiceFilename)

			ginkgo.By("creating Agnhost SVC")
			e2ekubectl.RunKubectlOrDieInput(ns, string(serviceJSON[:]), "create", "-f", "-")

			ginkgo.By("getting the original port")
			originalNodePort := e2ekubectl.RunKubectlOrDie(ns, "get", "service", "agnhost-primary", "-o", "jsonpath={.spec.ports[0].port}")

			ginkgo.By("applying the same configuration")
			e2ekubectl.RunKubectlOrDieInput(ns, string(serviceJSON[:]), "apply", "-f", "-")

			ginkgo.By("getting the port after applying configuration")
			currentNodePort := e2ekubectl.RunKubectlOrDie(ns, "get", "service", "agnhost-primary", "-o", "jsonpath={.spec.ports[0].port}")

			ginkgo.By("checking the result")
			if originalNodePort != currentNodePort {
				framework.Failf("port should keep the same")
			}
		})

		ginkgo.It("apply set/view last-applied", func(ctx context.Context) {
			deployment1Yaml := commonutils.SubstituteImageName(string(readTestFileOrDie(httpdDeployment1Filename)))
			deployment2Yaml := commonutils.SubstituteImageName(string(readTestFileOrDie(httpdDeployment2Filename)))
			deployment3Yaml := commonutils.SubstituteImageName(string(readTestFileOrDie(httpdDeployment3Filename)))

			ginkgo.By("deployment replicas number is 2")
			e2ekubectl.RunKubectlOrDieInput(ns, deployment1Yaml, "apply", "-f", "-")

			ginkgo.By("check the last-applied matches expectations annotations")
			output := e2ekubectl.RunKubectlOrDieInput(ns, deployment1Yaml, "apply", "view-last-applied", "-f", "-", "-o", "json")
			requiredString := "\"replicas\": 2"
			if !strings.Contains(output, requiredString) {
				framework.Failf("Missing %s in kubectl view-last-applied", requiredString)
			}

			ginkgo.By("apply file doesn't have replicas")
			e2ekubectl.RunKubectlOrDieInput(ns, deployment2Yaml, "apply", "set-last-applied", "-f", "-")

			ginkgo.By("check last-applied has been updated, annotations doesn't have replicas")
			output = e2ekubectl.RunKubectlOrDieInput(ns, deployment1Yaml, "apply", "view-last-applied", "-f", "-", "-o", "json")
			requiredString = "\"replicas\": 2"
			if strings.Contains(output, requiredString) {
				framework.Failf("Presenting %s in kubectl view-last-applied", requiredString)
			}

			ginkgo.By("scale set replicas to 3")
			httpdDeploy := "httpd-deployment"
			debugDiscovery()
			e2ekubectl.RunKubectlOrDie(ns, "scale", "deployment", httpdDeploy, "--replicas=3")

			ginkgo.By("apply file doesn't have replicas but image changed")
			e2ekubectl.RunKubectlOrDieInput(ns, deployment3Yaml, "apply", "-f", "-")

			ginkgo.By("verify replicas still is 3 and image has been updated")
			output = e2ekubectl.RunKubectlOrDieInput(ns, deployment3Yaml, "get", "-f", "-", "-o", "json")
			requiredItems := []string{"\"replicas\": 3", imageutils.GetE2EImage(imageutils.Httpd)}
			for _, item := range requiredItems {
				if !strings.Contains(output, item) {
					framework.Failf("Missing %s in kubectl apply", item)
				}
			}
		})
	})

	ginkgo.Describe("Kubectl diff", func() {
		/*
			Release: v1.19
			Testname: Kubectl, diff Deployment
			Description: Create a Deployment with httpd image. Declare the same Deployment with a different image, busybox. Diff of live Deployment with declared Deployment MUST include the difference between live and declared image.
		*/
		framework.ConformanceIt("should check if kubectl diff finds a difference for Deployments", func(ctx context.Context) {
			ginkgo.By("create deployment with httpd image")
			deployment := commonutils.SubstituteImageName(string(readTestFileOrDie(httpdDeployment3Filename)))
			e2ekubectl.RunKubectlOrDieInput(ns, deployment, "create", "-f", "-")

			ginkgo.By("verify diff finds difference between live and declared image")
			deployment = strings.Replace(deployment, imageutils.GetE2EImage(imageutils.Httpd), imageutils.GetE2EImage(imageutils.BusyBox), 1)
			if !strings.Contains(deployment, imageutils.GetE2EImage(imageutils.BusyBox)) {
				framework.Failf("Failed replacing image from %s to %s in:\n%s\n", imageutils.GetE2EImage(imageutils.Httpd), imageutils.GetE2EImage(imageutils.BusyBox), deployment)
			}
			output, err := e2ekubectl.RunKubectlInput(ns, deployment, "diff", "-f", "-")
			if err, ok := err.(*exec.ExitError); ok && err.ExitCode() == 1 {
				framework.Failf("Expected kubectl diff exit code of 1, but got %d: %v\n", err.ExitCode(), err)
			}
			requiredItems := []string{imageutils.GetE2EImage(imageutils.Httpd), imageutils.GetE2EImage(imageutils.BusyBox)}
			for _, item := range requiredItems {
				if !strings.Contains(output, item) {
					framework.Failf("Missing %s in kubectl diff output:\n%s\n%v\n", item, output, err)
				}
			}

			e2ekubectl.RunKubectlOrDieInput(ns, deployment, "delete", "-f", "-")
		})
	})

	ginkgo.Describe("Kubectl server-side dry-run", func() {
		/*
			Release: v1.19
			Testname: Kubectl, server-side dry-run Pod
			Description: The command 'kubectl run' must create a pod with the specified image name. After, the command 'kubectl patch pod -p {...} --dry-run=server' should update the Pod with the new image name and server-side dry-run enabled. The image name must not change.
		*/
		framework.ConformanceIt("should check if kubectl can dry-run update Pods", func(ctx context.Context) {
			httpdImage := imageutils.GetE2EImage(imageutils.Httpd)
			ginkgo.By("running the image " + httpdImage)
			podName := "e2e-test-httpd-pod"
			e2ekubectl.RunKubectlOrDie(ns, "run", podName, "--image="+httpdImage, podRunningTimeoutArg, "--labels=run="+podName)

			ginkgo.By("replace the image in the pod with server-side dry-run")
			specImage := fmt.Sprintf(`{"spec":{"containers":[{"name": "%s","image": "%s"}]}}`, podName, imageutils.GetE2EImage(imageutils.BusyBox))
			e2ekubectl.RunKubectlOrDie(ns, "patch", "pod", podName, "-p", specImage, "--dry-run=server")

			ginkgo.By("verifying the pod " + podName + " has the right image " + httpdImage)
			pod, err := c.CoreV1().Pods(ns).Get(ctx, podName, metav1.GetOptions{})
			if err != nil {
				framework.Failf("Failed getting pod %s: %v", podName, err)
			}
			containers := pod.Spec.Containers
			if checkContainersImage(containers, httpdImage) {
				framework.Failf("Failed creating pod with expected image %s", httpdImage)
			}

			e2ekubectl.RunKubectlOrDie(ns, "delete", "pods", podName)
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
				framework.Logf("%v\n%s", err, string(extension.GetValue().GetYaml()))
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
			framework.Failf("%v", err)
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

	ginkgo.Describe("Kubectl validation", func() {
		ginkgo.It("should create/apply a CR with unknown fields for CRD with no validation schema", func(ctx context.Context) {
			ginkgo.By("create CRD with no validation schema")
			crd, err := crd.CreateTestCRD(f)
			if err != nil {
				framework.Failf("failed to create test CRD: %v", err)
			}
			ginkgo.DeferCleanup(crd.CleanUp)

			ginkgo.By("sleep for 10s to wait for potential crd openapi publishing alpha feature")
			time.Sleep(10 * time.Second)

			meta := fmt.Sprintf(metaPattern, crd.Crd.Spec.Names.Kind, crd.Crd.Spec.Group, crd.Crd.Spec.Versions[0].Name, "test-cr")
			randomCR := fmt.Sprintf(`{%s,"a":{"b":[{"c":"d"}]}}`, meta)
			if err := createApplyCustomResource(randomCR, f.Namespace.Name, "test-cr", crd); err != nil {
				framework.Failf("%v", err)
			}
		})

		ginkgo.It("should create/apply a valid CR for CRD with validation schema", func(ctx context.Context) {
			ginkgo.By("prepare CRD with validation schema")
			crd, err := crd.CreateTestCRD(f, func(crd *apiextensionsv1.CustomResourceDefinition) {
				props := &apiextensionsv1.JSONSchemaProps{}
				if err := yaml.Unmarshal(schemaFoo, props); err != nil {
					framework.Failf("failed to unmarshal schema: %v", err)
				}
				for i := range crd.Spec.Versions {
					crd.Spec.Versions[i].Schema = &apiextensionsv1.CustomResourceValidation{OpenAPIV3Schema: props}
				}
			})
			if err != nil {
				framework.Failf("failed to create test CRD: %v", err)
			}
			ginkgo.DeferCleanup(crd.CleanUp)

			ginkgo.By("sleep for 10s to wait for potential crd openapi publishing alpha feature")
			time.Sleep(10 * time.Second)

			meta := fmt.Sprintf(metaPattern, crd.Crd.Spec.Names.Kind, crd.Crd.Spec.Group, crd.Crd.Spec.Versions[0].Name, "test-cr")
			validCR := fmt.Sprintf(`{%s,"spec":{"bars":[{"name":"test-bar"}]}}`, meta)
			if err := createApplyCustomResource(validCR, f.Namespace.Name, "test-cr", crd); err != nil {
				framework.Failf("%v", err)
			}
		})

		ginkgo.It("should create/apply an invalid/valid CR with arbitrary-extra properties for CRD with partially-specified validation schema", func(ctx context.Context) {
			ginkgo.By("prepare CRD with partially-specified validation schema")
			crd, err := crd.CreateTestCRD(f, func(crd *apiextensionsv1.CustomResourceDefinition) {
				props := &apiextensionsv1.JSONSchemaProps{}
				if err := yaml.Unmarshal(schemaFoo, props); err != nil {
					framework.Failf("failed to unmarshal schema: %v", err)
				}
				// Allow for arbitrary-extra properties.
				props.XPreserveUnknownFields = pointer.BoolPtr(true)
				for i := range crd.Spec.Versions {
					crd.Spec.Versions[i].Schema = &apiextensionsv1.CustomResourceValidation{OpenAPIV3Schema: props}
				}
			})
			if err != nil {
				framework.Failf("failed to create test CRD: %v", err)
			}
			ginkgo.DeferCleanup(crd.CleanUp)

			ginkgo.By("sleep for 10s to wait for potential crd openapi publishing alpha feature")
			time.Sleep(10 * time.Second)

			schema := schemaForGVK(schema.GroupVersionKind{Group: crd.Crd.Spec.Group, Version: crd.Crd.Spec.Versions[0].Name, Kind: crd.Crd.Spec.Names.Kind})
			gomega.Expect(schema).ToNot(gomega.BeNil(), "retrieving a schema for the crd")

			meta := fmt.Sprintf(metaPattern, crd.Crd.Spec.Names.Kind, crd.Crd.Spec.Group, crd.Crd.Spec.Versions[0].Name, "test-cr")

			// XPreserveUnknownFields is defined on the root of the schema so unknown fields within the spec
			// are still considered invalid
			invalidArbitraryCR := fmt.Sprintf(`{%s,"spec":{"bars":[{"name":"test-bar"}],"extraProperty":"arbitrary-value"}}`, meta)
			err = createApplyCustomResource(invalidArbitraryCR, f.Namespace.Name, "test-cr", crd)
			gomega.Expect(err).To(gomega.HaveOccurred(), "creating custom resource")

			if !strings.Contains(err.Error(), `unknown field "spec.extraProperty"`) {
				framework.Failf("incorrect error from createApplyCustomResource: %v", err)
			}

			// unknown fields on the root are considered valid
			validArbitraryCR := fmt.Sprintf(`{%s,"spec":{"bars":[{"name":"test-bar"}]},"extraProperty":"arbitrary-value"}`, meta)
			err = createApplyCustomResource(validArbitraryCR, f.Namespace.Name, "test-cr", crd)
			framework.ExpectNoError(err, "creating custom resource")
		})

		ginkgo.It("should detect unknown metadata fields in both the root and embedded object of a CR", func(ctx context.Context) {
			ginkgo.By("prepare CRD with x-kubernetes-embedded-resource: true")
			testCRD, err := crd.CreateTestCRD(f, func(crd *apiextensionsv1.CustomResourceDefinition) {
				props := &apiextensionsv1.JSONSchemaProps{}
				if err := yaml.Unmarshal(schemaFooEmbedded, props); err != nil {
					framework.Failf("failed to unmarshal schema: %v", err)
				}
				crd.Spec.Versions = []apiextensionsv1.CustomResourceDefinitionVersion{
					{
						Name:    "v1",
						Served:  true,
						Storage: true,
						Schema: &apiextensionsv1.CustomResourceValidation{
							OpenAPIV3Schema: props,
						},
					},
				}
			})
			if err != nil {
				framework.Failf("failed to create test CRD: %v", err)
			}
			ginkgo.DeferCleanup(testCRD.CleanUp)

			ginkgo.By("sleep for 10s to wait for potential crd openapi publishing alpha feature")
			time.Sleep(10 * time.Second)

			ginkgo.By("attempting to create a CR with unknown metadata fields at the root level")
			gvk := schema.GroupVersionKind{Group: testCRD.Crd.Spec.Group, Version: testCRD.Crd.Spec.Versions[0].Name, Kind: testCRD.Crd.Spec.Names.Kind}
			schema := schemaForGVK(gvk)
			gomega.Expect(schema).ToNot(gomega.BeNil(), "retrieving a schema for the crd")
			embeddedCRPattern := `

{%s,
  "spec": {
    "template": {
      "apiVersion": "foo/v1",
      "kind": "Sub",
      "metadata": {
        %s
        "name": "subobject",
        "namespace": "%s"
      }
    }
  }
}`
			meta := unknownFieldMetadataJSON(gvk, "test-cr")
			unknownRootMetaCR := fmt.Sprintf(embeddedCRPattern, meta, "", ns)
			_, err = e2ekubectl.RunKubectlInput(ns, unknownRootMetaCR, "create", "--validate=true", "-f", "-")
			if err == nil {
				framework.Failf("unexpected nil error when creating CR with unknown root metadata field")
			}
			if !(strings.Contains(err.Error(), `unknown field "unknownMeta"`) || strings.Contains(err.Error(), `unknown field "metadata.unknownMeta"`)) {
				framework.Failf("error missing root unknown metadata field, got: %v", err)
			}
			if strings.Contains(err.Error(), `unknown field "namespace"`) || strings.Contains(err.Error(), `unknown field "metadata.namespace"`) {
				framework.Failf("unexpected error, CR's root metadata namespace field unrecognized: %v", err)
			}

			ginkgo.By("attempting to create a CR with unknown metadata fields in the embedded object")
			metaEmbedded := fmt.Sprintf(metaPattern, testCRD.Crd.Spec.Names.Kind, testCRD.Crd.Spec.Group, testCRD.Crd.Spec.Versions[0].Name, "test-cr-embedded")
			unknownEmbeddedMetaCR := fmt.Sprintf(embeddedCRPattern, metaEmbedded, `"unknownMetaEmbedded": "bar",`, ns)
			_, err = e2ekubectl.RunKubectlInput(ns, unknownEmbeddedMetaCR, "create", "--validate=true", "-f", "-")
			if err == nil {
				framework.Failf("unexpected nil error when creating CR with unknown embedded metadata field")
			}
			if !(strings.Contains(err.Error(), `unknown field "unknownMetaEmbedded"`) || strings.Contains(err.Error(), `unknown field "spec.template.metadata.unknownMetaEmbedded"`)) {
				framework.Failf("error missing embedded unknown metadata field, got: %v", err)
			}
			if strings.Contains(err.Error(), `unknown field "namespace"`) || strings.Contains(err.Error(), `unknown field "spec.template.metadata.namespace"`) {
				framework.Failf("unexpected error, CR's embedded metadata namespace field unrecognized: %v", err)
			}
		})

		ginkgo.It("should detect unknown metadata fields of a typed object", func(ctx context.Context) {
			ginkgo.By("calling kubectl create deployment")
			invalidMetaDeployment := `
	{
		"apiVersion": "apps/v1",
		"kind": "Deployment",
		"metadata": {
			"name": "my-dep",
			"unknownMeta": "foo",
			"labels": {"app": "nginx"}
		},
		"spec": {
			"selector": {
				"matchLabels": {
					"app": "nginx"
				}
			},
			"template": {
				"metadata": {
					"labels": {
						"app": "nginx"
					}
				},
				"spec": {
					"containers": [{
						"name":  "nginx",
						"image": "nginx:latest"
					}]
				}
			}
		}
	}
		`
			_, err := e2ekubectl.RunKubectlInput(ns, invalidMetaDeployment, "create", "-f", "-")
			if err == nil {
				framework.Failf("unexpected nil error when creating deployment with unknown metadata field")
			}
			if !(strings.Contains(err.Error(), `unknown field "unknownMeta"`) || strings.Contains(err.Error(), `unknown field "metadata.unknownMeta"`)) {
				framework.Failf("error missing unknown metadata field, got: %v", err)
			}
			if strings.Contains(err.Error(), `unknown field "namespace"`) || strings.Contains(err.Error(), `unknown field "metadata.namespace"`) {
				framework.Failf("unexpected error, deployment's metadata namespace field unrecognized: %v", err)
			}

		})
	})

	ginkgo.Describe("Kubectl cluster-info", func() {
		/*
			Release: v1.9
			Testname: Kubectl, cluster info
			Description: Call kubectl to get cluster-info, output MUST contain cluster-info returned and Kubernetes control plane SHOULD be running.
		*/
		framework.ConformanceIt("should check if Kubernetes control plane services is included in cluster-info", func(ctx context.Context) {
			ginkgo.By("validating cluster-info")
			output := e2ekubectl.RunKubectlOrDie(ns, "cluster-info")
			// Can't check exact strings due to terminal control commands (colors)
			requiredItems := []string{"Kubernetes control plane", "is running at"}
			for _, item := range requiredItems {
				if !strings.Contains(output, item) {
					framework.Failf("Missing %s in kubectl cluster-info", item)
				}
			}
		})
	})

	ginkgo.Describe("Kubectl cluster-info dump", func() {
		ginkgo.It("should check if cluster-info dump succeeds", func(ctx context.Context) {
			ginkgo.By("running cluster-info dump")
			e2ekubectl.RunKubectlOrDie(ns, "cluster-info", "dump")
		})
	})

	ginkgo.Describe("Kubectl describe", func() {
		/*
			Release: v1.9
			Testname: Kubectl, describe pod or rc
			Description: Deploy an agnhost controller and an agnhost service. Kubectl describe pods SHOULD return the name, namespace, labels, state and other information as expected. Kubectl describe on rc, service, node and namespace SHOULD also return proper information.
		*/
		framework.ConformanceIt("should check if kubectl describe prints relevant information for rc and pods", func(ctx context.Context) {
			controllerJSON := commonutils.SubstituteImageName(string(readTestFileOrDie(agnhostControllerFilename)))
			serviceJSON := readTestFileOrDie(agnhostServiceFilename)

			e2ekubectl.RunKubectlOrDieInput(ns, controllerJSON, "create", "-f", "-")
			e2ekubectl.RunKubectlOrDieInput(ns, string(serviceJSON[:]), "create", "-f", "-")

			ginkgo.By("Waiting for Agnhost primary to start.")
			waitForOrFailWithDebug(ctx, 1)

			// Pod
			forEachPod(ctx, func(pod v1.Pod) {
				output := e2ekubectl.RunKubectlOrDie(ns, "describe", "pod", pod.Name)
				requiredStrings := [][]string{
					{"Name:", "agnhost-primary-"},
					{"Namespace:", ns},
					{"Node:"},
					{"Labels:", "app=agnhost"},
					{"role=primary"},
					{"Annotations:"},
					{"Status:", "Running"},
					{"IP:"},
					{"Controlled By:", "ReplicationController/agnhost-primary"},
					{"Image:", imageutils.GetE2EImage(imageutils.Agnhost)},
					{"State:", "Running"},
					{"QoS Class:", "BestEffort"},
				}
				checkOutput(output, requiredStrings)
			})

			// Rc
			requiredStrings := [][]string{
				{"Name:", "agnhost-primary"},
				{"Namespace:", ns},
				{"Selector:", "app=agnhost,role=primary"},
				{"Labels:", "app=agnhost"},
				{"role=primary"},
				{"Annotations:"},
				{"Replicas:", "1 current", "1 desired"},
				{"Pods Status:", "1 Running", "0 Waiting", "0 Succeeded", "0 Failed"},
				{"Pod Template:"},
				{"Image:", imageutils.GetE2EImage(imageutils.Agnhost)},
				{"Events:"}}
			checkKubectlOutputWithRetry(ns, requiredStrings, "describe", "rc", "agnhost-primary")

			// Service
			output := e2ekubectl.RunKubectlOrDie(ns, "describe", "service", "agnhost-primary")
			requiredStrings = [][]string{
				{"Name:", "agnhost-primary"},
				{"Namespace:", ns},
				{"Labels:", "app=agnhost"},
				{"role=primary"},
				{"Annotations:"},
				{"Selector:", "app=agnhost", "role=primary"},
				{"Type:", "ClusterIP"},
				{"IP:"},
				{"Port:", "<unset>", "6379/TCP"},
				{"Endpoints:"},
				{"Session Affinity:", "None"}}
			checkOutput(output, requiredStrings)

			// Node
			// It should be OK to list unschedulable Nodes here.
			nodes, err := c.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
			framework.ExpectNoError(err)
			node := nodes.Items[0]
			output = e2ekubectl.RunKubectlOrDie(ns, "describe", "node", node.Name)
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
			output = e2ekubectl.RunKubectlOrDie(ns, "describe", "namespace", ns)
			requiredStrings = [][]string{
				{"Name:", ns},
				{"Labels:"},
				{"Annotations:"},
				{"Status:", "Active"}}
			checkOutput(output, requiredStrings)

			// Quota and limitrange are skipped for now.
		})

		ginkgo.It("should check if kubectl describe prints relevant information for cronjob", func(ctx context.Context) {
			ginkgo.By("creating a cronjob")
			cronjobYaml := commonutils.SubstituteImageName(string(readTestFileOrDie("busybox-cronjob.yaml.in")))
			e2ekubectl.RunKubectlOrDieInput(ns, cronjobYaml, "create", "-f", "-")

			ginkgo.By("waiting for cronjob to start.")
			err := wait.PollImmediate(time.Second, time.Minute, func() (bool, error) {
				cj, err := c.BatchV1().CronJobs(ns).List(ctx, metav1.ListOptions{})
				if err != nil {
					return false, fmt.Errorf("Failed getting CronJob %s: %w", ns, err)
				}
				return len(cj.Items) > 0, nil
			})
			framework.ExpectNoError(err)

			ginkgo.By("verifying kubectl describe prints")
			output := e2ekubectl.RunKubectlOrDie(ns, "describe", "cronjob", "cronjob-test")
			requiredStrings := [][]string{
				{"Name:", "cronjob-test"},
				{"Namespace:", ns},
				{"Labels:"},
				{"Annotations:"},
				{"Schedule:", "*/1 * * * *"},
				{"Concurrency Policy:", "Allow"},
				{"Suspend:", "False"},
				{"Successful Job History Limit:", "3"},
				{"Failed Job History Limit:", "1"},
				{"Starting Deadline Seconds:", "30s"},
				{"Selector:"},
				{"Parallelism:"},
				{"Completions:"},
			}
			checkOutput(output, requiredStrings)
		})
	})

	ginkgo.Describe("Kubectl expose", func() {
		/*
			Release: v1.9
			Testname: Kubectl, create service, replication controller
			Description: Create a Pod running agnhost listening to port 6379. Using kubectl expose the agnhost primary replication controllers at port 1234. Validate that the replication controller is listening on port 1234 and the target port is set to 6379, port that agnhost primary is listening. Using kubectl expose the agnhost primary as a service at port 2345. The service MUST be listening on port 2345 and the target port is set to 6379, port that agnhost primary is listening.
		*/
		framework.ConformanceIt("should create services for rc", func(ctx context.Context) {
			controllerJSON := commonutils.SubstituteImageName(string(readTestFileOrDie(agnhostControllerFilename)))

			agnhostPort := 6379

			ginkgo.By("creating Agnhost RC")

			framework.Logf("namespace %v", ns)
			e2ekubectl.RunKubectlOrDieInput(ns, controllerJSON, "create", "-f", "-")

			// It may take a while for the pods to get registered in some cases, wait to be sure.
			ginkgo.By("Waiting for Agnhost primary to start.")
			waitForOrFailWithDebug(ctx, 1)
			forEachPod(ctx, func(pod v1.Pod) {
				framework.Logf("wait on agnhost-primary startup in %v ", ns)
				e2eoutput.LookForStringInLog(ns, pod.Name, "agnhost-primary", "Paused", framework.PodStartTimeout)
			})
			validateService := func(name string, servicePort int, timeout time.Duration) {
				err := wait.Poll(framework.Poll, timeout, func() (bool, error) {
					slices, err := c.DiscoveryV1().EndpointSlices(ns).List(ctx, metav1.ListOptions{LabelSelector: fmt.Sprintf("%s=%s", discoveryv1.LabelServiceName, name)})
					if err != nil {
						// log the real error
						framework.Logf("List endpointslices failed (interval %v): %v", framework.Poll, err)

						// if the error is API not found or could not find default credentials or TLS handshake timeout, try again
						if apierrors.IsNotFound(err) ||
							apierrors.IsUnauthorized(err) ||
							apierrors.IsServerTimeout(err) {
							err = nil
						}
						return false, err
					}

					uidToPort := e2eendpointslice.GetContainerPortsByPodUID(slices.Items)
					if len(uidToPort) == 0 {
						framework.Logf("No endpoint found, retrying")
						return false, nil
					}
					if len(uidToPort) > 1 {
						framework.Failf("Too many endpoints found")
					}
					for _, port := range uidToPort {
						if port[0] != agnhostPort {
							framework.Failf("Wrong endpoint port: %d", port[0])
						}
					}
					return true, nil
				})
				framework.ExpectNoError(err)

				e2eservice, err := c.CoreV1().Services(ns).Get(ctx, name, metav1.GetOptions{})
				framework.ExpectNoError(err)

				if len(e2eservice.Spec.Ports) != 1 {
					framework.Failf("1 port is expected")
				}
				port := e2eservice.Spec.Ports[0]
				if port.Port != int32(servicePort) {
					framework.Failf("Wrong service port: %d", port.Port)
				}
				if port.TargetPort.IntValue() != agnhostPort {
					framework.Failf("Wrong target port: %d", port.TargetPort.IntValue())
				}
			}

			ginkgo.By("exposing RC")
			e2ekubectl.RunKubectlOrDie(ns, "expose", "rc", "agnhost-primary", "--name=rm2", "--port=1234", fmt.Sprintf("--target-port=%d", agnhostPort))
			framework.ExpectNoError(e2enetwork.WaitForService(ctx, c, ns, "rm2", true, framework.Poll, framework.ServiceStartTimeout))
			validateService("rm2", 1234, framework.ServiceStartTimeout)

			ginkgo.By("exposing service")
			e2ekubectl.RunKubectlOrDie(ns, "expose", "service", "rm2", "--name=rm3", "--port=2345", fmt.Sprintf("--target-port=%d", agnhostPort))
			framework.ExpectNoError(e2enetwork.WaitForService(ctx, c, ns, "rm3", true, framework.Poll, framework.ServiceStartTimeout))
			validateService("rm3", 2345, framework.ServiceStartTimeout)
		})
	})

	ginkgo.Describe("Kubectl label", func() {
		var podYaml string
		ginkgo.BeforeEach(func(ctx context.Context) {
			ginkgo.By("creating the pod")
			podYaml = commonutils.SubstituteImageName(string(readTestFileOrDie("pause-pod.yaml.in")))
			e2ekubectl.RunKubectlOrDieInput(ns, podYaml, "create", "-f", "-")
			framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(ctx, c, pausePodName, ns, framework.PodStartTimeout))
		})
		ginkgo.AfterEach(func() {
			cleanupKubectlInputs(podYaml, ns, pausePodSelector)
		})

		/*
			Release: v1.9
			Testname: Kubectl, label update
			Description: When a Pod is running, update a Label using 'kubectl label' command. The label MUST be created in the Pod. A 'kubectl get pod' with -l option on the container MUST verify that the label can be read back. Use 'kubectl label label-' to remove the label. 'kubectl get pod' with -l option SHOULD not list the deleted label as the label is removed.
		*/
		framework.ConformanceIt("should update the label on a resource", func(ctx context.Context) {
			labelName := "testing-label"
			labelValue := "testing-label-value"

			ginkgo.By("adding the label " + labelName + " with value " + labelValue + " to a pod")
			e2ekubectl.RunKubectlOrDie(ns, "label", "pods", pausePodName, labelName+"="+labelValue)
			ginkgo.By("verifying the pod has the label " + labelName + " with the value " + labelValue)
			output := e2ekubectl.RunKubectlOrDie(ns, "get", "pod", pausePodName, "-L", labelName)
			if !strings.Contains(output, labelValue) {
				framework.Fail("Failed updating label " + labelName + " to the pod " + pausePodName)
			}

			ginkgo.By("removing the label " + labelName + " of a pod")
			e2ekubectl.RunKubectlOrDie(ns, "label", "pods", pausePodName, labelName+"-")
			ginkgo.By("verifying the pod doesn't have the label " + labelName)
			output = e2ekubectl.RunKubectlOrDie(ns, "get", "pod", pausePodName, "-L", labelName)
			if strings.Contains(output, labelValue) {
				framework.Fail("Failed removing label " + labelName + " of the pod " + pausePodName)
			}
		})
	})

	ginkgo.Describe("Kubectl copy", func() {
		var podYaml string
		ginkgo.BeforeEach(func(ctx context.Context) {
			ginkgo.By("creating the pod")
			podYaml = commonutils.SubstituteImageName(string(readTestFileOrDie("busybox-pod.yaml.in")))
			e2ekubectl.RunKubectlOrDieInput(ns, podYaml, "create", "-f", "-")
			framework.ExpectNoError(e2epod.WaitTimeoutForPodReadyInNamespace(ctx, c, busyboxPodName, ns, framework.PodStartTimeout))
		})
		ginkgo.AfterEach(func() {
			cleanupKubectlInputs(podYaml, ns, busyboxPodSelector)
		})

		/*
			Release: v1.12
			Testname: Kubectl, copy
			Description: When a Pod is running, copy a known file from it to a temporary local destination.
		*/
		ginkgo.It("should copy a file from a running Pod", func(ctx context.Context) {
			remoteContents := "foobar\n"
			podSource := fmt.Sprintf("%s:/root/foo/bar/foo.bar", busyboxPodName)
			tempDestination, err := os.CreateTemp(os.TempDir(), "copy-foobar")
			if err != nil {
				framework.Failf("Failed creating temporary destination file: %v", err)
			}

			ginkgo.By("specifying a remote filepath " + podSource + " on the pod")
			e2ekubectl.RunKubectlOrDie(ns, "cp", podSource, tempDestination.Name())
			ginkgo.By("verifying that the contents of the remote file " + podSource + " have been copied to a local file " + tempDestination.Name())
			localData, err := io.ReadAll(tempDestination)
			if err != nil {
				framework.Failf("Failed reading temporary local file: %v", err)
			}
			if string(localData) != remoteContents {
				framework.Failf("Failed copying remote file contents. Expected %s but got %s", remoteContents, string(localData))
			}
		})
	})

	ginkgo.Describe("Kubectl patch", func() {
		/*
			Release: v1.9
			Testname: Kubectl, patch to annotate
			Description: Start running agnhost and a replication controller. When the pod is running, using 'kubectl patch' command add annotations. The annotation MUST be added to running pods and SHOULD be able to read added annotations from each of the Pods running under the replication controller.
		*/
		framework.ConformanceIt("should add annotations for pods in rc", func(ctx context.Context) {
			controllerJSON := commonutils.SubstituteImageName(string(readTestFileOrDie(agnhostControllerFilename)))
			ginkgo.By("creating Agnhost RC")
			e2ekubectl.RunKubectlOrDieInput(ns, controllerJSON, "create", "-f", "-")
			ginkgo.By("Waiting for Agnhost primary to start.")
			waitForOrFailWithDebug(ctx, 1)
			ginkgo.By("patching all pods")
			forEachPod(ctx, func(pod v1.Pod) {
				e2ekubectl.RunKubectlOrDie(ns, "patch", "pod", pod.Name, "-p", "{\"metadata\":{\"annotations\":{\"x\":\"y\"}}}")
			})

			ginkgo.By("checking annotations")
			forEachPod(ctx, func(pod v1.Pod) {
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

	ginkgo.Describe("Kubectl version", func() {
		/*
			Release: v1.9
			Testname: Kubectl, version
			Description: The command 'kubectl version' MUST return the major, minor versions,  GitCommit, etc of the Client and the Server that the kubectl is configured to connect to.
		*/
		framework.ConformanceIt("should check is all data is printed", func(ctx context.Context) {
			versionString := e2ekubectl.RunKubectlOrDie(ns, "version")
			// we expect following values for: Major -> digit, Minor -> numeric followed by an optional '+',  GitCommit -> alphanumeric
			requiredItems := []string{"Client Version: ", "Server Version: "}
			for _, item := range requiredItems {
				// prior to 1.28 we printed long version information
				oldMatched, _ := regexp.MatchString(item+`version.Info\{Major:"\d", Minor:"\d+\+?", GitVersion:"v\d\.\d+\.[\d\w\-\.\+]+", GitCommit:"[0-9a-f]+"`, versionString)
				// 1.28+ prints short information
				newMatched, _ := regexp.MatchString(item+`v\d\.\d+\.[\d\w\-\.\+]+`, versionString)
				// due to backwards compatibility we need to match both until 1.30 most likely
				if !oldMatched && !newMatched {
					framework.Failf("Item %s value is not valid in %s\n", item, versionString)
				}
			}
		})
	})

	ginkgo.Describe("Kubectl run pod", func() {
		var podName string

		ginkgo.BeforeEach(func() {
			podName = "e2e-test-httpd-pod"
		})

		ginkgo.AfterEach(func() {
			e2ekubectl.RunKubectlOrDie(ns, "delete", "pods", podName)
		})

		/*
			Release: v1.9
			Testname: Kubectl, run pod
			Description: Command 'kubectl run' MUST create a pod, when a image name is specified in the run command. After the run command there SHOULD be a pod that should exist with one container running the specified image.
		*/
		framework.ConformanceIt("should create a pod from an image when restart is Never", func(ctx context.Context) {
			httpdImage := imageutils.GetE2EImage(imageutils.Httpd)
			ginkgo.By("running the image " + httpdImage)
			e2ekubectl.RunKubectlOrDie(ns, "run", podName, "--restart=Never", podRunningTimeoutArg, "--image="+httpdImage)
			ginkgo.By("verifying the pod " + podName + " was created")
			pod, err := c.CoreV1().Pods(ns).Get(ctx, podName, metav1.GetOptions{})
			if err != nil {
				framework.Failf("Failed getting pod %s: %v", podName, err)
			}
			containers := pod.Spec.Containers
			if checkContainersImage(containers, httpdImage) {
				framework.Failf("Failed creating pod %s with expected image %s", podName, httpdImage)
			}
			if pod.Spec.RestartPolicy != v1.RestartPolicyNever {
				framework.Failf("Failed creating a pod with correct restart policy for --restart=Never")
			}
		})
	})

	ginkgo.Describe("Kubectl replace", func() {
		var podName string

		ginkgo.BeforeEach(func() {
			podName = "e2e-test-httpd-pod"
		})

		ginkgo.AfterEach(func() {
			e2ekubectl.RunKubectlOrDie(ns, "delete", "pods", podName)
		})

		/*
			Release: v1.9
			Testname: Kubectl, replace
			Description: Command 'kubectl replace' on a existing Pod with a new spec MUST update the image of the container running in the Pod. A -f option to 'kubectl replace' SHOULD force to re-create the resource. The new Pod SHOULD have the container with new change to the image.
		*/
		framework.ConformanceIt("should update a single-container pod's image", func(ctx context.Context) {
			httpdImage := imageutils.GetE2EImage(imageutils.Httpd)
			ginkgo.By("running the image " + httpdImage)
			e2ekubectl.RunKubectlOrDie(ns, "run", podName, "--image="+httpdImage, podRunningTimeoutArg, "--labels=run="+podName)

			ginkgo.By("verifying the pod " + podName + " is running")
			label := labels.SelectorFromSet(labels.Set(map[string]string{"run": podName}))
			err := testutils.WaitForPodsWithLabelRunning(c, ns, label)
			if err != nil {
				framework.Failf("Failed getting pod %s: %v", podName, err)
			}

			ginkgo.By("verifying the pod " + podName + " was created")
			podJSON := e2ekubectl.RunKubectlOrDie(ns, "get", "pod", podName, "-o", "json")
			if !strings.Contains(podJSON, podName) {
				framework.Failf("Failed to find pod %s in [%s]", podName, podJSON)
			}

			ginkgo.By("replace the image in the pod")
			busyboxImage := imageutils.GetE2EImage(imageutils.BusyBox)
			podJSON = strings.Replace(podJSON, httpdImage, busyboxImage, 1)
			e2ekubectl.RunKubectlOrDieInput(ns, podJSON, "replace", "-f", "-")

			ginkgo.By("verifying the pod " + podName + " has the right image " + busyboxImage)
			pod, err := c.CoreV1().Pods(ns).Get(ctx, podName, metav1.GetOptions{})
			if err != nil {
				framework.Failf("Failed getting deployment %s: %v", podName, err)
			}
			containers := pod.Spec.Containers
			if checkContainersImage(containers, busyboxImage) {
				framework.Failf("Failed creating pod with expected image %s", busyboxImage)
			}
		})
	})

	ginkgo.Describe("Proxy server", func() {
		// TODO: test proxy options (static, prefix, etc)
		/*
			Release: v1.9
			Testname: Kubectl, proxy port zero
			Description: Start a proxy server on port zero by running 'kubectl proxy' with --port=0. Call the proxy server by requesting api versions from unix socket. The proxy server MUST provide at least one version string.
		*/
		framework.ConformanceIt("should support proxy with --port 0", func(ctx context.Context) {
			ginkgo.By("starting the proxy server")
			port, cmd, err := startProxyServer(ns)
			if cmd != nil {
				defer framework.TryKill(cmd)
			}
			if err != nil {
				framework.Failf("Failed to start proxy server: %v", err)
			}
			ginkgo.By("curling proxy /api/ output")
			localAddr := fmt.Sprintf("http://localhost:%d/api/", port)
			apiVersions, err := getAPIVersions(localAddr)
			if err != nil {
				framework.Failf("Expected at least one supported apiversion, got error %v", err)
			}
			if len(apiVersions.Versions) < 1 {
				framework.Failf("Expected at least one supported apiversion, got %v", apiVersions)
			}
		})

		/*
			Release: v1.9
			Testname: Kubectl, proxy socket
			Description: Start a proxy server on by running 'kubectl proxy' with --unix-socket=<some path>. Call the proxy server by requesting api versions from  http://locahost:0/api. The proxy server MUST provide at least one version string
		*/
		framework.ConformanceIt("should support --unix-socket=/path", func(ctx context.Context) {
			ginkgo.By("Starting the proxy")
			tmpdir, err := os.MkdirTemp("", "kubectl-proxy-unix")
			if err != nil {
				framework.Failf("Failed to create temporary directory: %v", err)
			}
			path := filepath.Join(tmpdir, "test")
			defer os.Remove(path)
			defer os.Remove(tmpdir)
			tk := e2ekubectl.NewTestKubeconfig(framework.TestContext.CertDir, framework.TestContext.Host, framework.TestContext.KubeConfig, framework.TestContext.KubeContext, framework.TestContext.KubectlPath, ns)
			cmd := tk.KubectlCmd("proxy", fmt.Sprintf("--unix-socket=%s", path))
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
			ginkgo.By("retrieving proxy /api/ output")
			_, err = curlUnix("http://unused/api", path)
			if err != nil {
				framework.Failf("Failed get of /api at %s: %v", path, err)
			}
		})
	})

	// This test must run [Serial] because it modifies the node so it doesn't allow pods to execute on
	// it, which will affect anything else running in parallel.
	f.Describe("Kubectl taint", framework.WithSerial(), func() {
		ginkgo.It("should update the taint on a node", func(ctx context.Context) {
			testTaint := v1.Taint{
				Key:    fmt.Sprintf("kubernetes.io/e2e-taint-key-001-%s", string(uuid.NewUUID())),
				Value:  "testing-taint-value",
				Effect: v1.TaintEffectNoSchedule,
			}

			nodeName := scheduling.GetNodeThatCanRunPod(ctx, f)

			ginkgo.By("adding the taint " + testTaint.ToString() + " to a node")
			runKubectlRetryOrDie(ns, "taint", "nodes", nodeName, testTaint.ToString())
			ginkgo.DeferCleanup(e2enode.RemoveTaintOffNode, f.ClientSet, nodeName, testTaint)

			ginkgo.By("verifying the node has the taint " + testTaint.ToString())
			output := runKubectlRetryOrDie(ns, "describe", "node", nodeName)
			requiredStrings := [][]string{
				{"Name:", nodeName},
				{"Taints:"},
				{testTaint.ToString()},
			}
			checkOutput(output, requiredStrings)

			ginkgo.By("removing the taint " + testTaint.ToString() + " of a node")
			runKubectlRetryOrDie(ns, "taint", "nodes", nodeName, testTaint.Key+":"+string(testTaint.Effect)+"-")
			ginkgo.By("verifying the node doesn't have the taint " + testTaint.Key)
			output = runKubectlRetryOrDie(ns, "describe", "node", nodeName)
			if strings.Contains(output, testTaint.Key) {
				framework.Fail("Failed removing taint " + testTaint.Key + " of the node " + nodeName)
			}
		})

		ginkgo.It("should remove all the taints with the same key off a node", func(ctx context.Context) {
			testTaint := v1.Taint{
				Key:    fmt.Sprintf("kubernetes.io/e2e-taint-key-002-%s", string(uuid.NewUUID())),
				Value:  "testing-taint-value",
				Effect: v1.TaintEffectNoSchedule,
			}

			nodeName := scheduling.GetNodeThatCanRunPod(ctx, f)

			ginkgo.By("adding the taint " + testTaint.ToString() + " to a node")
			runKubectlRetryOrDie(ns, "taint", "nodes", nodeName, testTaint.ToString())
			ginkgo.DeferCleanup(e2enode.RemoveTaintOffNode, f.ClientSet, nodeName,
				testTaint)

			ginkgo.By("verifying the node has the taint " + testTaint.ToString())
			output := runKubectlRetryOrDie(ns, "describe", "node", nodeName)
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
			runKubectlRetryOrDie(ns, "taint", "nodes", nodeName, newTestTaint.ToString())
			ginkgo.DeferCleanup(e2enode.RemoveTaintOffNode, f.ClientSet, nodeName, newTestTaint)

			ginkgo.By("verifying the node has the taint " + newTestTaint.ToString())
			output = runKubectlRetryOrDie(ns, "describe", "node", nodeName)
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
			runKubectlRetryOrDie(ns, "taint", "nodes", nodeName, noExecuteTaint.ToString())
			ginkgo.DeferCleanup(e2enode.RemoveTaintOffNode, f.ClientSet, nodeName, noExecuteTaint)

			ginkgo.By("verifying the node has the taint " + noExecuteTaint.ToString())
			output = runKubectlRetryOrDie(ns, "describe", "node", nodeName)
			requiredStrings = [][]string{
				{"Name:", nodeName},
				{"Taints:"},
				{noExecuteTaint.ToString()},
			}
			checkOutput(output, requiredStrings)

			ginkgo.By("removing all taints that have the same key " + testTaint.Key + " of the node")
			runKubectlRetryOrDie(ns, "taint", "nodes", nodeName, testTaint.Key+"-")
			ginkgo.By("verifying the node doesn't have the taints that have the same key " + testTaint.Key)
			output = runKubectlRetryOrDie(ns, "describe", "node", nodeName)
			if strings.Contains(output, testTaint.Key) {
				framework.Fail("Failed removing taints " + testTaint.Key + " of the node " + nodeName)
			}
		})
	})

	ginkgo.Describe("Kubectl events", func() {
		ginkgo.It("should show event when pod is created", func(ctx context.Context) {
			podName := "e2e-test-httpd-pod"
			httpdImage := imageutils.GetE2EImage(imageutils.Httpd)
			ginkgo.By("running the image " + httpdImage)
			e2ekubectl.RunKubectlOrDie(ns, "run", podName, "--image="+httpdImage, podRunningTimeoutArg, "--labels=run="+podName)

			ginkgo.By("verifying the pod " + podName + " is running")
			label := labels.SelectorFromSet(map[string]string{"run": podName})
			err := testutils.WaitForPodsWithLabelRunning(c, ns, label)
			if err != nil {
				framework.Failf("Failed getting pod %s: %v", podName, err)
			}

			ginkgo.By("show started event for this pod")
			events := e2ekubectl.RunKubectlOrDie(ns, "events", "--for=pod/"+podName)

			// replace multi spaces into single white space
			eventsStr := strings.Join(strings.Fields(strings.TrimSpace(events)), " ")
			if !strings.Contains(string(eventsStr), fmt.Sprintf("Normal Scheduled Pod/%s", podName)) {
				framework.Failf("failed to list expected event with pod name: %s, got: %s", podName, events)
			}

			ginkgo.By("expect not showing any WARNING message except timeouts")
			events = e2ekubectl.RunKubectlOrDie(ns, "events", "--types=WARNING", "--for=pod/"+podName)
			if events != "" && !strings.Contains(events, "timed out") {
				framework.Failf("unexpected non-timeout WARNING event fired, got: %s ", events)
			}
		})
	})

	ginkgo.Describe("Kubectl create quota", func() {
		ginkgo.It("should create a quota without scopes", func(ctx context.Context) {
			quotaName := "million"

			ginkgo.By("calling kubectl quota")
			e2ekubectl.RunKubectlOrDie(ns, "create", "quota", quotaName, "--hard=pods=1000000,services=1000000")

			ginkgo.By("verifying that the quota was created")
			quota, err := c.CoreV1().ResourceQuotas(ns).Get(ctx, quotaName, metav1.GetOptions{})
			if err != nil {
				framework.Failf("Failed getting quota %s: %v", quotaName, err)
			}

			if len(quota.Spec.Scopes) != 0 {
				framework.Failf("Expected empty scopes, got %v", quota.Spec.Scopes)
			}
			if len(quota.Spec.Hard) != 2 {
				framework.Failf("Expected two resources, got %v", quota.Spec.Hard)
			}
			r, found := quota.Spec.Hard[v1.ResourcePods]
			if expected := resource.MustParse("1000000"); !found || (&r).Cmp(expected) != 0 {
				framework.Failf("Expected pods=1000000, got %v", r)
			}
			r, found = quota.Spec.Hard[v1.ResourceServices]
			if expected := resource.MustParse("1000000"); !found || (&r).Cmp(expected) != 0 {
				framework.Failf("Expected services=1000000, got %v", r)
			}
		})

		ginkgo.It("should create a quota with scopes", func(ctx context.Context) {
			quotaName := "scopes"

			ginkgo.By("calling kubectl quota")
			e2ekubectl.RunKubectlOrDie(ns, "create", "quota", quotaName, "--hard=pods=1000000", "--scopes=BestEffort,NotTerminating")

			ginkgo.By("verifying that the quota was created")
			quota, err := c.CoreV1().ResourceQuotas(ns).Get(ctx, quotaName, metav1.GetOptions{})
			if err != nil {
				framework.Failf("Failed getting quota %s: %v", quotaName, err)
			}

			if len(quota.Spec.Scopes) != 2 {
				framework.Failf("Expected two scopes, got %v", quota.Spec.Scopes)
			}
			scopes := make(map[v1.ResourceQuotaScope]struct{})
			for _, scope := range quota.Spec.Scopes {
				scopes[scope] = struct{}{}
			}
			if _, found := scopes[v1.ResourceQuotaScopeBestEffort]; !found {
				framework.Failf("Expected BestEffort scope, got %v", quota.Spec.Scopes)
			}
			if _, found := scopes[v1.ResourceQuotaScopeNotTerminating]; !found {
				framework.Failf("Expected NotTerminating scope, got %v", quota.Spec.Scopes)
			}
		})

		ginkgo.It("should reject quota with invalid scopes", func(ctx context.Context) {
			quotaName := "scopes"

			ginkgo.By("calling kubectl quota")
			out, err := e2ekubectl.RunKubectl(ns, "create", "quota", quotaName, "--hard=hard=pods=1000000", "--scopes=Foo")
			if err == nil {
				framework.Failf("Expected kubectl to fail, but it succeeded: %s", out)
			}
		})
	})

	ginkgo.Describe("kubectl wait", func() {
		ginkgo.It("should ignore not found error with --for=delete", func(ctx context.Context) {
			ginkgo.By("calling kubectl wait --for=delete")
			e2ekubectl.RunKubectlOrDie(ns, "wait", "--for=delete", "pod/doesnotexist")
			e2ekubectl.RunKubectlOrDie(ns, "wait", "--for=delete", "pod", "--selector=app.kubernetes.io/name=noexist")
		})
	})

	ginkgo.Describe("kubectl subresource flag", func() {
		ginkgo.It("should not be used in a bulk GET", func() {
			ginkgo.By("calling kubectl get nodes --subresource=status")
			out, err := e2ekubectl.RunKubectl("", "get", "nodes", "--subresource=status")
			gomega.Expect(err).To(gomega.HaveOccurred(), fmt.Sprintf("Expected kubectl to fail, but it succeeded: %s", out))
			gomega.Expect(err).To(gomega.ContainSubstring("subresource cannot be used when bulk resources are specified"))
		})
		ginkgo.It("GET on status subresource of built-in type (node) returns identical info as GET on the built-in type", func(ctx context.Context) {
			ginkgo.By("first listing nodes in the cluster, and using first node of the list")
			nodes, err := c.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
			framework.ExpectNoError(err)
			gomega.Expect(nodes.Items).ToNot(gomega.BeEmpty())
			node := nodes.Items[0]
			// Avoid comparing values of fields that might end up
			// changing between the two invocations of kubectl. We
			// compare the name and version fields.
			ginkgo.By(fmt.Sprintf("calling kubectl get nodes %s", node.Name))
			outBuiltIn := e2ekubectl.RunKubectlOrDie("", "get", "nodes", node.Name,
				"--output=jsonpath='{.metadata.name}{.status.nodeInfo.kubeletVersion}'",
			)
			ginkgo.By(fmt.Sprintf("calling kubectl get nodes %s --subresource=status", node.Name))
			outStatusSubresource := e2ekubectl.RunKubectlOrDie("", "get", "nodes", node.Name,
				"--output=jsonpath='{.metadata.name}{.status.nodeInfo.kubeletVersion}'",
				"--subresource=status",
			)
			gomega.Expect(outBuiltIn).To(gomega.Equal(outStatusSubresource))
		})
	})
})

func getTestContextHost() string {
	if len(framework.TestContext.Host) > 0 {
		return framework.TestContext.Host
	}
	// if there is a kubeconfig, pick the first server from it
	if framework.TestContext.KubeConfig != "" {
		c, err := clientcmd.LoadFromFile(framework.TestContext.KubeConfig)
		if err == nil {
			for _, v := range c.Clusters {
				if v.Server != "" {
					framework.Logf("--host variable was not set, picking up the first server from %s",
						framework.TestContext.KubeConfig)
					return v.Server
				}
			}
		}
	}
	framework.Failf("--host variable must be set to the full URI to the api server on e2e run.")
	return ""
}

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
		framework.Failf("%v", err)
	}
}

func checkKubectlOutputWithRetry(namespace string, required [][]string, args ...string) {
	var pollErr error
	wait.PollImmediate(time.Second, time.Minute, func() (bool, error) {
		output := e2ekubectl.RunKubectlOrDie(namespace, args...)
		err := checkOutputReturnError(output, required)
		if err != nil {
			pollErr = err
			return false, nil
		}
		pollErr = nil
		return true, nil
	})
	if pollErr != nil {
		framework.Failf("%v", pollErr)
	}
	return
}

func checkContainersImage(containers []v1.Container, expectImage string) bool {
	return containers == nil || len(containers) != 1 || containers[0].Image != expectImage
}

func getAPIVersions(apiEndpoint string) (*metav1.APIVersions, error) {
	body, err := curl(apiEndpoint)
	if err != nil {
		return nil, fmt.Errorf("Failed http.Get of %s: %w", apiEndpoint, err)
	}
	var apiVersions metav1.APIVersions
	if err := json.Unmarshal([]byte(body), &apiVersions); err != nil {
		return nil, fmt.Errorf("Failed to parse /api output %s: %w", body, err)
	}
	return &apiVersions, nil
}

func startProxyServer(ns string) (int, *exec.Cmd, error) {
	// Specifying port 0 indicates we want the os to pick a random port.
	tk := e2ekubectl.NewTestKubeconfig(framework.TestContext.CertDir, framework.TestContext.Host, framework.TestContext.KubeConfig, framework.TestContext.KubeContext, framework.TestContext.KubectlPath, ns)
	cmd := tk.KubectlCmd("proxy", "-p", "0", "--disable-filter")
	stdout, stderr, err := framework.StartCmdAndStreamOutput(cmd)
	if err != nil {
		return -1, nil, err
	}
	buf := make([]byte, 128)
	var n int
	if n, err = stdout.Read(buf); err != nil {
		return -1, cmd, fmt.Errorf("Failed to read from kubectl proxy stdout: %w", err)
	}
	go func() {
		out, _ := io.ReadAll(stdout)
		framework.Logf("kubectl proxy stdout: %s", string(buf[:n])+string(out))
		stdout.Close()
	}()
	go func() {
		err, _ := io.ReadAll(stderr)
		framework.Logf("kubectl proxy stderr: %s", string(err))
		stderr.Close()
	}()
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
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	return string(body[:]), nil
}

func curl(url string) (string, error) {
	return curlTransport(url, utilnet.SetTransportDefaults(&http.Transport{}))
}

func validateGuestbookApp(ctx context.Context, c clientset.Interface, ns string) {
	framework.Logf("Waiting for all frontend pods to be Running.")
	label := labels.SelectorFromSet(labels.Set(map[string]string{"tier": "frontend", "app": "guestbook"}))
	err := testutils.WaitForPodsWithLabelRunning(c, ns, label)
	framework.ExpectNoError(err)
	framework.Logf("Waiting for frontend to serve content.")
	if !waitForGuestbookResponse(ctx, c, "get", "", `{"data":""}`, guestbookStartupTimeout, ns) {
		framework.Failf("Frontend service did not start serving content in %v seconds.", guestbookStartupTimeout.Seconds())
	}

	framework.Logf("Trying to add a new entry to the guestbook.")
	if !waitForGuestbookResponse(ctx, c, "set", "TestEntry", `{"message":"Updated"}`, guestbookResponseTimeout, ns) {
		framework.Failf("Cannot added new entry in %v seconds.", guestbookResponseTimeout.Seconds())
	}

	framework.Logf("Verifying that added entry can be retrieved.")
	if !waitForGuestbookResponse(ctx, c, "get", "", `{"data":"TestEntry"}`, guestbookResponseTimeout, ns) {
		framework.Failf("Entry to guestbook wasn't correctly added in %v seconds.", guestbookResponseTimeout.Seconds())
	}
}

// Returns whether received expected response from guestbook on time.
func waitForGuestbookResponse(ctx context.Context, c clientset.Interface, cmd, arg, expectedResponse string, timeout time.Duration, ns string) bool {
	for start := time.Now(); time.Since(start) < timeout && ctx.Err() == nil; time.Sleep(5 * time.Second) {
		res, err := makeRequestToGuestbook(ctx, c, cmd, arg, ns)
		if err == nil && res == expectedResponse {
			return true
		}
		framework.Logf("Failed to get response from guestbook. err: %v, response: %s", err, res)
	}
	return false
}

func makeRequestToGuestbook(ctx context.Context, c clientset.Interface, cmd, value string, ns string) (string, error) {
	proxyRequest, errProxy := e2eservice.GetServicesProxyRequest(c, c.CoreV1().RESTClient().Get())
	if errProxy != nil {
		return "", errProxy
	}

	ctx, cancel := context.WithTimeout(ctx, framework.SingleCallTimeout)
	defer cancel()

	result, err := proxyRequest.Namespace(ns).
		Name("frontend").
		Suffix("/guestbook").
		Param("cmd", cmd).
		Param("key", "messages").
		Param("value", value).
		Do(ctx).
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
		framework.Fail(err.Error())
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

func forEachReplicationController(ctx context.Context, c clientset.Interface, ns, selectorKey, selectorValue string, fn func(v1.ReplicationController)) {
	var rcs *v1.ReplicationControllerList
	var err error
	for t := time.Now(); time.Since(t) < framework.PodListTimeout && ctx.Err() == nil; time.Sleep(framework.Poll) {
		label := labels.SelectorFromSet(labels.Set(map[string]string{selectorKey: selectorValue}))
		options := metav1.ListOptions{LabelSelector: label.String()}
		rcs, err = c.CoreV1().ReplicationControllers(ns).List(ctx, options)
		framework.ExpectNoError(err)
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

func validateReplicationControllerConfiguration(rc v1.ReplicationController) {
	if rc.Name == "agnhost-primary" {
		if _, ok := rc.Annotations[v1.LastAppliedConfigAnnotation]; !ok {
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
func getUDData(jpgExpected string, ns string) func(context.Context, clientset.Interface, string) error {

	// getUDData validates data.json in the update-demo (returns nil if data is ok).
	return func(ctx context.Context, c clientset.Interface, podID string) error {
		framework.Logf("validating pod %s", podID)

		ctx, cancel := context.WithTimeout(ctx, framework.SingleCallTimeout)
		defer cancel()

		body, err := c.CoreV1().RESTClient().Get().
			Namespace(ns).
			Resource("pods").
			SubResource("proxy").
			Name(podID).
			Suffix("data.json").
			Do(ctx).
			Raw()

		if err != nil {
			if ctx.Err() != nil {
				framework.Failf("Failed to retrieve data from container: %v", err)
			}
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
		}
		return fmt.Errorf("data served up in container is inaccurate, %s didn't contain %s", data, jpgExpected)
	}
}

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

// createApplyCustomResource asserts that given CustomResource be created and applied
// without being rejected by kubectl validation
func createApplyCustomResource(resource, namespace, name string, crd *crd.TestCrd) error {
	ginkgo.By("successfully create CR")
	if _, err := e2ekubectl.RunKubectlInput(namespace, resource, "create", "--validate=true", "-f", "-"); err != nil {
		return fmt.Errorf("failed to create CR %s in namespace %s: %w", resource, namespace, err)
	}
	if _, err := e2ekubectl.RunKubectl(namespace, "delete", crd.Crd.Spec.Names.Plural, name); err != nil {
		return fmt.Errorf("failed to delete CR %s: %w", name, err)
	}
	ginkgo.By("successfully apply CR")
	if _, err := e2ekubectl.RunKubectlInput(namespace, resource, "apply", "--validate=true", "-f", "-"); err != nil {
		return fmt.Errorf("failed to apply CR %s in namespace %s: %w", resource, namespace, err)
	}
	if _, err := e2ekubectl.RunKubectl(namespace, "delete", crd.Crd.Spec.Names.Plural, name); err != nil {
		return fmt.Errorf("failed to delete CR %s: %w", name, err)
	}
	return nil
}

// trimDockerRegistry is the function for trimming the docker.io/library from the beginning of the imagename.
// If community docker installed it will not prefix the registry names with the dockerimages vs registry names prefixed with other runtimes or docker installed via RHEL extra repo.
// So this function will help to trim the docker.io/library if exists
func trimDockerRegistry(imagename string) string {
	imagename = strings.Replace(imagename, "docker.io/", "", 1)
	return strings.Replace(imagename, "library/", "", 1)
}

// validatorFn is the function which is individual tests will implement.
// we may want it to return more than just an error, at some point.
type validatorFn func(ctx context.Context, c clientset.Interface, podID string) error

// validateController is a generic mechanism for testing RC's that are running.
// It takes a container name, a test name, and a validator function which is plugged in by a specific test.
// "containername": this is grepped for.
// "containerImage" : this is the name of the image we expect to be launched.  Not to confuse w/ images (kitten.jpg)  which are validated.
// "testname":  which gets bubbled up to the logging/failure messages if errors happen.
// "validator" function: This function is given a podID and a client, and it can do some specific validations that way.
func validateController(ctx context.Context, c clientset.Interface, containerImage string, replicas int, containername string, testname string, validator validatorFn, ns string) {
	containerImage = trimDockerRegistry(containerImage)
	getPodsTemplate := "--template={{range.items}}{{.metadata.name}} {{end}}"

	getContainerStateTemplate := fmt.Sprintf(`--template={{if (exists . "status" "containerStatuses")}}{{range .status.containerStatuses}}{{if (and (eq .name "%s") (exists . "state" "running"))}}true{{end}}{{end}}{{end}}`, containername)

	getImageTemplate := fmt.Sprintf(`--template={{if (exists . "spec" "containers")}}{{range .spec.containers}}{{if eq .name "%s"}}{{.image}}{{end}}{{end}}{{end}}`, containername)

	ginkgo.By(fmt.Sprintf("waiting for all containers in %s pods to come up.", testname)) //testname should be selector
waitLoop:
	for start := time.Now(); time.Since(start) < framework.PodStartTimeout && ctx.Err() == nil; time.Sleep(5 * time.Second) {
		getPodsOutput := e2ekubectl.RunKubectlOrDie(ns, "get", "pods", "-o", "template", getPodsTemplate, "-l", testname)
		pods := strings.Fields(getPodsOutput)
		if numPods := len(pods); numPods != replicas {
			ginkgo.By(fmt.Sprintf("Replicas for %s: expected=%d actual=%d", testname, replicas, numPods))
			continue
		}
		var runningPods []string
		for _, podID := range pods {
			running := e2ekubectl.RunKubectlOrDie(ns, "get", "pods", podID, "-o", "template", getContainerStateTemplate)
			if running != "true" {
				framework.Logf("%s is created but not running", podID)
				continue waitLoop
			}

			currentImage := e2ekubectl.RunKubectlOrDie(ns, "get", "pods", podID, "-o", "template", getImageTemplate)
			currentImage = trimDockerRegistry(currentImage)
			if currentImage != containerImage {
				framework.Logf("%s is created but running wrong image; expected: %s, actual: %s", podID, containerImage, currentImage)
				continue waitLoop
			}

			// Call the generic validator function here.
			// This might validate for example, that (1) getting a url works and (2) url is serving correct content.
			if err := validator(ctx, c, podID); err != nil {
				framework.Logf("%s is running right image but validator function failed: %v", podID, err)
				continue waitLoop
			}

			framework.Logf("%s is verified up and running", podID)
			runningPods = append(runningPods, podID)
		}
		// If we reach here, then all our checks passed.
		if len(runningPods) == replicas {
			return
		}
	}
	// Reaching here means that one of more checks failed multiple times.  Assuming its not a race condition, something is broken.
	framework.Failf("Timed out after %v seconds waiting for %s pods to reach valid state", framework.PodStartTimeout.Seconds(), testname)
}

// mustListObjectsInNamespace queries all the objects we use for testing in the given namespace.
// Currently this is just ConfigMaps.
// We filter our "system" configmaps, like "kube-root-ca.crt".
func mustListObjectsInNamespace(ctx context.Context, c clientset.Interface, ns string) []runtime.Object {
	var objects []runtime.Object
	configMaps, err := c.CoreV1().ConfigMaps(ns).List(ctx, metav1.ListOptions{})
	if err != nil {
		framework.Failf("error listing configmaps: %v", err)
	}
	for i := range configMaps.Items {
		cm := &configMaps.Items[i]
		if cm.Name == "kube-root-ca.crt" {
			// Ignore system objects
			continue
		}
		objects = append(objects, cm)
	}
	return objects
}

// mustGetNames returns a slice containing the metadata.name for each object.
func mustGetNames(objects []runtime.Object) []string {
	var names []string
	for _, obj := range objects {
		metaAccessor, err := meta.Accessor(obj)
		if err != nil {
			framework.Failf("error getting accessor for %T: %v", obj, err)
		}
		name := metaAccessor.GetName()
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

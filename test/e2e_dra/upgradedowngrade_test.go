/*
Copyright 2022 The Kubernetes Authors.

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

package e2edra

import (
	"archive/tar"
	"compress/gzip"
	"context"
	_ "embed"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/version"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/errors"
	drautils "k8s.io/kubernetes/test/e2e/dra/utils"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2etestfiles "k8s.io/kubernetes/test/e2e/framework/testfiles"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/kubernetes/test/utils/localupcluster"
	admissionapi "k8s.io/pod-security-admission/api"
)

func init() {
	// -v=5 may be useful to debug driver operations, but usually isn't needed.
	ktesting.SetDefaultVerbosity(2)
}

var repoRoot = repoRootDefault()

func currentBinDir() (envName, content string) {
	envName = "KUBERNETES_SERVER_BIN_DIR"
	content, _ = os.LookupEnv(envName)
	return
}

// repoRootDefault figures out whether an E2E suite is invoked in its directory (as in `go test ./test/e2e`),
// directly in the root (as in `make test-e2e` or `ginkgo ./test/e2e`), or somewhere deep inside
// the _output directory (`ginkgo _output/bin/e2e.test` where `_output/bin` is actually a symlink).
func repoRootDefault() string {
	for i := 0; i < 10; i++ {
		path := "." + strings.Repeat("/..", i)
		if _, err := os.Stat(path + "/test/e2e/framework"); err == nil {
			return path
		}
	}
	// Traditional default.
	return "../../"
}

func TestUpgradeDowngrade(t *testing.T) {
	suiteConfig, reporterConfig := framework.CreateGinkgoConfig()
	ginkgo.RunSpecs(t, "DRA", suiteConfig, reporterConfig)
}

var _ = ginkgo.Describe("DRA upgrade/downgrade", func() {
	// Initialize the default values by registering flags. We don't actually expose those flags.
	var fs flag.FlagSet
	framework.RegisterCommonFlags(&fs)
	framework.RegisterClusterFlags(&fs)

	// Some other things normally done by test/e2e.
	e2etestfiles.AddFileSource(e2etestfiles.RootFileSource{Root: repoRoot})
	gomega.RegisterFailHandler(ginkgo.Fail)

	ginkgo.It("works", func(ctx context.Context) {
		// TODO: replace with helper code from https://github.com/kubernetes/kubernetes/pull/122481 should that get merged.
		tCtx := ktesting.Init(GinkgoContextTB())
		tCtx = ktesting.WithContext(tCtx, ctx)

		envName, dir := currentBinDir()
		if dir == "" {
			tCtx.Fatalf("%s must be set to test DRA upgrade/downgrade scenarios.", envName)
		}

		// Determine what we need to downgrade to.
		tCtx = ktesting.Begin(tCtx, "get source code version")
		gitVersion, _, err := sourceVersion(tCtx, repoRoot)
		tCtx.ExpectNoError(err, "determine source code version for repo root %q", repoRoot)
		version, err := version.ParseGeneric(gitVersion)
		tCtx.ExpectNoError(err, "parse version %s of repo root %q", gitVersion, repoRoot)
		major, previousMinor := version.Major(), version.Minor()-1
		tCtx = ktesting.End(tCtx)

		// KUBERNETES_SERVER_CACHE_DIR can be set to keep downloaded files across test restarts.
		binDir, cacheBinaries := os.LookupEnv("KUBERNETES_SERVER_CACHE_DIR")
		if !cacheBinaries {
			binDir = tCtx.TempDir()
		}
		haveBinaries := false

		// Get the previous release, if necessary.
		previousURL, previousVersion := serverDownloadURL(tCtx, major, previousMinor)
		if cacheBinaries {
			binDir = path.Join(binDir, previousVersion)
			_, err := os.Stat(path.Join(binDir, string(localupcluster.KubeClusterComponents[0])))
			if err == nil {
				haveBinaries = true
			}
		}
		if !haveBinaries {
			tCtx = ktesting.Begin(tCtx, fmt.Sprintf("download and unpack %s", previousURL))
			req, err := http.NewRequestWithContext(tCtx, http.MethodGet, previousURL, nil)
			tCtx.ExpectNoError(err, "construct request")
			response, err := http.DefaultClient.Do(req)
			tCtx.ExpectNoError(err, "download")
			defer func() {
				_ = response.Body.Close()
			}()
			decompress, err := gzip.NewReader(response.Body)
			tCtx.ExpectNoError(err, "construct gzip reader")
			unpack := tar.NewReader(decompress)
			for {
				header, err := unpack.Next()
				if err == io.EOF {
					break
				}
				base := path.Base(header.Name)
				if slices.Contains(localupcluster.KubeClusterComponents, localupcluster.KubeComponentName(base)) {
					data, err := io.ReadAll(unpack)
					tCtx.ExpectNoError(err, fmt.Sprintf("read content of %s", header.Name))
					tCtx.ExpectNoError(os.MkdirAll(binDir, 0755), "create directory for binaries")
					tCtx.ExpectNoError(os.WriteFile(path.Join(binDir, base), data, 0555), fmt.Sprintf("write content of %s", header.Name))
				}
			}
			tCtx = ktesting.End(tCtx)
		}

		tCtx = ktesting.Begin(tCtx, fmt.Sprintf("bring up v%d.%d", major, previousMinor))
		cluster := localupcluster.New(tCtx)
		localUpClusterEnv := map[string]string{
			"RUNTIME_CONFIG": "resource.k8s.io/v1beta1,resource.k8s.io/v1beta2",
			"FEATURE_GATES":  "DynamicResourceAllocation=true",
			// *not* needed because driver will run in "local filesystem" mode (= driver.IsLocal): "ALLOW_PRIVILEGED": "1",
		}
		cluster.Start(tCtx, binDir, localUpClusterEnv)
		tCtx = ktesting.End(tCtx)

		restConfig := cluster.LoadConfig(tCtx)
		restConfig.UserAgent = fmt.Sprintf("%s -- dra", restclient.DefaultKubernetesUserAgent())
		tCtx = ktesting.WithRESTConfig(tCtx, restConfig)
		// TODO: rewrite all DRA test code to use ktesting.TContext once https://github.com/kubernetes/kubernetes/pull/122481 is
		// merged, then we don't need to fake a Framework instance.
		f := &framework.Framework{
			BaseName:      "dra",
			Timeouts:      framework.NewTimeoutContext(),
			ClientSet:     tCtx.Client(),
			DynamicClient: tCtx.Dynamic(),

			// The driver containers have to run with sufficient privileges to
			// modify /var/lib/kubelet/plugins.
			NamespacePodSecurityLevel: admissionapi.LevelPrivileged,
		}
		f.SetClientConfig(restConfig)

		namespace, err := f.CreateNamespace(tCtx, f.BaseName, map[string]string{
			"e2e-framework": f.BaseName,
		})
		tCtx.ExpectNoError(err, "create namespace")
		f.Namespace = namespace
		f.UniqueName = namespace.Name

		tCtx = ktesting.Begin(tCtx, fmt.Sprintf("v%d.%d", major, previousMinor))

		tCtx.ExpectNoError(e2enode.WaitForAllNodesSchedulable(tCtx, tCtx.Client(), f.Timeouts.NodeSchedulable), "wait for all nodes to be schedulable")
		nodes := drautils.NewNodesNow(tCtx, f, 1, 1)

		// Opening sockets locally avoids intermittent errors and delays caused by proxying through the restarted apiserver.
		// We could speed up testing by shortening the sync delay in the ResourceSlice controller, but let's better
		// test the defaults.
		driver := drautils.NewDriverInstance(f)
		driver.IsLocal = true
		driver.Run(nodes, drautils.DriverResourcesNow(nodes, 1))
		b := drautils.NewBuilderNow(ctx, f, driver)

		claim := b.ExternalClaim()
		pod := b.PodExternal()
		b.Create(ctx, claim, pod)
		b.TestPod(ctx, f, pod)

		tCtx = ktesting.End(tCtx)

		tCtx = ktesting.Begin(tCtx, fmt.Sprintf("update to %s", gitVersion))
		// We could split this up into first updating the apiserver, then control plane components, then restarting kubelet.
		// For the purpose of this test here we we primarily care about full before/after comparisons, so not done yet.
		// TODO
		restoreOptions := cluster.Modify(tCtx, localupcluster.ModifyOptions{Upgrade: true, BinDir: dir})
		tCtx = ktesting.End(tCtx)

		// The kubelet wipes all ResourceSlices on a restart because it doesn't know which drivers were running.
		// Wait for the ResourceSlice controller in the driver to notice and recreate the ResourceSlices.
		tCtx = ktesting.Begin(tCtx, "wait for ResourceSlices")
		gomega.Eventually(ctx, driver.NewGetSlices()).WithTimeout(5 * time.Minute).Should(gomega.HaveField("Items", gomega.HaveLen(len(nodes.NodeNames))))
		tCtx = ktesting.End(tCtx)

		// Remove pod prepared by previous Kubernetes.
		framework.ExpectNoError(f.ClientSet.ResourceV1beta1().ResourceClaims(namespace.Name).Delete(ctx, claim.Name, metav1.DeleteOptions{}))
		framework.ExpectNoError(f.ClientSet.CoreV1().Pods(namespace.Name).Delete(ctx, pod.Name, metav1.DeleteOptions{}))
		framework.ExpectNoError(e2epod.WaitForPodNotFoundInNamespace(ctx, f.ClientSet, pod.Name, namespace.Name, f.Timeouts.PodDelete))

		// Create another claim and pod, this time using the latest Kubernetes.
		claim = b.ExternalClaim()
		pod = b.PodExternal()
		pod.Spec.ResourceClaims[0].ResourceClaimName = &claim.Name
		b.Create(ctx, claim, pod)
		b.TestPod(ctx, f, pod)

		// Roll back.
		tCtx = ktesting.Begin(tCtx, "downgrade")
		cluster.Modify(tCtx, restoreOptions)
		tCtx = ktesting.End(tCtx)

		// TODO: ensure that kube-controller-manager is up-and-running.
		// This works around https://github.com/kubernetes/kubernetes/issues/132334 and can be removed
		// once a fix for that is backported.
		tCtx = ktesting.Begin(tCtx, "wait for kube-controller-manager")
		ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) string {
			output, _ := cluster.GetSystemLogs(tCtx, localupcluster.KubeControllerManager)
			return output
		}).Should(gomega.ContainSubstring(`"Caches are synced" controller="resource_claim"`))
		tCtx = ktesting.End(tCtx)

		// We need to clean up explicitly because the normal
		// cleanup doesn't work (driver shuts down first).
		//
		// The retry loops are necessary because of a stale connection
		// to the restarted apiserver. Sometimes, attempts fail with "EOF" as error
		// or (even weirder) with
		//     getting *v1.Pod: pods "tester-2" is forbidden: User "kubernetes-admin" cannot get resource "pods" in API group "" in the namespace "dra-9021"
		ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) error {
			return f.ClientSet.ResourceV1beta1().ResourceClaims(namespace.Name).Delete(tCtx, claim.Name, metav1.DeleteOptions{})
		}).Should(gomega.Succeed(), "delete claim after downgrade")
		ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) error {
			return f.ClientSet.CoreV1().Pods(namespace.Name).Delete(tCtx, pod.Name, metav1.DeleteOptions{})
		}).Should(gomega.Succeed(), "delete pod after downgrade")
		ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) *v1.Pod {
			pod, err := f.ClientSet.CoreV1().Pods(namespace.Name).Get(tCtx, pod.Name, metav1.GetOptions{})
			if apierrors.IsNotFound(err) {
				return nil
			}
			tCtx.ExpectNoError(err, "get pod")
			return pod
		}).Should(gomega.BeNil(), "no pod after deletion after downgrade")
	})
})

// sourceVersion identifies the Kubernetes git version based on hack/print-workspace-status.sh.
//
// Adapted from https://github.com/kubernetes-sigs/kind/blob/3df64e784cc0ea74125b2a2e9877817418afa3af/pkg/build/nodeimage/internal/kube/source.go#L71-L104
func sourceVersion(tCtx ktesting.TContext, kubeRoot string) (gitVersion string, dockerTag string, err error) {
	// Get the version output.
	cmd := exec.CommandContext(tCtx, "hack/print-workspace-status.sh")
	cmd.Dir = kubeRoot
	output, err := cmd.Output()
	if err != nil {
		return "", "", err
	}

	// Parse it.
	for _, line := range strings.Split(string(output), "\n") {
		parts := strings.SplitN(line, " ", 2)
		if len(parts) != 2 {
			continue
		}
		switch parts[0] {
		case "gitVersion":
			gitVersion = parts[1]
		case "STABLE_DOCKER_TAG":
			dockerTag = parts[1]
		}
	}
	if gitVersion == "" {
		return "", "", errors.Errorf("could not obtain Kubernetes git version: %q", string(output))

	}
	if dockerTag == "" {
		return "", "", errors.Errorf("count not obtain docker tag: %q", string(output))
	}
	return
}

// func runCmdIn(tCtx ktesting.TContext, dir string, name string, args ...string) string {
// 	tCtx.Helper()
// 	tCtx.Logf("Running command: %s %s", name, strings.Join(args, " "))
// 	cmd := exec.CommandContext(tCtx, name, args...)
// 	cmd.Dir = dir
// 	var output strings.Builder
// 	reader, writer := io.Pipe()
// 	cmd.Stdout = writer
// 	cmd.Stderr = writer
// 	tCtx.ExpectNoError(cmd.Start(), "start %s command", name)
// 	scanner := bufio.NewScanner(reader)
// 	var wg sync.WaitGroup
// 	wg.Add(1)
// 	go func() {
// 		defer wg.Done()
// 		for scanner.Scan() {
// 			line := scanner.Text()
// 			line = strings.TrimSuffix(line, "\n")
// 			tCtx.Logf("%s: %s", name, line)
// 			output.WriteString(line)
// 			output.WriteByte('\n')
// 		}
// 	}()
// 	result := cmd.Wait()
// 	tCtx.ExpectNoError(writer.Close(), "close in-memory pipe")
// 	wg.Wait()
// 	tCtx.ExpectNoError(result, fmt.Sprintf("%s command failed, output:\n%s", name, output.String()))
// 	tCtx.ExpectNoError(scanner.Err(), "read %s command output", name)

// 	return output.String()
// }

// serverDownloadURL returns the full URL for a kubernetes-server archive matching
// the current GOOS/GOARCH for the given major/minor version of Kubernetes.
//
// This considers only proper releases.
func serverDownloadURL(tCtx ktesting.TContext, major, minor uint) (string, string) {
	tCtx.Helper()
	url := fmt.Sprintf("https://dl.k8s.io/release/stable-%d.%d.txt", major, minor)
	get, err := http.NewRequestWithContext(tCtx, http.MethodGet, url, nil)
	tCtx.ExpectNoError(err, "construct GET for %s", url)
	resp, err := http.DefaultClient.Do(get)
	tCtx.ExpectNoError(err, "get %s", url)
	if resp.StatusCode != http.StatusOK {
		tCtx.Fatalf("get %s: %d - %s", url, resp.StatusCode, resp.Status)
	}
	if resp.Body == nil {
		tCtx.Fatalf("empty response for %s", url)
	}
	defer func() {
		tCtx.ExpectNoError(resp.Body.Close(), "close response body")
	}()
	version, err := io.ReadAll(resp.Body)
	tCtx.ExpectNoError(err, "read response body for %s", url)
	return fmt.Sprintf("https://dl.k8s.io/release/%s/kubernetes-server-%s-%s.tar.gz", string(version), runtime.GOOS, runtime.GOARCH), string(version)
}

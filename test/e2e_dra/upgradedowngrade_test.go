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
	_ "embed"
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

	"github.com/onsi/gomega"

	"k8s.io/apimachinery/pkg/util/version"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/errors"
	drautils "k8s.io/kubernetes/test/e2e/dra/utils"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2etestfiles "k8s.io/kubernetes/test/e2e/framework/testfiles"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/kubernetes/test/utils/localupcluster"
)

var errHTTP404 = errors.New("resource not found (404)")

func init() {
	// -v=5 may be useful to debug driver operations, but usually isn't needed.
	ktesting.SetDefaultVerbosity(2)
}

// The overall flow of upgrade/downgrade testing is always the same:
//
//   - Bring up a cluster with the previous release.
//   - "Install" the test DRA driver with 8 devices for the one node in the cluster.
//     There is a DeviceClass for it.
//   - Step 1: run some test code.
//   - Upgrade the cluster to the current code.
//   - Step 2: run some more test code.
//   - Downgrade to the previous release again.
//   - Step 3: run some final test code.
//
// The "test code" gets registered here with a single function for each
// sub-test. That function then returns the next piece of code, which then
// returns the final code. Each callback function is executed as a sub-test.
// The builder is configured to not delete objects when that sub-test ends,
// so objects persist until the entire test is done.
//
// Each sub-test must be self-contained. They intentionally run in a random
// order. However, they share the same cluster and the 8 devices which are
// available there.
var subTests = map[string]initialTestFunc{
	"core DRA":                    coreDRA,
	"ResourceClaim device status": resourceClaimDeviceStatus,
	"DRA extended resources":      extendedResourcesUpgradeDowngrade,
}

type initialTestFunc func(tCtx ktesting.TContext, builder *drautils.Builder) upgradedTestFunc

type upgradedTestFunc func(tCtx ktesting.TContext) downgradedTestFunc

type downgradedTestFunc func(tCtx ktesting.TContext)

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

func TestUpgradeDowngrade(t *testing.T) { testUpgradeDowngrade(ktesting.Init(t)) }
func testUpgradeDowngrade(tCtx ktesting.TContext) {
	// Some other things normally done by test/e2e.
	e2etestfiles.AddFileSource(e2etestfiles.RootFileSource{Root: repoRoot})

	// Ideally we shouldn't have any code which directly calls gomega.Expect,
	// but we are not there yet (e.g. e2epod.MakePod). So for now we install
	// one fail handler which records failures in the main test context.
	gomega.RegisterFailHandler(func(message string, callerSkip ...int) {
		tCtx.Helper()
		tCtx.Fatal(message)
	})

	envName, dir := currentBinDir()
	if dir == "" {
		tCtx.Fatalf("%s must be set to test DRA upgrade/downgrade scenarios.", envName)
	}

	// Determine what we need to downgrade to.
	var major, previousMinor uint
	var gitVersion string

	tCtx.Step("get source code version", func(tCtx ktesting.TContext) {
		var err error
		gitVersion, _, err = sourceVersion(tCtx, repoRoot)
		tCtx.ExpectNoError(err, "determine source code version for repo root %q", repoRoot)
		version, err := version.ParseGeneric(gitVersion)
		tCtx.ExpectNoError(err, "parse version %s of repo root %q", gitVersion, repoRoot)
		major, previousMinor = version.Major(), version.Minor()-1
		if strings.Contains(gitVersion, "-alpha.0") {
			// All version up to and including x.y.z-alpha.0 are treated as if we were
			// still the previous minor version x.(y-1). There are two reason for this:
			//
			// - During code freeze around (at?) -rc.0, the master branch already
			//   identfies itself as the next release with -alpha.0. Without this
			//   special case, we would change the version skew testing from what
			//   has been tested and been known to work to something else, which
			//   can and at least once did break.
			//
			// - Early in the next cycle the differences compared to the previous
			//   release are small, so it's more interesting to go back further.
			previousMinor--
		}
		tCtx.Logf("got version: major: %d, minor: %d, previous minor: %d", major, version.Minor(), previousMinor)
	})

	// KUBERNETES_SERVER_CACHE_DIR can be set to keep downloaded files across test restarts.
	binDir, cacheBinaries := os.LookupEnv("KUBERNETES_SERVER_CACHE_DIR")
	if !cacheBinaries {
		binDir = tCtx.TempDir()
	}
	haveBinaries := false

	// Get the previous release.
	var previousURL, previousVersion string
	tCtx.Step("get previous release info", func(tCtx ktesting.TContext) {
		tCtx.Logf("stable release %d.%d", major, previousMinor)
		var err error
		previousURL, previousVersion, err = serverDownloadURL(tCtx, "stable", major, previousMinor)
		if errors.Is(err, errHTTP404) {
			tCtx.Logf("stable doesn't exist, get latest release %d.%d", major, previousMinor)
			previousURL, previousVersion, err = serverDownloadURL(tCtx, "latest", major, previousMinor)
		}
		tCtx.ExpectNoError(err)
		tCtx.Logf("got previous release version: %s, URL: %s", previousVersion, previousURL)
	})

	if cacheBinaries {
		binDir = path.Join(binDir, previousVersion)
		_, err := os.Stat(path.Join(binDir, string(localupcluster.KubeClusterComponents[0])))
		if err == nil {
			haveBinaries = true
		}
	}
	if !haveBinaries {
		tCtx.Step(fmt.Sprintf("download and unpack %s", previousURL), func(tCtx ktesting.TContext) {
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
		})
	}

	var cluster *localupcluster.Cluster
	tCtx.Step(fmt.Sprintf("bring up v%d.%d", major, previousMinor), func(tCtx ktesting.TContext) {
		cluster = localupcluster.New(tCtx)
		localUpClusterEnv := map[string]string{
			"RUNTIME_CONFIG": "resource.k8s.io/v1beta1,resource.k8s.io/v1beta2",
			"FEATURE_GATES":  "DynamicResourceAllocation=true,DRAExtendedResource=true",
			// *not* needed because driver will run in "local filesystem" mode (= driver.IsLocal): "ALLOW_PRIVILEGED": "1",
		}
		cluster.Start(tCtx, binDir, localUpClusterEnv)
	})

	restConfig := cluster.LoadConfig(tCtx)
	restConfig.UserAgent = fmt.Sprintf("%s -- dra", restclient.DefaultKubernetesUserAgent())
	tCtx = tCtx.WithRESTConfig(restConfig).WithNamespace("default")

	var nodes *drautils.Nodes
	tCtx.Step(fmt.Sprintf("v%d.%d", major, previousMinor), func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(e2enode.WaitForAllNodesSchedulable(tCtx, tCtx.Client(), 5*time.Minute), "wait for all nodes to be schedulable")
		nodes = drautils.NewNodesNow(tCtx, 1, 1)
	})

	// Opening sockets locally avoids intermittent errors and delays caused by proxying through the restarted apiserver.
	// We could speed up testing by shortening the sync delay in the ResourceSlice controller, but let's better
	// test the defaults.
	driver := drautils.NewDriverInstance(tCtx)
	driver.IsLocal = true
	driver.Run(tCtx, "/var/lib/kubelet", nodes, drautils.DriverResourcesNow(nodes, 8))
	b := drautils.NewBuilderNow(tCtx, driver)
	b.SkipCleanup = true

	upgradedTestFuncs := make(map[string]upgradedTestFunc, len(subTests))
	tCtx.Run("after-cluster-creation", func(tCtx ktesting.TContext) {
		for subTest, f := range subTests {
			tCtx.Run(subTest, func(tCtx ktesting.TContext) {
				// This only gets set if f doesn't panic because of a fatal error,
				// so below we won't continue if step 1 already failed.
				// Other sub-tests are not affected.
				upgradedTestFuncs[subTest] = f(tCtx, b)
			})
		}
	})

	// We could split this up into first updating the apiserver, then control plane components, then restarting kubelet.
	// For the purpose of this test here we we primarily care about full before/after comparisons, so not done yet.
	// TODO
	restoreOptions := cluster.Modify(tCtx.WithStep(fmt.Sprintf("update to %s", gitVersion)), localupcluster.ModifyOptions{Upgrade: true, BinDir: dir})

	// The kubelet wipes all ResourceSlices on a restart because it doesn't know which drivers were running.
	// Wait for the ResourceSlice controller in the driver to notice and recreate the ResourceSlices.
	tCtx.WithStep("wait for ResourceSlices").Eventually(driver.NewGetSlices()).WithTimeout(5 * time.Minute).Should(gomega.HaveField("Items", gomega.HaveLen(len(nodes.NodeNames))))

	downgradedTestFuncs := make(map[string]downgradedTestFunc, len(subTests))
	tCtx.Run("after-cluster-upgrade", func(tCtx ktesting.TContext) {
		for subTest, f := range upgradedTestFuncs {
			tCtx.Run(subTest, func(tCtx ktesting.TContext) {
				downgradedTestFuncs[subTest] = f(tCtx)
			})
		}
	})

	// Roll back.
	cluster.Modify(tCtx.WithStep("downgrade"), restoreOptions)

	tCtx.Run("after-cluster-downgrade", func(tCtx ktesting.TContext) {
		for subTest, f := range downgradedTestFuncs {
			tCtx.Run(subTest, func(tCtx ktesting.TContext) {
				f(tCtx)
			})
		}
	})
}

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
	tCtx.Logf("workspace status:\n%s", output)

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

// serverDownloadURL constructs a download URL for a Kubernetes server tarball based on the given
// prefix, major, and minor version numbers. It performs an HTTP GET request to retrieve the version
// string from a remote text file, then builds the final tarball URL using the retrieved version,
// the current OS, and architecture. If the version file is not found (HTTP 404), it returns
// errHTTP404 to allow the caller to try another prefix. Returns the tarball URL, the version string,
// or an error if any step fails.
// The function uses the provided testing context for logging and error handling.
//
// Parameters:
//   - tCtx: a ktesting.TContext used for test context and error handling.
//   - prefix: the release prefix (e.g., "stable", "latest").
//   - major: the major version number.
//   - minor: the minor version number.
//
// Returns:
//   - The constructed tarball download URL as a string.
//   - The version string as retrieved from the remote file.
//   - An error if the request fails, the response is invalid, or the version file is not found.
func serverDownloadURL(tCtx ktesting.TContext, prefix string, major, minor uint) (string, string, error) {
	tCtx.Helper()
	url := fmt.Sprintf("https://dl.k8s.io/release/%s-%d.%d.txt", prefix, major, minor)
	get, err := http.NewRequestWithContext(tCtx, http.MethodGet, url, nil)
	if err != nil {
		return "", "", fmt.Errorf("constructing GET for %s failed: %w", url, err)
	}
	resp, err := http.DefaultClient.Do(get)
	if err != nil {
		return "", "", fmt.Errorf("downloading %s failed: %w", url, err)
	}
	if resp.StatusCode == http.StatusNotFound {
		// Caller should be able to distinguish HTTP 404
		// to try another prefix (usually 'latest' if 'stable' returns 404)
		return "", "", errHTTP404
	}
	if resp.StatusCode != http.StatusOK {
		return "", "", fmt.Errorf("getting %s failed: status code: %d, status: %s", url, resp.StatusCode, resp.Status)
	}
	if resp.Body == nil {
		return "", "", fmt.Errorf("empty response for %s", url)
	}
	defer func() {
		tCtx.ExpectNoError(resp.Body.Close(), "close response body")
	}()
	version, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", "", fmt.Errorf("reading response body for %s failed: %w", url, err)
	}
	return fmt.Sprintf("https://dl.k8s.io/release/%s/kubernetes-server-%s-%s.tar.gz", string(version), runtime.GOOS, runtime.GOARCH), string(version), nil
}

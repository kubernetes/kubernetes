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
	"bytes"
	"compress/gzip"
	"context"
	_ "embed"
	"encoding/json"
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
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	resourceapiv1beta2 "k8s.io/api/resource/v1beta2"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apiruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/version"
	resourceapiac "k8s.io/client-go/applyconfigurations/resource/v1"
	resourceapiacv1beta2 "k8s.io/client-go/applyconfigurations/resource/v1beta2"
	restclient "k8s.io/client-go/rest"
	draapiv1beta2 "k8s.io/dynamic-resource-allocation/api/v1beta2"
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

var errHTTP404 = errors.New("resource not found (404)")

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
		tCtx = ktesting.End(tCtx)

		// KUBERNETES_SERVER_CACHE_DIR can be set to keep downloaded files across test restarts.
		binDir, cacheBinaries := os.LookupEnv("KUBERNETES_SERVER_CACHE_DIR")
		if !cacheBinaries {
			binDir = tCtx.TempDir()
		}
		haveBinaries := false

		// Get the previous release.
		tCtx = ktesting.Begin(tCtx, "get previous release info")
		tCtx.Logf("stable release %d.%d", major, previousMinor)
		previousURL, previousVersion, err := serverDownloadURL(tCtx, "stable", major, previousMinor)
		if errors.Is(err, errHTTP404) {
			tCtx.Logf("stable doesn't exist, get latest release %d.%d", major, previousMinor)
			previousURL, previousVersion, err = serverDownloadURL(tCtx, "latest", major, previousMinor)
		}
		tCtx.ExpectNoError(err)
		tCtx.Logf("got previous release version: %s, URL: %s", previousVersion, previousURL)
		tCtx = ktesting.End(tCtx)

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
		testResourceClaimDeviceStatusAfterUpgrade, testResourceClaimDeviceStatusAfterDowngrade := testResourceClaimDeviceStatus(tCtx, namespace.Name)

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
		testResourceClaimDeviceStatusAfterUpgrade()

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
		testResourceClaimDeviceStatusAfterDowngrade()

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

// testResourceClaimDeviceStatus corresponds to testResourceClaimDeviceStatus in test/integration/dra
// and was copied from there, therefore the unit-test style with tCtx and require.
func testResourceClaimDeviceStatus(tCtx ktesting.TContext, namespace string) (afterUpgrade, afterDowngrade func()) {
	claimName := "claim-with-device-status"
	claim := &resourceapiv1beta2.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      claimName,
		},
		Spec: resourceapiv1beta2.ResourceClaimSpec{
			Devices: resourceapiv1beta2.DeviceClaim{
				Requests: []resourceapiv1beta2.DeviceRequest{
					{
						Name: "foo",
						Exactly: &resourceapiv1beta2.ExactDeviceRequest{
							DeviceClassName: "foo",
						},
					},
				},
			},
		},
	}

	claim, err := tCtx.Client().ResourceV1beta2().ResourceClaims(namespace).Create(tCtx, claim, metav1.CreateOptions{})
	tCtx.ExpectNoError(err, "create ResourceClaim")

	// Add an allocation result.
	// A finalizer is required for that.
	finalizer := "test.example.com/my-test-finalizer"
	claim.Finalizers = append(claim.Finalizers, finalizer)
	claim, err = tCtx.Client().ResourceV1beta2().ResourceClaims(namespace).Update(tCtx, claim, metav1.UpdateOptions{})
	claim.Status.Allocation = &resourceapiv1beta2.AllocationResult{
		Devices: resourceapiv1beta2.DeviceAllocationResult{
			Results: []resourceapiv1beta2.DeviceRequestAllocationResult{
				{
					Request: "foo",
					Driver:  "one",
					Pool:    "global",
					Device:  "my-device",
				},
				{
					Request: "foo",
					Driver:  "two",
					Pool:    "global",
					Device:  "another-device",
				},
				{
					Request: "foo",
					Driver:  "three",
					Pool:    "global",
					Device:  "my-device",
				},
			},
		},
	}
	tCtx.ExpectNoError(err, "add finalizer")
	removeClaim := func(tCtx ktesting.TContext) {
		client := tCtx.Client().ResourceV1beta2()
		claim, err := client.ResourceClaims(namespace).Get(tCtx, claimName, metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			return
		}
		tCtx.ExpectNoError(err, "get claim to remove finalizer")
		if claim.Status.Allocation != nil {
			claim.Status.Allocation = nil
			claim, err = client.ResourceClaims(namespace).UpdateStatus(tCtx, claim, metav1.UpdateOptions{})
			tCtx.ExpectNoError(err, "remove allocation")
		}
		claim.Finalizers = nil
		claim, err = client.ResourceClaims(namespace).Update(tCtx, claim, metav1.UpdateOptions{})
		tCtx.ExpectNoError(err, "remove finalizer")
		err = client.ResourceClaims(namespace).Delete(tCtx, claim.Name, metav1.DeleteOptions{})
		tCtx.ExpectNoError(err, "delete claim")
	}
	tCtx.CleanupCtx(removeClaim)
	claim, err = tCtx.Client().ResourceV1beta2().ResourceClaims(namespace).UpdateStatus(tCtx, claim, metav1.UpdateOptions{})
	tCtx.ExpectNoError(err, "add allocation result")

	// Now adding the device status should work.
	deviceStatus := []resourceapiv1beta2.AllocatedDeviceStatus{{
		Driver: "one",
		Pool:   "global",
		Device: "my-device",
		Data: &apiruntime.RawExtension{
			Raw: []byte(`{"kind": "foo", "apiVersion": "dra.example.com/v1"}`),
		},
		NetworkData: &resourceapiv1beta2.NetworkDeviceData{
			InterfaceName: "net-1",
			IPs: []string{
				"10.9.8.0/24",
				"2001:db8::/64",
			},
			HardwareAddress: "ea:9f:cb:40:b1:7b",
		},
	}}
	claim.Status.Devices = deviceStatus
	tCtx.ExpectNoError(err, "add device status")
	require.Equal(tCtx, deviceStatus, claim.Status.Devices, "after adding device status")

	// Strip the RawExtension. SSA re-encodes it, which causes negligble differences that nonetheless break assert.Equal.
	claim.Status.Devices[0].Data = nil
	deviceStatus[0].Data = nil
	claim, err = tCtx.Client().ResourceV1beta2().ResourceClaims(namespace).UpdateStatus(tCtx, claim, metav1.UpdateOptions{})
	tCtx.ExpectNoError(err, "add device status")
	require.Equal(tCtx, deviceStatus, claim.Status.Devices, "after stripping RawExtension")

	// Exercise SSA.
	deviceStatusAC := resourceapiacv1beta2.AllocatedDeviceStatus().
		WithDriver("two").
		WithPool("global").
		WithDevice("another-device").
		WithNetworkData(resourceapiacv1beta2.NetworkDeviceData().WithInterfaceName("net-2"))
	deviceStatus = append(deviceStatus, resourceapiv1beta2.AllocatedDeviceStatus{
		Driver: "two",
		Pool:   "global",
		Device: "another-device",
		NetworkData: &resourceapiv1beta2.NetworkDeviceData{
			InterfaceName: "net-2",
		},
	})
	claimAC := resourceapiacv1beta2.ResourceClaim(claim.Name, claim.Namespace).
		WithStatus(resourceapiacv1beta2.ResourceClaimStatus().WithDevices(deviceStatusAC))
	claim, err = tCtx.Client().ResourceV1beta2().ResourceClaims(namespace).ApplyStatus(tCtx, claimAC, metav1.ApplyOptions{
		Force:        true,
		FieldManager: "manager-1",
	})
	tCtx.ExpectNoError(err, "apply device status two")
	require.Equal(tCtx, deviceStatus, claim.Status.Devices, "after applying device status two")

	deviceStatusAC = resourceapiacv1beta2.AllocatedDeviceStatus().
		WithDriver("three").
		WithPool("global").
		WithDevice("my-device").
		WithNetworkData(resourceapiacv1beta2.NetworkDeviceData().WithInterfaceName("net-3"))
	deviceStatus = append(deviceStatus, resourceapiv1beta2.AllocatedDeviceStatus{
		Driver: "three",
		Pool:   "global",
		Device: "my-device",
		NetworkData: &resourceapiv1beta2.NetworkDeviceData{
			InterfaceName: "net-3",
		},
	})
	claimAC = resourceapiacv1beta2.ResourceClaim(claim.Name, claim.Namespace).
		WithStatus(resourceapiacv1beta2.ResourceClaimStatus().WithDevices(deviceStatusAC))
	claim, err = tCtx.Client().ResourceV1beta2().ResourceClaims(namespace).ApplyStatus(tCtx, claimAC, metav1.ApplyOptions{
		Force:        true,
		FieldManager: "manager-2",
	})
	tCtx.ExpectNoError(err, "apply device status three")
	require.Equal(tCtx, deviceStatus, claim.Status.Devices, "after applying device status three")
	var buffer bytes.Buffer
	encoder := json.NewEncoder(&buffer)
	encoder.SetIndent("   ", "   ")
	tCtx.ExpectNoError(encoder.Encode(claim))
	tCtx.Logf("Final ResourceClaim:\n%s", buffer.String())

	afterUpgrade = func() {
		// Update one entry, remove the other.
		deviceStatusAC := resourceapiac.AllocatedDeviceStatus().
			WithDriver("two").
			WithPool("global").
			WithDevice("another-device").
			WithNetworkData(resourceapiac.NetworkDeviceData().WithInterfaceName("yet-another-net"))
		deviceStatus[1].NetworkData.InterfaceName = "yet-another-net"
		claimAC := resourceapiac.ResourceClaim(claim.Name, claim.Namespace).
			WithStatus(resourceapiac.ResourceClaimStatus().WithDevices(deviceStatusAC))
		claim, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).ApplyStatus(tCtx, claimAC, metav1.ApplyOptions{
			Force:        true,
			FieldManager: "manager-1",
		})
		tCtx.ExpectNoError(err, "update device status two")

		var deviceStatusV1 []resourceapi.AllocatedDeviceStatus
		for _, status := range deviceStatus {
			var statusV1 resourceapi.AllocatedDeviceStatus
			tCtx.ExpectNoError(draapiv1beta2.Convert_v1beta2_AllocatedDeviceStatus_To_v1_AllocatedDeviceStatus(&status, &statusV1, nil))
			deviceStatusV1 = append(deviceStatusV1, statusV1)
		}
		require.Equal(tCtx, deviceStatusV1, claim.Status.Devices, "after updating device status two")
	}
	afterDowngrade = func() {
		claimAC := resourceapiacv1beta2.ResourceClaim(claim.Name, claim.Namespace)
		deviceStatus = deviceStatus[0:2]
		claim, err := tCtx.Client().ResourceV1beta2().ResourceClaims(namespace).ApplyStatus(tCtx, claimAC, metav1.ApplyOptions{
			Force:        true,
			FieldManager: "manager-2",
		})
		tCtx.ExpectNoError(err, "remove device status three")
		require.Equal(tCtx, deviceStatus, claim.Status.Devices, "after removing device status three")

		// The cleanup order is so that we have to run this explicitly now.
		// The tCtx.CleanupCtx is more for the sake of completeness.
		removeClaim(tCtx)
	}
	return
}

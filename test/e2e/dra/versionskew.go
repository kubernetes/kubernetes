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

package dra

import (
	"bufio"
	"context"
	_ "embed"
	"fmt"
	"io"
	"os/exec"
	"path"
	"strings"
	"sync"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/version"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/errors"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/utils/kindcluster"
	"k8s.io/kubernetes/test/utils/ktesting"
	admissionapi "k8s.io/pod-security-admission/api"
)

//go:embed kind.yaml
var kindConfig string

// Upgrade/downgrade tests are slow, partly because "make quick-release" below
// and BuildImage in kindcluster.go are slow.
//
// If both were done once, they don't need to be repeated unless something
// changes. This is not checked, so both steps are always done.
// Developers who have gone through the full test and want to re-run
// it more quickly can comment out the invocations.
// Beware that the apiserver not coming up again can be caused by not
// having up-to-date release images, so run "make quick-release" manually.

var _ = framework.SIGDescribe("node")(framework.WithLabel("DRA"), feature.DynamicResourceAllocation, framework.WithFeatureGate(features.DynamicResourceAllocation) /* TODO: add framework.WithSlow() once we have pull-kubernetes-kind-dra-slow */ /* TODO: add feature.KindCommand once jobs are updated*/, func() {

	ginkgo.It("upgrade/downgrade works", func(ctx context.Context) {
		// TODO: replace with helper code from https://github.com/kubernetes/kubernetes/pull/122481 should that get merged.
		tCtx := ktesting.Init(GinkgoContextTB())
		tCtx = ktesting.WithContext(tCtx, ctx)

		tCtx = ktesting.Begin(tCtx, "get source code version")
		gitVersion, dockerTag, err := sourceVersion(tCtx, framework.TestContext.RepoRoot)
		tCtx.ExpectNoError(err, "determine source code version for repo root %q", framework.TestContext.RepoRoot)
		version, err := version.ParseGeneric(gitVersion)
		tCtx.ExpectNoError(err, "parse version %s of repo root %q", gitVersion, framework.TestContext.RepoRoot)
		major, previousMinor := version.Major(), version.Minor()-1
		tCtx = ktesting.End(tCtx)

		clusterName := "dra"
		tCtx = ktesting.Begin(tCtx, fmt.Sprintf("bring up v%d.%d", major, previousMinor))
		previousImageSource := kindcluster.ServerDownloadURL(tCtx, major, previousMinor)
		previousImageName := kindcluster.BuildImage(tCtx, fmt.Sprintf("%d.%d", major, previousMinor), previousImageSource)
		cluster := kindcluster.New(tCtx)
		cluster.Start(tCtx, clusterName, kindConfig, previousImageName)
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
		nodes := NewNodesNow(tCtx, f, 1, 1)
		driver := NewDriverInstance(f)
		driver.Run(nodes, driverResourcesNow(nodes, 1))
		b := newBuilderNow(ctx, f, driver)

		claim := b.externalClaim()
		pod := b.podExternal()
		b.create(ctx, claim, pod)
		b.testPod(ctx, f, pod)

		tCtx = ktesting.End(tCtx)

		tCtx = ktesting.Begin(tCtx, fmt.Sprintf("update to %s", gitVersion))
		repoRoot := framework.TestContext.RepoRoot
		// We could split this up into first updating the apiserver, then control plane components, then restarting kubelet.
		// For the purpose of this test here we we primarily care about full before/after comparisons, so not done yet.
		releaseImagesDir, kubelet := buildReleaseImages(tCtx, repoRoot)
		restoreOptions := cluster.Modify(tCtx, kindcluster.ModifyOptions{Upgrade: true, DockerTag: dockerTag, ArchSuffix: "-amd64", ReleaseImagesDir: releaseImagesDir, KubeletBinary: kubelet})
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
		claim = b.externalClaim()
		pod = b.podExternal()
		pod.Spec.ResourceClaims[0].ResourceClaimName = &claim.Name
		b.create(ctx, claim, pod)
		b.testPod(ctx, f, pod)

		// Roll back.
		tCtx = ktesting.Begin(tCtx, "downgrade")
		cluster.Modify(tCtx, restoreOptions)
		tCtx = ktesting.End(tCtx)

		// TODO: ensure that kube-controller-manager is up-and-running.
		// This works around https://github.com/kubernetes/kubernetes/issues/132334 and can be removed
		// once a fix for that is backported.
		tCtx = ktesting.Begin(tCtx, "wait for kube-controller-manager")
		ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) string {
			return cluster.GetSystemLogs(tCtx, "kube-controller-manager")
		}).Should(gomega.And(
			gomega.ContainSubstring(`successfully renewed lease kube-system/kube-controller-manager`),
			gomega.ContainSubstring(`"Caches populated" type="*v1beta1.ResourceClaim"`),
			gomega.ContainSubstring(`"Caches populated" type="*v1.Pod"`),
		))
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

func buildReleaseImages(tCtx ktesting.TContext, repoRoot string) (string, string) {
	// Comment out to save time if already built.
	runCmdIn(tCtx, repoRoot, "make", "quick-release")
	return path.Join(repoRoot, "_output/release-images/amd64"), path.Join(repoRoot, "_output/dockerized/bin/linux/amd64/kubelet") // TODO (?) find kubelet
}

func runCmdIn(tCtx ktesting.TContext, dir string, name string, args ...string) string {
	tCtx.Helper()
	tCtx.Logf("Running command: %s %s", name, strings.Join(args, " "))
	cmd := exec.CommandContext(tCtx, name, args...)
	cmd.Dir = dir
	var output strings.Builder
	reader, writer := io.Pipe()
	cmd.Stdout = writer
	cmd.Stderr = writer
	tCtx.ExpectNoError(cmd.Start(), "start %s command", name)
	scanner := bufio.NewScanner(reader)
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for scanner.Scan() {
			line := scanner.Text()
			line = strings.TrimSuffix(line, "\n")
			tCtx.Logf("%s: %s", name, line)
			output.WriteString(line)
			output.WriteByte('\n')
		}
	}()
	result := cmd.Wait()
	tCtx.ExpectNoError(writer.Close(), "close in-memory pipe")
	wg.Wait()
	tCtx.ExpectNoError(result, fmt.Sprintf("%s command failed, output:\n%s", name, output.String()))
	tCtx.ExpectNoError(scanner.Err(), "read %s command output", name)

	return output.String()
}

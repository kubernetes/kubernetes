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

	"github.com/onsi/ginkgo/v2"

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

var _ = framework.SIGDescribe("node")(framework.WithLabel("DRA"), feature.DynamicResourceAllocation, framework.WithFeatureGate(features.DynamicResourceAllocation), feature.KindCommand, func() {

	ginkgo.It("upgrade/downgrade works", func(ctx context.Context) {
		// TODO: replace with helper code from https://github.com/kubernetes/kubernetes/pull/122481 should that get merged.
		tCtx := ktesting.Init(GinkgoContextTB())
		tCtx = ktesting.WithContext(tCtx, ctx)
		// imageSource := framework.TestContext.RepoRoot

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
		releaseImageDir, kubelet := buildReleaseImages(tCtx, repoRoot)
		cluster.UpdateAll(tCtx, dockerTag, releaseImageDir, kubelet)

		// We need to clean up explicitly because the normal
		// cleanup doesn't work (driver shuts down first).
		framework.ExpectNoError(f.ClientSet.ResourceV1beta1().ResourceClaims(namespace.Name).Delete(ctx, claim.Name, metav1.DeleteOptions{}))
		framework.ExpectNoError(f.ClientSet.CoreV1().Pods(namespace.Name).Delete(ctx, pod.Name, metav1.DeleteOptions{}))
		framework.ExpectNoError(e2epod.WaitForPodNotFoundInNamespace(ctx, f.ClientSet, pod.Name, namespace.Name, f.Timeouts.PodDelete))
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
	runCmdIn(tCtx, repoRoot, "make", "quick-release")
	return path.Join(repoRoot, "_output/release-images/amd64"), path.Join(repoRoot, "_output/dockerized/bin/linux/amd64/kubelet") // TODO (?) find kubelet
}

func runCmdIn(tCtx ktesting.TContext, dir string, name string, args ...string) string {
	tCtx.Helper()
	tCtx.Logf("Running command: %s %s", strings.Join(args, " "))
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
	tCtx.ExpectNoError(cmd.Wait(), fmt.Sprintf("%s command failed, output:\n%s", name, output.String()))
	writer.Close()
	wg.Wait()
	tCtx.ExpectNoError(scanner.Err(), "read %s command output", name)

	return output.String()
}

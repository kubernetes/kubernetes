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
	"context"
	_ "embed"
	"fmt"

	"github.com/onsi/ginkgo/v2"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	restclient "k8s.io/client-go/rest"
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

		clusterName := "dra"
		version := "HEAD"
		imageSource := framework.TestContext.RepoRoot
		var imageName string
		ktesting.Step(tCtx, "build HEAD image", func(tCtx ktesting.TContext) {
			imageName = kindcluster.BuildImage(tCtx, version, imageSource)
		})

		cluster := kindcluster.New(tCtx)
		ktesting.Step(tCtx, "start HEAD cluster", func(tCtx ktesting.TContext) {
			cluster.Start(tCtx, clusterName, kindConfig, imageName)
		})

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

		tCtx.ExpectNoError(e2enode.WaitForAllNodesSchedulable(tCtx, tCtx.Client(), f.Timeouts.NodeSchedulable), "wait for all nodes to be schedulable")
		nodes := NewNodesNow(tCtx, f, 1, 1)
		driver := NewDriverInstance(f)
		driver.Run(nodes, driverResourcesNow(nodes, 1))
		b := newBuilderNow(ctx, f, driver)

		claim := b.externalClaim()
		pod := b.podExternal()
		b.create(ctx, claim, pod)
		b.testPod(ctx, f, pod)

		// We need to clean up explicitly because the normal
		// cleanup doesn't work (driver shuts down first).
		framework.ExpectNoError(f.ClientSet.ResourceV1beta1().ResourceClaims(namespace.Name).Delete(ctx, claim.Name, metav1.DeleteOptions{}))
		framework.ExpectNoError(f.ClientSet.CoreV1().Pods(namespace.Name).Delete(ctx, pod.Name, metav1.DeleteOptions{}))
		framework.ExpectNoError(e2epod.WaitForPodNotFoundInNamespace(ctx, f.ClientSet, pod.Name, namespace.Name, f.Timeouts.PodDelete))
	})
})

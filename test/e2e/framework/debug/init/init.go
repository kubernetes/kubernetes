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

// Package init sets debug.DumpAllNamespaceInfo as implementation in the framework
// and enables log size verification and resource gathering.
package init

import (
	"context"
	"sync"
	"time"

	"github.com/onsi/ginkgo/v2"
	"k8s.io/kubernetes/test/e2e/framework"
	e2edebug "k8s.io/kubernetes/test/e2e/framework/debug"
)

func init() {
	framework.NewFrameworkExtensions = append(framework.NewFrameworkExtensions,
		func(f *framework.Framework) {
			f.DumpAllNamespaceInfo = func(ctx context.Context, f *framework.Framework, ns string) {
				e2edebug.DumpAllNamespaceInfo(ctx, f.ClientSet, ns)
			}

			if framework.TestContext.GatherLogsSizes {
				ginkgo.BeforeEach(func() {
					var wg sync.WaitGroup
					wg.Add(1)
					ctx, cancel := context.WithCancel(context.Background())
					verifier := e2edebug.NewLogsVerifier(ctx, f.ClientSet)
					go func() {
						defer wg.Done()
						verifier.Run(ctx)
					}()
					ginkgo.DeferCleanup(func() {
						ginkgo.By("Gathering log sizes data", func() {
							cancel()
							wg.Wait()
							f.TestSummaries = append(f.TestSummaries, verifier.GetSummary())
						})
					})
				})
			}

			if framework.TestContext.GatherKubeSystemResourceUsageData != "false" &&
				framework.TestContext.GatherKubeSystemResourceUsageData != "none" {
				ginkgo.BeforeEach(func(ctx context.Context) {
					var nodeMode e2edebug.NodesSet
					switch framework.TestContext.GatherKubeSystemResourceUsageData {
					case "master":
						nodeMode = e2edebug.MasterNodes
					case "masteranddns":
						nodeMode = e2edebug.MasterAndDNSNodes
					default:
						nodeMode = e2edebug.AllNodes
					}

					gatherer, err := e2edebug.NewResourceUsageGatherer(ctx, f.ClientSet, e2edebug.ResourceGathererOptions{
						InKubemark:                  framework.ProviderIs("kubemark"),
						Nodes:                       nodeMode,
						ResourceDataGatheringPeriod: 60 * time.Second,
						ProbeDuration:               15 * time.Second,
						PrintVerboseLogs:            false,
					}, nil)
					if err != nil {
						framework.Logf("Error while creating NewResourceUsageGatherer: %v", err)
						return
					}

					go gatherer.StartGatheringData(ctx)
					ginkgo.DeferCleanup(func() {
						ginkgo.By("Collecting resource usage data", func() {
							summary, resourceViolationError := gatherer.StopAndSummarize([]int{90, 99, 100}, nil /* no constraints */)
							// Always record the summary, even if there was an error.
							f.TestSummaries = append(f.TestSummaries, summary)
							// Now fail if there was an error.
							framework.ExpectNoError(resourceViolationError)
						})
					})
				})
			}
		},
	)
}

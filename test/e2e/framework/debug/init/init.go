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
	"sync"
	"time"

	"github.com/onsi/ginkgo/v2"
	"k8s.io/kubernetes/test/e2e/framework"
	e2edebug "k8s.io/kubernetes/test/e2e/framework/debug"
)

var (
	// TODO: this variable used to be a field in framework.Framework. It is
	// not clear how it was ever set. https://grep.app/search?q=AddonResourceConstraints
	// returns only the default initialization with an empty map. Perhaps it can be removed?

	// Constraints that passed to a check which is executed after data is gathered to
	// see if 99% of results are within acceptable bounds. It has to be injected in the test,
	// as expectations vary greatly. Constraints are grouped by the container names.
	AddonResourceConstraints map[string]e2edebug.ResourceConstraint
)

func init() {
	framework.NewFrameworkExtensions = append(framework.NewFrameworkExtensions,
		func(f *framework.Framework) {
			f.DumpAllNamespaceInfo = func(f *framework.Framework, ns string) {
				e2edebug.DumpAllNamespaceInfo(f.ClientSet, ns)
			}

			if framework.TestContext.GatherLogsSizes {
				var (
					wg           sync.WaitGroup
					closeChannel chan bool
					verifier     *e2edebug.LogsSizeVerifier
				)

				ginkgo.BeforeEach(func() {
					wg.Add(1)
					closeChannel = make(chan bool)
					verifier = e2edebug.NewLogsVerifier(f.ClientSet, closeChannel)
					go func() {
						defer wg.Done()
						verifier.Run()
					}()
					ginkgo.DeferCleanup(func() {
						ginkgo.By("Gathering log sizes data", func() {
							close(closeChannel)
							wg.Wait()
							f.TestSummaries = append(f.TestSummaries, verifier.GetSummary())
						})
					})
				})
			}

			if framework.TestContext.GatherKubeSystemResourceUsageData != "false" &&
				framework.TestContext.GatherKubeSystemResourceUsageData != "none" {
				ginkgo.BeforeEach(func() {
					var nodeMode e2edebug.NodesSet
					switch framework.TestContext.GatherKubeSystemResourceUsageData {
					case "master":
						nodeMode = e2edebug.MasterNodes
					case "masteranddns":
						nodeMode = e2edebug.MasterAndDNSNodes
					default:
						nodeMode = e2edebug.AllNodes
					}

					gatherer, err := e2edebug.NewResourceUsageGatherer(f.ClientSet, e2edebug.ResourceGathererOptions{
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

					go gatherer.StartGatheringData()
					ginkgo.DeferCleanup(func() {
						ginkgo.By("Collecting resource usage data", func() {
							summary, resourceViolationError := gatherer.StopAndSummarize([]int{90, 99, 100}, AddonResourceConstraints)
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

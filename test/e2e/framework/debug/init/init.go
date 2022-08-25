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
// and enables log size verification.
package init

import (
	"sync"

	"github.com/onsi/ginkgo/v2"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/framework/debug"
)

func init() {
	framework.NewFrameworkExtensions = append(framework.NewFrameworkExtensions,
		func(f *framework.Framework) {
			f.DumpAllNamespaceInfo = func(f *framework.Framework, ns string) {
				debug.DumpAllNamespaceInfo(f.ClientSet, ns)
			}

			if framework.TestContext.GatherLogsSizes {
				var (
					wg           sync.WaitGroup
					closeChannel chan bool
					verifier     *debug.LogsSizeVerifier
				)

				ginkgo.BeforeEach(func() {
					wg.Add(1)
					closeChannel = make(chan bool)
					verifier = debug.NewLogsVerifier(f.ClientSet, closeChannel)
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
		},
	)
}

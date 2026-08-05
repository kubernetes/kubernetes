/*
Copyright The Kubernetes Authors.

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

package withprovider

import (
	"context"
	"testing"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/ginkgo/v2/types"
	"github.com/stretchr/testify/assert"

	"k8s.io/kubernetes/test/e2e/framework"
)

// TestWithProvider runs a small suite of specs tagged with
// framework.WithProvider and checks that:
//   - specs whose provider list doesn't match TestContext.Provider get
//     skipped at runtime instead of running,
//   - specs whose provider list does match run normally,
//   - the "[Provider:...]" tag shows up in the spec text (for visibility in
//     e.g. --list-tests output) but is not registered as a real, filterable
//     Ginkgo label (unlike other tags such as [Slow]),
//   - this works regardless of whether the It body takes no arguments,
//     a context.Context or a ginkgo.SpecContext,
//   - this also works when framework.WithProvider is passed to a Context
//     instead of directly to the It's below it (the check is then injected
//     as a BeforeEach instead of running directly in the It).
func TestWithProvider(t *testing.T) {
	oldProvider := framework.TestContext.Provider
	framework.TestContext.Provider = "gce"
	t.Cleanup(func() { framework.TestContext.Provider = oldProvider })

	var ran []string
	ginkgo.Describe("wrap", func() {
		framework.It("aws only", framework.WithProvider("aws"), func() {
			ran = append(ran, "aws only")
		})
		framework.It("gce only", framework.WithProvider("gce"), func() {
			ran = append(ran, "gce only")
		})
		framework.It("aws or gce", framework.WithProvider("aws", "gce"), func() {
			ran = append(ran, "aws or gce")
		})
		framework.It("aws only ctx", framework.WithProvider("aws"), func(ctx context.Context) {
			ran = append(ran, "aws only ctx")
		})
		framework.It("gce only ctx", framework.WithProvider("gce"), func(ctx context.Context) {
			ran = append(ran, "gce only ctx")
		})
		framework.It("aws only spec ctx", framework.WithProvider("aws"), func(ctx ginkgo.SpecContext) {
			ran = append(ran, "aws only spec ctx")
		})
		framework.It("gce only spec ctx", framework.WithProvider("gce"), func(ctx ginkgo.SpecContext) {
			ran = append(ran, "gce only spec ctx")
		})
		framework.Context("aws context", framework.WithProvider("aws"), func() {
			framework.It("inner one", func() {
				ran = append(ran, "context aws inner one")
			})
			framework.It("inner two", func() {
				ran = append(ran, "context aws inner two")
			})
		})
		framework.Context("gce context", framework.WithProvider("gce"), func() {
			framework.It("inner", func() {
				ran = append(ran, "context gce inner")
			})
		})
	})

	var report types.Report
	ginkgo.ReportAfterSuite("capture report", func(r types.Report) {
		report = r
	})

	suiteConfig, reporterConfig := framework.CreateGinkgoConfig()
	fakeT := &testing.T{}
	ginkgo.RunSpecs(fakeT, "WithProvider Suite", suiteConfig, reporterConfig)

	assert.False(t, fakeT.Failed(), "suite run should not fail")
	assert.ElementsMatch(t, []string{
		"gce only",
		"aws or gce",
		"gce only ctx",
		"gce only spec ctx",
		"context gce inner",
	}, ran, "only specs whose provider list includes the configured provider should run")

	states := map[string]types.SpecState{}
	labels := map[string][]string{}
	for _, spec := range report.SpecReports {
		states[spec.FullText()] = spec.State
		labels[spec.FullText()] = spec.Labels()
	}

	assert.Equal(t, types.SpecStateSkipped, states["wrap aws only [Provider:aws]"])
	assert.Equal(t, types.SpecStatePassed, states["wrap gce only [Provider:gce]"])
	assert.Equal(t, types.SpecStatePassed, states["wrap aws or gce [Provider:aws,gce]"])
	assert.Equal(t, types.SpecStateSkipped, states["wrap aws only ctx [Provider:aws]"])
	assert.Equal(t, types.SpecStatePassed, states["wrap gce only ctx [Provider:gce]"])
	assert.Equal(t, types.SpecStateSkipped, states["wrap aws only spec ctx [Provider:aws]"])
	assert.Equal(t, types.SpecStatePassed, states["wrap gce only spec ctx [Provider:gce]"])
	assert.Equal(t, types.SpecStateSkipped, states["wrap aws context [Provider:aws] inner one"])
	assert.Equal(t, types.SpecStateSkipped, states["wrap aws context [Provider:aws] inner two"])
	assert.Equal(t, types.SpecStatePassed, states["wrap gce context [Provider:gce] inner"])

	// The provider constraint must be visible in the text, but not usable as
	// a "--label-filter" label like [Slow] or [Serial] are.
	assert.NotContains(t, labels["wrap aws only [Provider:aws]"], "Provider:aws")
	assert.NotContains(t, labels["wrap aws context [Provider:aws] inner one"], "Provider:aws")
}

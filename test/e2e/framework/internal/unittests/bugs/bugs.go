/*
Copyright 2023 The Kubernetes Authors.

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

package bugs

import (
	"bytes"
	"testing"

	"github.com/onsi/ginkgo/v2"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/framework/internal/unittests/bugs/features"
)

// The line number of the following code is checked in BugOutput below.
// Be careful when moving it around or changing the import statements above.
// Here are some intentionally blank lines that can be removed to compensate
// for future additional import statements.
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
// This must be line #50.

func helper() {
	framework.RecordBug(framework.NewBug("new bug", 0))
	framework.RecordBug(framework.NewBug("parent", 1))
}

func RecordBugs() {
	helper()
	framework.RecordBug(framework.Bug{FileName: "buggy/buggy.go", LineNumber: 100, Message: "hello world"})
	framework.RecordBug(framework.Bug{FileName: "some/relative/path/buggy.go", LineNumber: 200, Message: "    with spaces    \n"})
}

var (
	validFeature     = framework.ValidFeatures.Add("feature-foo")
	validEnvironment = framework.ValidEnvironments.Add("Linux")
)

func Describe() {
	// Normally a single line would be better, but this is an extreme example and
	// thus uses multiple.
	framework.SIGDescribe("testing")("abc",
		// Bugs in parameters will be attributed to the Describe call, not the line of the parameter.
		"",        // buggy: not needed
		" space1", // buggy: leading white space
		"space2 ", // buggy: trailing white space
		framework.WithFeature("no-such-feature"),
		framework.WithFeature(validFeature),
		framework.WithEnvironment("no-such-env"),
		framework.WithEnvironment(validEnvironment),
		framework.WithFeatureGate("no-such-feature-gate"),
		framework.WithFeatureGate(features.Alpha),
		framework.WithFeatureGate(features.Beta),
		framework.WithFeatureGate(features.BetaDefaultOff),
		framework.WithFeatureGate(features.GA),
		framework.WithConformance(),
		framework.WithNodeConformance(),
		framework.WithSlow(),
		framework.WithSerial(),
		framework.WithDisruptive(),
		framework.WithLabel("custom-label"),
		"xyz", // okay, becomes part of the final text
		func() {
			f := framework.NewDefaultFramework("abc")

			framework.Context("y", framework.WithLabel("foo"), func() {
				framework.It("should", f.WithLabel("bar"), func() {
				})
			})

			f.Context("x", f.WithLabel("foo"), func() {
				f.It("should", f.WithLabel("bar"), func() {
				})
			})
		},
	)

	framework.SIGDescribe("123")
}

const (
	numBugs   = 3
	bugOutput = `ERROR: bugs.go:53: new bug
ERROR: bugs.go:58: parent
ERROR: bugs.go:71: empty strings as separators are unnecessary and need to be removed
ERROR: bugs.go:71: trailing or leading spaces are unnecessary and need to be removed: " space1"
ERROR: bugs.go:71: trailing or leading spaces are unnecessary and need to be removed: "space2 "
ERROR: bugs.go:76: WithFeature: unknown feature "no-such-feature"
ERROR: bugs.go:78: WithEnvironment: unknown environment "no-such-env"
ERROR: bugs.go:80: WithFeatureGate: the feature gate "no-such-feature-gate" is unknown
ERROR: bugs.go:107: SIG label must be lowercase, no spaces and no sig- prefix, got instead: "123"
ERROR: buggy/buggy.go:100: hello world
ERROR: some/relative/path/buggy.go:200: with spaces
`
	// Used by unittests/list-tests. It's sorted by test name, not source code location.
	ListTestsOutput = `The following spec names can be used with 'ginkgo run --focus/skip':
    ../bugs/bugs.go:101: [sig-testing] abc   space1 space2  [Feature:no-such-feature] [Feature:feature-foo] [Environment:no-such-env] [Environment:Linux] [FeatureGate:no-such-feature-gate] [Feature:OffByDefault] [FeatureGate:TestAlphaFeature] [Alpha] [Feature:OffByDefault] [FeatureGate:TestBetaFeature] [Beta] [FeatureGate:TestBetaDefaultOffFeature] [Beta] [Feature:OffByDefault] [FeatureGate:TestGAFeature] [Conformance] [NodeConformance] [Slow] [Serial] [Disruptive] [custom-label] xyz x [foo] should [bar]
    ../bugs/bugs.go:96: [sig-testing] abc   space1 space2  [Feature:no-such-feature] [Feature:feature-foo] [Environment:no-such-env] [Environment:Linux] [FeatureGate:no-such-feature-gate] [Feature:OffByDefault] [FeatureGate:TestAlphaFeature] [Alpha] [Feature:OffByDefault] [FeatureGate:TestBetaFeature] [Beta] [FeatureGate:TestBetaDefaultOffFeature] [Beta] [Feature:OffByDefault] [FeatureGate:TestGAFeature] [Conformance] [NodeConformance] [Slow] [Serial] [Disruptive] [custom-label] xyz y [foo] should [bar]

`

	// Used by unittests/list-labels.
	ListLabelsOutput = `The following labels can be used with 'ginkgo run --label-filter':
    Alpha
    Beta
    BetaOffByDefault
    Conformance
    Disruptive
    Environment:Linux
    Environment:no-such-env
    Feature:OffByDefault
    Feature:feature-foo
    Feature:no-such-feature
    FeatureGate:TestAlphaFeature
    FeatureGate:TestBetaDefaultOffFeature
    FeatureGate:TestBetaFeature
    FeatureGate:TestGAFeature
    FeatureGate:no-such-feature-gate
    NodeConformance
    Serial
    Slow
    bar
    custom-label
    foo
    sig-testing

`
)

func GetGinkgoOutput(t *testing.T) string {
	var buffer bytes.Buffer
	ginkgo.GinkgoWriter.TeeTo(&buffer)
	t.Cleanup(ginkgo.GinkgoWriter.ClearTeeWriters)

	suiteConfig, reporterConfig := framework.CreateGinkgoConfig()
	fakeT := &testing.T{}
	ginkgo.RunSpecs(fakeT, "Buggy Suite", suiteConfig, reporterConfig)

	return buffer.String()
}

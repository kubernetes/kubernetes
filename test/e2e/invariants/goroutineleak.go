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

// please speak to SIG-Testing leads before adding anything to this package
// see: https://git.k8s.io/enhancements/keps/sig-testing/5468-invariant-testing
package invariants

import (
	"context"
	"flag"
	"regexp"

	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/invariants/goroutineleak"

	"github.com/onsi/ginkgo/v2"
	ginkgotypes "github.com/onsi/ginkgo/v2/types"
)

// checks for goroutines which the Go runtime has determined can never be
// unblocked, i.e. permanently leaked goroutines.
const invariantGoroutineLeaksLeafText = "should enable checking for goroutine leaks"

const (
	// defaultNamespacesRE matches namespaces whose pods may be scraped when
	// pod scraping is enabled: anything containing the word "system". The
	// same convention as the logcheck package, for the same reason: there is
	// no API for identifying system namespaces
	// (https://github.com/kubernetes/enhancements/issues/5708).
	defaultGoroutineLeakNamespacesRE = `system`

	// defaultNodesRE matches all nodes.
	defaultGoroutineLeakNodesRE = `.*`
)

// goroutineLeakConfig determines what gets scraped and whether findings fail
// the suite.
type goroutineLeakConfig struct {
	namespaces goroutineLeakRegexp
	nodes      goroutineLeakRegexp

	// pods enables scraping kube-controller-manager and kube-scheduler
	// through the pod proxy. This is opt-in because in a default kubeadm
	// cluster those components bind to 127.0.0.1 and their delegated
	// authorizer rejects the identity the API server proxies with. A job
	// which enables this must also start them with --bind-address=0.0.0.0
	// and add the profile path to --authorization-always-allow-paths.
	pods bool

	// enforce turns findings into a suite failure. It defaults to false so
	// that the check can be rolled out in reporting-only mode first and
	// cannot introduce flakes into jobs which are not watching it.
	enforce bool
}

// goroutineLeakRegexp implements flag.Value for a regular expression.
type goroutineLeakRegexp struct {
	re *regexp.Regexp
}

var _ flag.Value = &goroutineLeakRegexp{}

func (r *goroutineLeakRegexp) String() string {
	if r.re == nil {
		return ""
	}
	return r.re.String()
}

func (r *goroutineLeakRegexp) Set(expr string) error {
	re, err := regexp.Compile(expr)
	if err != nil {
		// This already starts with "error parsing regexp" and the caller adds
		// the expression string, so no need to wrap the error here.
		return err
	}
	r.re = re
	return nil
}

var goroutineLeakChecks = goroutineLeakConfig{
	namespaces: goroutineLeakRegexp{re: regexp.MustCompile(defaultGoroutineLeakNamespacesRE)},
	nodes:      goroutineLeakRegexp{re: regexp.MustCompile(defaultGoroutineLeakNodesRE)},
}

// RegisterGoroutineLeakFlags adds command line flags for configuring the
// goroutine leak invariant. They have "goroutineleak" as prefix.
func RegisterGoroutineLeakFlags(fs *flag.FlagSet) {
	fs.Var(&goroutineLeakChecks.namespaces, "goroutineleak-namespaces-regexp",
		"pods in namespaces matching this regular expression get scraped when -goroutineleak-pods is set")
	fs.Var(&goroutineLeakChecks.nodes, "goroutineleak-nodes-regexp",
		"kubelets on nodes matching this regular expression get scraped")
	fs.BoolVar(&goroutineLeakChecks.pods, "goroutineleak-pods", false,
		"enables scraping kube-controller-manager and kube-scheduler through the pod proxy; requires a cluster configured for it")
	fs.BoolVar(&goroutineLeakChecks.enforce, "goroutineleak-enforce", false,
		"turns leaked goroutines into a suite failure instead of only reporting them")
}

var _ = framework.SIGDescribe("testing")("Invariant Goroutine Leaks", func() {
	// this test is a sentinel for selecting the report after suite logic
	//
	// this allows us to run it by default in most jobs, but it can be opted-out,
	// does not run when selecting Conformance, and it can be tagged Flaky
	// if we encounter issues with it
	ginkgo.It(invariantGoroutineLeaksLeafText, func() {})
})

var _ = ginkgo.ReportAfterSuite("Invariant Goroutine Leaks", func(ctx ginkgo.SpecContext, report ginkgo.Report) {
	// skip early if we are in dry-run mode and didn't really run any tests
	if report.SuiteConfig.DryRun {
		return
	}
	// check if we ran the 'should enable checking for goroutine leaks' "test"
	invariantsSelected := false
	for _, spec := range report.SpecReports {
		if spec.LeafNodeText == invariantGoroutineLeaksLeafText {
			invariantsSelected = spec.State.Is(ginkgotypes.SpecStatePassed)
			break
		}
	}
	// skip if the associated "test" was skipped
	if !invariantsSelected {
		return
	}
	// otherwise actually check invariants now
	checkInvariantGoroutineLeaks(ctx)
})

func checkInvariantGoroutineLeaks(ctx context.Context) {
	config, err := framework.LoadConfig()
	if err != nil {
		framework.Failf("error loading client config: %v", err)
	}
	c, err := clientset.NewForConfig(config)
	if err != nil {
		framework.Failf("error loading client config: %v", err)
	}

	results := []goroutineleak.Result{goroutineleak.CheckAPIServer(ctx, c)}
	results = append(results, goroutineleak.CheckKubelets(ctx, c, goroutineLeakChecks.nodes.re)...)
	if goroutineLeakChecks.pods {
		results = append(results, goroutineleak.CheckPods(ctx, c, goroutineLeakChecks.namespaces.re)...)
	}

	// Always report what was found, including components which reported no
	// leaks, so that a check which examined nothing is distinguishable from
	// one which passed.
	ginkgo.GinkgoWriter.Print(goroutineleak.Report(results))

	if !goroutineLeakChecks.enforce {
		return
	}
	if failure := goroutineleak.Failure(results); failure != "" {
		framework.Failf("%s", failure)
	}
}

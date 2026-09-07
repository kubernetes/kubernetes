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
	"net"
	"regexp"

	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/invariants/goroutineleak"

	"github.com/onsi/ginkgo/v2"
	ginkgotypes "github.com/onsi/ginkgo/v2/types"
)

// checks for goroutines which the Go runtime has determined can never be
// unblocked, in other words permanently leaked goroutines
const invariantGoroutineLeaksLeafText = "should enable checking for goroutine leaks"

// defaultGoroutineLeakNodesRE is the default regular expression for which
// nodes should have their kubelet checked.
const defaultGoroutineLeakNodesRE = `.*`

// goroutineLeakCheck determines what gets checked.
type goroutineLeakCheck struct {
	// nodes contains a regular expression which determines which nodes have
	// their kubelet checked.
	nodes goroutineLeakRegexp
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

var enabledGoroutineLeakCheck = goroutineLeakCheck{
	nodes: goroutineLeakRegexp{re: regexp.MustCompile(defaultGoroutineLeakNodesRE)},
}

// RegisterGoroutineLeakFlags adds command line flags for configuring the
// goroutine leak invariant to the given flag set. They have "goroutineleak"
// as prefix.
func RegisterGoroutineLeakFlags(fs *flag.FlagSet) {
	fs.Var(&enabledGoroutineLeakCheck.nodes, "goroutineleak-nodes-regexp",
		"all kubelets on nodes matching this regular expression get checked")
}

// podDialer adapts the e2e pod dialer to the interface used by the
// goroutineleak package, which stays free of e2e framework imports so that it
// can be tested and reused without a cluster.
type podDialer struct {
	dialer *e2epod.Dialer
}

func (d podDialer) DialPod(ctx context.Context, namespace, podName string, port int) (net.Conn, error) {
	return d.dialer.DialContainerPort(ctx, e2epod.Addr{
		Namespace: namespace,
		PodName:   podName,
		Port:      port,
	})
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
	results = append(results, goroutineleak.CheckKubelets(ctx, c, enabledGoroutineLeakCheck.nodes.re)...)
	results = append(results, checkControlPlanePods(ctx, c, config)...)

	// Report what was checked, including components which reported no leaks,
	// so that a check which examined nothing is distinguishable from one
	// which passed.
	ginkgo.GinkgoWriter.Print(goroutineleak.Report(results))

	if failure := goroutineleak.Failure(results); failure != "" {
		framework.Failf("%s", failure)
	}
}

func checkControlPlanePods(ctx context.Context, c clientset.Interface, config *restclient.Config) []goroutineleak.Result {
	dialer := podDialer{dialer: e2epod.NewDialer(c, config)}
	return goroutineleak.CheckControlPlanePods(ctx, c, config, dialer)
}

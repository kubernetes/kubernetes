package steps

import (
	"fmt"
	"os"
	"path"
	"strings"
	"testing"

	// "testing"
	// "k8s.io/kubernetes/test/e2e"
	"github.com/DATA-DOG/godog"
	"github.com/onsi/ginkgo"
	"github.com/onsi/ginkgo/config"
	"github.com/onsi/ginkgo/reporters"
	"k8s.io/klog"
	"k8s.io/kubernetes/test/e2e/framework"
)

// Suite is an interface that describes the parts of the godog.Suite interface
// used within the package
type Suite interface {
	Step(interface{}, interface{})
	// BeforeScenario(func(interface{}))
	// BeforeSuite(func())
	// AfterSuite(func())
	// AfterScenario(func(interface{}))
	// AfterScenario(func(interface{}))
}

var testingMain *testing.M

func existingTest(existingTestName string) error {
	// Our Gingo Focus needs to escape all brackets: [] with \
	config.GinkgoConfig.SkipString = ""
	quotedName := strings.ReplaceAll(
		strings.ReplaceAll(
			existingTestName,
			"[", "\\["),
		"[", "\\[")
	config.GinkgoConfig.FocusString = quotedName
	// os.Exit(m.Run())
	// Run tests through the Ginkgo runner with output to console + JUnit for Jenkins
	var r []ginkgo.Reporter
	if framework.TestContext.ReportDir != "" {
		// TODO: we should probably only be trying to create this directory once
		// rather than once-per-Ginkgo-node.
		if err := os.MkdirAll(framework.TestContext.ReportDir, 0755); err != nil {
			klog.Errorf("Failed creating report directory: %v", err)
		} else {
			r = append(r, reporters.NewJUnitReporter(path.Join(framework.TestContext.ReportDir, fmt.Sprintf("junit_%v%v.xml", framework.TestContext.ReportPrefix, existingTestName))))
		}
	}
	klog.Infof("Starting godog e2e run %q", framework.RunID)
	testingMain.Run()
	// Once we set FocusString, I think we can just run m.Run()
	// ginkgo.RunSpecsWithDefaultAndCustomReporters(
	// 	e2e.ginkoTest,
	// 	"GoDog driven Kubernetes e2e suite",
	// 	r)
	return godog.ErrPending
}

func iRunTheTest() error {
	return godog.ErrPending
}

func theExistingTestWillPass() error {
	return godog.ErrPending
}

func thisIsFine() error {
	return godog.ErrPending
}

func FirstStepsFeatureContext(s *godog.Suite, m *testing.M) {
	testingMain = m
	s.Step(`^existing test "([^"]*)"$`, existingTest)
	s.Step(`^I run the test$`, iRunTheTest)
	s.Step(`^the existing test will pass$`, theExistingTestWillPass)
	s.Step(`^this is fine$`, thisIsFine)
	// s.BeforeScenario(func(interface{})) {}
}

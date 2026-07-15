package internal

import (
	"io"
	"os"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2/internal/interrupt_handler"
	"github.com/onsi/ginkgo/v2/reporters"
	"github.com/onsi/ginkgo/v2/types"
)

// ForwardingOutputInterceptor wraps a real OutputInterceptor but forwards
// captured output to stdout in real-time. This allows stdout/stderr to be
// captured in test results while still showing output during test execution.
type ForwardingOutputInterceptor struct {
	interceptor OutputInterceptor
	stdoutClone *os.File
}

// NewForwardingOutputInterceptor creates an output interceptor that captures
// stdout/stderr while also forwarding it to a clone of stdout in real-time.
// The stdout clone is created using dupStdout() (platform-specific) before
// the interceptor redirects output, ensuring output is always forwarded to
// the original terminal.
func NewForwardingOutputInterceptor() *ForwardingOutputInterceptor {
	// Create a clone of stdout BEFORE the interceptor can redirect it.
	// This ensures we always have a handle to the original terminal for
	// forwarding output. The dupStdout() function is platform-specific.
	stdoutClone := dupStdout()
	if stdoutClone == nil {
		// If we can't dup stdout, fall back to NoopOutputInterceptor behavior
		return &ForwardingOutputInterceptor{
			interceptor: NoopOutputInterceptor{},
			stdoutClone: nil,
		}
	}

	return &ForwardingOutputInterceptor{
		interceptor: NewOutputInterceptor(),
		stdoutClone: stdoutClone,
	}
}

func (f *ForwardingOutputInterceptor) StartInterceptingOutput() {
	if f.stdoutClone != nil {
		f.interceptor.StartInterceptingOutputAndForwardTo(f.stdoutClone)
	} else {
		f.interceptor.StartInterceptingOutput()
	}
}

func (f *ForwardingOutputInterceptor) StartInterceptingOutputAndForwardTo(w io.Writer) {
	f.interceptor.StartInterceptingOutputAndForwardTo(w)
}

func (f *ForwardingOutputInterceptor) StopInterceptingAndReturnOutput() string {
	return f.interceptor.StopInterceptingAndReturnOutput()
}

func (f *ForwardingOutputInterceptor) PauseIntercepting() {
	f.interceptor.PauseIntercepting()
}

func (f *ForwardingOutputInterceptor) ResumeIntercepting() {
	f.interceptor.ResumeIntercepting()
}

func (f *ForwardingOutputInterceptor) Shutdown() {
	f.interceptor.Shutdown()
	if f.stdoutClone != nil {
		f.stdoutClone.Close()
		f.stdoutClone = nil
	}
}

type AnnotateFunc func(testName string, test types.TestSpec)

func (suite *Suite) SetAnnotateFn(fn AnnotateFunc) {
	suite.annotateFn = fn
}

func (suite *Suite) GetReport() types.Report {
	return suite.report
}

func (suite *Suite) WalkTests(fn AnnotateFunc) {
	if suite.phase != PhaseBuildTree {
		panic("cannot run before building the tree = call suite.BuildTree() first")
	}
	ApplyNestedFocusPolicyToTree(suite.tree)
	specs := GenerateSpecsFromTreeRoot(suite.tree)
	for _, spec := range specs {
		fn(spec.Text(), spec)
	}
}

func (suite *Suite) InPhaseBuildTree() bool {
	return suite.phase == PhaseBuildTree
}

func (suite *Suite) ClearBeforeAndAfterSuiteNodes() {
	// Don't build the tree multiple times, it results in multiple initing of tests
	if !suite.InPhaseBuildTree() {
		suite.BuildTree()
	}
	newNodes := Nodes{}
	for _, node := range suite.suiteNodes {
		if node.NodeType == types.NodeTypeBeforeSuite || node.NodeType == types.NodeTypeAfterSuite || node.NodeType == types.NodeTypeSynchronizedBeforeSuite || node.NodeType == types.NodeTypeSynchronizedAfterSuite {
			continue
		}
		newNodes = append(newNodes, node)
	}
	suite.suiteNodes = newNodes
}

func (suite *Suite) RunSpec(spec types.TestSpec, suiteLabels Labels, suiteDescription, suitePath string, failer *Failer, writer WriterInterface, suiteConfig types.SuiteConfig, reporterConfig types.ReporterConfig) (bool, bool) {
	if suite.phase != PhaseBuildTree {
		panic("cannot run before building the tree = call suite.BuildTree() first")
	}

	suite.phase = PhaseRun
	suite.client = nil
	suite.failer = failer
	suite.reporter = reporters.NewDefaultReporter(reporterConfig, writer)
	suite.writer = writer
	if strings.ToLower(suiteConfig.OutputInterceptorMode) == "none" {
		suite.outputInterceptor = NoopOutputInterceptor{}
	} else {
		suite.outputInterceptor = NewForwardingOutputInterceptor()
	}
	if suite.config.Timeout > 0 {
		suite.deadline = time.Now().Add(suiteConfig.Timeout)
	}
	suite.interruptHandler = interrupt_handler.NewInterruptHandler(nil)
	suite.config = suiteConfig

	success := suite.runSpecs(suiteDescription, suiteLabels, SemVerConstraints{}, ComponentSemVerConstraints{}, suitePath, false, []Spec{spec.(Spec)})

	return success, false
}

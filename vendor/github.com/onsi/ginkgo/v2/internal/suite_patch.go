package internal

import (
	"github.com/onsi/ginkgo/v2/internal/interrupt_handler"
	"github.com/onsi/ginkgo/v2/reporters"
	"github.com/onsi/ginkgo/v2/types"
)

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

func (suite *Suite) ClearBeforeAndAfterSuiteNodes() {
	suite.BuildTree()
	newNodes := Nodes{}
	for _, node := range suite.suiteNodes {
		if node.NodeType == types.NodeTypeBeforeSuite || node.NodeType == types.NodeTypeAfterSuite || node.NodeType == types.NodeTypeSynchronizedBeforeSuite || node.NodeType == types.NodeTypeSynchronizedAfterSuite {
			continue
		}
		newNodes = append(newNodes, node)
	}
	suite.suiteNodes = newNodes
}

func (suite *Suite) RunSpec(spec types.TestSpec, suiteLabels Labels, suitePath string, failer *Failer, writer WriterInterface, suiteConfig types.SuiteConfig) (bool, bool) {
	if suite.phase != PhaseBuildTree {
		panic("cannot run before building the tree = call suite.BuildTree() first")
	}

	suite.phase = PhaseRun
	suite.client = nil
	suite.failer = failer
	suite.reporter = reporters.NoopReporter{}
	suite.writer = writer
	suite.outputInterceptor = NoopOutputInterceptor{}
	suite.interruptHandler = interrupt_handler.NewInterruptHandler(suiteConfig.Timeout, nil)
	suite.config = suiteConfig

	success := suite.runSpecs("", suiteLabels, suitePath, false, []Spec{spec.(Spec)})

	return success, false
}

package suite

import (
	"math/rand"
	"time"

	"github.com/onsi/ginkgo/config"
	"github.com/onsi/ginkgo/internal/containernode"
	"github.com/onsi/ginkgo/internal/failer"
	"github.com/onsi/ginkgo/internal/leafnodes"
	"github.com/onsi/ginkgo/internal/spec"
	"github.com/onsi/ginkgo/internal/specrunner"
	"github.com/onsi/ginkgo/internal/writer"
	"github.com/onsi/ginkgo/reporters"
	"github.com/onsi/ginkgo/types"
)

type ginkgoTestingT interface {
	Fail()
}

type Suite struct {
	topLevelContainer *containernode.ContainerNode
	currentContainer  *containernode.ContainerNode
	containerIndex    int
	beforeSuiteNode   leafnodes.SuiteNode
	afterSuiteNode    leafnodes.SuiteNode
	runner            *specrunner.SpecRunner
	failer            *failer.Failer
	running           bool
}

func New(failer *failer.Failer) *Suite {
	topLevelContainer := containernode.New("[Top Level]", types.FlagTypeNone, types.CodeLocation{})

	return &Suite{
		topLevelContainer: topLevelContainer,
		currentContainer:  topLevelContainer,
		failer:            failer,
		containerIndex:    1,
	}
}

func (suite *Suite) Run(t ginkgoTestingT, description string, reporters []reporters.Reporter, writer writer.WriterInterface, config config.GinkgoConfigType) (bool, bool) {
	if config.ParallelTotal < 1 {
		panic("ginkgo.parallel.total must be >= 1")
	}

	if config.ParallelNode > config.ParallelTotal || config.ParallelNode < 1 {
		panic("ginkgo.parallel.node is one-indexed and must be <= ginkgo.parallel.total")
	}

	r := rand.New(rand.NewSource(config.RandomSeed))
	suite.topLevelContainer.Shuffle(r)
	specs := suite.generateSpecs(description, config)
	suite.runner = specrunner.New(description, suite.beforeSuiteNode, specs, suite.afterSuiteNode, reporters, writer, config)

	suite.running = true
	success := suite.runner.Run()
	if !success {
		t.Fail()
	}
	return success, specs.HasProgrammaticFocus()
}

func (suite *Suite) generateSpecs(description string, config config.GinkgoConfigType) *spec.Specs {
	specsSlice := []*spec.Spec{}
	suite.topLevelContainer.BackPropagateProgrammaticFocus()
	for _, collatedNodes := range suite.topLevelContainer.Collate() {
		specsSlice = append(specsSlice, spec.New(collatedNodes.Subject, collatedNodes.Containers, config.EmitSpecProgress))
	}

	specs := spec.NewSpecs(specsSlice)

	if config.RandomizeAllSpecs {
		specs.Shuffle(rand.New(rand.NewSource(config.RandomSeed)))
	}

	specs.ApplyFocus(description, config.FocusString, config.SkipString)

	if config.SkipMeasurements {
		specs.SkipMeasurements()
	}

	if config.ParallelTotal > 1 {
		specs.TrimForParallelization(config.ParallelTotal, config.ParallelNode)
	}

	return specs
}

func (suite *Suite) CurrentRunningSpecSummary() (*types.SpecSummary, bool) {
	return suite.runner.CurrentSpecSummary()
}

func (suite *Suite) SetBeforeSuiteNode(body interface{}, codeLocation types.CodeLocation, timeout time.Duration) {
	if suite.beforeSuiteNode != nil {
		panic("You may only call BeforeSuite once!")
	}
	suite.beforeSuiteNode = leafnodes.NewBeforeSuiteNode(body, codeLocation, timeout, suite.failer)
}

func (suite *Suite) SetAfterSuiteNode(body interface{}, codeLocation types.CodeLocation, timeout time.Duration) {
	if suite.afterSuiteNode != nil {
		panic("You may only call AfterSuite once!")
	}
	suite.afterSuiteNode = leafnodes.NewAfterSuiteNode(body, codeLocation, timeout, suite.failer)
}

func (suite *Suite) SetSynchronizedBeforeSuiteNode(bodyA interface{}, bodyB interface{}, codeLocation types.CodeLocation, timeout time.Duration) {
	if suite.beforeSuiteNode != nil {
		panic("You may only call BeforeSuite once!")
	}
	suite.beforeSuiteNode = leafnodes.NewSynchronizedBeforeSuiteNode(bodyA, bodyB, codeLocation, timeout, suite.failer)
}

func (suite *Suite) SetSynchronizedAfterSuiteNode(bodyA interface{}, bodyB interface{}, codeLocation types.CodeLocation, timeout time.Duration) {
	if suite.afterSuiteNode != nil {
		panic("You may only call AfterSuite once!")
	}
	suite.afterSuiteNode = leafnodes.NewSynchronizedAfterSuiteNode(bodyA, bodyB, codeLocation, timeout, suite.failer)
}

func (suite *Suite) PushContainerNode(text string, body func(), flag types.FlagType, codeLocation types.CodeLocation) {
	container := containernode.New(text, flag, codeLocation)
	suite.currentContainer.PushContainerNode(container)

	previousContainer := suite.currentContainer
	suite.currentContainer = container
	suite.containerIndex++

	body()

	suite.containerIndex--
	suite.currentContainer = previousContainer
}

func (suite *Suite) PushItNode(text string, body interface{}, flag types.FlagType, codeLocation types.CodeLocation, timeout time.Duration) {
	if suite.running {
		suite.failer.Fail("You may only call It from within a Describe or Context", codeLocation)
	}
	suite.currentContainer.PushSubjectNode(leafnodes.NewItNode(text, body, flag, codeLocation, timeout, suite.failer, suite.containerIndex))
}

func (suite *Suite) PushMeasureNode(text string, body interface{}, flag types.FlagType, codeLocation types.CodeLocation, samples int) {
	if suite.running {
		suite.failer.Fail("You may only call Measure from within a Describe or Context", codeLocation)
	}
	suite.currentContainer.PushSubjectNode(leafnodes.NewMeasureNode(text, body, flag, codeLocation, samples, suite.failer, suite.containerIndex))
}

func (suite *Suite) PushBeforeEachNode(body interface{}, codeLocation types.CodeLocation, timeout time.Duration) {
	if suite.running {
		suite.failer.Fail("You may only call BeforeEach from within a Describe or Context", codeLocation)
	}
	suite.currentContainer.PushSetupNode(leafnodes.NewBeforeEachNode(body, codeLocation, timeout, suite.failer, suite.containerIndex))
}

func (suite *Suite) PushJustBeforeEachNode(body interface{}, codeLocation types.CodeLocation, timeout time.Duration) {
	if suite.running {
		suite.failer.Fail("You may only call JustBeforeEach from within a Describe or Context", codeLocation)
	}
	suite.currentContainer.PushSetupNode(leafnodes.NewJustBeforeEachNode(body, codeLocation, timeout, suite.failer, suite.containerIndex))
}

func (suite *Suite) PushAfterEachNode(body interface{}, codeLocation types.CodeLocation, timeout time.Duration) {
	if suite.running {
		suite.failer.Fail("You may only call AfterEach from within a Describe or Context", codeLocation)
	}
	suite.currentContainer.PushSetupNode(leafnodes.NewAfterEachNode(body, codeLocation, timeout, suite.failer, suite.containerIndex))
}

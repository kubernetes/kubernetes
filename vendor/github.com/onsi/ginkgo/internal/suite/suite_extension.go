package suite

import (
	"math/rand"
	"strings"

	"github.com/onsi/ginkgo/config"
	"github.com/onsi/ginkgo/internal/containernode"
	"github.com/onsi/ginkgo/internal/leafnodes"
	"github.com/onsi/ginkgo/internal/spec"
	"github.com/onsi/ginkgo/internal/spec_iterator"
	"github.com/onsi/ginkgo/types"
)

func (suite *Suite) Iterator(config config.GinkgoConfigType) spec_iterator.SpecIterator {
	specsSlice := []*spec.Spec{}
	for _, collatedNodes := range suite.topLevelContainer.Collate() {
		specsSlice = append(specsSlice, spec.New(collatedNodes.Subject, collatedNodes.Containers, config.EmitSpecProgress))
	}

	specs := spec.NewSpecs(specsSlice)

	if config.RandomizeAllSpecs {
		specs.Shuffle(rand.New(rand.NewSource(config.RandomSeed)))
	}

	if config.SkipMeasurements {
		specs.SkipMeasurements()
	}
	return spec_iterator.NewSerialIterator(specs.Specs())
}

func (suite *Suite) WalkTests(fn func(testName, parentName string, test types.TestNode)) {
	suite.topLevelContainer.BackPropagateProgrammaticFocus()
	for _, collatedNodes := range suite.topLevelContainer.Collate() {
		itNode, ok := collatedNodes.Subject.(*leafnodes.ItNode)
		if !ok {
			continue
		}
		fn(collatedNodes.Subject.Text(), containerName(collatedNodes.Containers), itNode)
	}
}

func containerName(containers []*containernode.ContainerNode) string {
	var b strings.Builder
	b.Grow(len(containers) * 20)
	for i, container := range containers {
		if i > 0 {
			b.WriteRune(' ')
		}
		b.WriteString(container.Text())
	}
	return b.String()
}

func (suite *Suite) ClearBeforeSuiteNode() {
	suite.beforeSuiteNode = nil
}

func (suite *Suite) ClearAfterSuiteNode() {
	suite.afterSuiteNode = nil
}

func (suite *Suite) GetBeforeSuiteNode() leafnodes.SuiteNode {
	return suite.beforeSuiteNode
}

func (suite *Suite) GetAfterSuiteNode() leafnodes.SuiteNode {
	return suite.afterSuiteNode
}

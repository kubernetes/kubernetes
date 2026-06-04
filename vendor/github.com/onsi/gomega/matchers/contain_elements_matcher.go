package matchers

import (
	"fmt"

	"github.com/onsi/gomega/format"
	"github.com/onsi/gomega/matchers/internal/miter"
	"github.com/onsi/gomega/matchers/support/goraph/bipartitegraph"
)

type ContainElementsMatcher struct {
	Elements        []any
	missingElements []any
}

func (matcher *ContainElementsMatcher) Match(actual any) (success bool, err error) {
	if !isArrayOrSlice(actual) && !isMap(actual) && !miter.IsIter(actual) {
		return false, fmt.Errorf("ContainElements matcher expects an array/slice/map/iter.Seq/iter.Seq2.  Got:\n%s", format.Object(actual, 1))
	}

	matchers := matchers(matcher.Elements)
	bipartiteGraph, err := bipartitegraph.NewBipartiteGraph(valuesOf(actual), matchers, neighbours)
	if err != nil {
		return false, err
	}

	edges := bipartiteGraph.LargestMatching()
	if len(edges) == len(matchers) {
		return true, nil
	}

	_, missingMatchers := bipartiteGraph.FreeLeftRight(edges)
	matcher.missingElements = equalMatchersToElements(missingMatchers)

	return false, nil
}

func (matcher *ContainElementsMatcher) FailureMessage(actual any) (message string) {
	message = format.Message(actual, "to contain elements", presentable(matcher.Elements))
	return appendMissingElements(message, matcher.missingElements)
}

func (matcher *ContainElementsMatcher) NegatedFailureMessage(actual any) (message string) {
	return format.Message(actual, "not to contain elements", presentable(matcher.Elements))
}

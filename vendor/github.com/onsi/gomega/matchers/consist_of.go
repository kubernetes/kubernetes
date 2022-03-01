// untested sections: 3

package matchers

import (
	"fmt"
	"reflect"

	"github.com/onsi/gomega/format"
	"github.com/onsi/gomega/matchers/support/goraph/bipartitegraph"
)

type ConsistOfMatcher struct {
	Elements        []interface{}
	missingElements []interface{}
	extraElements   []interface{}
}

func (matcher *ConsistOfMatcher) Match(actual interface{}) (success bool, err error) {
	if !isArrayOrSlice(actual) && !isMap(actual) {
		return false, fmt.Errorf("ConsistOf matcher expects an array/slice/map.  Got:\n%s", format.Object(actual, 1))
	}

	matchers := matchers(matcher.Elements)
	values := valuesOf(actual)

	bipartiteGraph, err := bipartitegraph.NewBipartiteGraph(values, matchers, neighbours)
	if err != nil {
		return false, err
	}

	edges := bipartiteGraph.LargestMatching()
	if len(edges) == len(values) && len(edges) == len(matchers) {
		return true, nil
	}

	var missingMatchers []interface{}
	matcher.extraElements, missingMatchers = bipartiteGraph.FreeLeftRight(edges)
	matcher.missingElements = equalMatchersToElements(missingMatchers)

	return false, nil
}

func neighbours(value, matcher interface{}) (bool, error) {
	match, err := matcher.(omegaMatcher).Match(value)
	return match && err == nil, nil
}

func equalMatchersToElements(matchers []interface{}) (elements []interface{}) {
	for _, matcher := range matchers {
		equalMatcher, ok := matcher.(*EqualMatcher)
		if ok {
			matcher = equalMatcher.Expected
		}
		elements = append(elements, matcher)
	}
	return
}

func matchers(expectedElems []interface{}) (matchers []interface{}) {
	elems := expectedElems
	if len(expectedElems) == 1 && isArrayOrSlice(expectedElems[0]) {
		elems = []interface{}{}
		value := reflect.ValueOf(expectedElems[0])
		for i := 0; i < value.Len(); i++ {
			elems = append(elems, value.Index(i).Interface())
		}
	}

	for _, e := range elems {
		matcher, isMatcher := e.(omegaMatcher)
		if !isMatcher {
			matcher = &EqualMatcher{Expected: e}
		}
		matchers = append(matchers, matcher)
	}
	return
}

func valuesOf(actual interface{}) []interface{} {
	value := reflect.ValueOf(actual)
	values := []interface{}{}
	if isMap(actual) {
		keys := value.MapKeys()
		for i := 0; i < value.Len(); i++ {
			values = append(values, value.MapIndex(keys[i]).Interface())
		}
	} else {
		for i := 0; i < value.Len(); i++ {
			values = append(values, value.Index(i).Interface())
		}
	}

	return values
}

func (matcher *ConsistOfMatcher) FailureMessage(actual interface{}) (message string) {
	message = format.Message(actual, "to consist of", matcher.Elements)
	message = appendMissingElements(message, matcher.missingElements)
	if len(matcher.extraElements) > 0 {
		message = fmt.Sprintf("%s\nthe extra elements were\n%s", message,
			format.Object(matcher.extraElements, 1))
	}
	return
}

func appendMissingElements(message string, missingElements []interface{}) string {
	if len(missingElements) == 0 {
		return message
	}
	return fmt.Sprintf("%s\nthe missing elements were\n%s", message,
		format.Object(missingElements, 1))
}

func (matcher *ConsistOfMatcher) NegatedFailureMessage(actual interface{}) (message string) {
	return format.Message(actual, "not to consist of", matcher.Elements)
}

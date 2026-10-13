package matchers

import (
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/onsi/gomega/format"
	"go.yaml.in/yaml/v3"
)

type MatchYAMLMatcher struct {
	YAMLToMatch      any
	firstFailurePath []any
	// the 1-based number of the first mismatched document when comparing
	// multi-document streams; 0 otherwise
	firstFailureDocument int
}

func (matcher *MatchYAMLMatcher) Match(actual any) (success bool, err error) {
	matcher.firstFailurePath = nil
	matcher.firstFailureDocument = 0

	actualString, expectedString, err := matcher.toStrings(actual)
	if err != nil {
		return false, err
	}

	adocs, err := decodeYAMLDocuments(actualString)
	if err != nil {
		return false, fmt.Errorf("Actual '%s' should be valid YAML, but it is not.\nUnderlying error:%s", actualString, err)
	}
	edocs, err := decodeYAMLDocuments(expectedString)
	if err != nil {
		return false, fmt.Errorf("Expected '%s' should be valid YAML, but it is not.\nUnderlying error:%s", expectedString, err)
	}

	if len(adocs) != len(edocs) {
		return false, nil
	}
	for i := range adocs {
		var equal bool
		equal, matcher.firstFailurePath = deepEqual(adocs[i], edocs[i])
		if !equal {
			if len(adocs) > 1 {
				matcher.firstFailureDocument = i + 1
			}
			return false, nil
		}
	}
	return true, nil
}

func (matcher *MatchYAMLMatcher) FailureMessage(actual any) (message string) {
	actualString, expectedString, _ := matcher.toNormalisedStrings(actual)
	return matcher.formattedMessage(format.Message(actualString, "to match YAML of", expectedString))
}

func (matcher *MatchYAMLMatcher) NegatedFailureMessage(actual any) (message string) {
	actualString, expectedString, _ := matcher.toNormalisedStrings(actual)
	return matcher.formattedMessage(format.Message(actualString, "not to match YAML of", expectedString))
}

func (matcher *MatchYAMLMatcher) formattedMessage(comparisonMessage string) string {
	if matcher.firstFailureDocument > 0 {
		comparisonMessage = fmt.Sprintf("%s\n\nfirst mismatched document: %d (counting from 1)", comparisonMessage, matcher.firstFailureDocument)
	}
	return formattedMessage(comparisonMessage, matcher.firstFailurePath)
}

func (matcher *MatchYAMLMatcher) toNormalisedStrings(actual any) (actualFormatted, expectedFormatted string, err error) {
	actualString, expectedString, err := matcher.toStrings(actual)
	return normalise(actualString), normalise(expectedString), err
}

func normalise(input string) string {
	docs, err := decodeYAMLDocuments(input)
	if err != nil {
		panic(err) // unreachable since Match already decodes the input
	}
	outputs := make([]string, len(docs))
	for i, doc := range docs {
		output, err := yaml.Marshal(doc)
		if err != nil {
			panic(err) // untested section, unreachable since we decode above
		}
		outputs[i] = string(output)
	}
	return strings.TrimSpace(strings.Join(outputs, "---\n"))
}

// decodeYAMLDocuments decodes every document in a YAML stream.
//
// Empty documents - such as those produced by a leading or trailing "---" - are
// skipped, so "---\na: 1\n---\n" is the single document "a: 1".  A stream with
// no documents at all is treated as a single null document, as yaml.Unmarshal
// does.
func decodeYAMLDocuments(input string) ([]any, error) {
	docs := []any{}
	decoder := yaml.NewDecoder(strings.NewReader(input))
	for {
		var node yaml.Node
		err := decoder.Decode(&node)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, err
		}
		if isEmptyYAMLDocument(&node) {
			continue
		}
		var doc any
		if err := node.Decode(&doc); err != nil {
			return nil, err
		}
		docs = append(docs, doc)
	}
	if len(docs) == 0 {
		docs = append(docs, nil)
	}
	return docs, nil
}

// isEmptyYAMLDocument reports whether a document has no content at all.
// Explicit nulls (e.g. "~" or "null") are content.
func isEmptyYAMLDocument(document *yaml.Node) bool {
	if len(document.Content) != 1 {
		return false
	}
	content := document.Content[0]
	return content.Kind == yaml.ScalarNode && content.ShortTag() == "!!null" && content.Value == "" &&
		content.Style == 0 && content.Anchor == ""
}

func (matcher *MatchYAMLMatcher) toStrings(actual any) (actualFormatted, expectedFormatted string, err error) {
	actualString, ok := toString(actual)
	if !ok {
		return "", "", fmt.Errorf("MatchYAMLMatcher matcher requires a string, stringer, or []byte.  Got actual:\n%s", format.Object(actual, 1))
	}
	expectedString, ok := toString(matcher.YAMLToMatch)
	if !ok {
		return "", "", fmt.Errorf("MatchYAMLMatcher matcher requires a string, stringer, or []byte.  Got expected:\n%s", format.Object(matcher.YAMLToMatch, 1))
	}

	return actualString, expectedString, nil
}

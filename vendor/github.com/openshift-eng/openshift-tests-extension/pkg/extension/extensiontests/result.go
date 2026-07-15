package extensiontests

import (
	"bytes"
	_ "embed"
	"encoding/json"
	"fmt"
	"strings"
	"text/template"

	"github.com/openshift-eng/openshift-tests-extension/pkg/junit"
)

func (results ExtensionTestResults) Walk(walkFn func(*ExtensionTestResult)) {
	for i := range results {
		walkFn(results[i])
	}
}

// AddDetails adds additional information to an ExtensionTestResult. Value must marshal to JSON.
func (result *ExtensionTestResult) AddDetails(name string, value interface{}) {
	result.Details = append(result.Details, Details{Name: name, Value: value})
}

func (result ExtensionTestResult) ToJUnit() *junit.TestCase {
	tc := &junit.TestCase{
		Name:     result.Name,
		Duration: float64(result.Duration) / 1000.0,
	}
	switch result.Result {
	case ResultFailed:
		tc.FailureOutput = &junit.FailureOutput{
			Message: result.Error,
			Output:  result.Error,
		}
	case ResultSkipped:
		messages := []string{}
		for _, detail := range result.Details {
			messages = append(messages, fmt.Sprintf("%s: %s", detail.Name, detail.Value))
		}
		tc.SkipMessage = &junit.SkipMessage{
			Message: strings.Join(messages, "\n"),
		}
	case ResultPassed:
		tc.SystemOut = result.Output
	}

	return tc
}

func (results ExtensionTestResults) ToJUnit(suiteName string) junit.TestSuite {
	suite := junit.TestSuite{
		Name: suiteName,
	}

	results.Walk(func(result *ExtensionTestResult) {
		suite.NumTests++
		switch result.Result {
		case ResultFailed:
			suite.NumFailed++
		case ResultSkipped:
			suite.NumSkipped++
		case ResultPassed:
			// do nothing
		default:
			panic(fmt.Sprintf("unknown result type: %s", result.Result))
		}

		suite.TestCases = append(suite.TestCases, result.ToJUnit())
	})

	return suite
}

//go:embed viewer.html
var viewerHtml []byte

// RenderResultsHTML renders the HTML viewer template with the provided JSON data.
// The caller is responsible for marshaling their results to JSON. This allows
// callers with different result struct types to use the same HTML viewer.
func RenderResultsHTML(jsonData []byte, suiteName string) ([]byte, error) {
	tmpl, err := template.New("viewer").Parse(string(viewerHtml))
	if err != nil {
		return nil, fmt.Errorf("failed to parse template: %w", err)
	}
	var out bytes.Buffer
	if err := tmpl.Execute(&out, struct {
		Data      string
		SuiteName string
	}{
		string(jsonData),
		suiteName,
	}); err != nil {
		return nil, fmt.Errorf("failed to execute template: %w", err)
	}
	return out.Bytes(), nil
}

func (results ExtensionTestResults) ToHTML(suiteName string) ([]byte, error) {
	encoded, err := json.Marshal(results)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal extension test results: %w", err)
	}
	// pare down the output if there's a lot, we want this to load in some reasonable amount of time
	if len(encoded) > 2<<20 {
		// n.b. this is wasteful, but we want to mutate our inputs in a safe manner, so the encode/decode/encode
		// pass is useful as a deep copy
		var copiedResults ExtensionTestResults
		if err := json.Unmarshal(encoded, &copiedResults); err != nil {
			return nil, fmt.Errorf("failed to unmarshal extension test results: %w", err)
		}
		copiedResults.Walk(func(result *ExtensionTestResult) {
			if result.Result == ResultPassed {
				result.Error = ""
				result.Output = ""
				result.Details = nil
			}
		})
		encoded, err = json.Marshal(copiedResults)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal extension test results: %w", err)
		}
	}
	return RenderResultsHTML(encoded, suiteName)
}

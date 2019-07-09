package godog

import "github.com/DATA-DOG/godog/gherkin"

// examples is a helper func to cast gherkin.Examples
// or gherkin.BaseExamples if its empty
// @TODO: this should go away with gherkin update
func examples(ex interface{}) (*gherkin.Examples, bool) {
	t, ok := ex.(*gherkin.Examples)
	return t, ok
}

// means there are no scenarios or they do not have steps
func isEmptyFeature(ft *gherkin.Feature) bool {
	for _, def := range ft.ScenarioDefinitions {
		if !isEmptyScenario(def) {
			return false
		}
	}
	return true
}

// means scenario dooes not have steps
func isEmptyScenario(def interface{}) bool {
	switch t := def.(type) {
	case *gherkin.Scenario:
		if len(t.Steps) > 0 {
			return false
		}
	case *gherkin.ScenarioOutline:
		if len(t.Steps) > 0 {
			return false
		}
	}
	return true
}

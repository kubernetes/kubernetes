package godog

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"path/filepath"
	"reflect"
	"regexp"
	"strconv"
	"strings"

	"github.com/DATA-DOG/godog/gherkin"
)

// SuiteContext provides steps for godog suite execution and
// can be used for meta-testing of godog features/steps themselves.
//
// Beware, steps or their definitions might change without backward
// compatibility guarantees. A typical user of the godog library should never
// need this, rather it is provided for those developing add-on libraries for godog.
//
// For an example of how to use, see godog's own `features/` and `suite_test.go`.
func SuiteContext(s *Suite, additionalContextInitializers ...func(suite *Suite)) {
	c := &suiteContext{
		extraCIs: additionalContextInitializers,
	}

	// apply any additional context intializers to modify the context that the
	// meta-tests will be run in
	for _, ci := range additionalContextInitializers {
		ci(s)
	}

	s.BeforeScenario(c.ResetBeforeEachScenario)

	s.Step(`^(?:a )?feature path "([^"]*)"$`, c.featurePath)
	s.Step(`^I parse features$`, c.parseFeatures)
	s.Step(`^I'm listening to suite events$`, c.iAmListeningToSuiteEvents)
	s.Step(`^I run feature suite$`, c.iRunFeatureSuite)
	s.Step(`^I run feature suite with tags "([^"]*)"$`, c.iRunFeatureSuiteWithTags)
	s.Step(`^I run feature suite with formatter "([^"]*)"$`, c.iRunFeatureSuiteWithFormatter)
	s.Step(`^(?:a )?feature "([^"]*)"(?: file)?:$`, c.aFeatureFile)
	s.Step(`^the suite should have (passed|failed)$`, c.theSuiteShouldHave)

	s.Step(`^I should have ([\d]+) features? files?:$`, c.iShouldHaveNumFeatureFiles)
	s.Step(`^I should have ([\d]+) scenarios? registered$`, c.numScenariosRegistered)
	s.Step(`^there (was|were) ([\d]+) "([^"]*)" events? fired$`, c.thereWereNumEventsFired)
	s.Step(`^there was event triggered before scenario "([^"]*)"$`, c.thereWasEventTriggeredBeforeScenario)
	s.Step(`^these events had to be fired for a number of times:$`, c.theseEventsHadToBeFiredForNumberOfTimes)

	s.Step(`^(?:a )?failing step`, c.aFailingStep)
	s.Step(`^this step should fail`, c.aFailingStep)
	s.Step(`^the following steps? should be (passed|failed|skipped|undefined|pending):`, c.followingStepsShouldHave)
	s.Step(`^all steps should (?:be|have|have been) (passed|failed|skipped|undefined|pending)$`, c.allStepsShouldHave)
	s.Step(`^the undefined step snippets should be:$`, c.theUndefinedStepSnippetsShouldBe)

	// event stream
	s.Step(`^the following events should be fired:$`, c.thereShouldBeEventsFired)

	// lt
	s.Step(`^savybių aplankas "([^"]*)"$`, c.featurePath)
	s.Step(`^aš išskaitau savybes$`, c.parseFeatures)
	s.Step(`^aš turėčiau turėti ([\d]+) savybių failus:$`, c.iShouldHaveNumFeatureFiles)

	s.Step(`^(?:a )?pending step$`, func() error {
		return ErrPending
	})
	s.Step(`^(?:a )?passing step$`, func() error {
		return nil
	})

	// Introduced to test formatter/cucumber.feature
	s.Step(`^the rendered json will be as follows:$`, c.theRenderJSONWillBe)

	s.Step(`^(?:a )?failing multistep$`, func() Steps {
		return Steps{"passing step", "failing step"}
	})

	s.Step(`^(?:a |an )?undefined multistep$`, func() Steps {
		return Steps{"passing step", "undefined step", "passing step"}
	})

	s.Step(`^(?:a )?passing multistep$`, func() Steps {
		return Steps{"passing step", "passing step", "passing step"}
	})

	s.Step(`^(?:a )?failing nested multistep$`, func() Steps {
		return Steps{"passing step", "passing multistep", "failing multistep"}
	})
}

type firedEvent struct {
	name string
	args []interface{}
}

type suiteContext struct {
	paths       []string
	testedSuite *Suite
	extraCIs    []func(suite *Suite)
	events      []*firedEvent
	out         bytes.Buffer
}

func (s *suiteContext) ResetBeforeEachScenario(interface{}) {
	// reset whole suite with the state
	s.out.Reset()
	s.paths = []string{}
	s.testedSuite = &Suite{}
	// our tested suite will have the same context registered
	SuiteContext(s.testedSuite, s.extraCIs...)
	// reset all fired events
	s.events = []*firedEvent{}
}

func (s *suiteContext) iRunFeatureSuiteWithTags(tags string) error {
	if err := s.parseFeatures(); err != nil {
		return err
	}
	for _, feat := range s.testedSuite.features {
		applyTagFilter(tags, feat.Feature)
	}
	s.testedSuite.fmt = testFormatterFunc("godog", &s.out)
	s.testedSuite.run()
	s.testedSuite.fmt.Summary()
	return nil
}

func (s *suiteContext) iRunFeatureSuiteWithFormatter(name string) error {
	f := FindFmt(name)
	if f == nil {
		return fmt.Errorf(`formatter "%s" is not available`, name)
	}
	s.testedSuite.fmt = f("godog", &s.out)
	if err := s.parseFeatures(); err != nil {
		return err
	}
	s.testedSuite.run()
	s.testedSuite.fmt.Summary()
	return nil
}

func (s *suiteContext) thereShouldBeEventsFired(doc *gherkin.DocString) error {
	actual := strings.Split(strings.TrimSpace(s.out.String()), "\n")
	expect := strings.Split(strings.TrimSpace(doc.Content), "\n")
	if len(expect) != len(actual) {
		return fmt.Errorf("expected %d events, but got %d", len(expect), len(actual))
	}

	type ev struct {
		Event string
	}

	for i, event := range actual {
		exp := strings.TrimSpace(expect[i])
		var act ev
		if err := json.Unmarshal([]byte(event), &act); err != nil {
			return fmt.Errorf("failed to read event data: %v", err)
		}

		if act.Event != exp {
			return fmt.Errorf(`expected event: "%s" at position: %d, but actual was "%s"`, exp, i, act.Event)
		}
	}
	return nil
}

func (s *suiteContext) cleanupSnippet(snip string) string {
	lines := strings.Split(strings.TrimSpace(snip), "\n")
	for i := 0; i < len(lines); i++ {
		lines[i] = strings.TrimSpace(lines[i])
	}
	return strings.Join(lines, "\n")
}

func (s *suiteContext) theUndefinedStepSnippetsShouldBe(body *gherkin.DocString) error {
	f, ok := s.testedSuite.fmt.(*testFormatter)
	if !ok {
		return fmt.Errorf("this step requires testFormatter, but there is: %T", s.testedSuite.fmt)
	}
	actual := s.cleanupSnippet(f.snippets())
	expected := s.cleanupSnippet(body.Content)
	if actual != expected {
		return fmt.Errorf("snippets do not match actual: %s", f.snippets())
	}
	return nil
}

func (s *suiteContext) followingStepsShouldHave(status string, steps *gherkin.DocString) error {
	var expected = strings.Split(steps.Content, "\n")
	var actual, unmatched, matched []string

	f, ok := s.testedSuite.fmt.(*testFormatter)
	if !ok {
		return fmt.Errorf("this step requires testFormatter, but there is: %T", s.testedSuite.fmt)
	}
	switch status {
	case "passed":
		for _, st := range f.passed {
			actual = append(actual, st.step.Text)
		}
	case "failed":
		for _, st := range f.failed {
			actual = append(actual, st.step.Text)
		}
	case "skipped":
		for _, st := range f.skipped {
			actual = append(actual, st.step.Text)
		}
	case "undefined":
		for _, st := range f.undefined {
			actual = append(actual, st.step.Text)
		}
	case "pending":
		for _, st := range f.pending {
			actual = append(actual, st.step.Text)
		}
	default:
		return fmt.Errorf("unexpected step status wanted: %s", status)
	}

	if len(expected) > len(actual) {
		return fmt.Errorf("number of expected %s steps: %d is less than actual %s steps: %d", status, len(expected), status, len(actual))
	}

	for _, a := range actual {
		for _, e := range expected {
			if a == e {
				matched = append(matched, e)
				break
			}
		}
	}

	if len(matched) >= len(expected) {
		return nil
	}
	for _, s := range expected {
		var found bool
		for _, m := range matched {
			if s == m {
				found = true
				break
			}
		}
		if !found {
			unmatched = append(unmatched, s)
		}
	}

	return fmt.Errorf("the steps: %s - are not %s", strings.Join(unmatched, ", "), status)
}

func (s *suiteContext) allStepsShouldHave(status string) error {
	f, ok := s.testedSuite.fmt.(*testFormatter)
	if !ok {
		return fmt.Errorf("this step requires testFormatter, but there is: %T", s.testedSuite.fmt)
	}

	total := len(f.passed) + len(f.failed) + len(f.skipped) + len(f.undefined) + len(f.pending)
	var actual int
	switch status {
	case "passed":
		actual = len(f.passed)
	case "failed":
		actual = len(f.failed)
	case "skipped":
		actual = len(f.skipped)
	case "undefined":
		actual = len(f.undefined)
	case "pending":
		actual = len(f.pending)
	default:
		return fmt.Errorf("unexpected step status wanted: %s", status)
	}

	if total > actual {
		return fmt.Errorf("number of expected %s steps: %d is less than actual %s steps: %d", status, total, status, actual)
	}
	return nil
}

func (s *suiteContext) iAmListeningToSuiteEvents() error {
	s.testedSuite.BeforeSuite(func() {
		s.events = append(s.events, &firedEvent{"BeforeSuite", []interface{}{}})
	})
	s.testedSuite.AfterSuite(func() {
		s.events = append(s.events, &firedEvent{"AfterSuite", []interface{}{}})
	})
	s.testedSuite.BeforeFeature(func(ft *gherkin.Feature) {
		s.events = append(s.events, &firedEvent{"BeforeFeature", []interface{}{ft}})
	})
	s.testedSuite.AfterFeature(func(ft *gherkin.Feature) {
		s.events = append(s.events, &firedEvent{"AfterFeature", []interface{}{ft}})
	})
	s.testedSuite.BeforeScenario(func(scenario interface{}) {
		s.events = append(s.events, &firedEvent{"BeforeScenario", []interface{}{scenario}})
	})
	s.testedSuite.AfterScenario(func(scenario interface{}, err error) {
		s.events = append(s.events, &firedEvent{"AfterScenario", []interface{}{scenario, err}})
	})
	s.testedSuite.BeforeStep(func(step *gherkin.Step) {
		s.events = append(s.events, &firedEvent{"BeforeStep", []interface{}{step}})
	})
	s.testedSuite.AfterStep(func(step *gherkin.Step, err error) {
		s.events = append(s.events, &firedEvent{"AfterStep", []interface{}{step, err}})
	})
	return nil
}

func (s *suiteContext) aFailingStep() error {
	return fmt.Errorf("intentional failure")
}

// parse a given feature file body as a feature
func (s *suiteContext) aFeatureFile(name string, body *gherkin.DocString) error {
	ft, err := gherkin.ParseFeature(strings.NewReader(body.Content))
	s.testedSuite.features = append(s.testedSuite.features, &feature{Feature: ft, Path: name})
	return err
}

func (s *suiteContext) featurePath(path string) error {
	s.paths = append(s.paths, path)
	return nil
}

func (s *suiteContext) parseFeatures() error {
	fts, err := parseFeatures("", s.paths)
	if err != nil {
		return err
	}
	s.testedSuite.features = append(s.testedSuite.features, fts...)
	return nil
}

func (s *suiteContext) theSuiteShouldHave(state string) error {
	if s.testedSuite.failed && state == "passed" {
		return fmt.Errorf("the feature suite has failed")
	}
	if !s.testedSuite.failed && state == "failed" {
		return fmt.Errorf("the feature suite has passed")
	}
	return nil
}

func (s *suiteContext) iShouldHaveNumFeatureFiles(num int, files *gherkin.DocString) error {
	if len(s.testedSuite.features) != num {
		return fmt.Errorf("expected %d features to be parsed, but have %d", num, len(s.testedSuite.features))
	}
	expected := strings.Split(files.Content, "\n")
	var actual []string
	for _, ft := range s.testedSuite.features {
		actual = append(actual, ft.Path)
	}
	if len(expected) != len(actual) {
		return fmt.Errorf("expected %d feature paths to be parsed, but have %d", len(expected), len(actual))
	}
	for i := 0; i < len(expected); i++ {
		var matched bool
		split := strings.Split(expected[i], "/")
		exp := filepath.Join(split...)
		for j := 0; j < len(actual); j++ {
			split = strings.Split(actual[j], "/")
			act := filepath.Join(split...)
			if exp == act {
				matched = true
				break
			}
		}
		if !matched {
			return fmt.Errorf(`expected feature path "%s" at position: %d, was not parsed, actual are %+v`, exp, i, actual)
		}
	}
	return nil
}

func (s *suiteContext) iRunFeatureSuite() error {
	if err := s.parseFeatures(); err != nil {
		return err
	}
	s.testedSuite.fmt = testFormatterFunc("godog", &s.out)
	s.testedSuite.run()
	s.testedSuite.fmt.Summary()

	return nil
}

func (s *suiteContext) numScenariosRegistered(expected int) (err error) {
	var num int
	for _, ft := range s.testedSuite.features {
		num += len(ft.ScenarioDefinitions)
	}
	if num != expected {
		err = fmt.Errorf("expected %d scenarios to be registered, but got %d", expected, num)
	}
	return
}

func (s *suiteContext) thereWereNumEventsFired(_ string, expected int, typ string) error {
	var num int
	for _, event := range s.events {
		if event.name == typ {
			num++
		}
	}
	if num != expected {
		return fmt.Errorf("expected %d %s events to be fired, but got %d", expected, typ, num)
	}
	return nil
}

func (s *suiteContext) thereWasEventTriggeredBeforeScenario(expected string) error {
	var found []string
	for _, event := range s.events {
		if event.name != "BeforeScenario" {
			continue
		}

		var name string
		switch t := event.args[0].(type) {
		case *gherkin.Scenario:
			name = t.Name
		case *gherkin.ScenarioOutline:
			name = t.Name
		}
		if name == expected {
			return nil
		}

		found = append(found, name)
	}

	if len(found) == 0 {
		return fmt.Errorf("before scenario event was never triggered or listened")
	}

	return fmt.Errorf(`expected "%s" scenario, but got these fired %s`, expected, `"`+strings.Join(found, `", "`)+`"`)
}

func (s *suiteContext) theseEventsHadToBeFiredForNumberOfTimes(tbl *gherkin.DataTable) error {
	if len(tbl.Rows[0].Cells) != 2 {
		return fmt.Errorf("expected two columns for event table row, got: %d", len(tbl.Rows[0].Cells))
	}

	for _, row := range tbl.Rows {
		num, err := strconv.ParseInt(row.Cells[1].Value, 10, 0)
		if err != nil {
			return err
		}
		if err := s.thereWereNumEventsFired("", int(num), row.Cells[0].Value); err != nil {
			return err
		}
	}
	return nil
}

func (s *suiteContext) theRenderJSONWillBe(docstring *gherkin.DocString) error {
	loc := regexp.MustCompile(`"suite_context.go:\d+"`)
	var expected []cukeFeatureJSON
	if err := json.Unmarshal([]byte(loc.ReplaceAllString(docstring.Content, `"suite_context.go:0"`)), &expected); err != nil {
		return err
	}

	var actual []cukeFeatureJSON
	replaced := loc.ReplaceAllString(s.out.String(), `"suite_context.go:0"`)
	if err := json.Unmarshal([]byte(replaced), &actual); err != nil {
		return err
	}

	if !reflect.DeepEqual(expected, actual) {
		return fmt.Errorf("expected json does not match actual: %s", replaced)
	}
	return nil
}

type testFormatter struct {
	basefmt
	scenarios []interface{}
}

func testFormatterFunc(suite string, out io.Writer) Formatter {
	return &testFormatter{
		basefmt: basefmt{
			started: timeNowFunc(),
			indent:  2,
			out:     out,
		},
	}
}

func (f *testFormatter) Node(node interface{}) {
	f.basefmt.Node(node)
	switch t := node.(type) {
	case *gherkin.Scenario:
		f.scenarios = append(f.scenarios, t)
	case *gherkin.ScenarioOutline:
		f.scenarios = append(f.scenarios, t)
	}
}

func (f *testFormatter) Summary() {}

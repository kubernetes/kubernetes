package godog

import (
	"bytes"
	"fmt"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/DATA-DOG/godog/gherkin"
)

var errorInterface = reflect.TypeOf((*error)(nil)).Elem()
var typeOfBytes = reflect.TypeOf([]byte(nil))

type feature struct {
	*gherkin.Feature
	Content   []byte `json:"-"`
	Path      string `json:"path"`
	scenarios map[int]bool
	order     int
}

// ErrUndefined is returned in case if step definition was not found
var ErrUndefined = fmt.Errorf("step is undefined")

// ErrPending should be returned by step definition if
// step implementation is pending
var ErrPending = fmt.Errorf("step implementation is pending")

// Suite allows various contexts
// to register steps and event handlers.
//
// When running a test suite, the instance of Suite
// is passed to all functions (contexts), which
// have it as a first and only argument.
//
// Note that all event hooks does not catch panic errors
// in order to have a trace information. Only step
// executions are catching panic error since it may
// be a context specific error.
type Suite struct {
	steps    []*StepDef
	features []*feature
	fmt      Formatter

	failed        bool
	randomSeed    int64
	stopOnFailure bool
	strict        bool

	// suite event handlers
	beforeSuiteHandlers    []func()
	beforeFeatureHandlers  []func(*gherkin.Feature)
	beforeScenarioHandlers []func(interface{})
	beforeStepHandlers     []func(*gherkin.Step)
	afterStepHandlers      []func(*gherkin.Step, error)
	afterScenarioHandlers  []func(interface{}, error)
	afterFeatureHandlers   []func(*gherkin.Feature)
	afterSuiteHandlers     []func()
}

// Step allows to register a *StepDef in Godog
// feature suite, the definition will be applied
// to all steps matching the given Regexp expr.
//
// It will panic if expr is not a valid regular
// expression or stepFunc is not a valid step
// handler.
//
// Note that if there are two definitions which may match
// the same step, then only the first matched handler
// will be applied.
//
// If none of the *StepDef is matched, then
// ErrUndefined error will be returned when
// running steps.
func (s *Suite) Step(expr interface{}, stepFunc interface{}) {
	var regex *regexp.Regexp

	switch t := expr.(type) {
	case *regexp.Regexp:
		regex = t
	case string:
		regex = regexp.MustCompile(t)
	case []byte:
		regex = regexp.MustCompile(string(t))
	default:
		panic(fmt.Sprintf("expecting expr to be a *regexp.Regexp or a string, got type: %T", expr))
	}

	v := reflect.ValueOf(stepFunc)
	typ := v.Type()
	if typ.Kind() != reflect.Func {
		panic(fmt.Sprintf("expected handler to be func, but got: %T", stepFunc))
	}

	if typ.NumOut() != 1 {
		panic(fmt.Sprintf("expected handler to return only one value, but it has: %d", typ.NumOut()))
	}

	def := &StepDef{
		Handler: stepFunc,
		Expr:    regex,
		hv:      v,
	}

	typ = typ.Out(0)
	switch typ.Kind() {
	case reflect.Interface:
		if !typ.Implements(errorInterface) {
			panic(fmt.Sprintf("expected handler to return an error, but got: %s", typ.Kind()))
		}
	case reflect.Slice:
		if typ.Elem().Kind() != reflect.String {
			panic(fmt.Sprintf("expected handler to return []string for multistep, but got: []%s", typ.Kind()))
		}
		def.nested = true
	default:
		panic(fmt.Sprintf("expected handler to return an error or []string, but got: %s", typ.Kind()))
	}

	s.steps = append(s.steps, def)
}

// BeforeSuite registers a function or method
// to be run once before suite runner.
//
// Use it to prepare the test suite for a spin.
// Connect and prepare database for instance...
func (s *Suite) BeforeSuite(fn func()) {
	s.beforeSuiteHandlers = append(s.beforeSuiteHandlers, fn)
}

// BeforeFeature registers a function or method
// to be run once before every feature execution.
//
// If godog is run with concurrency option, it will
// run every feature per goroutine. So user may choose
// whether to isolate state within feature context or
// scenario.
//
// Best practice is not to have any state dependency on
// every scenario, but in some cases if VM for example
// needs to be started it may take very long for each
// scenario to restart it.
//
// Use it wisely and avoid sharing state between scenarios.
func (s *Suite) BeforeFeature(fn func(*gherkin.Feature)) {
	s.beforeFeatureHandlers = append(s.beforeFeatureHandlers, fn)
}

// BeforeScenario registers a function or method
// to be run before every scenario or scenario outline.
//
// The interface argument may be *gherkin.Scenario
// or *gherkin.ScenarioOutline
//
// It is a good practice to restore the default state
// before every scenario so it would be isolated from
// any kind of state.
func (s *Suite) BeforeScenario(fn func(interface{})) {
	s.beforeScenarioHandlers = append(s.beforeScenarioHandlers, fn)
}

// BeforeStep registers a function or method
// to be run before every scenario
func (s *Suite) BeforeStep(fn func(*gherkin.Step)) {
	s.beforeStepHandlers = append(s.beforeStepHandlers, fn)
}

// AfterStep registers an function or method
// to be run after every scenario
//
// It may be convenient to return a different kind of error
// in order to print more state details which may help
// in case of step failure
//
// In some cases, for example when running a headless
// browser, to take a screenshot after failure.
func (s *Suite) AfterStep(fn func(*gherkin.Step, error)) {
	s.afterStepHandlers = append(s.afterStepHandlers, fn)
}

// AfterScenario registers an function or method
// to be run after every scenario or scenario outline
//
// The interface argument may be *gherkin.Scenario
// or *gherkin.ScenarioOutline
func (s *Suite) AfterScenario(fn func(interface{}, error)) {
	s.afterScenarioHandlers = append(s.afterScenarioHandlers, fn)
}

// AfterFeature registers a function or method
// to be run once after feature executed all scenarios.
func (s *Suite) AfterFeature(fn func(*gherkin.Feature)) {
	s.afterFeatureHandlers = append(s.afterFeatureHandlers, fn)
}

// AfterSuite registers a function or method
// to be run once after suite runner
func (s *Suite) AfterSuite(fn func()) {
	s.afterSuiteHandlers = append(s.afterSuiteHandlers, fn)
}

func (s *Suite) run() {
	// run before suite handlers
	for _, f := range s.beforeSuiteHandlers {
		f()
	}
	// run features
	for _, f := range s.features {
		s.runFeature(f)
		if s.failed && s.stopOnFailure {
			// stop on first failure
			break
		}
	}
	// run after suite handlers
	for _, f := range s.afterSuiteHandlers {
		f()
	}
}

func (s *Suite) matchStep(step *gherkin.Step) *StepDef {
	def := s.matchStepText(step.Text)
	if def != nil && step.Argument != nil {
		def.args = append(def.args, step.Argument)
	}
	return def
}

func (s *Suite) runStep(step *gherkin.Step, prevStepErr error) (err error) {
	// run before step handlers
	for _, f := range s.beforeStepHandlers {
		f(step)
	}

	match := s.matchStep(step)
	s.fmt.Defined(step, match)

	// user multistep definitions may panic
	defer func() {
		if e := recover(); e != nil {
			err = &traceError{
				msg:   fmt.Sprintf("%v", e),
				stack: callStack(),
			}
		}

		if prevStepErr != nil {
			return
		}

		if err == ErrUndefined {
			return
		}

		switch err {
		case nil:
			s.fmt.Passed(step, match)
		case ErrPending:
			s.fmt.Pending(step, match)
		default:
			s.fmt.Failed(step, match, err)
		}

		// run after step handlers
		for _, f := range s.afterStepHandlers {
			f(step, err)
		}
	}()

	if undef, err := s.maybeUndefined(step.Text, step.Argument); err != nil {
		return err
	} else if len(undef) > 0 {
		if match != nil {
			match = &StepDef{
				args:      match.args,
				hv:        match.hv,
				Expr:      match.Expr,
				Handler:   match.Handler,
				nested:    match.nested,
				undefined: undef,
			}
		}
		s.fmt.Undefined(step, match)
		return ErrUndefined
	}

	if prevStepErr != nil {
		s.fmt.Skipped(step, match)
		return nil
	}

	err = s.maybeSubSteps(match.run())
	return
}

func (s *Suite) maybeUndefined(text string, arg interface{}) ([]string, error) {
	step := s.matchStepText(text)
	if nil == step {
		return []string{text}, nil
	}

	var undefined []string
	if !step.nested {
		return undefined, nil
	}

	if arg != nil {
		step.args = append(step.args, arg)
	}

	for _, next := range step.run().(Steps) {
		lines := strings.Split(next, "\n")
		// @TODO: we cannot currently parse table or content body from nested steps
		if len(lines) > 1 {
			return undefined, fmt.Errorf("nested steps cannot be multiline and have table or content body argument")
		}
		if len(lines[0]) > 0 && lines[0][len(lines[0])-1] == ':' {
			return undefined, fmt.Errorf("nested steps cannot be multiline and have table or content body argument")
		}
		undef, err := s.maybeUndefined(next, nil)
		if err != nil {
			return undefined, err
		}
		undefined = append(undefined, undef...)
	}
	return undefined, nil
}

func (s *Suite) maybeSubSteps(result interface{}) error {
	if nil == result {
		return nil
	}

	if err, ok := result.(error); ok {
		return err
	}

	steps, ok := result.(Steps)
	if !ok {
		return fmt.Errorf("unexpected error, should have been []string: %T - %+v", result, result)
	}

	for _, text := range steps {
		if def := s.matchStepText(text); def == nil {
			return ErrUndefined
		} else if err := s.maybeSubSteps(def.run()); err != nil {
			return fmt.Errorf("%s: %+v", text, err)
		}
	}
	return nil
}

func (s *Suite) matchStepText(text string) *StepDef {
	for _, h := range s.steps {
		if m := h.Expr.FindStringSubmatch(text); len(m) > 0 {
			var args []interface{}
			for _, m := range m[1:] {
				args = append(args, m)
			}

			// since we need to assign arguments
			// better to copy the step definition
			return &StepDef{
				args:    args,
				hv:      h.hv,
				Expr:    h.Expr,
				Handler: h.Handler,
				nested:  h.nested,
			}
		}
	}
	return nil
}

func (s *Suite) runSteps(steps []*gherkin.Step) (err error) {
	for _, step := range steps {
		stepErr := s.runStep(step, err)
		switch stepErr {
		case ErrUndefined:
			// do not overwrite failed error
			if err == ErrUndefined || err == nil {
				err = stepErr
			}
		case ErrPending:
			err = stepErr
		case nil:
		default:
			err = stepErr
		}
	}
	return
}

func (s *Suite) runOutline(outline *gherkin.ScenarioOutline, b *gherkin.Background) (failErr error) {
	s.fmt.Node(outline)

	for _, ex := range outline.Examples {
		example, hasExamples := examples(ex)
		if !hasExamples {
			// @TODO: may need to print empty example node, but
			// for backward compatibility, cannot cast to *gherkin.ExamplesBase
			// at the moment
			continue
		}

		s.fmt.Node(example)
		placeholders := example.TableHeader.Cells
		groups := example.TableBody

		for _, group := range groups {
			if !isEmptyScenario(outline) {
				for _, f := range s.beforeScenarioHandlers {
					f(outline)
				}
			}
			var steps []*gherkin.Step
			for _, outlineStep := range outline.Steps {
				text := outlineStep.Text
				for i, placeholder := range placeholders {
					text = strings.Replace(text, "<"+placeholder.Value+">", group.Cells[i].Value, -1)
				}

				// translate argument
				arg := outlineStep.Argument
				switch t := outlineStep.Argument.(type) {
				case *gherkin.DataTable:
					tbl := &gherkin.DataTable{
						Node: t.Node,
						Rows: make([]*gherkin.TableRow, len(t.Rows)),
					}
					for i, row := range t.Rows {
						cells := make([]*gherkin.TableCell, len(row.Cells))
						for j, cell := range row.Cells {
							trans := cell.Value
							for i, placeholder := range placeholders {
								trans = strings.Replace(trans, "<"+placeholder.Value+">", group.Cells[i].Value, -1)
							}
							cells[j] = &gherkin.TableCell{
								Node:  cell.Node,
								Value: trans,
							}
						}
						tbl.Rows[i] = &gherkin.TableRow{
							Node:  row.Node,
							Cells: cells,
						}
					}
					arg = tbl
				case *gherkin.DocString:
					trans := t.Content
					for i, placeholder := range placeholders {
						trans = strings.Replace(trans, "<"+placeholder.Value+">", group.Cells[i].Value, -1)
					}
					arg = &gherkin.DocString{
						Node:        t.Node,
						Content:     trans,
						ContentType: t.ContentType,
						Delimitter:  t.Delimitter,
					}
				}

				// clone a step
				step := &gherkin.Step{
					Node:     outlineStep.Node,
					Text:     text,
					Keyword:  outlineStep.Keyword,
					Argument: arg,
				}
				steps = append(steps, step)
			}

			// run example table row
			s.fmt.Node(group)

			if b != nil {
				steps = append(b.Steps, steps...)
			}

			err := s.runSteps(steps)

			if !isEmptyScenario(outline) {
				for _, f := range s.afterScenarioHandlers {
					f(outline, err)
				}
			}

			if s.shouldFail(err) {
				failErr = err
				if s.stopOnFailure {
					return
				}
			}
		}
	}
	return
}

func (s *Suite) shouldFail(err error) bool {
	if err == nil {
		return false
	}

	if err == ErrUndefined || err == ErrPending {
		return s.strict
	}

	return true
}

func (s *Suite) runFeature(f *feature) {
	if !isEmptyFeature(f.Feature) {
		for _, fn := range s.beforeFeatureHandlers {
			fn(f.Feature)
		}
	}

	s.fmt.Feature(f.Feature, f.Path, f.Content)

	// make a local copy of the feature scenario defenitions,
	// then shuffle it if we are randomizing scenarios
	scenarios := make([]interface{}, len(f.ScenarioDefinitions))
	if s.randomSeed != 0 {
		r := rand.New(rand.NewSource(s.randomSeed))
		perm := r.Perm(len(f.ScenarioDefinitions))
		for i, v := range perm {
			scenarios[v] = f.ScenarioDefinitions[i]
		}
	} else {
		copy(scenarios, f.ScenarioDefinitions)
	}

	defer func() {
		if !isEmptyFeature(f.Feature) {
			for _, fn := range s.afterFeatureHandlers {
				fn(f.Feature)
			}
		}
	}()

	for _, scenario := range scenarios {
		var err error
		if f.Background != nil {
			s.fmt.Node(f.Background)
		}
		switch t := scenario.(type) {
		case *gherkin.ScenarioOutline:
			err = s.runOutline(t, f.Background)
		case *gherkin.Scenario:
			err = s.runScenario(t, f.Background)
		}
		if s.shouldFail(err) {
			s.failed = true
			if s.stopOnFailure {
				return
			}
		}
	}
}

func (s *Suite) runScenario(scenario *gherkin.Scenario, b *gherkin.Background) (err error) {
	if isEmptyScenario(scenario) {
		s.fmt.Node(scenario)
		return ErrUndefined
	}

	// run before scenario handlers
	for _, f := range s.beforeScenarioHandlers {
		f(scenario)
	}

	s.fmt.Node(scenario)

	// background
	steps := scenario.Steps
	if b != nil {
		steps = append(b.Steps, steps...)
	}

	// scenario
	err = s.runSteps(steps)

	// run after scenario handlers
	for _, f := range s.afterScenarioHandlers {
		f(scenario, err)
	}

	return
}

func (s *Suite) printStepDefinitions(w io.Writer) {
	var longest int
	for _, def := range s.steps {
		n := utf8.RuneCountInString(def.Expr.String())
		if longest < n {
			longest = n
		}
	}
	for _, def := range s.steps {
		n := utf8.RuneCountInString(def.Expr.String())
		location := def.definitionID()
		spaces := strings.Repeat(" ", longest-n)
		fmt.Fprintln(w, yellow(def.Expr.String())+spaces, black("# "+location))
	}
	if len(s.steps) == 0 {
		fmt.Fprintln(w, "there were no contexts registered, could not find any step definition..")
	}
}

var pathLineRe = regexp.MustCompile(`:([\d]+)$`)

func extractFeaturePathLine(p string) (string, int) {
	line := -1
	retPath := p
	if m := pathLineRe.FindStringSubmatch(p); len(m) > 0 {
		if i, err := strconv.Atoi(m[1]); err == nil {
			line = i
			retPath = p[:strings.LastIndexByte(p, ':')]
		}
	}
	return retPath, line
}

func parseFeatures(filter string, paths []string) ([]*feature, error) {
	byPath := make(map[string]*feature)
	var order int
	for _, pat := range paths {
		// check if line number is specified
		path, line := extractFeaturePathLine(pat)
		var err error
		// parse features
		err = filepath.Walk(path, func(p string, f os.FileInfo, err error) error {
			if err == nil && !f.IsDir() && strings.HasSuffix(p, ".feature") {
				reader, err := os.Open(p)
				if err != nil {
					return err
				}
				var buf bytes.Buffer
				ft, err := gherkin.ParseFeature(io.TeeReader(reader, &buf))
				reader.Close()
				if err != nil {
					return fmt.Errorf("%s - %v", p, err)
				}

				feat := byPath[p]
				if feat == nil {
					feat = &feature{
						Path:      p,
						Feature:   ft,
						Content:   buf.Bytes(),
						scenarios: make(map[int]bool),
						order:     order,
					}
					order++
					byPath[p] = feat
				}
				// filter scenario by line number
				for _, def := range ft.ScenarioDefinitions {
					var ln int
					switch t := def.(type) {
					case *gherkin.Scenario:
						ln = t.Location.Line
					case *gherkin.ScenarioOutline:
						ln = t.Location.Line
					}
					if line == -1 || ln == line {
						feat.scenarios[ln] = true
					}
				}
			}
			return err
		})
		// check error
		switch {
		case os.IsNotExist(err):
			return nil, fmt.Errorf(`feature path "%s" is not available`, path)
		case os.IsPermission(err):
			return nil, fmt.Errorf(`feature path "%s" is not accessible`, path)
		case err != nil:
			return nil, err
		}
	}
	return filterFeatures(filter, byPath), nil
}

type sortByOrderGiven []*feature

func (s sortByOrderGiven) Len() int           { return len(s) }
func (s sortByOrderGiven) Less(i, j int) bool { return s[i].order < s[j].order }
func (s sortByOrderGiven) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

func filterFeatures(tags string, collected map[string]*feature) (features []*feature) {
	for _, ft := range collected {
		var scenarios []interface{}
		for _, def := range ft.ScenarioDefinitions {
			var ln int
			switch t := def.(type) {
			case *gherkin.Scenario:
				ln = t.Location.Line
			case *gherkin.ScenarioOutline:
				ln = t.Location.Line
			}
			if ft.scenarios[ln] {
				scenarios = append(scenarios, def)
			}
		}
		ft.ScenarioDefinitions = scenarios
		applyTagFilter(tags, ft.Feature)
		features = append(features, ft)
	}

	sort.Sort(sortByOrderGiven(features))

	return features
}

func applyTagFilter(tags string, ft *gherkin.Feature) {
	if len(tags) == 0 {
		return
	}

	var scenarios []interface{}
	for _, scenario := range ft.ScenarioDefinitions {
		switch t := scenario.(type) {
		case *gherkin.ScenarioOutline:
			var allExamples []*gherkin.Examples
			for _, examples := range t.Examples {
				if matchesTags(tags, allTags(ft, t, examples)) {
					allExamples = append(allExamples, examples)
				}
			}
			t.Examples = allExamples
			if len(t.Examples) > 0 {
				scenarios = append(scenarios, scenario)
			}
		case *gherkin.Scenario:
			if matchesTags(tags, allTags(ft, t)) {
				scenarios = append(scenarios, scenario)
			}
		}
	}
	ft.ScenarioDefinitions = scenarios
}

func allTags(nodes ...interface{}) []string {
	var tags, tmp []string
	for _, node := range nodes {
		var gr []*gherkin.Tag
		switch t := node.(type) {
		case *gherkin.Feature:
			gr = t.Tags
		case *gherkin.ScenarioOutline:
			gr = t.Tags
		case *gherkin.Scenario:
			gr = t.Tags
		case *gherkin.Examples:
			gr = t.Tags
		}

		for _, gtag := range gr {
			tag := strings.TrimSpace(gtag.Name)
			if tag[0] == '@' {
				tag = tag[1:]
			}
			copy(tmp, tags)
			var found bool
			for _, tg := range tmp {
				if tg == tag {
					found = true
					break
				}
			}
			if !found {
				tags = append(tags, tag)
			}
		}
	}
	return tags
}

func hasTag(tags []string, tag string) bool {
	for _, t := range tags {
		if t == tag {
			return true
		}
	}
	return false
}

// based on http://behat.readthedocs.org/en/v2.5/guides/6.cli.html#gherkin-filters
func matchesTags(filter string, tags []string) (ok bool) {
	ok = true
	for _, andTags := range strings.Split(filter, "&&") {
		var okComma bool
		for _, tag := range strings.Split(andTags, ",") {
			tag = strings.Replace(strings.TrimSpace(tag), "@", "", -1)
			if tag[0] == '~' {
				tag = tag[1:]
				okComma = !hasTag(tags, tag) || okComma
			} else {
				okComma = hasTag(tags, tag) || okComma
			}
		}
		ok = ok && okComma
	}
	return
}

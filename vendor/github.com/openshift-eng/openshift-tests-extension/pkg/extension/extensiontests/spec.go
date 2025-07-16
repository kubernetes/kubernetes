package extensiontests

import (
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/checker/decls"
	"github.com/google/cel-go/common/types"

	"github.com/openshift-eng/openshift-tests-extension/pkg/dbtime"
	"github.com/openshift-eng/openshift-tests-extension/pkg/flags"
)

// Walk iterates over all test specs, and executions the function provided. The test spec can be mutated.
func (specs ExtensionTestSpecs) Walk(walkFn func(*ExtensionTestSpec)) ExtensionTestSpecs {
	for i := range specs {
		walkFn(specs[i])
	}

	return specs
}

type SelectFunction func(spec *ExtensionTestSpec) bool

// Select filters the ExtensionTestSpecs to only those that match the provided SelectFunction
func (specs ExtensionTestSpecs) Select(selectFn SelectFunction) ExtensionTestSpecs {
	filtered := ExtensionTestSpecs{}
	for _, spec := range specs {
		if selectFn(spec) {
			filtered = append(filtered, spec)
		}
	}

	return filtered
}

// MustSelect filters the ExtensionTestSpecs to only those that match the provided SelectFunction.
// if no specs are selected, it will throw an error
func (specs ExtensionTestSpecs) MustSelect(selectFn SelectFunction) (ExtensionTestSpecs, error) {
	filtered := specs.Select(selectFn)
	if len(filtered) == 0 {
		return filtered, fmt.Errorf("no specs selected with specified SelectFunctions")
	}

	return filtered, nil
}

// SelectAny filters the ExtensionTestSpecs to only those that match any of the provided SelectFunctions
func (specs ExtensionTestSpecs) SelectAny(selectFns []SelectFunction) ExtensionTestSpecs {
	filtered := ExtensionTestSpecs{}
	for _, spec := range specs {
		for _, selectFn := range selectFns {
			if selectFn(spec) {
				filtered = append(filtered, spec)
				break
			}
		}
	}

	return filtered
}

// MustSelectAny filters the ExtensionTestSpecs to only those that match any of the provided SelectFunctions.
// if no specs are selected, it will throw an error
func (specs ExtensionTestSpecs) MustSelectAny(selectFns []SelectFunction) (ExtensionTestSpecs, error) {
	filtered := specs.SelectAny(selectFns)
	if len(filtered) == 0 {
		return filtered, fmt.Errorf("no specs selected with specified SelectFunctions")
	}

	return filtered, nil
}

// SelectAll filters the ExtensionTestSpecs to only those that match all the provided SelectFunctions
func (specs ExtensionTestSpecs) SelectAll(selectFns []SelectFunction) ExtensionTestSpecs {
	filtered := ExtensionTestSpecs{}
	for _, spec := range specs {
		anyFalse := false
		for _, selectFn := range selectFns {
			if !selectFn(spec) {
				anyFalse = true
				break
			}
		}
		if !anyFalse {
			filtered = append(filtered, spec)
		}
	}

	return filtered
}

// MustSelectAll filters the ExtensionTestSpecs to only those that match all the provided SelectFunctions.
// if no specs are selected, it will throw an error
func (specs ExtensionTestSpecs) MustSelectAll(selectFns []SelectFunction) (ExtensionTestSpecs, error) {
	filtered := specs.SelectAll(selectFns)
	if len(filtered) == 0 {
		return filtered, fmt.Errorf("no specs selected with specified SelectFunctions")
	}

	return filtered, nil
}

// ModuleTestsOnly ensures that ginkgo tests from vendored sources aren't selected.
func ModuleTestsOnly() SelectFunction {
	return func(spec *ExtensionTestSpec) bool {
		for _, cl := range spec.CodeLocations {
			if strings.Contains(cl, "/vendor/") {
				return false
			}
		}

		return true
	}
}

// AllTestsIncludingVendored is an alternative to ModuleTestsOnly, which would explicitly opt-in
// to including vendored tests.
func AllTestsIncludingVendored() SelectFunction {
	return func(spec *ExtensionTestSpec) bool {
		return true
	}
}

// NameContains returns a function that selects specs whose name contains the provided string
func NameContains(name string) SelectFunction {
	return func(spec *ExtensionTestSpec) bool {
		return strings.Contains(spec.Name, name)
	}
}

// NameContainsAll returns a function that selects specs whose name contains each of the provided contents strings
func NameContainsAll(contents ...string) SelectFunction {
	return func(spec *ExtensionTestSpec) bool {
		for _, content := range contents {
			if !strings.Contains(spec.Name, content) {
				return false
			}
		}
		return true
	}
}

// HasLabel returns a function that selects specs with the provided label
func HasLabel(label string) SelectFunction {
	return func(spec *ExtensionTestSpec) bool {
		return spec.Labels.Has(label)
	}
}

// HasTagWithValue returns a function that selects specs containing a tag with the provided key and value
func HasTagWithValue(key, value string) SelectFunction {
	return func(spec *ExtensionTestSpec) bool {
		return spec.Tags[key] == value
	}
}

// WithLifecycle returns a function that selects specs with the provided Lifecycle
func WithLifecycle(lifecycle Lifecycle) SelectFunction {
	return func(spec *ExtensionTestSpec) bool {
		return spec.Lifecycle == lifecycle
	}
}

func (specs ExtensionTestSpecs) Names() []string {
	var names []string
	for _, spec := range specs {
		names = append(names, spec.Name)
	}
	return names
}

// Run executes all the specs in parallel, up to maxConcurrent at the same time. Results
// are written to the given ResultWriter after each spec has completed execution.  BeforeEach,
// BeforeAll, AfterEach, AfterAll hooks are executed when specified. "Each" hooks must be thread
// safe. Returns an error if any test spec failed, indicating the quantity of failures.
func (specs ExtensionTestSpecs) Run(w ResultWriter, maxConcurrent int) error {
	queue := make(chan *ExtensionTestSpec)
	failures := atomic.Int64{}

	// Execute beforeAll
	for _, spec := range specs {
		for _, beforeAllTask := range spec.beforeAll {
			beforeAllTask.Run()
		}
	}

	// Feed the queue
	go func() {
		specs.Walk(func(spec *ExtensionTestSpec) {
			queue <- spec
		})
		close(queue)
	}()

	// Start consumers
	var wg sync.WaitGroup
	for i := 0; i < maxConcurrent; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for spec := range queue {
				for _, beforeEachTask := range spec.beforeEach {
					beforeEachTask.Run(*spec)
				}

				res := runSpec(spec)
				if res.Result == ResultFailed {
					failures.Add(1)
				}

				for _, afterEachTask := range spec.afterEach {
					afterEachTask.Run(res)
				}

				// We can't assume the runner will set the name of a test; it may not know it. Even if
				// it does, we may want to modify it (e.g. k8s-tests for annotations currently).
				res.Name = spec.Name
				w.Write(res)
			}
		}()
	}

	// Wait for all consumers to finish
	wg.Wait()

	// Execute afterAll
	for _, spec := range specs {
		for _, afterAllTask := range spec.afterAll {
			afterAllTask.Run()
		}
	}

	failCount := failures.Load()
	if failCount > 0 {
		return fmt.Errorf("%d tests failed", failCount)
	}
	return nil
}

// AddBeforeAll adds a function to be run once before all tests start executing.
func (specs ExtensionTestSpecs) AddBeforeAll(fn func()) {
	task := &OneTimeTask{fn: fn}
	specs.Walk(func(spec *ExtensionTestSpec) {
		spec.beforeAll = append(spec.beforeAll, task)
	})
}

// AddAfterAll adds a function to be run once after all tests have finished.
func (specs ExtensionTestSpecs) AddAfterAll(fn func()) {
	task := &OneTimeTask{fn: fn}
	specs.Walk(func(spec *ExtensionTestSpec) {
		spec.afterAll = append(spec.afterAll, task)
	})
}

// AddBeforeEach adds a function that runs before each test starts executing. The ExtensionTestSpec is
// passed in for contextual information, but must not be modified. The provided function must be thread
// safe.
func (specs ExtensionTestSpecs) AddBeforeEach(fn func(spec ExtensionTestSpec)) {
	task := &SpecTask{fn: fn}
	specs.Walk(func(spec *ExtensionTestSpec) {
		spec.beforeEach = append(spec.beforeEach, task)
	})
}

// AddAfterEach adds a function that runs after each test has finished executing. The ExtensionTestResult
// can be modified if needed. The provided function must be thread safe.
func (specs ExtensionTestSpecs) AddAfterEach(fn func(task *ExtensionTestResult)) {
	task := &TestResultTask{fn: fn}
	specs.Walk(func(spec *ExtensionTestSpec) {
		spec.afterEach = append(spec.afterEach, task)
	})
}

// MustFilter filters specs using the given celExprs.  Each celExpr is OR'd together, if any
// match the spec is included in the filtered set. If your CEL expression is invalid or filtering
// otherwise fails, this function panics.
func (specs ExtensionTestSpecs) MustFilter(celExprs []string) ExtensionTestSpecs {
	specs, err := specs.Filter(celExprs)
	if err != nil {
		panic(fmt.Sprintf("filter did not succeed: %s", err.Error()))
	}

	return specs
}

// Filter filters specs using the given celExprs.  Each celExpr is OR'd together, if any
// match the spec is included in the filtered set.
func (specs ExtensionTestSpecs) Filter(celExprs []string) (ExtensionTestSpecs, error) {
	var filteredSpecs ExtensionTestSpecs

	// Empty filters returns all
	if len(celExprs) == 0 {
		return specs, nil
	}

	env, err := cel.NewEnv(
		cel.Declarations(
			decls.NewVar("source", decls.String),
			decls.NewVar("name", decls.String),
			decls.NewVar("originalName", decls.String),
			decls.NewVar("labels", decls.NewListType(decls.String)),
			decls.NewVar("codeLocations", decls.NewListType(decls.String)),
			decls.NewVar("tags", decls.NewMapType(decls.String, decls.String)),
		),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create CEL environment: %w", err)
	}

	// OR all expressions together
	for _, spec := range specs {
		include := false
		for _, celExpr := range celExprs {
			prg, err := programForCEL(env, celExpr)
			if err != nil {
				return nil, err
			}
			out, _, err := prg.Eval(map[string]interface{}{
				"name":          spec.Name,
				"source":        spec.Source,
				"originalName":  spec.OriginalName,
				"labels":        spec.Labels.UnsortedList(),
				"codeLocations": spec.CodeLocations,
				"tags":          spec.Tags,
			})
			if err != nil {
				return nil, fmt.Errorf("error evaluating CEL expression: %v", err)
			}

			// If any CEL expression evaluates to true, include the TestSpec
			if out == types.True {
				include = true
				break
			}
		}
		if include {
			filteredSpecs = append(filteredSpecs, spec)
		}
	}

	return filteredSpecs, nil
}

func programForCEL(env *cel.Env, celExpr string) (cel.Program, error) {
	// Parse CEL expression
	ast, iss := env.Parse(celExpr)
	if iss.Err() != nil {
		return nil, fmt.Errorf("error parsing CEL expression '%s': %v", celExpr, iss.Err())
	}

	// Check the AST
	checked, iss := env.Check(ast)
	if iss.Err() != nil {
		return nil, fmt.Errorf("error checking CEL expression '%s': %v", celExpr, iss.Err())
	}

	// Create a CEL program from the checked AST
	prg, err := env.Program(checked)
	if err != nil {
		return nil, fmt.Errorf("error creating CEL program: %v", err)
	}
	return prg, nil
}

// FilterByEnvironment checks both the Include and Exclude fields of the ExtensionTestSpec to return those specs which match.
// Tests will be included by default unless they are explicitly excluded. If Include is specified, only those tests matching
// the CEL expression will be included.
//
// See helper functions in extensiontests/environment.go to craft CEL expressions
func (specs ExtensionTestSpecs) FilterByEnvironment(envFlags flags.EnvironmentalFlags) (ExtensionTestSpecs, error) {
	var filteredSpecs ExtensionTestSpecs
	if envFlags.IsEmpty() {
		return specs, nil
	}

	env, err := cel.NewEnv(
		cel.Declarations(
			decls.NewVar("apiGroups", decls.NewListType(decls.String)),
			decls.NewVar("architecture", decls.String),
			decls.NewVar("externalConnectivity", decls.String),
			decls.NewVar("fact_keys", decls.NewListType(decls.String)),
			decls.NewVar("facts", decls.NewMapType(decls.String, decls.String)),
			decls.NewVar("featureGates", decls.NewListType(decls.String)),
			decls.NewVar("network", decls.String),
			decls.NewVar("networkStack", decls.String),
			decls.NewVar("optionalCapabilities", decls.NewListType(decls.String)),
			decls.NewVar("platform", decls.String),
			decls.NewVar("topology", decls.String),
			decls.NewVar("upgrade", decls.String),
			decls.NewVar("version", decls.String),
		),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create CEL environment: %w", err)
	}
	factKeys := make([]string, len(envFlags.Facts))
	for k := range envFlags.Facts {
		factKeys = append(factKeys, k)
	}
	vars := map[string]interface{}{
		"apiGroups":            envFlags.APIGroups,
		"architecture":         envFlags.Architecture,
		"externalConnectivity": envFlags.ExternalConnectivity,
		"fact_keys":            factKeys,
		"facts":                envFlags.Facts,
		"featureGates":         envFlags.FeatureGates,
		"network":              envFlags.Network,
		"networkStack":         envFlags.NetworkStack,
		"optionalCapabilities": envFlags.OptionalCapabilities,
		"platform":             envFlags.Platform,
		"topology":             envFlags.Topology,
		"upgrade":              envFlags.Upgrade,
		"version":              envFlags.Version,
	}

	for _, spec := range specs {
		envSel := spec.EnvironmentSelector
		// If there is no include or exclude CEL, include it implicitly
		if envSel.IsEmpty() {
			filteredSpecs = append(filteredSpecs, spec)
			continue
		}

		if envSel.Exclude != "" {
			prg, err := programForCEL(env, envSel.Exclude)
			if err != nil {
				return nil, err
			}
			out, _, err := prg.Eval(vars)
			if err != nil {
				return nil, fmt.Errorf("error evaluating CEL expression: %v", err)
			}
			// If it is explicitly excluded, don't check include
			if out == types.True {
				continue
			}
		}

		if envSel.Include != "" {
			prg, err := programForCEL(env, envSel.Include)
			if err != nil {
				return nil, err
			}
			out, _, err := prg.Eval(vars)
			if err != nil {
				return nil, fmt.Errorf("error evaluating CEL expression: %v", err)
			}

			if out == types.True {
				filteredSpecs = append(filteredSpecs, spec)
			}
		} else { // If it hasn't been excluded, and there is no "include" it will be implicitly included
			filteredSpecs = append(filteredSpecs, spec)
		}

	}

	return filteredSpecs, nil
}

// AddLabel adds the labels to each spec.
func (specs ExtensionTestSpecs) AddLabel(labels ...string) ExtensionTestSpecs {
	for i := range specs {
		specs[i].Labels.Insert(labels...)
	}

	return specs
}

// RemoveLabel removes the labels from each spec.
func (specs ExtensionTestSpecs) RemoveLabel(labels ...string) ExtensionTestSpecs {
	for i := range specs {
		specs[i].Labels.Delete(labels...)
	}

	return specs
}

// SetTag specifies a key/value pair for each spec.
func (specs ExtensionTestSpecs) SetTag(key, value string) ExtensionTestSpecs {
	for i := range specs {
		specs[i].Tags[key] = value
	}

	return specs
}

// UnsetTag removes the specified key from each spec.
func (specs ExtensionTestSpecs) UnsetTag(key string) ExtensionTestSpecs {
	for i := range specs {
		delete(specs[i].Tags, key)
	}

	return specs
}

// Include adds the specified CEL expression to explicitly include tests by environment to each spec
func (specs ExtensionTestSpecs) Include(includeCEL string) ExtensionTestSpecs {
	for _, spec := range specs {
		spec.Include(includeCEL)
	}
	return specs
}

// Exclude adds the specified CEL expression to explicitly exclude tests by environment to each spec
func (specs ExtensionTestSpecs) Exclude(excludeCEL string) ExtensionTestSpecs {
	for _, spec := range specs {
		spec.Exclude(excludeCEL)
	}
	return specs
}

// Include adds the specified CEL expression to explicitly include tests by environment.
// If there is already an "include" defined, it will OR the expressions together
func (spec *ExtensionTestSpec) Include(includeCEL string) *ExtensionTestSpec {
	existingInclude := spec.EnvironmentSelector.Include
	if existingInclude != "" {
		includeCEL = fmt.Sprintf("(%s) || (%s)", existingInclude, includeCEL)
	}

	spec.EnvironmentSelector.Include = includeCEL
	return spec
}

// Exclude adds the specified CEL expression to explicitly exclude tests by environment.
// If there is already an "exclude" defined, it will OR the expressions together
func (spec *ExtensionTestSpec) Exclude(excludeCEL string) *ExtensionTestSpec {
	existingExclude := spec.EnvironmentSelector.Exclude
	if existingExclude != "" {
		excludeCEL = fmt.Sprintf("(%s) || (%s)", existingExclude, excludeCEL)
	}

	spec.EnvironmentSelector.Exclude = excludeCEL
	return spec
}

func runSpec(spec *ExtensionTestSpec) *ExtensionTestResult {
	startTime := time.Now().UTC()
	res := spec.Run()
	duration := time.Since(startTime)
	endTime := startTime.Add(duration).UTC()
	if res == nil {
		// this shouldn't happen
		panic(fmt.Sprintf("test produced no result: %s", spec.Name))
	}

	res.Lifecycle = spec.Lifecycle

	// If the runner doesn't populate this info, we should set it
	if res.StartTime == nil {
		res.StartTime = dbtime.Ptr(startTime)
	}
	if res.EndTime == nil {
		res.EndTime = dbtime.Ptr(endTime)
	}
	if res.Duration == 0 {
		res.Duration = duration.Milliseconds()
	}

	return res
}

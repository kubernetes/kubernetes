package extensiontests

import (
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/checker/decls"
	"github.com/google/cel-go/common/types"

	"github.com/openshift-eng/openshift-tests-extension/pkg/dbtime"
)

// Walk iterates over all test specs, and executions the function provided. The test spec can be mutated.
func (specs ExtensionTestSpecs) Walk(walkFn func(*ExtensionTestSpec)) ExtensionTestSpecs {
	for i := range specs {
		walkFn(specs[i])
	}

	return specs
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

			out, _, err := prg.Eval(map[string]interface{}{
				"name":         spec.Name,
				"source":       spec.Source,
				"originalName": spec.OriginalName,
				"labels":       spec.Labels.UnsortedList(),
				"tags":         spec.Tags,
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

package extension

import (
	"fmt"
	"strings"

	et "github.com/openshift-eng/openshift-tests-extension/pkg/extension/extensiontests"
	"github.com/openshift-eng/openshift-tests-extension/pkg/util/sets"
	"github.com/openshift-eng/openshift-tests-extension/pkg/version"
)

func NewExtension(product, kind, name string) *Extension {
	return &Extension{
		APIVersion: CurrentExtensionAPIVersion,
		Source: Source{
			Commit:       version.CommitFromGit,
			BuildDate:    version.BuildDate,
			GitTreeState: version.GitTreeState,
		},
		Component: Component{
			Product: product,
			Kind:    kind,
			Name:    name,
		},
	}
}

func (e *Extension) GetSuite(name string) (*Suite, error) {
	var suite *Suite

	for _, s := range e.Suites {
		if s.Name == name {
			suite = &s
			break
		}
	}

	if suite == nil {
		return nil, fmt.Errorf("no such suite: %s", name)
	}

	return suite, nil
}

func (e *Extension) GetSpecs() et.ExtensionTestSpecs {
	return e.specs
}

func (e *Extension) AddSpecs(specs et.ExtensionTestSpecs) {
	specs.Walk(func(spec *et.ExtensionTestSpec) {
		spec.Source = e.Component.Identifier()
	})

	e.specs = append(e.specs, specs...)
}

// IgnoreObsoleteTests allows removal of a test.
func (e *Extension) IgnoreObsoleteTests(testNames ...string) {
	if e.obsoleteTests == nil {
		e.obsoleteTests = sets.New[string](testNames...)
	} else {
		e.obsoleteTests.Insert(testNames...)
	}
}

// FindRemovedTestsWithoutRename compares the current set of test specs against oldSpecs, including consideration of the original name,
// we return an error.  Can be used to detect test renames or removals.
func (e *Extension) FindRemovedTestsWithoutRename(oldSpecs et.ExtensionTestSpecs) ([]string, error) {
	currentSpecs := e.GetSpecs()
	currentMap := make(map[string]bool)

	// Populate current specs into a map for quick lookup by both Name and OriginalName.
	for _, spec := range currentSpecs {
		currentMap[spec.Name] = true
		if spec.OriginalName != "" {
			currentMap[spec.OriginalName] = true
		}
	}

	var removedTests []string

	// Check oldSpecs against current specs.
	for _, oldSpec := range oldSpecs {
		// Skip if the test is marked as obsolete.
		if e.obsoleteTests.Has(oldSpec.Name) {
			continue
		}

		// Check if oldSpec is missing in currentSpecs by both Name and OriginalName.
		if !currentMap[oldSpec.Name] && (oldSpec.OriginalName == "" || !currentMap[oldSpec.OriginalName]) {
			removedTests = append(removedTests, oldSpec.Name)
		}
	}

	// Return error if any removed tests were found.
	if len(removedTests) > 0 {
		return removedTests, fmt.Errorf("tests removed without rename: %v", removedTests)
	}

	return nil, nil
}

// AddGlobalSuite adds a suite whose qualifiers will apply to all tests,
// not just this one.  Allowing a developer to create a composed suite of
// tests from many sources.
func (e *Extension) AddGlobalSuite(suite Suite) *Extension {
	if e.Suites == nil {
		e.Suites = []Suite{suite}
	} else {
		e.Suites = append(e.Suites, suite)
	}

	return e
}

// AddSuite adds a suite whose qualifiers will only apply to tests present
// in its own extension.
func (e *Extension) AddSuite(suite Suite) *Extension {
	expr := fmt.Sprintf("source == %q", e.Component.Identifier())
	if len(suite.Qualifiers) == 0 {
		suite.Qualifiers = []string{expr}
	} else {
		for i := range suite.Qualifiers {
			suite.Qualifiers[i] = fmt.Sprintf("(%s) && (%s)",
				expr, suite.Qualifiers[i])
		}
	}

	e.AddGlobalSuite(suite)
	return e
}

func (e *Extension) RegisterImage(image Image) *Extension {
	e.Images = append(e.Images, image)
	return e
}

func (e *Extension) FindSpecsByName(names ...string) (et.ExtensionTestSpecs, error) {
	var specs et.ExtensionTestSpecs
	var notFound []string

	for _, name := range names {
		found := false
		for i := range e.specs {
			if e.specs[i].Name == name {
				specs = append(specs, e.specs[i])
				found = true
				break
			}
		}
		if !found {
			notFound = append(notFound, name)
		}
	}

	if len(notFound) > 0 {
		return nil, fmt.Errorf("no such tests: %s", strings.Join(notFound, ", "))
	}

	return specs, nil
}

func (e *Component) Identifier() string {
	return fmt.Sprintf("%s:%s:%s", e.Product, e.Kind, e.Name)
}

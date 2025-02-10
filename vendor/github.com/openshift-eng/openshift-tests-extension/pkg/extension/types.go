package extension

import (
	"k8s.io/apimachinery/pkg/util/sets"

	"github.com/openshift-eng/openshift-tests-extension/pkg/extension/extensiontests"
)

const CurrentExtensionAPIVersion = "v1.0"

// Extension represents an extension to openshift-tests.
type Extension struct {
	APIVersion string    `json:"apiVersion"`
	Source     Source    `json:"source"`
	Component  Component `json:"component"`

	// Suites that the extension wants to advertise/participate in.
	Suites []Suite `json:"suites"`

	Images []Image `json:"images"`

	// Private data
	specs         extensiontests.ExtensionTestSpecs
	obsoleteTests sets.Set[string]
}

// Source contains the details of the commit and source URL.
type Source struct {
	// Commit from which this binary was compiled.
	Commit string `json:"commit"`
	// BuildDate ISO8601 string of when the binary was built
	BuildDate string `json:"build_date"`
	// GitTreeState lets you know the status of the git tree (clean/dirty)
	GitTreeState string `json:"git_tree_state"`
	// SourceURL contains the url of the git repository (if known) that this extension was built from.
	SourceURL string `json:"source_url,omitempty"`
}

// Component represents the component the binary acts on.
type Component struct {
	// The product this component is part of.
	Product string `json:"product"`
	// The type of the component.
	Kind string `json:"type"`
	// The name of the component.
	Name string `json:"name"`
}

// Suite represents additional suites the extension wants to advertise.
type Suite struct {
	// The name of the suite.
	Name string `json:"name"`
	// Parent suites this suite is part of.
	Parents []string `json:"parents,omitempty"`
	// Qualifiers are CEL expressions that are OR'd together for test selection that are members of the suite.
	Qualifiers []string `json:"qualifiers,omitempty"`
}

type Image struct {
	Index    int    `json:"index"`
	Registry string `json:"registry"`
	Name     string `json:"name"`
	Version  string `json:"version"`
}

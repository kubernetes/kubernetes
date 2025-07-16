package extension

import (
	"time"

	"github.com/openshift-eng/openshift-tests-extension/pkg/extension/extensiontests"
	"github.com/openshift-eng/openshift-tests-extension/pkg/util/sets"
)

const CurrentExtensionAPIVersion = "v1.1"

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

type ClusterStability string

var (
	// ClusterStabilityStable means that at no point during testing do we expect a component to take downtime and upgrades are not happening.
	ClusterStabilityStable ClusterStability = "Stable"

	// ClusterStabilityDisruptive means that the suite is expected to induce outages to the cluster.
	ClusterStabilityDisruptive ClusterStability = "Disruptive"

	// ClusterStabilityUpgrade was previously defined, but was removed by @deads2k. Please contact him if you find a use
	// case for it and needs to be reintroduced.
	// ClusterStabilityUpgrade    ClusterStability = "Upgrade"
)

// Suite represents additional suites the extension wants to advertise. Child suites when being executed in the context
// of a parent will have their count, parallelism, stability, and timeout options superseded by the parent's suite.
type Suite struct {
	Name        string `json:"name"`
	Description string `json:"description"`

	// Parents are the parent suites this suite is part of.
	Parents []string `json:"parents,omitempty"`
	// Qualifiers are CEL expressions that are OR'd together for test selection that are members of the suite.
	Qualifiers []string `json:"qualifiers,omitempty"`

	// Count is the default number of times to execute each test in this suite.
	Count int `json:"count,omitempty"`
	// Parallelism is the maximum parallelism of this suite.
	Parallelism int `json:"parallelism,omitempty"`
	// ClusterStability informs openshift-tests whether this entire test suite is expected to be disruptive or not
	// to normal cluster operations.
	ClusterStability ClusterStability `json:"clusterStability,omitempty"`
	// TestTimeout is the default timeout for tests in this suite.
	TestTimeout *time.Duration `json:"testTimeout,omitempty"`
}

type Image struct {
	Index    int    `json:"index"`
	Registry string `json:"registry"`
	Name     string `json:"name"`
	Version  string `json:"version"`
}

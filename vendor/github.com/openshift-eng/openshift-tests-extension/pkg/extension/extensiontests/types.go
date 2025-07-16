package extensiontests

import (
	"github.com/openshift-eng/openshift-tests-extension/pkg/dbtime"
	"github.com/openshift-eng/openshift-tests-extension/pkg/util/sets"
)

type Lifecycle string

var LifecycleInforming Lifecycle = "informing"
var LifecycleBlocking Lifecycle = "blocking"

type ExtensionTestSpecs []*ExtensionTestSpec

type ExtensionTestSpec struct {
	Name string `json:"name"`

	// OriginalName contains the very first name this test was ever known as, used to preserve
	// history across all names.
	OriginalName string `json:"originalName,omitempty"`

	// Labels are single string values to apply to the test spec
	Labels sets.Set[string] `json:"labels"`

	// Tags are key:value pairs
	Tags map[string]string `json:"tags,omitempty"`

	// Resources gives optional information about what's required to run this test.
	Resources Resources `json:"resources"`

	// Source is the origin of the test.
	Source string `json:"source"`

	// CodeLocations are the files where the spec originates from.
	CodeLocations []string `json:"codeLocations,omitempty"`

	// Lifecycle informs the executor whether the test is informing only, and should not cause the
	// overall job run to fail, or if it's blocking where a failure of the test is fatal.
	// Informing lifecycle tests can be used temporarily to gather information about a test's stability.
	// Tests must not remain informing forever.
	Lifecycle Lifecycle `json:"lifecycle"`

	// EnvironmentSelector allows for CEL expressions to be used to control test inclusion
	EnvironmentSelector EnvironmentSelector `json:"environmentSelector,omitempty"`

	// Run invokes a test
	Run func() *ExtensionTestResult `json:"-"`

	// Hook functions
	afterAll   []*OneTimeTask
	beforeAll  []*OneTimeTask
	afterEach  []*TestResultTask
	beforeEach []*SpecTask
}

type Resources struct {
	Isolation Isolation `json:"isolation"`
	Memory    string    `json:"memory,omitempty"`
	Duration  string    `json:"duration,omitempty"`
	Timeout   string    `json:"timeout,omitempty"`
}

type Isolation struct {
	Mode     string   `json:"mode,omitempty"`
	Conflict []string `json:"conflict,omitempty"`
}

type EnvironmentSelector struct {
	Include string `json:"include,omitempty"`
	Exclude string `json:"exclude,omitempty"`
}

func (e EnvironmentSelector) IsEmpty() bool {
	return e.Include == "" && e.Exclude == ""
}

type ExtensionTestResults []*ExtensionTestResult

type Result string

var ResultPassed Result = "passed"
var ResultSkipped Result = "skipped"
var ResultFailed Result = "failed"

type ExtensionTestResult struct {
	Name      string         `json:"name"`
	Lifecycle Lifecycle      `json:"lifecycle"`
	Duration  int64          `json:"duration"`
	StartTime *dbtime.DBTime `json:"startTime"`
	EndTime   *dbtime.DBTime `json:"endTime"`
	Result    Result         `json:"result"`
	Output    string         `json:"output"`
	Error     string         `json:"error,omitempty"`
	Details   []Details      `json:"details,omitempty"`
}

// Details are human-readable messages to further explain skips, timeouts, etc.
// It can also be used to provide contemporaneous information about failures
// that may not be easily returned by must-gather. For larger artifacts (greater than
// 10KB, write them to $EXTENSION_ARTIFACTS_DIR.
type Details struct {
	Name  string      `json:"name"`
	Value interface{} `json:"value"`
}

package v1

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// TestReporting is used for origin (and potentially others) to report the test names for a given FeatureGate into
// the payload for later analysis on a per-payload basis.
// This doesn't need any CRD because it's never stored in the cluster.
//
// Compatibility level 4: No compatibility is provided, the API can change at any point for any reason. These capabilities should not be used by applications needing long term support.
// +openshift:compatibility-gen:internal
type TestReporting struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// +kubebuilder:validation:Required
	// +required
	Spec TestReportingSpec `json:"spec"`
	// status holds observed values from the cluster. They may not be overridden.
	// +optional
	Status TestReportingStatus `json:"status"`
}

type TestReportingSpec struct {
	// TestsForFeatureGates is a list, indexed by FeatureGate and includes information about testing.
	TestsForFeatureGates []FeatureGateTests `json:"testsForFeatureGates"`
}

type FeatureGateTests struct {
	// FeatureGate is the name of the FeatureGate as it appears in The FeatureGate CR instance.
	FeatureGate string `json:"featureGate"`

	// Tests contains an item for every TestName
	Tests []TestDetails `json:"tests"`
}

type TestDetails struct {
	// TestName is the name of the test as it appears in junit XMLs.
	// It does not include the suite name since the same test can be executed in many suites.
	TestName string `json:"testName"`
}

type TestReportingStatus struct {
}

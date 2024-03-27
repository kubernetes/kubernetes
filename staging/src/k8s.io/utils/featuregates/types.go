package featuregates

import "testing"

type FeatureGate interface {
	Name() string
	// Enabled indicates whether the gate is true or false.
	// It can be called *after* flags have been parsed.  If used prior to flag parsing, the results can be inconsistent.
	Enabled() bool
	// SetEnabledForTesting returns a function that will restore the previous enabled value.
	//Normal usage is `defer FeatureGate.SetEnabledForTesting()`
	SetEnabledForTesting(t *testing.T, enabled bool) RestoreFunc

	// stabilityLevel is used to bind to flags
	stabilityLevel() StabilityLevel
	// defaultValue is used to expose to flags
	defaultValue() bool
	// lockToDefalt is used to ensure flags and env vars aren't improperly specified
	lockToDefault() bool
	// setEnabled is used to when the flags are specified
	setEnabled(bool)
	// PreventOverridingDefault is called after the default value has been used to specify flags.
	preventOverridingDefault()
}

// RestoreFunc is returned for calls to SetEnabledForTest and should be deferred to put the value back the way it was.
type RestoreFunc func()

// MutableFeatureGate is an interface to assert when overriding default values.  It is not commonly used, but it is
// commonly implemented.  Use with care.
type MutableFeatureGate interface {
	FeatureGate
	// OverrideDefaultValue allows specifying a different default in a particular binary.
	// In general, this should only be called in the main for a particular binary.
	OverrideDefaultValue(bool)
}

type FeatureSet interface {
	// FeatureGates returns a list of all the FeatureGates.
	// Since general usage is to directly reference the gate, this isn't expected to be used often.
	featureGates() []FeatureGate
	// PreventAdditionalFeatureGates stops the list of gates of changing, but it doesn't affect the value of those gates.
	preventAdditionalFeatureGates()
}

type MutableFeatureSet interface {
	AddFeatureGates(...FeatureGate) error
	AddFeatureGatesOrDie(...FeatureGate)

	AddFeatureSets(...FeatureSet) error
	AddFeatureSetsOrDie(...FeatureSet)

	featureGates() []FeatureGate

	preventAdditionalFeatureGates()
}

type StabilityLevel string

const (
	// Values for PreRelease.
	Alpha = StabilityLevel("ALPHA")
	Beta  = StabilityLevel("BETA")
	GA    = StabilityLevel("")

	// Deprecated
	Deprecated = StabilityLevel("DEPRECATED")
)

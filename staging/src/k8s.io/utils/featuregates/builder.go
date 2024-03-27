package featuregates

import "fmt"

// FeatureGateBuilder is used to construct featureGates.  This is a builder pattern, so we can easily provide
// validation for field values and completeness when creating the FeatureGate.
type FeatureGateBuilder struct {
	name           string
	withEnvVar     bool
	defaultValue   bool
	lockToDefault  bool
	stabilityLevel StabilityLevel
}

// NewFeatureGate returns a convenient way to build featureGates that are no controllable via envvars. For most
// library-based callers, NewEnvVarFeatureGate is a better choice because it allows the FeatureGate to be controlled
// via an environment variable when not explicitly bound.
func NewFeatureGate(name string) *FeatureGateBuilder {
	return &FeatureGateBuilder{
		name: name,
	}
}

// NewEnvVarFeatureGate is a good choice for libraries that want to allow control of featureGates without rebuilding
// binaries and without explicitly wiring to a flag.
func NewEnvVarFeatureGate(name string) *FeatureGateBuilder {
	return &FeatureGateBuilder{
		name:       name,
		withEnvVar: true,
	}
}

func (b *FeatureGateBuilder) EnableByDefault() *FeatureGateBuilder {
	b.defaultValue = true
	return b
}

func (b *FeatureGateBuilder) LockToDefault() *FeatureGateBuilder {
	b.lockToDefault = true
	return b
}

func (b *FeatureGateBuilder) Alpha() *FeatureGateBuilder {
	b.stabilityLevel = Alpha
	return b
}
func (b *FeatureGateBuilder) Beta() *FeatureGateBuilder {
	b.stabilityLevel = Beta
	return b
}
func (b *FeatureGateBuilder) Stable() *FeatureGateBuilder {
	b.stabilityLevel = GA
	return b
}
func (b *FeatureGateBuilder) Deprecated() *FeatureGateBuilder {
	b.stabilityLevel = Deprecated
	return b
}

func (b *FeatureGateBuilder) ToFeatureGate() (FeatureGate, error) {
	if len(b.name) == 0 {
		return nil, fmt.Errorf("name is required")
	}
	if b.withEnvVar {
		return &envvarFeatureGate{
			name:              b.name,
			enabled:           b.defaultValue,
			defaultVal:        b.defaultValue,
			lockToDefaultVal:  b.lockToDefault,
			stabilityLevelVal: b.stabilityLevel,
		}, nil
	}

	return &simpleFeatureGate{
		name:              b.name,
		enabled:           b.defaultValue,
		defaultVal:        b.defaultValue,
		lockToDefaultVal:  b.lockToDefault,
		stabilityLevelVal: b.stabilityLevel,
	}, nil
}

func (b *FeatureGateBuilder) ToFeatureGateOrDie() FeatureGate {
	ret, err := b.ToFeatureGate()
	if err != nil {
		panic(err)
	}
	return ret
}

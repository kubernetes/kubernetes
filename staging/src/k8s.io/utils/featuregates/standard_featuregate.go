package featuregates

import (
	"sync"
	"testing"
)

type simpleFeatureGate struct {
	name              string
	enabled           bool
	defaultVal        bool
	lockToDefaultVal  bool
	stabilityLevelVal StabilityLevel

	lock   sync.Mutex
	closed bool
}

func (f *simpleFeatureGate) stabilityLevel() StabilityLevel {
	return f.stabilityLevelVal
}

func (f *simpleFeatureGate) setEnabled(in bool) {
	f.enabled = in
}

func (f *simpleFeatureGate) Name() string {
	return f.name
}

func (f *simpleFeatureGate) Enabled() bool {
	return f.enabled
}

func (f *simpleFeatureGate) defaultValue() bool {
	return f.defaultVal
}

func (f *simpleFeatureGate) lockToDefault() bool {
	return f.lockToDefaultVal
}

func (f *simpleFeatureGate) OverrideDefaultValue(in bool) {
	f.lock.Lock()
	defer f.lock.Unlock()

	if f.closed {
		panic("cannot override default because the flags have already been built")
	}
	f.defaultVal = in
}

func (f *simpleFeatureGate) preventOverridingDefault() {
	f.lock.Lock()
	defer f.lock.Unlock()

	f.closed = true
}

func (f *simpleFeatureGate) SetEnabledForTesting(t *testing.T, enabled bool) RestoreFunc {
	t.Helper()

	oldEnabled := f.enabled
	f.enabled = enabled

	return func() {
		f.enabled = oldEnabled
	}
}

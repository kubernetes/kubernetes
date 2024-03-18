package featuregates

import (
	"fmt"
	"k8s.io/klog/v2"
	"os"
	"strconv"
	"sync"
	"testing"

	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
)

type envvarFeatureGate struct {
	name              string
	defaultVal        bool
	lockToDefaultVal  bool
	stabilityLevelVal StabilityLevel

	lock    sync.RWMutex
	closed  bool
	enabled bool
	// readEnvVarsOnce guards reading environmental variables
	readEnvVarsOnce sync.Once
	readAtLeastOnce bool
}

func (f *envvarFeatureGate) stabilityLevel() StabilityLevel {
	return f.stabilityLevelVal
}

func (f *envvarFeatureGate) Name() string {
	return f.name
}

func (f *envvarFeatureGate) Enabled() bool {
	f.readEnvVarValue()

	f.lock.RLock()
	if f.readAtLeastOnce {
		defer f.lock.RUnlock()
		return f.enabled
	}

	f.lock.RUnlock()
	f.lock.Lock()
	defer f.lock.Unlock()
	f.readAtLeastOnce = true
	return f.enabled
}

func (f *envvarFeatureGate) defaultValue() bool {
	return f.defaultVal
}

func (f *envvarFeatureGate) setEnabled(in bool) {
	f.lock.Lock()
	defer f.lock.Unlock()

	if f.readAtLeastOnce {
		panic("cannot setEnabled after the gate has been read by the library")
	}

	f.enabled = in
}

func (f *envvarFeatureGate) OverrideDefaultValue(in bool) {
	f.lock.Lock()
	defer f.lock.Unlock()

	if f.closed {
		panic("cannot override default because the flags have already been built")
	}
	f.defaultVal = in
}

func (f *envvarFeatureGate) preventOverridingDefault() {
	f.lock.Lock()
	defer f.lock.Unlock()

	f.closed = true
}

func (f *envvarFeatureGate) lockToDefault() bool {
	return f.lockToDefaultVal
}

func (f *envvarFeatureGate) readEnvVarValue() {
	f.readEnvVarsOnce.Do(func() {
		feature := f.Name()
		featureState, featureStateSet := os.LookupEnv(fmt.Sprintf("KUBE_FEATURE_%s", feature))
		if !featureStateSet {
			return
		}
		klog.V(1).InfoS("Feature gate updated state", "feature", feature, "enabled", featureState)

		boolVal, boolErr := strconv.ParseBool(featureState)
		switch {
		case boolErr != nil:
			utilruntime.HandleError(fmt.Errorf("cannot set feature gate %q to %q, due to %v", feature, featureState, boolErr))
		case f.lockToDefault():
			if boolVal != f.defaultValue() {
				utilruntime.HandleError(fmt.Errorf("cannot set feature gate %q to %q, feature is locked to %v", feature, featureState, f.defaultValue()))
				break
			}
		default:
			f.lock.Lock()
			f.enabled = boolVal
			defer f.lock.Unlock()
		}
	})
}

func (f *envvarFeatureGate) SetEnabledForTesting(t *testing.T, enabled bool) RestoreFunc {
	t.Helper()

	oldEnabled := f.enabled
	f.enabled = enabled

	return func() {
		f.enabled = oldEnabled
	}
}

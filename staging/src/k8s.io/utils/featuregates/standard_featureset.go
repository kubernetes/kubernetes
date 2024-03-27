package featuregates

import (
	"fmt"
	"sort"
	"sync"
)

type simpleFeatureSet struct {
	nameToFeatureGate map[string]FeatureGate

	lock   sync.Mutex
	closed bool
}

func NewSimpleFeatureSet() MutableFeatureSet {
	return &simpleFeatureSet{
		nameToFeatureGate: map[string]FeatureGate{},
	}
}

func (f *simpleFeatureSet) addFeatureGate(in FeatureGate) error {
	f.lock.Lock()
	defer f.lock.Unlock()
	if f.closed {
		return fmt.Errorf("this FeatureSet is closed")
	}

	existing, ok := f.nameToFeatureGate[in.Name()]
	if existing == in {
		// adding the exact same instance of the same FeatureGate is acceptable.
		// could be done with a %p, but the unit test suggests this is working ok.
		return nil
	}
	if ok {
		return fmt.Errorf("featureGate/%q already added", in.Name())
	}
	f.nameToFeatureGate[in.Name()] = in
	return nil
}

func (f *simpleFeatureSet) AddFeatureGates(in ...FeatureGate) error {
	errs := []error{}
	for i := range in {
		if err := f.addFeatureGate(in[i]); err != nil {
			errs = append(errs, err)
		}
	}
	if len(errs) == 0 {
		return nil
	}

	// TODO, do something nicer here.  Shall we move aggregated errors?
	return errs[0]
}

func (f *simpleFeatureSet) AddFeatureGatesOrDie(in ...FeatureGate) {
	if err := f.AddFeatureGates(in...); err != nil {
		panic(err)
	}
}

func (f *simpleFeatureSet) addFeatureSet(in FeatureSet) error {
	inGates := in.featureGates()
	if err := f.AddFeatureGates(inGates...); err != nil {
		return err
	}
	return nil
}

func (f *simpleFeatureSet) AddFeatureSets(inSets ...FeatureSet) error {
	errs := []error{}
	for i := range inSets {
		if err := f.addFeatureSet(inSets[i]); err != nil {
			errs = append(errs, err)
		}
	}
	if len(errs) == 0 {
		return nil
	}

	// TODO, do something nicer here.  Shall we move aggregated errors?
	return errs[0]
}

func (f *simpleFeatureSet) AddFeatureSetsOrDie(in ...FeatureSet) {
	if err := f.AddFeatureSets(in...); err != nil {
		panic(err)
	}
}

func (f *simpleFeatureSet) featureGates() []FeatureGate {
	keys := []string{}
	for k := range f.nameToFeatureGate {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	ret := []FeatureGate{}
	for _, fgName := range keys {
		ret = append(ret, f.nameToFeatureGate[fgName])
	}
	return ret
}

func (f *simpleFeatureSet) preventAdditionalFeatureGates() {
	f.lock.Lock()
	defer f.lock.Unlock()
	// TODO(mtaufen): Shouldn't we just close it on the first Set/SetFromMap instead?
	// Not all components expose a feature gates flag using this AddFlags method, and
	// in the future, all components will completely stop exposing a feature gates flag,
	// in favor of componentconfig.
	f.closed = true

	for _, featureGate := range f.nameToFeatureGate {
		featureGate.preventOverridingDefault()
	}
}

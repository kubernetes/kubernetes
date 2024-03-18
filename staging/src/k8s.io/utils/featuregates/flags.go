package featuregates

import (
	"fmt"
	"github.com/spf13/pflag"
	"sort"
	"strconv"
	"strings"
)

const (
	flagName = "feature-gates"

	// allAlphaGate is a global toggle for alpha features. Per-feature key
	// values override the default set by allAlphaGate. Examples:
	//   AllAlpha=false,NewFeature=true  will result in newFeature=true
	//   AllAlpha=true,NewFeature=false  will result in newFeature=false
	allAlphaGate = "AllAlpha"

	// allBetaGate is a global toggle for beta features. Per-feature key
	// values override the default set by allBetaGate. Examples:
	//   AllBeta=false,NewFeature=true  will result in NewFeature=true
	//   AllBeta=true,NewFeature=false  will result in NewFeature=false
	allBetaGate = "AllBeta"
)

// Trying to avoid pflag dependency
type Value interface {
	String() string
	Set(string) error
	Type() string
}

// Trying to avoid pflag dependency
type FlagSet interface {
	Var(value Value, name string, usage string)
}

type FeatureSetOptions struct {
	delegate FeatureSet
}

func NewFeatureSetOptions(featureSet FeatureSet) *FeatureSetOptions {
	return &FeatureSetOptions{
		delegate: featureSet,
	}
}

// AddFlags adds a flag for setting global feature gates to the specified FlagSet.
func (f *FeatureSetOptions) AddFlags(fs *pflag.FlagSet) {
	f.delegate.preventAdditionalFeatureGates()

	flagImpl := &featureGateFlagValue{
		delegate: f.delegate,
	}

	helpStrings := []string{
		fmt.Sprintf("%s=true|false (%s - default=%t)", allAlphaGate, Alpha, false),
		fmt.Sprintf("%s=true|false (%s - default=%t)", allBetaGate, Beta, false),
	}
	for _, featureGate := range f.delegate.featureGates() {
		helpStrings = append(helpStrings, fmt.Sprintf("%s=true|false (%s - default=%t)", featureGate.Name(), featureGate.stabilityLevel(), featureGate.defaultValue()))
	}
	sort.Strings(helpStrings)

	fs.Var(flagImpl, flagName, ""+
		"A set of key=value pairs that describe feature gates for alpha/experimental features. "+
		"Options are:\n"+strings.Join(helpStrings, "\n"))
}

type featureGateFlagValue struct {
	delegate FeatureSet
}

// Set parses a string of the form "key1=value1,key2=value2,..." into a
// map[string]bool of known keys or returns an error.
func (f *featureGateFlagValue) Set(value string) error {
	m := make(map[string]bool)
	for _, s := range strings.Split(value, ",") {
		if len(s) == 0 {
			continue
		}
		arr := strings.SplitN(s, "=", 2)
		k := strings.TrimSpace(arr[0])
		if len(arr) != 2 {
			return fmt.Errorf("missing bool value for %s", k)
		}
		v := strings.TrimSpace(arr[1])
		boolValue, err := strconv.ParseBool(v)
		if err != nil {
			return fmt.Errorf("invalid value of %s=%s, err: %v", k, v, err)
		}
		m[k] = boolValue
	}
	return f.setOnDelegate(m)
}

func (f *featureGateFlagValue) setOnDelegate(m map[string]bool) error {
	featureGateNames := map[string]FeatureGate{}
	for i := range f.delegate.featureGates() {
		featureGate := f.delegate.featureGates()[i]
		featureGateNames[featureGate.Name()] = featureGate
	}

	for featureGateName, enabled := range m {
		switch {
		case featureGateName == allAlphaGate:
			for _, featureGate := range f.delegate.featureGates() {
				if featureGate.stabilityLevel() == Alpha {
					featureGate.setEnabled(enabled)
				}
			}
			continue
		case featureGateName == allBetaGate:
			for _, featureGate := range f.delegate.featureGates() {
				if featureGate.stabilityLevel() == Beta {
					featureGate.setEnabled(enabled)
				}
			}
			continue
		}

		featureGate, exists := featureGateNames[featureGateName]
		if !exists {
			return fmt.Errorf("unrecognized feature gate: %s", featureGateName)
		}
		if featureGate.lockToDefault() && featureGate.defaultValue() != enabled {
			return fmt.Errorf("cannot set feature gate %v to %v, feature is locked to %v", featureGateName, enabled, featureGate.defaultValue())
		}
		featureGate.setEnabled(enabled)
	}

	return nil
}

// String returns a string containing all enabled feature gates, formatted as "key1=value1,key2=value2,...".
func (f *featureGateFlagValue) String() string {
	pairs := []string{}
	knownFeatureGates := f.delegate.featureGates()
	for _, featureGate := range knownFeatureGates {
		pairs = append(pairs, fmt.Sprintf("%s=%t", featureGate.Name(), featureGate.defaultValue()))
	}
	pairs = append(pairs, fmt.Sprintf("%s=false", allAlphaGate))
	pairs = append(pairs, fmt.Sprintf("%s=false", allBetaGate))
	sort.Strings(pairs)

	return strings.Join(pairs, ",")
}

func (f *featureGateFlagValue) Type() string {
	return "mapStringBool"
}

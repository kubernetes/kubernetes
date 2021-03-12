package viper

import "github.com/spf13/pflag"

// FlagValueSet is an interface that users can implement
// to bind a set of flags to viper.
type FlagValueSet interface {
	VisitAll(fn func(FlagValue))
}

// FlagValue is an interface that users can implement
// to bind different flags to viper.
type FlagValue interface {
	HasChanged() bool
	Name() string
	ValueString() string
	ValueType() string
}

// pflagValueSet is a wrapper around *pflag.ValueSet
// that implements FlagValueSet.
type pflagValueSet struct {
	flags *pflag.FlagSet
}

// VisitAll iterates over all *pflag.Flag inside the *pflag.FlagSet.
func (p pflagValueSet) VisitAll(fn func(flag FlagValue)) {
	p.flags.VisitAll(func(flag *pflag.Flag) {
		fn(pflagValue{flag})
	})
}

// pflagValue is a wrapper aroung *pflag.flag
// that implements FlagValue
type pflagValue struct {
	flag *pflag.Flag
}

// HasChanged returns whether the flag has changes or not.
func (p pflagValue) HasChanged() bool {
	return p.flag.Changed
}

// Name returns the name of the flag.
func (p pflagValue) Name() string {
	return p.flag.Name
}

// ValueString returns the value of the flag as a string.
func (p pflagValue) ValueString() string {
	return p.flag.Value.String()
}

// ValueType returns the type of the flag as a string.
func (p pflagValue) ValueType() string {
	return p.flag.Value.Type()
}

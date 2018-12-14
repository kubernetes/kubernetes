package flags

import (
	"errors"

	"github.com/spf13/pflag"
)

// ConfigFlagSet wraps a pflag.FlagSet, separating defaulting, parsing and application.
type ConfigFlagSet struct {
	fs         pflag.FlagSet
	applyFuncs map[string]func()
	parsed     bool
}

// Parse processes the command line flags, but does not apply them.
func (fs *ConfigFlagSet) Parse(arguments []string) error {
	if fs.parsed {
		return errors.New("ConfigFlagSet already parsed")
	}
	fs.parsed = true
	return fs.fs.Parse(arguments)
}

// Apply copies the parsed flag values (those which were passed in the command line) to the destination.
func (fs *ConfigFlagSet) Apply() {
	if !fs.parsed {
		panic("ConfigFlagSet not parsed yet")
	}
	for name, apply := range fs.applyFuncs {
		if fs.fs.Changed(name) {
			apply()
		}
	}
}

func (fs *ConfigFlagSet) StringSliceVar(target *[]string, name string, usage string, def []string, merge func([]string)) {
	tmp := make([]string, len(def), 0)
	fs.fs.StringSliceVar(&tmp, name, def, usage)
	if fs.applyFuncs == nil {
		fs.applyFuncs = map[string]func(){}
	}
	fs.applyFuncs[name] = func() {
		if merge == nil {
			*target = tmp
		} else {
			merge(tmp)
		}
	}
}

func (fs *ConfigFlagSet) Int32Var(target *int32, name string, usage string, def int32, merge func(int32)) {
	var tmp int32
	fs.fs.Int32Var(&tmp, name, def, usage)
	if fs.applyFuncs == nil {
		fs.applyFuncs = map[string]func(){}
	}
	fs.applyFuncs[name] = func() {
		if merge == nil {
			*target = tmp
		} else {
			merge(tmp)
		}
	}
}

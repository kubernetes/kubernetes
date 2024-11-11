package flags

import (
	"github.com/spf13/pflag"

	"github.com/openshift-eng/openshift-tests-extension/pkg/extension"
)

// ComponentFlags contains information for specifying the component.
type ComponentFlags struct {
	Component string
}

func NewComponentFlags() *ComponentFlags {
	return &ComponentFlags{
		Component: extension.DefaultExtension,
	}
}

func (f *ComponentFlags) BindFlags(fs *pflag.FlagSet) {
	fs.StringVar(&f.Component,
		"component",
		f.Component,
		"specify the component to enable")
}

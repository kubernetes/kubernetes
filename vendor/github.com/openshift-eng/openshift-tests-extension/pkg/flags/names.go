package flags

import (
	"github.com/spf13/pflag"
)

// NamesFlags contains information for specifying multiple test names.
type NamesFlags struct {
	Names []string
}

func NewNamesFlags() *NamesFlags {
	return &NamesFlags{
		Names: []string{},
	}
}

func (f *NamesFlags) BindFlags(fs *pflag.FlagSet) {
	fs.StringArrayVarP(&f.Names,
		"names",
		"n",
		f.Names,
		"specify test name (can be specified multiple times)")
}

package flags

import (
	"github.com/spf13/pflag"
)

// SuiteFlags contains information for specifying the suite.
type SuiteFlags struct {
	Suite string
}

func NewSuiteFlags() *SuiteFlags {
	return &SuiteFlags{}
}

func (f *SuiteFlags) BindFlags(fs *pflag.FlagSet) {
	fs.StringVar(&f.Suite,
		"suite",
		f.Suite,
		"specify the suite to use")
}

package featuregate

import (
	"flag"

	"github.com/spf13/pflag"
)

// flagSet is the interface implemented by pflag.FlagSet, with
// just those methods defined which are needed by addFlags.
type flagSet interface {
	Var(value pflag.Value, name string, usage string)
}

// goFlagSet implements flagSet for a stdlib flag.FlagSet.
type goFlagSet struct {
	*flag.FlagSet
}

func (fs goFlagSet) Var(value pflag.Value, name string, usage string) {
	fs.FlagSet.Var(value, name, usage)
}

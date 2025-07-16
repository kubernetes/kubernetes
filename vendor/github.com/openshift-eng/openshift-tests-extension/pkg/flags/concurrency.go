package flags

import "github.com/spf13/pflag"

// ConcurrencyFlags contains information for configuring concurrency
type ConcurrencyFlags struct {
	MaxConcurency int
}

func NewConcurrencyFlags() *ConcurrencyFlags {
	return &ConcurrencyFlags{
		MaxConcurency: 10,
	}
}

func (f *ConcurrencyFlags) BindFlags(fs *pflag.FlagSet) {
	fs.IntVarP(&f.MaxConcurency,
		"max-concurrency",
		"c",
		f.MaxConcurency,
		"maximum number of tests to run in parallel",
	)
}

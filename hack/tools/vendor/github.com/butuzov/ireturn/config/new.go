package config

import (
	"errors"
	"flag"
	"strings"

	"github.com/butuzov/ireturn/types"
)

var ErrCollisionOfInterests = errors.New("can't have both `-accept` and `-reject` specified at same time")

//nolint: exhaustivestruct
func DefaultValidatorConfig() *allowConfig {
	return allowAll([]string{
		types.NameEmpty,  // "empty": empty interfaces (interface{})
		types.NameError,  // "error": for all error's
		types.NameAnon,   // "anon": for all empty interfaces with methods (interface {Method()})
		types.NameStdLib, // "std": for all standard library packages
	})
}

// New is factory function that return allowConfig or rejectConfig depending
// on provided arguments.
func New(fs *flag.FlagSet) (interface{}, error) {
	var (
		allowList  = toSlice(getFlagVal(fs, "allow"))
		rejectList = toSlice(getFlagVal(fs, "reject"))
	)

	// can't have both at same time.
	if len(allowList) != 0 && len(rejectList) != 0 {
		return nil, ErrCollisionOfInterests
	}

	switch {
	case len(allowList) > 0:
		return allowAll(allowList), nil
	case len(rejectList) > 0:
		return rejectAll(rejectList), nil
	}

	// can have none at same time.
	return nil, nil
}

// both constants used to cleanup items provided in comma separated list.
const (
	SepTab   string = " "
	SepSpace string = "	"
)

func toSlice(s string) []string {
	var results []string

	for _, pattern := range strings.Split(s, ",") {
		pattern = strings.Trim(pattern, SepTab+SepSpace)
		if pattern != "" {
			results = append(results, pattern)
		}
	}

	return results
}

func getFlagVal(fs *flag.FlagSet, name string) string {
	flg := fs.Lookup(name)

	if flg == nil {
		return ""
	}

	return flg.Value.String()
}

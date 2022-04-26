package config

import (
	"regexp"

	"github.com/butuzov/ireturn/types"
)

// defaultConfig is core of the validation, ...
// todo(butuzov): write proper intro...

type defaultConfig struct {
	List []string

	// private fields (for search optimization look ups)
	init  bool
	quick uint8
	list  []*regexp.Regexp
}

func (config *defaultConfig) Has(i types.IFace) bool {
	if !config.init {
		config.compileList()
		config.init = true
	}

	if config.quick&uint8(i.Type) > 0 {
		return true
	}

	// not a named interface (because error, interface{}, anon interface has keywords.)
	if i.Type&types.NamedInterface == 0 && i.Type&types.NamedStdInterface == 0 {
		return false
	}

	for _, re := range config.list {
		if re.MatchString(i.Name) {
			return true
		}
	}

	return false
}

// compileList will transform text list into a bitmask for quick searches and
// slice of regular expressions for quick searches.
func (config *defaultConfig) compileList() {
	for _, str := range config.List {
		switch str {
		case types.NameError:
			config.quick |= uint8(types.ErrorInterface)
		case types.NameEmpty:
			config.quick |= uint8(types.EmptyInterface)
		case types.NameAnon:
			config.quick |= uint8(types.AnonInterface)
		case types.NameStdLib:
			config.quick |= uint8(types.NamedStdInterface)
		}

		// allow to parse regular expressions
		// todo(butuzov): how can we log error in golangci-lint?
		if re, err := regexp.Compile(str); err == nil {
			config.list = append(config.list, re)
		}
	}
}

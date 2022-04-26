package config

import "github.com/butuzov/ireturn/types"

// allowConfig specifies a list of interfaces (keywords, patters and regular expressions)
// that are allowed by ireturn as valid to return, any non listed interface are rejected.
type allowConfig struct {
	*defaultConfig
}

func allowAll(patterns []string) *allowConfig {
	return &allowConfig{&defaultConfig{List: patterns}}
}

func (ac *allowConfig) IsValid(i types.IFace) bool {
	return ac.Has(i)
}

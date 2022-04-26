package config

import (
	"regexp"
	"strings"
)

type Config struct {
	Checks           map[string]bool
	IgnoredNumbers   map[string]struct{}
	IgnoredFunctions []*regexp.Regexp
	IgnoredFiles     []*regexp.Regexp
}

type Option func(config *Config)

func DefaultConfig() *Config {
	return &Config{
		Checks: map[string]bool{},
		IgnoredNumbers: map[string]struct{}{
			"0":   {},
			"0.0": {},
			"1":   {},
			"1.0": {},
		},
		IgnoredFiles: []*regexp.Regexp{
			regexp.MustCompile(`_test.go`),
		},
		IgnoredFunctions: []*regexp.Regexp{
			regexp.MustCompile(`time.Date`),
		},
	}
}

func WithOptions(options ...Option) *Config {
	c := DefaultConfig()

	for _, option := range options {
		option(c)
	}

	return c
}

func WithIgnoredFunctions(excludes string) Option {
	return func(config *Config) {
		for _, exclude := range strings.Split(excludes, ",") {
			if exclude == "" {
				continue
			}
			config.IgnoredFunctions = append(config.IgnoredFunctions, regexp.MustCompile(exclude))
		}
	}
}

func WithIgnoredFiles(excludes string) Option {
	return func(config *Config) {
		for _, exclude := range strings.Split(excludes, ",") {
			if exclude == "" {
				continue
			}
			config.IgnoredFiles = append(config.IgnoredFiles, regexp.MustCompile(exclude))
		}
	}
}

func WithIgnoredNumbers(numbers string) Option {
	return func(config *Config) {
		for _, number := range strings.Split(numbers, ",") {
			if number == "" {
				continue
			}
			config.IgnoredNumbers[config.removeDigitSeparator(number)] = struct{}{}
		}
	}
}

func WithCustomChecks(checks string) Option {
	return func(config *Config) {
		if checks == "" {
			return
		}

		for name, _ := range config.Checks {
			config.Checks[name] = false
		}

		for _, name := range strings.Split(checks, ",") {
			if name == "" {
				continue
			}
			config.Checks[name] = true
		}
	}
}

func (c *Config) IsCheckEnabled(name string) bool {
	return c.Checks[name]
}

func (c *Config) IsIgnoredNumber(number string) bool {
	_, ok := c.IgnoredNumbers[c.removeDigitSeparator(number)]
	return ok
}

func (c *Config) IsIgnoredFunction(f string) bool {
	for _, pattern := range c.IgnoredFunctions {
		if pattern.MatchString(f) {
			return true
		}
	}

	return false
}

func (c *Config) removeDigitSeparator(number string) string {
	return strings.Replace(number, "_", "", -1)
}

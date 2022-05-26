package lint

// Arguments is type used for the arguments of a rule.
type Arguments = []interface{}

// RuleConfig is type used for the rule configuration.
type RuleConfig struct {
	Arguments Arguments
	Severity  Severity
	Disabled  bool
}

// RulesConfig defines the config for all rules.
type RulesConfig = map[string]RuleConfig

// DirectiveConfig is type used for the linter directive configuration.
type DirectiveConfig struct {
	Severity Severity
}

// DirectivesConfig defines the config for all directives.
type DirectivesConfig = map[string]DirectiveConfig

// Config defines the config of the linter.
type Config struct {
	IgnoreGeneratedHeader bool `toml:"ignoreGeneratedHeader"`
	Confidence            float64
	Severity              Severity
	EnableAllRules        bool             `toml:"enableAllRules"`
	Rules                 RulesConfig      `toml:"rule"`
	ErrorCode             int              `toml:"errorCode"`
	WarningCode           int              `toml:"warningCode"`
	Directives            DirectivesConfig `toml:"directive"`
	Exclude               []string         `toml:"exclude"`
}

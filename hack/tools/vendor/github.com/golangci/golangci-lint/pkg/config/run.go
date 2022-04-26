package config

import "time"

// Run encapsulates the config options for running the linter analysis.
type Run struct {
	IsVerbose           bool `mapstructure:"verbose"`
	Silent              bool
	CPUProfilePath      string
	MemProfilePath      string
	TracePath           string
	Concurrency         int
	PrintResourcesUsage bool `mapstructure:"print-resources-usage"`

	Config   string // The path to the golangci config file, as specified with the --config argument.
	NoConfig bool

	Args []string

	Go string `mapstructure:"go"`

	BuildTags           []string `mapstructure:"build-tags"`
	ModulesDownloadMode string   `mapstructure:"modules-download-mode"`

	ExitCodeIfIssuesFound int  `mapstructure:"issues-exit-code"`
	AnalyzeTests          bool `mapstructure:"tests"`

	// Deprecated: Deadline exists for historical compatibility
	// and should not be used. To set run timeout use Timeout instead.
	Deadline time.Duration
	Timeout  time.Duration

	PrintVersion       bool
	SkipFiles          []string `mapstructure:"skip-files"`
	SkipDirs           []string `mapstructure:"skip-dirs"`
	UseDefaultSkipDirs bool     `mapstructure:"skip-dirs-use-default"`

	AllowParallelRunners bool `mapstructure:"allow-parallel-runners"`
	AllowSerialRunners   bool `mapstructure:"allow-serial-runners"`
}

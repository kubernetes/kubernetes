package config

const (
	OutFormatJSON              = "json"
	OutFormatLineNumber        = "line-number"
	OutFormatColoredLineNumber = "colored-line-number"
	OutFormatTab               = "tab"
	OutFormatCheckstyle        = "checkstyle"
	OutFormatCodeClimate       = "code-climate"
	OutFormatHTML              = "html"
	OutFormatJunitXML          = "junit-xml"
	OutFormatGithubActions     = "github-actions"
)

var OutFormats = []string{
	OutFormatColoredLineNumber,
	OutFormatLineNumber,
	OutFormatJSON,
	OutFormatTab,
	OutFormatCheckstyle,
	OutFormatCodeClimate,
	OutFormatHTML,
	OutFormatJunitXML,
	OutFormatGithubActions,
}

type Output struct {
	Format              string
	Color               string
	PrintIssuedLine     bool   `mapstructure:"print-issued-lines"`
	PrintLinterName     bool   `mapstructure:"print-linter-name"`
	UniqByLine          bool   `mapstructure:"uniq-by-line"`
	SortResults         bool   `mapstructure:"sort-results"`
	PrintWelcomeMessage bool   `mapstructure:"print-welcome"`
	PathPrefix          string `mapstructure:"path-prefix"`
}

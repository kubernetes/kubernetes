package imports

import (
	"sort"
	"strings"
	"unicode"

	"github.com/daixiang0/gci/pkg/configuration"
	"github.com/daixiang0/gci/pkg/constants"
)

type ImportDef struct {
	Alias         string
	QuotedPath    string
	PrefixComment []string
	InlineComment string
}

func (i ImportDef) Path() string {
	return strings.TrimSuffix(strings.TrimPrefix(i.QuotedPath, string('"')), string('"'))
}

// Validate checks whether the contents are valid for an import
func (i ImportDef) Validate() error {
	err := checkAlias(i.Alias)
	if err != nil {
		return ValidationError{err}
	}
	if !strings.HasPrefix(i.QuotedPath, string('"')) {
		return MissingOpeningQuotesError
	}
	if !strings.HasSuffix(i.QuotedPath, string('"')) {
		return MissingClosingQuotesError
	}
	return nil
}

func checkAlias(alias string) error {
	for idx, r := range alias {
		if !unicode.IsLetter(r) {
			if r != '_' && r != '.' {
				if idx == 0 || !unicode.IsDigit(r) {
					// aliases may not start with a digit
					return InvalidCharacterError{r, alias}
				}
			}
		}
	}
	return nil
}

func (i ImportDef) String() string {
	return i.QuotedPath
}

// useful for logging statements
func (i ImportDef) UnquotedString() string {
	return strings.Trim(i.QuotedPath, "\"")
}

func (i ImportDef) Format(cfg configuration.FormatterConfiguration) string {
	linePrefix := constants.Indent
	var output string
	if cfg.NoPrefixComments == false || i.QuotedPath == `"C"` {
		for _, prefixComment := range i.PrefixComment {
			output += linePrefix + prefixComment + constants.Linebreak
		}
	}
	output += linePrefix
	if i.Alias != "" {
		output += i.Alias + constants.Blank
	}
	output += i.QuotedPath
	if cfg.NoInlineComments == false {
		if i.InlineComment != "" {
			output += constants.Blank + i.InlineComment
		}
	}
	output += constants.Linebreak
	return output
}

func SortImportsByPath(imports []ImportDef) []ImportDef {
	sort.Slice(
		imports,
		func(i, j int) bool {
			return sort.StringsAreSorted([]string{imports[i].Path(), imports[j].Path()})
		},
	)
	return imports
}

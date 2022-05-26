// Package nolintlint provides a linter to ensure that all //nolint directives are followed by explanations
package nolintlint

import (
	"fmt"
	"go/ast"
	"go/token"
	"regexp"
	"strings"
	"unicode"

	"github.com/golangci/golangci-lint/pkg/result"
)

type BaseIssue struct {
	fullDirective                     string
	directiveWithOptionalLeadingSpace string
	position                          token.Position
	replacement                       *result.Replacement
}

//nolint:gocritic // TODO must be change in the future.
func (b BaseIssue) Position() token.Position {
	return b.position
}

//nolint:gocritic // TODO must be change in the future.
func (b BaseIssue) Replacement() *result.Replacement {
	return b.replacement
}

type ExtraLeadingSpace struct {
	BaseIssue
}

//nolint:gocritic // TODO must be change in the future.
func (i ExtraLeadingSpace) Details() string {
	return fmt.Sprintf("directive `%s` should not have more than one leading space", i.fullDirective)
}

//nolint:gocritic // TODO must be change in the future.
func (i ExtraLeadingSpace) String() string { return toString(i) }

type NotMachine struct {
	BaseIssue
}

//nolint:gocritic // TODO must be change in the future.
func (i NotMachine) Details() string {
	expected := i.fullDirective[:2] + strings.TrimLeftFunc(i.fullDirective[2:], unicode.IsSpace)
	return fmt.Sprintf("directive `%s` should be written without leading space as `%s`",
		i.fullDirective, expected)
}

//nolint:gocritic // TODO must be change in the future.
func (i NotMachine) String() string { return toString(i) }

type NotSpecific struct {
	BaseIssue
}

//nolint:gocritic // TODO must be change in the future.
func (i NotSpecific) Details() string {
	return fmt.Sprintf("directive `%s` should mention specific linter such as `%s:my-linter`",
		i.fullDirective, i.directiveWithOptionalLeadingSpace)
}

//nolint:gocritic // TODO must be change in the future.
func (i NotSpecific) String() string { return toString(i) }

type ParseError struct {
	BaseIssue
}

//nolint:gocritic // TODO must be change in the future.
func (i ParseError) Details() string {
	return fmt.Sprintf("directive `%s` should match `%s[:<comma-separated-linters>] [// <explanation>]`",
		i.fullDirective,
		i.directiveWithOptionalLeadingSpace)
}

//nolint:gocritic // TODO must be change in the future.
func (i ParseError) String() string { return toString(i) }

type NoExplanation struct {
	BaseIssue
	fullDirectiveWithoutExplanation string
}

//nolint:gocritic // TODO must be change in the future.
func (i NoExplanation) Details() string {
	return fmt.Sprintf("directive `%s` should provide explanation such as `%s // this is why`",
		i.fullDirective, i.fullDirectiveWithoutExplanation)
}

//nolint:gocritic // TODO must be change in the future.
func (i NoExplanation) String() string { return toString(i) }

type UnusedCandidate struct {
	BaseIssue
	ExpectedLinter string
}

//nolint:gocritic // TODO must be change in the future.
func (i UnusedCandidate) Details() string {
	details := fmt.Sprintf("directive `%s` is unused", i.fullDirective)
	if i.ExpectedLinter != "" {
		details += fmt.Sprintf(" for linter %q", i.ExpectedLinter)
	}
	return details
}

//nolint:gocritic // TODO must be change in the future.
func (i UnusedCandidate) String() string { return toString(i) }

func toString(i Issue) string {
	return fmt.Sprintf("%s at %s", i.Details(), i.Position())
}

type Issue interface {
	Details() string
	Position() token.Position
	String() string
	Replacement() *result.Replacement
}

type Needs uint

const (
	NeedsMachineOnly Needs = 1 << iota
	NeedsSpecific
	NeedsExplanation
	NeedsUnused
	NeedsAll = NeedsMachineOnly | NeedsSpecific | NeedsExplanation
)

var commentPattern = regexp.MustCompile(`^//\s*(nolint)(:\s*[\w-]+\s*(?:,\s*[\w-]+\s*)*)?\b`)

// matches a complete nolint directive
var fullDirectivePattern = regexp.MustCompile(`^//\s*nolint(?::(\s*[\w-]+\s*(?:,\s*[\w-]+\s*)*))?\s*(//.*)?\s*\n?$`)

type Linter struct {
	needs           Needs // indicates which linter checks to perform
	excludeByLinter map[string]bool
}

// NewLinter creates a linter that enforces that the provided directives fulfill the provided requirements
func NewLinter(needs Needs, excludes []string) (*Linter, error) {
	excludeByName := make(map[string]bool)
	for _, e := range excludes {
		excludeByName[e] = true
	}

	return &Linter{
		needs:           needs,
		excludeByLinter: excludeByName,
	}, nil
}

var leadingSpacePattern = regexp.MustCompile(`^//(\s*)`)
var trailingBlankExplanation = regexp.MustCompile(`\s*(//\s*)?$`)

//nolint:funlen,gocyclo
func (l Linter) Run(fset *token.FileSet, nodes ...ast.Node) ([]Issue, error) {
	var issues []Issue

	for _, node := range nodes {
		file, ok := node.(*ast.File)
		if !ok {
			continue
		}

		for _, c := range file.Comments {
			for _, comment := range c.List {
				if !commentPattern.MatchString(comment.Text) {
					continue
				}

				// check for a space between the "//" and the directive
				leadingSpaceMatches := leadingSpacePattern.FindStringSubmatch(comment.Text)

				var leadingSpace string
				if len(leadingSpaceMatches) > 0 {
					leadingSpace = leadingSpaceMatches[1]
				}

				directiveWithOptionalLeadingSpace := comment.Text
				if len(leadingSpace) > 0 {
					split := strings.Split(strings.SplitN(comment.Text, ":", 2)[0], "//")
					directiveWithOptionalLeadingSpace = "// " + strings.TrimSpace(split[1])
				}

				pos := fset.Position(comment.Pos())
				end := fset.Position(comment.End())

				base := BaseIssue{
					fullDirective:                     comment.Text,
					directiveWithOptionalLeadingSpace: directiveWithOptionalLeadingSpace,
					position:                          pos,
				}

				// check for, report and eliminate leading spaces, so we can check for other issues
				if len(leadingSpace) > 0 {
					removeWhitespace := &result.Replacement{
						Inline: &result.InlineFix{
							StartCol:  pos.Column + 1,
							Length:    len(leadingSpace),
							NewString: "",
						},
					}
					if (l.needs & NeedsMachineOnly) != 0 {
						issue := NotMachine{BaseIssue: base}
						issue.BaseIssue.replacement = removeWhitespace
						issues = append(issues, issue)
					} else if len(leadingSpace) > 1 {
						issue := ExtraLeadingSpace{BaseIssue: base}
						issue.BaseIssue.replacement = removeWhitespace
						issue.BaseIssue.replacement.Inline.NewString = " " // assume a single space was intended
						issues = append(issues, issue)
					}
				}

				fullMatches := fullDirectivePattern.FindStringSubmatch(comment.Text)
				if len(fullMatches) == 0 {
					issues = append(issues, ParseError{BaseIssue: base})
					continue
				}

				lintersText, explanation := fullMatches[1], fullMatches[2]
				var linters []string
				if len(lintersText) > 0 {
					lls := strings.Split(lintersText, ",")
					linters = make([]string, 0, len(lls))
					rangeStart := (pos.Column - 1) + len("//") + len(leadingSpace) + len("nolint:")
					for i, ll := range lls {
						rangeEnd := rangeStart + len(ll)
						if i < len(lls)-1 {
							rangeEnd++ // include trailing comma
						}
						trimmedLinterName := strings.TrimSpace(ll)
						if trimmedLinterName != "" {
							linters = append(linters, trimmedLinterName)
						}
						rangeStart = rangeEnd
					}
				}

				if (l.needs & NeedsSpecific) != 0 {
					if len(linters) == 0 {
						issues = append(issues, NotSpecific{BaseIssue: base})
					}
				}

				// when detecting unused directives, we send all the directives through and filter them out in the nolint processor
				if (l.needs & NeedsUnused) != 0 {
					removeNolintCompletely := &result.Replacement{
						Inline: &result.InlineFix{
							StartCol:  pos.Column - 1,
							Length:    end.Column - pos.Column,
							NewString: "",
						},
					}

					if len(linters) == 0 {
						issue := UnusedCandidate{BaseIssue: base}
						issue.replacement = removeNolintCompletely
						issues = append(issues, issue)
					} else {
						for _, linter := range linters {
							issue := UnusedCandidate{BaseIssue: base, ExpectedLinter: linter}
							// only offer replacement if there is a single linter
							// because of issues around commas and the possibility of all
							// linters being removed
							if len(linters) == 1 {
								issue.replacement = removeNolintCompletely
							}
							issues = append(issues, issue)
						}
					}
				}

				if (l.needs&NeedsExplanation) != 0 && (explanation == "" || strings.TrimSpace(explanation) == "//") {
					needsExplanation := len(linters) == 0 // if no linters are mentioned, we must have explanation
					// otherwise, check if we are excluding all the mentioned linters
					for _, ll := range linters {
						if !l.excludeByLinter[ll] { // if a linter does require explanation
							needsExplanation = true
							break
						}
					}

					if needsExplanation {
						fullDirectiveWithoutExplanation := trailingBlankExplanation.ReplaceAllString(comment.Text, "")
						issues = append(issues, NoExplanation{
							BaseIssue:                       base,
							fullDirectiveWithoutExplanation: fullDirectiveWithoutExplanation,
						})
					}
				}
			}
		}
	}

	return issues, nil
}

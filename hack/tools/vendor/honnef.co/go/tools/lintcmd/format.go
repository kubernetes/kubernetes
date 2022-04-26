package lintcmd

import (
	"encoding/json"
	"fmt"
	"go/token"
	"io"
	"os"
	"path/filepath"
	"text/tabwriter"

	"honnef.co/go/tools/analysis/lint"
	"honnef.co/go/tools/sarif"
)

func shortPath(path string) string {
	cwd, err := os.Getwd()
	if err != nil {
		return path
	}
	if rel, err := filepath.Rel(cwd, path); err == nil && len(rel) < len(path) {
		return rel
	}
	return path
}

func relativePositionString(pos token.Position) string {
	s := shortPath(pos.Filename)
	if pos.IsValid() {
		if s != "" {
			s += ":"
		}
		s += fmt.Sprintf("%d:%d", pos.Line, pos.Column)
	}
	if s == "" {
		s = "-"
	}
	return s
}

type statter interface {
	Stats(total, errors, warnings, ignored int)
}

type complexFormatter interface {
	Start(checks []*lint.Analyzer)
	End()
}

type formatter interface {
	Format(p problem)
}

type textFormatter struct {
	W io.Writer
}

func (o textFormatter) Format(p problem) {
	fmt.Fprintf(o.W, "%s: %s\n", relativePositionString(p.Position), p.String())
	for _, r := range p.Related {
		fmt.Fprintf(o.W, "\t%s: %s\n", relativePositionString(r.Position), r.Message)
	}
}

type nullFormatter struct{}

func (nullFormatter) Format(problem) {}

type jsonFormatter struct {
	W io.Writer
}

func (o jsonFormatter) Format(p problem) {
	type location struct {
		File   string `json:"file"`
		Line   int    `json:"line"`
		Column int    `json:"column"`
	}
	type related struct {
		Location location `json:"location"`
		End      location `json:"end"`
		Message  string   `json:"message"`
	}
	jp := struct {
		Code     string    `json:"code"`
		Severity string    `json:"severity,omitempty"`
		Location location  `json:"location"`
		End      location  `json:"end"`
		Message  string    `json:"message"`
		Related  []related `json:"related,omitempty"`
	}{
		Code:     p.Category,
		Severity: p.Severity.String(),
		Location: location{
			File:   p.Position.Filename,
			Line:   p.Position.Line,
			Column: p.Position.Column,
		},
		End: location{
			File:   p.End.Filename,
			Line:   p.End.Line,
			Column: p.End.Column,
		},
		Message: p.Message,
	}
	for _, r := range p.Related {
		jp.Related = append(jp.Related, related{
			Location: location{
				File:   r.Position.Filename,
				Line:   r.Position.Line,
				Column: r.Position.Column,
			},
			End: location{
				File:   r.End.Filename,
				Line:   r.End.Line,
				Column: r.End.Column,
			},
			Message: r.Message,
		})
	}
	_ = json.NewEncoder(o.W).Encode(jp)
}

type stylishFormatter struct {
	W io.Writer

	prevFile string
	tw       *tabwriter.Writer
}

func (o *stylishFormatter) Format(p problem) {
	pos := p.Position
	if pos.Filename == "" {
		pos.Filename = "-"
	}

	if pos.Filename != o.prevFile {
		if o.prevFile != "" {
			o.tw.Flush()
			fmt.Fprintln(o.W)
		}
		fmt.Fprintln(o.W, pos.Filename)
		o.prevFile = pos.Filename
		o.tw = tabwriter.NewWriter(o.W, 0, 4, 2, ' ', 0)
	}
	fmt.Fprintf(o.tw, "  (%d, %d)\t%s\t%s\n", pos.Line, pos.Column, p.Category, p.Message)
	for _, r := range p.Related {
		fmt.Fprintf(o.tw, "    (%d, %d)\t\t  %s\n", r.Position.Line, r.Position.Column, r.Message)
	}
}

func (o *stylishFormatter) Stats(total, errors, warnings, ignored int) {
	if o.tw != nil {
		o.tw.Flush()
		fmt.Fprintln(o.W)
	}
	fmt.Fprintf(o.W, " ✖ %d problems (%d errors, %d warnings, %d ignored)\n",
		total, errors, warnings, ignored)
}

type sarifFormatter struct {
	checks []*lint.Analyzer
	run    sarif.Run
}

func (o *sarifFormatter) Start(checks []*lint.Analyzer) {
	// XXX URI-escape all locations

	cwd, _ := os.Getwd()
	o.checks = checks
	o.run = sarif.Run{
		Tool: sarif.Tool{
			Driver: sarif.ToolComponent{
				Name: "Staticcheck",
				// XXX version.Version is useless when it's "devel"
				// Version: version.Version, // XXX pass this through from the command
				// XXX SemanticVersion is useless when it's "devel"
				// XXX SemanticVersion shouldn't have the leading "v"
				// SemanticVersion: version.MachineVersion, // XXX pass this through from the command
				InformationURI: "https://staticcheck.io",
			},
		},
		Invocations: []sarif.Invocation{{
			CommandLine: "XXX",
			Arguments:   os.Args[1:],
			WorkingDirectory: sarif.ArtifactLocation{
				URI: cwd,
			},
			ExecutionSuccessful: true,
		}},
	}
	for _, c := range checks {
		o.run.Tool.Driver.Rules = append(o.run.Tool.Driver.Rules,
			sarif.ReportingDescriptor{
				// We don't set Name, as Name and ID mustn't be identical.
				ID: c.Analyzer.Name,
				ShortDescription: sarif.Message{
					Text:     c.Doc.Title,
					Markdown: c.Doc.Title,
				},
				HelpURI: "https://staticcheck.io/docs/checks#" + c.Analyzer.Name,
				// We use our markdown as the plain text version, too. We
				// use very little markdown, primarily quotations,
				// indented code blocks and backticks. All of these are
				// fine as plain text, too.
				Help: sarif.Message{
					// OPT(dh): don't compute the string multiple times
					Text:     c.Doc.String(),
					Markdown: c.Doc.String(),
				},
			})
	}
}

func (o *sarifFormatter) sarifLevel(p problem) string {
	// OPT(dh): this is O(n*m) for n analyzers and m problems
	for _, c := range o.checks {
		if c.Analyzer.Name == p.Category {
			switch c.Doc.Severity {
			case lint.SeverityNone:
				// no configured severity, default to warning
				return "warning"
			case lint.SeverityError:
				return "error"
			case lint.SeverityDeprecated:
				return "warning"
			case lint.SeverityWarning:
				return "warning"
			case lint.SeverityInfo:
				return "note"
			case lint.SeverityHint:
				return "note"
			default:
				// unreachable
				return "none"
			}
		}
	}
	// unreachable
	return "none"
}

// XXX don't add "Available since" (and other metadata) to rule descriptions.
// XXX set properties.tags – we can use different tags for the
//   staticcheck, simple, stylecheck and unused checks, so users can
//   filter their results
// XXX columnKind

func (o *sarifFormatter) Format(p problem) {
	// Notes on GitHub-specific restrictions:
	//
	// Result.Message needs to either have ID or Text set. Markdown
	// gets ignored. Text isn't treated verbatim however: Markdown
	// formatting gets stripped, except for links.
	//
	// GitHub does not display RelatedLocations. The only way to make
	// use of them is to link to them (via their ID) in the
	// Result.Message. And even then, it will only show the referred
	// line of code, not the message. We can duplicate the messages in
	// the Result.Message, but we can't even indent them, because
	// leading whitespace gets stripped.
	//
	// GitHub does use the Markdown version of rule help, but it
	// renders it the way it renders comments on issues – that is, it
	// turns line breaks into hard line breaks, even though it
	// shouldn't.
	//
	// GitHub doesn't make use of the tool's URI or version, nor of
	// the help URIs of rules.
	//
	// There does not seem to be a way of using SARIF for "normal" CI,
	// without results showing up as code scanning alerts. Also, a
	// SARIF file containing only warnings, no errors, will not fail
	// CI. GitHub does display some parts of SARIF results in PRs, but
	// most of the useful parts of SARIF, such as help text of rules,
	// is only accessible via the code scanning alerts, which are only
	// accessible by users with write permissions.
	//
	// Result.Suppressions is being ignored.
	//
	//
	// Notes on other tools
	//
	// VS Code Sarif viewer
	//
	// The Sarif viewer in VS Code displays the full message in the
	// tabular view, removing newlines. That makes our multi-line
	// messages (which we use as a workaround for missing related
	// information) very ugly.
	//
	// Much like GitHub, the Sarif viewer does not make related
	// information visible unless we explicitly refer to it in the
	// message.
	//
	// Suggested fixes are not exposed in any way.
	//
	// It only shows the shortDescription or fullDescription of a
	// rule, not its help. We can't put the help in fullDescription,
	// because the fullDescription isn't meant to be that long. For
	// example, GitHub displays it in a single line, under the
	// shortDescription.
	//
	// VS Code can filter based on Result.Suppressions, but it doesn't
	// display our suppression message. Also, by default, suppressed
	// results get shown, and the column indicating that a result is
	// suppressed is hidden, which makes for a confusing experience.
	//
	// When a rule has only an ID, no name, VS Code displays a
	// prominent dash in place of the name. When the name and ID are
	// identical, it prints both. However, we can't make them
	// identical, as SARIF requires that either the ID and name are
	// different, or that the name is omitted.

	r := sarif.Result{
		RuleID:  p.Category,
		Kind:    sarif.Fail,
		Level:   o.sarifLevel(p),
		Message: sarif.Message{Text: p.Message},
	}
	r.Locations = []sarif.Location{{
		PhysicalLocation: sarif.PhysicalLocation{
			ArtifactLocation: sarif.ArtifactLocation{
				URI:   "file://" + p.Position.Filename,
				Index: -1,
			},
			Region: sarif.Region{
				StartLine:   p.Position.Line,
				StartColumn: p.Position.Column, // XXX sarif is 1-based; are we?
				EndLine:     p.End.Line,
				EndColumn:   p.End.Column,
			},
		},
	}}
	for _, fix := range p.SuggestedFixes {
		sfix := sarif.Fix{
			Description: sarif.Message{Text: fix.Message},
		}
		// file name -> replacements
		changes := map[string][]sarif.Replacement{}
		for _, edit := range fix.TextEdits {
			changes[edit.Position.Filename] = append(changes[edit.Position.Filename], sarif.Replacement{
				DeletedRegion: sarif.Region{
					StartLine:   edit.Position.Line,
					StartColumn: edit.Position.Column,
					EndLine:     edit.End.Line,
					EndColumn:   edit.End.Column,
				},
				InsertedContent: sarif.ArtifactContent{
					Text: string(edit.NewText),
				},
			})
		}
		for path, replacements := range changes {
			sfix.ArtifactChanges = append(sfix.ArtifactChanges, sarif.ArtifactChange{
				ArtifactLocation: sarif.ArtifactLocation{
					URI: "file://" + path,
				},
				Replacements: replacements,
			})
		}
		r.Fixes = append(r.Fixes, sfix)
	}
	for i, related := range p.Related {
		r.Message.Text += fmt.Sprintf("\n\t[%s](%d)", related.Message, i+1)

		r.RelatedLocations = append(r.RelatedLocations,
			sarif.Location{
				ID: i + 1,
				Message: &sarif.Message{
					Text: related.Message,
				},
				PhysicalLocation: sarif.PhysicalLocation{
					ArtifactLocation: sarif.ArtifactLocation{
						URI:   "file://" + related.Position.Filename,
						Index: -1,
					},
					Region: sarif.Region{
						StartLine:   related.Position.Line,
						StartColumn: related.Position.Column,
						EndLine:     related.End.Line,
						EndColumn:   related.End.Column,
					},
				},
			})
	}

	if p.Severity == severityIgnored {
		r.Suppressions = []sarif.Suppression{{
			Kind:          "inSource",
			Justification: "to be filled in",
		}}
	} else {
		// We want an empty slice, not nil. SARIF differentiates
		// between the two. An empty slice means that the problem
		// wasn't suppressed, while nil means that we don't have the
		// information available.
		r.Suppressions = []sarif.Suppression{}
	}
	o.run.Results = append(o.run.Results, r)
}

func (o *sarifFormatter) End() {
	// XXX populate artifacts
	json.NewEncoder(os.Stdout).Encode(sarif.Log{
		Version: sarif.Version,
		Schema:  sarif.Schema,
		Runs:    []sarif.Run{o.run},
	})
}

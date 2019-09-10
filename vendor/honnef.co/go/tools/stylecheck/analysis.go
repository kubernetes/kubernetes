package stylecheck

import (
	"flag"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"honnef.co/go/tools/config"
	"honnef.co/go/tools/facts"
	"honnef.co/go/tools/internal/passes/buildssa"
	"honnef.co/go/tools/lint/lintutil"
)

func newFlagSet() flag.FlagSet {
	fs := flag.NewFlagSet("", flag.PanicOnError)
	fs.Var(lintutil.NewVersionFlag(), "go", "Target Go version")
	return *fs
}

var Analyzers = map[string]*analysis.Analyzer{
	"ST1000": {
		Name:     "ST1000",
		Run:      CheckPackageComment,
		Doc:      Docs["ST1000"].String(),
		Requires: []*analysis.Analyzer{},
		Flags:    newFlagSet(),
	},
	"ST1001": {
		Name:     "ST1001",
		Run:      CheckDotImports,
		Doc:      Docs["ST1001"].String(),
		Requires: []*analysis.Analyzer{facts.Generated, config.Analyzer},
		Flags:    newFlagSet(),
	},
	"ST1003": {
		Name:     "ST1003",
		Run:      CheckNames,
		Doc:      Docs["ST1003"].String(),
		Requires: []*analysis.Analyzer{facts.Generated, config.Analyzer},
		Flags:    newFlagSet(),
	},
	"ST1005": {
		Name:     "ST1005",
		Run:      CheckErrorStrings,
		Doc:      Docs["ST1005"].String(),
		Requires: []*analysis.Analyzer{buildssa.Analyzer},
		Flags:    newFlagSet(),
	},
	"ST1006": {
		Name:     "ST1006",
		Run:      CheckReceiverNames,
		Doc:      Docs["ST1006"].String(),
		Requires: []*analysis.Analyzer{buildssa.Analyzer, facts.Generated},
		Flags:    newFlagSet(),
	},
	"ST1008": {
		Name:     "ST1008",
		Run:      CheckErrorReturn,
		Doc:      Docs["ST1008"].String(),
		Requires: []*analysis.Analyzer{buildssa.Analyzer},
		Flags:    newFlagSet(),
	},
	"ST1011": {
		Name:  "ST1011",
		Run:   CheckTimeNames,
		Doc:   Docs["ST1011"].String(),
		Flags: newFlagSet(),
	},
	"ST1012": {
		Name:     "ST1012",
		Run:      CheckErrorVarNames,
		Doc:      Docs["ST1012"].String(),
		Requires: []*analysis.Analyzer{config.Analyzer},
		Flags:    newFlagSet(),
	},
	"ST1013": {
		Name:     "ST1013",
		Run:      CheckHTTPStatusCodes,
		Doc:      Docs["ST1013"].String(),
		Requires: []*analysis.Analyzer{facts.Generated, facts.TokenFile, config.Analyzer},
		Flags:    newFlagSet(),
	},
	"ST1015": {
		Name:     "ST1015",
		Run:      CheckDefaultCaseOrder,
		Doc:      Docs["ST1015"].String(),
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated, facts.TokenFile},
		Flags:    newFlagSet(),
	},
	"ST1016": {
		Name:     "ST1016",
		Run:      CheckReceiverNamesIdentical,
		Doc:      Docs["ST1016"].String(),
		Requires: []*analysis.Analyzer{buildssa.Analyzer},
		Flags:    newFlagSet(),
	},
	"ST1017": {
		Name:     "ST1017",
		Run:      CheckYodaConditions,
		Doc:      Docs["ST1017"].String(),
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated, facts.TokenFile},
		Flags:    newFlagSet(),
	},
	"ST1018": {
		Name:     "ST1018",
		Run:      CheckInvisibleCharacters,
		Doc:      Docs["ST1018"].String(),
		Requires: []*analysis.Analyzer{inspect.Analyzer},
		Flags:    newFlagSet(),
	},
}

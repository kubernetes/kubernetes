package simple

import (
	"flag"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
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
	"S1000": {
		Name:     "S1000",
		Run:      LintSingleCaseSelect,
		Doc:      Docs["S1000"].String(),
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
		Flags:    newFlagSet(),
	},
	"S1001": {
		Name:     "S1001",
		Run:      LintLoopCopy,
		Doc:      Docs["S1001"].String(),
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
		Flags:    newFlagSet(),
	},
	"S1002": {
		Name:     "S1002",
		Run:      LintIfBoolCmp,
		Doc:      Docs["S1002"].String(),
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
		Flags:    newFlagSet(),
	},
	"S1003": {
		Name:     "S1003",
		Run:      LintStringsContains,
		Doc:      Docs["S1003"].String(),
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
		Flags:    newFlagSet(),
	},
	"S1004": {
		Name:     "S1004",
		Run:      LintBytesCompare,
		Doc:      Docs["S1004"].String(),
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
		Flags:    newFlagSet(),
	},
	"S1005": {
		Name:     "S1005",
		Run:      LintUnnecessaryBlank,
		Doc:      Docs["S1005"].String(),
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
		Flags:    newFlagSet(),
	},
	"S1006": {
		Name:     "S1006",
		Run:      LintForTrue,
		Doc:      Docs["S1006"].String(),
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
		Flags:    newFlagSet(),
	},
	"S1007": {
		Name:     "S1007",
		Run:      LintRegexpRaw,
		Doc:      Docs["S1007"].String(),
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
		Flags:    newFlagSet(),
	},
	"S1008": {
		Name:     "S1008",
		Run:      LintIfReturn,
		Doc:      Docs["S1008"].String(),
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
		Flags:    newFlagSet(),
	},
	"S1009": {
		Name:     "S1009",
		Run:      LintRedundantNilCheckWithLen,
		Doc:      Docs["S1009"].String(),
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
		Flags:    newFlagSet(),
	},
	"S1010": {
		Name:     "S1010",
		Run:      LintSlicing,
		Doc:      Docs["S1010"].String(),
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
		Flags:    newFlagSet(),
	},
	"S1011": {
		Name:     "S1011",
		Run:      LintLoopAppend,
		Doc:      Docs["S1011"].String(),
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
		Flags:    newFlagSet(),
	},
	"S1012": {
		Name:     "S1012",
		Run:      LintTimeSince,
		Doc:      Docs["S1012"].String(),
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
		Flags:    newFlagSet(),
	},
	"S1016": {
		Name:     "S1016",
		Run:      LintSimplerStructConversion,
		Doc:      Docs["S1016"].String(),
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
		Flags:    newFlagSet(),
	},
	"S1017": {
		Name:     "S1017",
		Run:      LintTrim,
		Doc:      Docs["S1017"].String(),
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
		Flags:    newFlagSet(),
	},
	"S1018": {
		Name:     "S1018",
		Run:      LintLoopSlide,
		Doc:      Docs["S1018"].String(),
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
		Flags:    newFlagSet(),
	},
	"S1019": {
		Name:     "S1019",
		Run:      LintMakeLenCap,
		Doc:      Docs["S1019"].String(),
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
		Flags:    newFlagSet(),
	},
	"S1020": {
		Name:     "S1020",
		Run:      LintAssertNotNil,
		Doc:      Docs["S1020"].String(),
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
		Flags:    newFlagSet(),
	},
	"S1021": {
		Name:     "S1021",
		Run:      LintDeclareAssign,
		Doc:      Docs["S1021"].String(),
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
		Flags:    newFlagSet(),
	},
	"S1023": {
		Name:     "S1023",
		Run:      LintRedundantBreak,
		Doc:      Docs["S1023"].String(),
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
		Flags:    newFlagSet(),
	},
	"S1024": {
		Name:     "S1024",
		Run:      LintTimeUntil,
		Doc:      Docs["S1024"].String(),
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
		Flags:    newFlagSet(),
	},
	"S1025": {
		Name:     "S1025",
		Run:      LintRedundantSprintf,
		Doc:      Docs["S1025"].String(),
		Requires: []*analysis.Analyzer{buildssa.Analyzer, inspect.Analyzer, facts.Generated},
		Flags:    newFlagSet(),
	},
	"S1028": {
		Name:     "S1028",
		Run:      LintErrorsNewSprintf,
		Doc:      Docs["S1028"].String(),
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
		Flags:    newFlagSet(),
	},
	"S1029": {
		Name:     "S1029",
		Run:      LintRangeStringRunes,
		Doc:      Docs["S1029"].String(),
		Requires: []*analysis.Analyzer{buildssa.Analyzer},
		Flags:    newFlagSet(),
	},
	"S1030": {
		Name:     "S1030",
		Run:      LintBytesBufferConversions,
		Doc:      Docs["S1030"].String(),
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
		Flags:    newFlagSet(),
	},
	"S1031": {
		Name:     "S1031",
		Run:      LintNilCheckAroundRange,
		Doc:      Docs["S1031"].String(),
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
		Flags:    newFlagSet(),
	},
	"S1032": {
		Name:     "S1032",
		Run:      LintSortHelpers,
		Doc:      Docs["S1032"].String(),
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
		Flags:    newFlagSet(),
	},
	"S1033": {
		Name:     "S1033",
		Run:      LintGuardedDelete,
		Doc:      Docs["S1033"].String(),
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
		Flags:    newFlagSet(),
	},
	"S1034": {
		Name:     "S1034",
		Run:      LintSimplifyTypeSwitch,
		Doc:      Docs["S1034"].String(),
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
		Flags:    newFlagSet(),
	},
}

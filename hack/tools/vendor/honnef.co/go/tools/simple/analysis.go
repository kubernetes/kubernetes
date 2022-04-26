package simple

import (
	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"honnef.co/go/tools/analysis/facts"
	"honnef.co/go/tools/analysis/lint"
	"honnef.co/go/tools/internal/passes/buildir"
)

var Analyzers = lint.InitializeAnalyzers(Docs, map[string]*analysis.Analyzer{
	"S1000": {
		Run:      CheckSingleCaseSelect,
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
	},
	"S1001": {
		Run:      CheckLoopCopy,
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
	},
	"S1002": {
		Run:      CheckIfBoolCmp,
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
	},
	"S1003": {
		Run:      CheckStringsContains,
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
	},
	"S1004": {
		Run:      CheckBytesCompare,
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
	},
	"S1005": {
		Run:      CheckUnnecessaryBlank,
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
	},
	"S1006": {
		Run:      CheckForTrue,
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
	},
	"S1007": {
		Run:      CheckRegexpRaw,
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
	},
	"S1008": {
		Run:      CheckIfReturn,
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
	},
	"S1009": {
		Run:      CheckRedundantNilCheckWithLen,
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
	},
	"S1010": {
		Run:      CheckSlicing,
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
	},
	"S1011": {
		Run:      CheckLoopAppend,
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
	},
	"S1012": {
		Run:      CheckTimeSince,
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
	},
	"S1016": {
		Run:      CheckSimplerStructConversion,
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
	},
	"S1017": {
		Run:      CheckTrim,
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
	},
	"S1018": {
		Run:      CheckLoopSlide,
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
	},
	"S1019": {
		Run:      CheckMakeLenCap,
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
	},
	"S1020": {
		Run:      CheckAssertNotNil,
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
	},
	"S1021": {
		Run:      CheckDeclareAssign,
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
	},
	"S1023": {
		Run:      CheckRedundantBreak,
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
	},
	"S1024": {
		Run:      CheckTimeUntil,
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
	},
	"S1025": {
		Run:      CheckRedundantSprintf,
		Requires: []*analysis.Analyzer{buildir.Analyzer, inspect.Analyzer, facts.Generated},
	},
	"S1028": {
		Run:      CheckErrorsNewSprintf,
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
	},
	"S1029": {
		Run:      CheckRangeStringRunes,
		Requires: []*analysis.Analyzer{buildir.Analyzer},
	},
	"S1030": {
		Run:      CheckBytesBufferConversions,
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
	},
	"S1031": {
		Run:      CheckNilCheckAroundRange,
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
	},
	"S1032": {
		Run:      CheckSortHelpers,
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
	},
	"S1033": {
		Run:      CheckGuardedDelete,
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
	},
	"S1034": {
		Run:      CheckSimplifyTypeSwitch,
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
	},
	"S1035": {
		Run:      CheckRedundantCanonicalHeaderKey,
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
	},
	"S1036": {
		Run:      CheckUnnecessaryGuard,
		Requires: []*analysis.Analyzer{inspect.Analyzer},
	},
	"S1037": {
		Run:      CheckElaborateSleep,
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
	},
	"S1038": {
		Run:      CheckPrintSprintf,
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
	},
	"S1039": {
		Run:      CheckSprintLiteral,
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
	},
	"S1040": {
		Run:      CheckSameTypeTypeAssertion,
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated},
	},
})

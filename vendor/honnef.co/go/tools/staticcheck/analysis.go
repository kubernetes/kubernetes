package staticcheck

import (
	"honnef.co/go/tools/facts"
	"honnef.co/go/tools/internal/passes/buildir"
	"honnef.co/go/tools/lint/lintutil"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
)

func makeCallCheckerAnalyzer(rules map[string]CallCheck, extraReqs ...*analysis.Analyzer) *analysis.Analyzer {
	reqs := []*analysis.Analyzer{buildir.Analyzer, facts.TokenFile}
	reqs = append(reqs, extraReqs...)
	return &analysis.Analyzer{
		Run:      callChecker(rules),
		Requires: reqs,
	}
}

var Analyzers = lintutil.InitializeAnalyzers(Docs, map[string]*analysis.Analyzer{
	"SA1000": makeCallCheckerAnalyzer(checkRegexpRules),
	"SA1001": {
		Run:      CheckTemplate,
		Requires: []*analysis.Analyzer{inspect.Analyzer},
	},
	"SA1002": makeCallCheckerAnalyzer(checkTimeParseRules),
	"SA1003": makeCallCheckerAnalyzer(checkEncodingBinaryRules),
	"SA1004": {
		Run:      CheckTimeSleepConstant,
		Requires: []*analysis.Analyzer{inspect.Analyzer},
	},
	"SA1005": {
		Run:      CheckExec,
		Requires: []*analysis.Analyzer{inspect.Analyzer},
	},
	"SA1006": {
		Run:      CheckUnsafePrintf,
		Requires: []*analysis.Analyzer{inspect.Analyzer},
	},
	"SA1007": makeCallCheckerAnalyzer(checkURLsRules),
	"SA1008": {
		Run:      CheckCanonicalHeaderKey,
		Requires: []*analysis.Analyzer{inspect.Analyzer},
	},
	"SA1010": makeCallCheckerAnalyzer(checkRegexpFindAllRules),
	"SA1011": makeCallCheckerAnalyzer(checkUTF8CutsetRules),
	"SA1012": {
		Run:      CheckNilContext,
		Requires: []*analysis.Analyzer{inspect.Analyzer},
	},
	"SA1013": {
		Run:      CheckSeeker,
		Requires: []*analysis.Analyzer{inspect.Analyzer},
	},
	"SA1014": makeCallCheckerAnalyzer(checkUnmarshalPointerRules),
	"SA1015": {
		Run:      CheckLeakyTimeTick,
		Requires: []*analysis.Analyzer{buildir.Analyzer},
	},
	"SA1016": {
		Run:      CheckUntrappableSignal,
		Requires: []*analysis.Analyzer{inspect.Analyzer},
	},
	"SA1017": makeCallCheckerAnalyzer(checkUnbufferedSignalChanRules),
	"SA1018": makeCallCheckerAnalyzer(checkStringsReplaceZeroRules),
	"SA1019": {
		Run:      CheckDeprecated,
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Deprecated, facts.Generated},
	},
	"SA1020": makeCallCheckerAnalyzer(checkListenAddressRules),
	"SA1021": makeCallCheckerAnalyzer(checkBytesEqualIPRules),
	"SA1023": {
		Run:      CheckWriterBufferModified,
		Requires: []*analysis.Analyzer{buildir.Analyzer},
	},
	"SA1024": makeCallCheckerAnalyzer(checkUniqueCutsetRules),
	"SA1025": {
		Run:      CheckTimerResetReturnValue,
		Requires: []*analysis.Analyzer{buildir.Analyzer},
	},
	"SA1026": makeCallCheckerAnalyzer(checkUnsupportedMarshal),
	"SA1027": makeCallCheckerAnalyzer(checkAtomicAlignment),
	"SA1028": makeCallCheckerAnalyzer(checkSortSliceRules),
	"SA1029": makeCallCheckerAnalyzer(checkWithValueKeyRules),

	"SA2000": {
		Run:      CheckWaitgroupAdd,
		Requires: []*analysis.Analyzer{inspect.Analyzer},
	},
	"SA2001": {
		Run:      CheckEmptyCriticalSection,
		Requires: []*analysis.Analyzer{inspect.Analyzer},
	},
	"SA2002": {
		Run:      CheckConcurrentTesting,
		Requires: []*analysis.Analyzer{buildir.Analyzer},
	},
	"SA2003": {
		Run:      CheckDeferLock,
		Requires: []*analysis.Analyzer{buildir.Analyzer},
	},

	"SA3000": {
		Run:      CheckTestMainExit,
		Requires: []*analysis.Analyzer{inspect.Analyzer},
	},
	"SA3001": {
		Run:      CheckBenchmarkN,
		Requires: []*analysis.Analyzer{inspect.Analyzer},
	},

	"SA4000": {
		Run:      CheckLhsRhsIdentical,
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.TokenFile, facts.Generated},
	},
	"SA4001": {
		Run:      CheckIneffectiveCopy,
		Requires: []*analysis.Analyzer{inspect.Analyzer},
	},
	"SA4003": {
		Run:      CheckExtremeComparison,
		Requires: []*analysis.Analyzer{inspect.Analyzer},
	},
	"SA4004": {
		Run:      CheckIneffectiveLoop,
		Requires: []*analysis.Analyzer{inspect.Analyzer},
	},
	"SA4006": {
		Run:      CheckUnreadVariableValues,
		Requires: []*analysis.Analyzer{buildir.Analyzer, facts.Generated},
	},
	"SA4008": {
		Run:      CheckLoopCondition,
		Requires: []*analysis.Analyzer{buildir.Analyzer},
	},
	"SA4009": {
		Run:      CheckArgOverwritten,
		Requires: []*analysis.Analyzer{buildir.Analyzer},
	},
	"SA4010": {
		Run:      CheckIneffectiveAppend,
		Requires: []*analysis.Analyzer{buildir.Analyzer},
	},
	"SA4011": {
		Run:      CheckScopedBreak,
		Requires: []*analysis.Analyzer{inspect.Analyzer},
	},
	"SA4012": {
		Run:      CheckNaNComparison,
		Requires: []*analysis.Analyzer{buildir.Analyzer},
	},
	"SA4013": {
		Run:      CheckDoubleNegation,
		Requires: []*analysis.Analyzer{inspect.Analyzer},
	},
	"SA4014": {
		Run:      CheckRepeatedIfElse,
		Requires: []*analysis.Analyzer{inspect.Analyzer},
	},
	"SA4015": makeCallCheckerAnalyzer(checkMathIntRules),
	"SA4016": {
		Run:      CheckSillyBitwiseOps,
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.TokenFile},
	},
	"SA4017": {
		Run:      CheckPureFunctions,
		Requires: []*analysis.Analyzer{buildir.Analyzer, facts.Purity},
	},
	"SA4018": {
		Run:      CheckSelfAssignment,
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated, facts.TokenFile, facts.Purity},
	},
	"SA4019": {
		Run:      CheckDuplicateBuildConstraints,
		Requires: []*analysis.Analyzer{facts.Generated},
	},
	"SA4020": {
		Run:      CheckUnreachableTypeCases,
		Requires: []*analysis.Analyzer{inspect.Analyzer},
	},
	"SA4021": {
		Run:      CheckSingleArgAppend,
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated, facts.TokenFile},
	},

	"SA5000": {
		Run:      CheckNilMaps,
		Requires: []*analysis.Analyzer{buildir.Analyzer},
	},
	"SA5001": {
		Run:      CheckEarlyDefer,
		Requires: []*analysis.Analyzer{inspect.Analyzer},
	},
	"SA5002": {
		Run:      CheckInfiniteEmptyLoop,
		Requires: []*analysis.Analyzer{inspect.Analyzer},
	},
	"SA5003": {
		Run:      CheckDeferInInfiniteLoop,
		Requires: []*analysis.Analyzer{inspect.Analyzer},
	},
	"SA5004": {
		Run:      CheckLoopEmptyDefault,
		Requires: []*analysis.Analyzer{inspect.Analyzer},
	},
	"SA5005": {
		Run:      CheckCyclicFinalizer,
		Requires: []*analysis.Analyzer{buildir.Analyzer},
	},
	"SA5007": {
		Run:      CheckInfiniteRecursion,
		Requires: []*analysis.Analyzer{buildir.Analyzer},
	},
	"SA5008": {
		Run:      CheckStructTags,
		Requires: []*analysis.Analyzer{inspect.Analyzer},
	},
	"SA5009": makeCallCheckerAnalyzer(checkPrintfRules),
	"SA5010": {
		Run:      CheckImpossibleTypeAssertion,
		Requires: []*analysis.Analyzer{buildir.Analyzer, facts.TokenFile},
	},
	"SA5011": {
		Run:      CheckMaybeNil,
		Requires: []*analysis.Analyzer{buildir.Analyzer},
	},

	"SA6000": makeCallCheckerAnalyzer(checkRegexpMatchLoopRules),
	"SA6001": {
		Run:      CheckMapBytesKey,
		Requires: []*analysis.Analyzer{buildir.Analyzer},
	},
	"SA6002": makeCallCheckerAnalyzer(checkSyncPoolValueRules),
	"SA6003": {
		Run:      CheckRangeStringRunes,
		Requires: []*analysis.Analyzer{buildir.Analyzer},
	},
	"SA6005": {
		Run:      CheckToLowerToUpperComparison,
		Requires: []*analysis.Analyzer{inspect.Analyzer},
	},

	"SA9001": {
		Run:      CheckDubiousDeferInChannelRangeLoop,
		Requires: []*analysis.Analyzer{inspect.Analyzer},
	},
	"SA9002": {
		Run:      CheckNonOctalFileMode,
		Requires: []*analysis.Analyzer{inspect.Analyzer},
	},
	"SA9003": {
		Run:      CheckEmptyBranch,
		Requires: []*analysis.Analyzer{buildir.Analyzer, facts.TokenFile, facts.Generated},
	},
	"SA9004": {
		Run:      CheckMissingEnumTypesInDeclaration,
		Requires: []*analysis.Analyzer{inspect.Analyzer},
	},
	// Filtering generated code because it may include empty structs generated from data models.
	"SA9005": makeCallCheckerAnalyzer(checkNoopMarshal, facts.Generated),

	"SA4022": {
		Run:      CheckAddressIsNil,
		Requires: []*analysis.Analyzer{inspect.Analyzer},
	},
})

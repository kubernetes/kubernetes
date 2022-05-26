package stylecheck

import (
	"honnef.co/go/tools/analysis/facts"
	"honnef.co/go/tools/analysis/lint"
	"honnef.co/go/tools/config"
	"honnef.co/go/tools/internal/passes/buildir"
	"honnef.co/go/tools/internal/sharedcheck"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
)

var Analyzers = lint.InitializeAnalyzers(Docs, map[string]*analysis.Analyzer{
	"ST1000": {
		Run: CheckPackageComment,
	},
	"ST1001": {
		Run:      CheckDotImports,
		Requires: []*analysis.Analyzer{facts.Generated, config.Analyzer},
	},
	"ST1003": {
		Run:      CheckNames,
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated, config.Analyzer},
	},
	"ST1005": {
		Run:      CheckErrorStrings,
		Requires: []*analysis.Analyzer{buildir.Analyzer},
	},
	"ST1006": {
		Run:      CheckReceiverNames,
		Requires: []*analysis.Analyzer{buildir.Analyzer, facts.Generated},
	},
	"ST1008": {
		Run:      CheckErrorReturn,
		Requires: []*analysis.Analyzer{buildir.Analyzer},
	},
	"ST1011": {
		Run:      CheckTimeNames,
		Requires: []*analysis.Analyzer{inspect.Analyzer},
	},
	"ST1012": {
		Run: CheckErrorVarNames,
	},
	"ST1013": {
		Run: CheckHTTPStatusCodes,
		// TODO(dh): why does this depend on facts.TokenFile?
		Requires: []*analysis.Analyzer{facts.Generated, facts.TokenFile, config.Analyzer, inspect.Analyzer},
	},
	"ST1015": {
		Run:      CheckDefaultCaseOrder,
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated, facts.TokenFile},
	},
	"ST1016": {
		Run:      CheckReceiverNamesIdentical,
		Requires: []*analysis.Analyzer{buildir.Analyzer, facts.Generated},
	},
	"ST1017": {
		Run:      CheckYodaConditions,
		Requires: []*analysis.Analyzer{inspect.Analyzer, facts.Generated, facts.TokenFile},
	},
	"ST1018": {
		Run:      CheckInvisibleCharacters,
		Requires: []*analysis.Analyzer{inspect.Analyzer},
	},
	"ST1019": {
		Run:      CheckDuplicatedImports,
		Requires: []*analysis.Analyzer{facts.Generated},
	},
	"ST1020": {
		Run:      CheckExportedFunctionDocs,
		Requires: []*analysis.Analyzer{facts.Generated, inspect.Analyzer},
	},
	"ST1021": {
		Run:      CheckExportedTypeDocs,
		Requires: []*analysis.Analyzer{facts.Generated, inspect.Analyzer},
	},
	"ST1022": {
		Run:      CheckExportedVarDocs,
		Requires: []*analysis.Analyzer{facts.Generated, inspect.Analyzer},
	},
	"ST1023": sharedcheck.RedundantTypeInDeclarationChecker("should", false),
})

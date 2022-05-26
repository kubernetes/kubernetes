package analyzer

import (
	"flag"
)

const (
	FlagNameConstRequireSingleConst = "const-require-single-const"
	FlagNameConstRequireGrouping    = "const-require-grouping"

	FlagNameImportRequireSingleImport = "import-require-single-import"
	FlagNameImportRequireGrouping     = "import-require-grouping"

	FlagNameTypeRequireSingleType = "type-require-single-type"
	FlagNameTypeRequireGrouping   = "type-require-grouping"

	FlagNameVarRequireSingleVar = "var-require-single-var"
	FlagNameVarRequireGrouping  = "var-require-grouping"
)

func Flags() flag.FlagSet {
	fs := flag.NewFlagSet(Name, flag.ExitOnError)

	fs.Bool(FlagNameConstRequireSingleConst, false, "require the use of a single global 'const' declaration only")
	fs.Bool(FlagNameConstRequireGrouping, false, "require the use of grouped global 'const' declarations")

	fs.Bool(FlagNameImportRequireSingleImport, false, "require the use of a single 'import' declaration only")
	fs.Bool(FlagNameImportRequireGrouping, false, "require the use of grouped 'import' declarations")

	fs.Bool(FlagNameTypeRequireSingleType, false, "require the use of a single global 'type' declaration only")
	fs.Bool(FlagNameTypeRequireGrouping, false, "require the use of grouped global 'type' declarations")

	fs.Bool(FlagNameVarRequireSingleVar, false, "require the use of a single global 'var' declaration only")
	fs.Bool(FlagNameVarRequireGrouping, false, "require the use of grouped global 'var' declarations")

	return *fs
}

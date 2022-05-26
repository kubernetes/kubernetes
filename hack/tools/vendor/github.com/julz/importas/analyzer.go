package importas

import (
	"fmt"
	"go/ast"
	"go/types"
	"strconv"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
)

var config = &Config{
	RequiredAlias: make(map[string]string),
}

var Analyzer = &analysis.Analyzer{
	Name: "importas",
	Doc:  "Enforces consistent import aliases",
	Run:  run,

	Flags: flags(config),

	Requires: []*analysis.Analyzer{inspect.Analyzer},
}

func run(pass *analysis.Pass) (interface{}, error) {
	return runWithConfig(config, pass)
}

func runWithConfig(config *Config, pass *analysis.Pass) (interface{}, error) {
	if err := config.CompileRegexp(); err != nil {
		return nil, err
	}

	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)
	inspect.Preorder([]ast.Node{(*ast.ImportSpec)(nil)}, func(n ast.Node) {
		visitImportSpecNode(config, n.(*ast.ImportSpec), pass)
	})

	return nil, nil
}

func visitImportSpecNode(config *Config, node *ast.ImportSpec, pass *analysis.Pass) {
	if !config.DisallowUnaliased && node.Name == nil {
		return
	}

	alias := ""
	if node.Name != nil {
		alias = node.Name.String()
	}

	if alias == "." {
		return // Dot aliases are generally used in tests, so ignore.
	}

	if strings.HasPrefix(alias, "_") {
		return // Used by go test and for auto-includes, not a conflict.
	}

	path, err := strconv.Unquote(node.Path.Value)
	if err != nil {
		pass.Reportf(node.Pos(), "import not quoted")
	}

	if required, exists := config.AliasFor(path); exists && required != alias {
		message := fmt.Sprintf("import %q imported as %q but must be %q according to config", path, alias, required)
		if alias == "" {
			message = fmt.Sprintf("import %q imported without alias but must be with alias %q according to config", path, required)
		}

		pass.Report(analysis.Diagnostic{
			Pos:     node.Pos(),
			End:     node.End(),
			Message: message,
			SuggestedFixes: []analysis.SuggestedFix{{
				Message:   "Use correct alias",
				TextEdits: findEdits(node, pass.TypesInfo.Uses, path, alias, required),
			}},
		})
	} else if !exists && config.DisallowExtraAliases {
		pass.Report(analysis.Diagnostic{
			Pos:     node.Pos(),
			End:     node.End(),
			Message: fmt.Sprintf("import %q has alias %q which is not part of config", path, alias),
			SuggestedFixes: []analysis.SuggestedFix{{
				Message:   "remove alias",
				TextEdits: findEdits(node, pass.TypesInfo.Uses, path, alias, ""),
			}},
		})
	}
}

func findEdits(node ast.Node, uses map[*ast.Ident]types.Object, importPath, original, required string) []analysis.TextEdit {
	// Edit the actual import line.
	importLine := strconv.Quote(importPath)
	if required != "" {
		importLine = required + " " + importLine
	}
	result := []analysis.TextEdit{{
		Pos:     node.Pos(),
		End:     node.End(),
		NewText: []byte(importLine),
	}}

	packageReplacement := required
	if required == "" {
		packageParts := strings.Split(importPath, "/")
		if len(packageParts) != 0 {
			packageReplacement = packageParts[len(packageParts)-1]
		} else {
			// fall back to original
			packageReplacement = original
		}
	}

	// Edit all the uses of the alias in the code.
	for use, pkg := range uses {
		pkgName, ok := pkg.(*types.PkgName)
		if !ok {
			// skip identifiers that aren't pointing at a PkgName.
			continue
		}

		if pkgName.Pos() != node.Pos() {
			// skip identifiers pointing to a different import statement.
			continue
		}

		result = append(result, analysis.TextEdit{
			Pos:     use.Pos(),
			End:     use.End(),
			NewText: []byte(packageReplacement),
		})
	}

	return result
}

package analyzer

import (
	"flag"
	"fmt"
	"go/ast"
	"go/types"
	"path"
	"strings"

	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"

	"golang.org/x/tools/go/analysis"
)

// Analyzer that checks if all struct's fields are initialized
var Analyzer = &analysis.Analyzer{
	Name:     "exhaustivestruct",
	Doc:      "Checks if all struct's fields are initialized",
	Run:      run,
	Requires: []*analysis.Analyzer{inspect.Analyzer},
	Flags:    newFlagSet(),
}

// StructPatternList is a comma separated list of expressions to match struct packages and names
// The struct packages have the form example.com/package.ExampleStruct
// The matching patterns can use matching syntax from https://pkg.go.dev/path#Match
// If this list is empty, all structs are tested.
var StructPatternList string

func newFlagSet() flag.FlagSet {
	fs := flag.NewFlagSet("", flag.PanicOnError)
	fs.StringVar(&StructPatternList, "struct_patterns", "", "This is a comma separated list of expressions to match struct packages and names")
	return *fs
}

func run(pass *analysis.Pass) (interface{}, error) {
	splitFn := func(c rune) bool { return c == ',' }
	inspector := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)
	structPatterns := strings.FieldsFunc(StructPatternList, splitFn)
	// validate the pattern syntax
	for _, pattern := range structPatterns {
		_, err := path.Match(pattern, "")
		if err != nil {
			return nil, fmt.Errorf("invalid struct pattern %s: %w", pattern, err)
		}
	}

	nodeFilter := []ast.Node{
		(*ast.CompositeLit)(nil),
		(*ast.ReturnStmt)(nil),
	}

	var returnStmt *ast.ReturnStmt

	inspector.Preorder(nodeFilter, func(node ast.Node) {
		var name string

		compositeLit, ok := node.(*ast.CompositeLit)
		if !ok {
			// Keep track of the last return statement whilte iterating
			retLit, ok := node.(*ast.ReturnStmt)
			if ok {
				returnStmt = retLit
			}
			return
		}

		i, ok := compositeLit.Type.(*ast.Ident)

		if ok {
			name = i.Name
		} else {
			s, ok := compositeLit.Type.(*ast.SelectorExpr)

			if !ok {
				return
			}

			name = s.Sel.Name
		}

		if compositeLit.Type == nil {
			return
		}

		t := pass.TypesInfo.TypeOf(compositeLit.Type)

		if t == nil {
			return
		}

		if len(structPatterns) > 0 {
			shouldLint := false
			for _, pattern := range structPatterns {
				// We check the patterns for vailidy ahead of time, so we don't need to check the error here
				if match, _ := path.Match(pattern, t.String()); match {
					shouldLint = true
					break
				}
			}
			if !shouldLint {
				return
			}
		}

		str, ok := t.Underlying().(*types.Struct)

		if !ok {
			return
		}

		// Don't report an error if:
		// 1. This composite literal contains no fields and
		// 2. It's in a return statement and
		// 3. The return statement contains a non-nil error
		if len(compositeLit.Elts) == 0 {
			// Check if this composite is one of the results the last return statement
			isInResults := false
			if returnStmt != nil {
				for _, result := range returnStmt.Results {
					compareComposite, ok := result.(*ast.CompositeLit)
					if ok {
						if compareComposite == compositeLit {
							isInResults = true
						}
					}
				}
			}
			nonNilError := false
			if isInResults {
				// Check if any of the results has an error type and if that error is set to non-nil (if it's set to nil, the type would be "untyped nil")
				for _, result := range returnStmt.Results {
					if pass.TypesInfo.TypeOf(result).String() == "error" {
						nonNilError = true
					}
				}
			}

			if nonNilError {
				return
			}
		}

		samePackage := strings.HasPrefix(t.String(), pass.Pkg.Path()+".")

		missing := []string{}

		for i := 0; i < str.NumFields(); i++ {
			fieldName := str.Field(i).Name()
			exists := false

			if !samePackage && !str.Field(i).Exported() {
				continue
			}

			for eIndex, e := range compositeLit.Elts {
				if k, ok := e.(*ast.KeyValueExpr); ok {
					if i, ok := k.Key.(*ast.Ident); ok {
						if i.Name == fieldName {
							exists = true
							break
						}
					}
				} else {
					if eIndex == i {
						exists = true
						break
					}
				}
			}

			if !exists {
				missing = append(missing, fieldName)
			}
		}

		if len(missing) == 1 {
			pass.Reportf(node.Pos(), "%s is missing in %s", missing[0], name)
		} else if len(missing) > 1 {
			pass.Reportf(node.Pos(), "%s are missing in %s", strings.Join(missing, ", "), name)
		}
	})

	return nil, nil
}

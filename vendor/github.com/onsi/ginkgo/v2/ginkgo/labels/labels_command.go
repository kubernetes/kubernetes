package labels

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"sort"
	"strconv"
	"strings"

	"github.com/onsi/ginkgo/v2/ginkgo/command"
	"github.com/onsi/ginkgo/v2/ginkgo/internal"
	"github.com/onsi/ginkgo/v2/types"
	"golang.org/x/tools/go/ast/inspector"
)

func BuildLabelsCommand() command.Command {
	var cliConfig = types.NewDefaultCLIConfig()

	flags, err := types.BuildLabelsCommandFlagSet(&cliConfig)
	if err != nil {
		panic(err)
	}

	return command.Command{
		Name:     "labels",
		Usage:    "ginkgo labels <FLAGS> <PACKAGES>",
		Flags:    flags,
		ShortDoc: "List labels detected in the passed-in packages (or the package in the current directory if left blank).",
		DocLink:  "spec-labels",
		Command: func(args []string, _ []string) {
			ListLabels(args, cliConfig)
		},
	}
}

func ListLabels(args []string, cliConfig types.CLIConfig) {
	suites := internal.FindSuites(args, cliConfig, false).WithoutState(internal.TestSuiteStateSkippedByFilter)
	if len(suites) == 0 {
		command.AbortWith("Found no test suites")
	}
	for _, suite := range suites {
		labels := fetchLabelsFromPackage(suite.Path)
		if len(labels) == 0 {
			fmt.Printf("%s: No labels found\n", suite.PackageName)
		} else {
			fmt.Printf("%s: [%s]\n", suite.PackageName, strings.Join(labels, ", "))
		}
	}
}

func fetchLabelsFromPackage(packagePath string) []string {
	fset := token.NewFileSet()
	parsedPackages, err := parser.ParseDir(fset, packagePath, nil, 0)
	command.AbortIfError("Failed to parse package source:", err)

	files := []*ast.File{}
	hasTestPackage := false
	for key, pkg := range parsedPackages {
		if strings.HasSuffix(key, "_test") {
			hasTestPackage = true
			for _, file := range pkg.Files {
				files = append(files, file)
			}
		}
	}
	if !hasTestPackage {
		for _, pkg := range parsedPackages {
			for _, file := range pkg.Files {
				files = append(files, file)
			}
		}
	}

	seen := map[string]bool{}
	labels := []string{}
	ispr := inspector.New(files)
	ispr.Preorder([]ast.Node{&ast.CallExpr{}}, func(n ast.Node) {
		potentialLabels := fetchLabels(n.(*ast.CallExpr))
		for _, label := range potentialLabels {
			if !seen[label] {
				seen[label] = true
				labels = append(labels, strconv.Quote(label))
			}
		}
	})

	sort.Strings(labels)
	return labels
}

func fetchLabels(callExpr *ast.CallExpr) []string {
	out := []string{}
	switch expr := callExpr.Fun.(type) {
	case *ast.Ident:
		if expr.Name != "Label" {
			return out
		}
	case *ast.SelectorExpr:
		if expr.Sel.Name != "Label" {
			return out
		}
	default:
		return out
	}
	for _, arg := range callExpr.Args {
		switch expr := arg.(type) {
		case *ast.BasicLit:
			if expr.Kind == token.STRING {
				unquoted, err := strconv.Unquote(expr.Value)
				if err != nil {
					unquoted = expr.Value
				}
				validated, err := types.ValidateAndCleanupLabel(unquoted, types.CodeLocation{})
				if err == nil {
					out = append(out, validated)
				}
			}
		}
	}
	return out
}

package nodot

import (
	"fmt"
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"path/filepath"
	"strings"
)

func ApplyNoDot(data []byte) ([]byte, error) {
	sections, err := generateNodotSections()
	if err != nil {
		return nil, err
	}

	for _, section := range sections {
		data = section.createOrUpdateIn(data)
	}

	return data, nil
}

type nodotSection struct {
	name         string
	pkg          string
	declarations []string
	types        []string
}

func (s nodotSection) createOrUpdateIn(data []byte) []byte {
	renames := map[string]string{}

	contents := string(data)

	lines := strings.Split(contents, "\n")

	comment := "// Declarations for " + s.name

	newLines := []string{}
	for _, line := range lines {
		if line == comment {
			continue
		}

		words := strings.Split(line, " ")
		lastWord := words[len(words)-1]

		if s.containsDeclarationOrType(lastWord) {
			renames[lastWord] = words[1]
			continue
		}

		newLines = append(newLines, line)
	}

	if len(newLines[len(newLines)-1]) > 0 {
		newLines = append(newLines, "")
	}

	newLines = append(newLines, comment)

	for _, typ := range s.types {
		name, ok := renames[s.prefix(typ)]
		if !ok {
			name = typ
		}
		newLines = append(newLines, fmt.Sprintf("type %s %s", name, s.prefix(typ)))
	}

	for _, decl := range s.declarations {
		name, ok := renames[s.prefix(decl)]
		if !ok {
			name = decl
		}
		newLines = append(newLines, fmt.Sprintf("var %s = %s", name, s.prefix(decl)))
	}

	newLines = append(newLines, "")

	newContents := strings.Join(newLines, "\n")

	return []byte(newContents)
}

func (s nodotSection) prefix(declOrType string) string {
	return s.pkg + "." + declOrType
}

func (s nodotSection) containsDeclarationOrType(word string) bool {
	for _, declaration := range s.declarations {
		if s.prefix(declaration) == word {
			return true
		}
	}

	for _, typ := range s.types {
		if s.prefix(typ) == word {
			return true
		}
	}

	return false
}

func generateNodotSections() ([]nodotSection, error) {
	sections := []nodotSection{}

	declarations, err := getExportedDeclerationsForPackage("github.com/onsi/ginkgo", "ginkgo_dsl.go", "GINKGO_VERSION", "GINKGO_PANIC")
	if err != nil {
		return nil, err
	}
	sections = append(sections, nodotSection{
		name:         "Ginkgo DSL",
		pkg:          "ginkgo",
		declarations: declarations,
		types:        []string{"Done", "Benchmarker"},
	})

	declarations, err = getExportedDeclerationsForPackage("github.com/onsi/gomega", "gomega_dsl.go", "GOMEGA_VERSION")
	if err != nil {
		return nil, err
	}
	sections = append(sections, nodotSection{
		name:         "Gomega DSL",
		pkg:          "gomega",
		declarations: declarations,
	})

	declarations, err = getExportedDeclerationsForPackage("github.com/onsi/gomega", "matchers.go")
	if err != nil {
		return nil, err
	}
	sections = append(sections, nodotSection{
		name:         "Gomega Matchers",
		pkg:          "gomega",
		declarations: declarations,
	})

	return sections, nil
}

func getExportedDeclerationsForPackage(pkgPath string, filename string, blacklist ...string) ([]string, error) {
	pkg, err := build.Import(pkgPath, ".", 0)
	if err != nil {
		return []string{}, err
	}

	declarations, err := getExportedDeclarationsForFile(filepath.Join(pkg.Dir, filename))
	if err != nil {
		return []string{}, err
	}

	blacklistLookup := map[string]bool{}
	for _, declaration := range blacklist {
		blacklistLookup[declaration] = true
	}

	filteredDeclarations := []string{}
	for _, declaration := range declarations {
		if blacklistLookup[declaration] {
			continue
		}
		filteredDeclarations = append(filteredDeclarations, declaration)
	}

	return filteredDeclarations, nil
}

func getExportedDeclarationsForFile(path string) ([]string, error) {
	fset := token.NewFileSet()
	tree, err := parser.ParseFile(fset, path, nil, 0)
	if err != nil {
		return []string{}, err
	}

	declarations := []string{}
	ast.FileExports(tree)
	for _, decl := range tree.Decls {
		switch x := decl.(type) {
		case *ast.GenDecl:
			switch s := x.Specs[0].(type) {
			case *ast.ValueSpec:
				declarations = append(declarations, s.Names[0].Name)
			}
		case *ast.FuncDecl:
			declarations = append(declarations, x.Name.Name)
		}
	}

	return declarations, nil
}

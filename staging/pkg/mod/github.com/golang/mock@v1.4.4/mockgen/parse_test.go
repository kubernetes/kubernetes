package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

func TestFileParser_ParseFile(t *testing.T) {
	fs := token.NewFileSet()
	file, err := parser.ParseFile(fs, "internal/tests/custom_package_name/greeter/greeter.go", nil, 0)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	p := fileParser{
		fileSet:            fs,
		imports:            make(map[string]string),
		importedInterfaces: make(map[string]map[string]*ast.InterfaceType),
	}

	pkg, err := p.parseFile("", file)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	checkGreeterImports(t, p.imports)

	expectedName := "greeter"
	if pkg.Name != expectedName {
		t.Fatalf("Expected name to be %v but got %v", expectedName, pkg.Name)
	}

	expectedInterfaceName := "InputMaker"
	if pkg.Interfaces[0].Name != expectedInterfaceName {
		t.Fatalf("Expected interface name to be %v but got %v", expectedInterfaceName, pkg.Interfaces[0].Name)
	}
}

func TestFileParser_ParsePackage(t *testing.T) {
	fs := token.NewFileSet()
	_, err := parser.ParseFile(fs, "internal/tests/custom_package_name/greeter/greeter.go", nil, 0)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	p := fileParser{
		fileSet:            fs,
		imports:            make(map[string]string),
		importedInterfaces: make(map[string]map[string]*ast.InterfaceType),
	}

	err = p.parsePackage("github.com/golang/mock/mockgen/internal/tests/custom_package_name/greeter")
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	checkGreeterImports(t, p.imports)
}

func TestImportsOfFile(t *testing.T) {
	fs := token.NewFileSet()
	file, err := parser.ParseFile(fs, "internal/tests/custom_package_name/greeter/greeter.go", nil, 0)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	imports, _ := importsOfFile(file)
	checkGreeterImports(t, imports)
}

func checkGreeterImports(t *testing.T, imports map[string]string) {
	// check that imports have stdlib package "fmt"
	if fmtPackage, ok := imports["fmt"]; !ok {
		t.Errorf("Expected imports to have key \"fmt\"")
	} else {
		expectedFmtPackage := "fmt"
		if fmtPackage != expectedFmtPackage {
			t.Errorf("Expected fmt key to have value %s but got %s", expectedFmtPackage, fmtPackage)
		}
	}

	// check that imports have package named "validator"
	if validatorPackage, ok := imports["validator"]; !ok {
		t.Errorf("Expected imports to have key \"fmt\"")
	} else {
		expectedValidatorPackage := "github.com/golang/mock/mockgen/internal/tests/custom_package_name/validator"
		if validatorPackage != expectedValidatorPackage {
			t.Errorf("Expected validator key to have value %s but got %s", expectedValidatorPackage, validatorPackage)
		}
	}

	// check that imports have package named "client"
	if clientPackage, ok := imports["client"]; !ok {
		t.Errorf("Expected imports to have key \"client\"")
	} else {
		expectedClientPackage := "github.com/golang/mock/mockgen/internal/tests/custom_package_name/client/v1"
		if clientPackage != expectedClientPackage {
			t.Errorf("Expected client key to have value %s but got %s", expectedClientPackage, clientPackage)
		}
	}

	// check that imports don't have package named "v1"
	if _, ok := imports["v1"]; ok {
		t.Errorf("Expected import not to have key \"v1\"")
	}
}

func Benchmark_parseFile(b *testing.B) {
	source := "internal/tests/performance/big_interface/big_interface.go"
	for n := 0; n < b.N; n++ {
		sourceMode(source)
	}
}

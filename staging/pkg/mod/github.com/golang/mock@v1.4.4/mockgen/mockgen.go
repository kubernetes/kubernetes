// Copyright 2010 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// MockGen generates mock implementations of Go interfaces.
package main

// TODO: This does not support recursive embedded interfaces.
// TODO: This does not support embedding package-local interfaces in a separate file.

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"go/build"
	"go/format"
	"go/token"
	"io"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"unicode"

	"github.com/golang/mock/mockgen/model"
)

const (
	gomockImportPath = "github.com/golang/mock/gomock"
)

var (
	version = ""
	commit  = "none"
	date    = "unknown"
)

var (
	source          = flag.String("source", "", "(source mode) Input Go source file; enables source mode.")
	destination     = flag.String("destination", "", "Output file; defaults to stdout.")
	mockNames       = flag.String("mock_names", "", "Comma-separated interfaceName=mockName pairs of explicit mock names to use. Mock names default to 'Mock'+ interfaceName suffix.")
	packageOut      = flag.String("package", "", "Package of the generated code; defaults to the package of the input with a 'mock_' prefix.")
	selfPackage     = flag.String("self_package", "", "The full package import path for the generated code. The purpose of this flag is to prevent import cycles in the generated code by trying to include its own package. This can happen if the mock's package is set to one of its inputs (usually the main one) and the output is stdio so mockgen cannot detect the final output package. Setting this flag will then tell mockgen which import to exclude.")
	writePkgComment = flag.Bool("write_package_comment", true, "Writes package documentation comment (godoc) if true.")
	copyrightFile   = flag.String("copyright_file", "", "Copyright file used to add copyright header")

	debugParser = flag.Bool("debug_parser", false, "Print out parser results only.")
	showVersion = flag.Bool("version", false, "Print version.")
)

func main() {
	flag.Usage = usage
	flag.Parse()

	if *showVersion {
		printVersion()
		return
	}

	var pkg *model.Package
	var err error
	var packageName string
	if *source != "" {
		pkg, err = sourceMode(*source)
	} else {
		if flag.NArg() != 2 {
			usage()
			log.Fatal("Expected exactly two arguments")
		}
		packageName = flag.Arg(0)
		if packageName == "." {
			dir, err := os.Getwd()
			if err != nil {
				log.Fatalf("Get current directory failed: %v", err)
			}
			packageName, err = packageNameOfDir(dir)
			if err != nil {
				log.Fatalf("Parse package name failed: %v", err)
			}
		}
		pkg, err = reflectMode(packageName, strings.Split(flag.Arg(1), ","))
	}
	if err != nil {
		log.Fatalf("Loading input failed: %v", err)
	}

	if *debugParser {
		pkg.Print(os.Stdout)
		return
	}

	dst := os.Stdout
	if len(*destination) > 0 {
		if err := os.MkdirAll(filepath.Dir(*destination), os.ModePerm); err != nil {
			log.Fatalf("Unable to create directory: %v", err)
		}
		f, err := os.Create(*destination)
		if err != nil {
			log.Fatalf("Failed opening destination file: %v", err)
		}
		defer f.Close()
		dst = f
	}

	outputPackageName := *packageOut
	if outputPackageName == "" {
		// pkg.Name in reflect mode is the base name of the import path,
		// which might have characters that are illegal to have in package names.
		outputPackageName = "mock_" + sanitize(pkg.Name)
	}

	// outputPackagePath represents the fully qualified name of the package of
	// the generated code. Its purposes are to prevent the module from importing
	// itself and to prevent qualifying type names that come from its own
	// package (i.e. if there is a type called X then we want to print "X" not
	// "package.X" since "package" is this package). This can happen if the mock
	// is output into an already existing package.
	outputPackagePath := *selfPackage
	if len(outputPackagePath) == 0 && len(*destination) > 0 {
		dst, _ := filepath.Abs(filepath.Dir(*destination))
		for _, prefix := range build.Default.SrcDirs() {
			if strings.HasPrefix(dst, prefix) {
				if rel, err := filepath.Rel(prefix, dst); err == nil {
					outputPackagePath = rel
					break
				}
			}
		}
	}

	g := new(generator)
	if *source != "" {
		g.filename = *source
	} else {
		g.srcPackage = packageName
		g.srcInterfaces = flag.Arg(1)
	}

	if *mockNames != "" {
		g.mockNames = parseMockNames(*mockNames)
	}
	if *copyrightFile != "" {
		header, err := ioutil.ReadFile(*copyrightFile)
		if err != nil {
			log.Fatalf("Failed reading copyright file: %v", err)
		}

		g.copyrightHeader = string(header)
	}
	if err := g.Generate(pkg, outputPackageName, outputPackagePath); err != nil {
		log.Fatalf("Failed generating mock: %v", err)
	}
	if _, err := dst.Write(g.Output()); err != nil {
		log.Fatalf("Failed writing to destination: %v", err)
	}
}

func parseMockNames(names string) map[string]string {
	mocksMap := make(map[string]string)
	for _, kv := range strings.Split(names, ",") {
		parts := strings.SplitN(kv, "=", 2)
		if len(parts) != 2 || parts[1] == "" {
			log.Fatalf("bad mock names spec: %v", kv)
		}
		mocksMap[parts[0]] = parts[1]
	}
	return mocksMap
}

func usage() {
	_, _ = io.WriteString(os.Stderr, usageText)
	flag.PrintDefaults()
}

const usageText = `mockgen has two modes of operation: source and reflect.

Source mode generates mock interfaces from a source file.
It is enabled by using the -source flag. Other flags that
may be useful in this mode are -imports and -aux_files.
Example:
	mockgen -source=foo.go [other options]

Reflect mode generates mock interfaces by building a program
that uses reflection to understand interfaces. It is enabled
by passing two non-flag arguments: an import path, and a
comma-separated list of symbols.
Example:
	mockgen database/sql/driver Conn,Driver

`

type generator struct {
	buf                       bytes.Buffer
	indent                    string
	mockNames                 map[string]string // may be empty
	filename                  string            // may be empty
	srcPackage, srcInterfaces string            // may be empty
	copyrightHeader           string

	packageMap map[string]string // map from import path to package name
}

func (g *generator) p(format string, args ...interface{}) {
	fmt.Fprintf(&g.buf, g.indent+format+"\n", args...)
}

func (g *generator) in() {
	g.indent += "\t"
}

func (g *generator) out() {
	if len(g.indent) > 0 {
		g.indent = g.indent[0 : len(g.indent)-1]
	}
}

func removeDot(s string) string {
	if len(s) > 0 && s[len(s)-1] == '.' {
		return s[0 : len(s)-1]
	}
	return s
}

// sanitize cleans up a string to make a suitable package name.
func sanitize(s string) string {
	t := ""
	for _, r := range s {
		if t == "" {
			if unicode.IsLetter(r) || r == '_' {
				t += string(r)
				continue
			}
		} else {
			if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' {
				t += string(r)
				continue
			}
		}
		t += "_"
	}
	if t == "_" {
		t = "x"
	}
	return t
}

func (g *generator) Generate(pkg *model.Package, outputPkgName string, outputPackagePath string) error {
	if outputPkgName != pkg.Name && *selfPackage == "" {
		// reset outputPackagePath if it's not passed in through -self_package
		outputPackagePath = ""
	}

	if g.copyrightHeader != "" {
		lines := strings.Split(g.copyrightHeader, "\n")
		for _, line := range lines {
			g.p("// %s", line)
		}
		g.p("")
	}

	g.p("// Code generated by MockGen. DO NOT EDIT.")
	if g.filename != "" {
		g.p("// Source: %v", g.filename)
	} else {
		g.p("// Source: %v (interfaces: %v)", g.srcPackage, g.srcInterfaces)
	}
	g.p("")

	// Get all required imports, and generate unique names for them all.
	im := pkg.Imports()
	im[gomockImportPath] = true

	// Only import reflect if it's used. We only use reflect in mocked methods
	// so only import if any of the mocked interfaces have methods.
	for _, intf := range pkg.Interfaces {
		if len(intf.Methods) > 0 {
			im["reflect"] = true
			break
		}
	}

	// Sort keys to make import alias generation predictable
	sortedPaths := make([]string, len(im))
	x := 0
	for pth := range im {
		sortedPaths[x] = pth
		x++
	}
	sort.Strings(sortedPaths)

	packagesName := createPackageMap(sortedPaths)

	g.packageMap = make(map[string]string, len(im))
	localNames := make(map[string]bool, len(im))
	for _, pth := range sortedPaths {
		base, ok := packagesName[pth]
		if !ok {
			base = sanitize(path.Base(pth))
		}

		// Local names for an imported package can usually be the basename of the import path.
		// A couple of situations don't permit that, such as duplicate local names
		// (e.g. importing "html/template" and "text/template"), or where the basename is
		// a keyword (e.g. "foo/case").
		// try base0, base1, ...
		pkgName := base
		i := 0
		for localNames[pkgName] || token.Lookup(pkgName).IsKeyword() {
			pkgName = base + strconv.Itoa(i)
			i++
		}

		// Avoid importing package if source pkg == output pkg
		if pth == pkg.PkgPath && outputPkgName == pkg.Name {
			continue
		}

		g.packageMap[pth] = pkgName
		localNames[pkgName] = true
	}

	if *writePkgComment {
		g.p("// Package %v is a generated GoMock package.", outputPkgName)
	}
	g.p("package %v", outputPkgName)
	g.p("")
	g.p("import (")
	g.in()
	for pkgPath, pkgName := range g.packageMap {
		if pkgPath == outputPackagePath {
			continue
		}
		g.p("%v %q", pkgName, pkgPath)
	}
	for _, pkgPath := range pkg.DotImports {
		g.p(". %q", pkgPath)
	}
	g.out()
	g.p(")")

	for _, intf := range pkg.Interfaces {
		if err := g.GenerateMockInterface(intf, outputPackagePath); err != nil {
			return err
		}
	}

	return nil
}

// The name of the mock type to use for the given interface identifier.
func (g *generator) mockName(typeName string) string {
	if mockName, ok := g.mockNames[typeName]; ok {
		return mockName
	}

	return "Mock" + typeName
}

func (g *generator) GenerateMockInterface(intf *model.Interface, outputPackagePath string) error {
	mockType := g.mockName(intf.Name)

	g.p("")
	g.p("// %v is a mock of %v interface", mockType, intf.Name)
	g.p("type %v struct {", mockType)
	g.in()
	g.p("ctrl     *gomock.Controller")
	g.p("recorder *%vMockRecorder", mockType)
	g.out()
	g.p("}")
	g.p("")

	g.p("// %vMockRecorder is the mock recorder for %v", mockType, mockType)
	g.p("type %vMockRecorder struct {", mockType)
	g.in()
	g.p("mock *%v", mockType)
	g.out()
	g.p("}")
	g.p("")

	// TODO: Re-enable this if we can import the interface reliably.
	// g.p("// Verify that the mock satisfies the interface at compile time.")
	// g.p("var _ %v = (*%v)(nil)", typeName, mockType)
	// g.p("")

	g.p("// New%v creates a new mock instance", mockType)
	g.p("func New%v(ctrl *gomock.Controller) *%v {", mockType, mockType)
	g.in()
	g.p("mock := &%v{ctrl: ctrl}", mockType)
	g.p("mock.recorder = &%vMockRecorder{mock}", mockType)
	g.p("return mock")
	g.out()
	g.p("}")
	g.p("")

	// XXX: possible name collision here if someone has EXPECT in their interface.
	g.p("// EXPECT returns an object that allows the caller to indicate expected use")
	g.p("func (m *%v) EXPECT() *%vMockRecorder {", mockType, mockType)
	g.in()
	g.p("return m.recorder")
	g.out()
	g.p("}")

	g.GenerateMockMethods(mockType, intf, outputPackagePath)

	return nil
}

func (g *generator) GenerateMockMethods(mockType string, intf *model.Interface, pkgOverride string) {
	for _, m := range intf.Methods {
		g.p("")
		_ = g.GenerateMockMethod(mockType, m, pkgOverride)
		g.p("")
		_ = g.GenerateMockRecorderMethod(mockType, m)
	}
}

func makeArgString(argNames, argTypes []string) string {
	args := make([]string, len(argNames))
	for i, name := range argNames {
		// specify the type only once for consecutive args of the same type
		if i+1 < len(argTypes) && argTypes[i] == argTypes[i+1] {
			args[i] = name
		} else {
			args[i] = name + " " + argTypes[i]
		}
	}
	return strings.Join(args, ", ")
}

// GenerateMockMethod generates a mock method implementation.
// If non-empty, pkgOverride is the package in which unqualified types reside.
func (g *generator) GenerateMockMethod(mockType string, m *model.Method, pkgOverride string) error {
	argNames := g.getArgNames(m)
	argTypes := g.getArgTypes(m, pkgOverride)
	argString := makeArgString(argNames, argTypes)

	rets := make([]string, len(m.Out))
	for i, p := range m.Out {
		rets[i] = p.Type.String(g.packageMap, pkgOverride)
	}
	retString := strings.Join(rets, ", ")
	if len(rets) > 1 {
		retString = "(" + retString + ")"
	}
	if retString != "" {
		retString = " " + retString
	}

	ia := newIdentifierAllocator(argNames)
	idRecv := ia.allocateIdentifier("m")

	g.p("// %v mocks base method", m.Name)
	g.p("func (%v *%v) %v(%v)%v {", idRecv, mockType, m.Name, argString, retString)
	g.in()
	g.p("%s.ctrl.T.Helper()", idRecv)

	var callArgs string
	if m.Variadic == nil {
		if len(argNames) > 0 {
			callArgs = ", " + strings.Join(argNames, ", ")
		}
	} else {
		// Non-trivial. The generated code must build a []interface{},
		// but the variadic argument may be any type.
		idVarArgs := ia.allocateIdentifier("varargs")
		idVArg := ia.allocateIdentifier("a")
		g.p("%s := []interface{}{%s}", idVarArgs, strings.Join(argNames[:len(argNames)-1], ", "))
		g.p("for _, %s := range %s {", idVArg, argNames[len(argNames)-1])
		g.in()
		g.p("%s = append(%s, %s)", idVarArgs, idVarArgs, idVArg)
		g.out()
		g.p("}")
		callArgs = ", " + idVarArgs + "..."
	}
	if len(m.Out) == 0 {
		g.p(`%v.ctrl.Call(%v, %q%v)`, idRecv, idRecv, m.Name, callArgs)
	} else {
		idRet := ia.allocateIdentifier("ret")
		g.p(`%v := %v.ctrl.Call(%v, %q%v)`, idRet, idRecv, idRecv, m.Name, callArgs)

		// Go does not allow "naked" type assertions on nil values, so we use the two-value form here.
		// The value of that is either (x.(T), true) or (Z, false), where Z is the zero value for T.
		// Happily, this coincides with the semantics we want here.
		retNames := make([]string, len(rets))
		for i, t := range rets {
			retNames[i] = ia.allocateIdentifier(fmt.Sprintf("ret%d", i))
			g.p("%s, _ := %s[%d].(%s)", retNames[i], idRet, i, t)
		}
		g.p("return " + strings.Join(retNames, ", "))
	}

	g.out()
	g.p("}")
	return nil
}

func (g *generator) GenerateMockRecorderMethod(mockType string, m *model.Method) error {
	argNames := g.getArgNames(m)

	var argString string
	if m.Variadic == nil {
		argString = strings.Join(argNames, ", ")
	} else {
		argString = strings.Join(argNames[:len(argNames)-1], ", ")
	}
	if argString != "" {
		argString += " interface{}"
	}

	if m.Variadic != nil {
		if argString != "" {
			argString += ", "
		}
		argString += fmt.Sprintf("%s ...interface{}", argNames[len(argNames)-1])
	}

	ia := newIdentifierAllocator(argNames)
	idRecv := ia.allocateIdentifier("mr")

	g.p("// %v indicates an expected call of %v", m.Name, m.Name)
	g.p("func (%s *%vMockRecorder) %v(%v) *gomock.Call {", idRecv, mockType, m.Name, argString)
	g.in()
	g.p("%s.mock.ctrl.T.Helper()", idRecv)

	var callArgs string
	if m.Variadic == nil {
		if len(argNames) > 0 {
			callArgs = ", " + strings.Join(argNames, ", ")
		}
	} else {
		if len(argNames) == 1 {
			// Easy: just use ... to push the arguments through.
			callArgs = ", " + argNames[0] + "..."
		} else {
			// Hard: create a temporary slice.
			idVarArgs := ia.allocateIdentifier("varargs")
			g.p("%s := append([]interface{}{%s}, %s...)",
				idVarArgs,
				strings.Join(argNames[:len(argNames)-1], ", "),
				argNames[len(argNames)-1])
			callArgs = ", " + idVarArgs + "..."
		}
	}
	g.p(`return %s.mock.ctrl.RecordCallWithMethodType(%s.mock, "%s", reflect.TypeOf((*%s)(nil).%s)%s)`, idRecv, idRecv, m.Name, mockType, m.Name, callArgs)

	g.out()
	g.p("}")
	return nil
}

func (g *generator) getArgNames(m *model.Method) []string {
	argNames := make([]string, len(m.In))
	for i, p := range m.In {
		name := p.Name
		if name == "" || name == "_" {
			name = fmt.Sprintf("arg%d", i)
		}
		argNames[i] = name
	}
	if m.Variadic != nil {
		name := m.Variadic.Name
		if name == "" {
			name = fmt.Sprintf("arg%d", len(m.In))
		}
		argNames = append(argNames, name)
	}
	return argNames
}

func (g *generator) getArgTypes(m *model.Method, pkgOverride string) []string {
	argTypes := make([]string, len(m.In))
	for i, p := range m.In {
		argTypes[i] = p.Type.String(g.packageMap, pkgOverride)
	}
	if m.Variadic != nil {
		argTypes = append(argTypes, "..."+m.Variadic.Type.String(g.packageMap, pkgOverride))
	}
	return argTypes
}

type identifierAllocator map[string]struct{}

func newIdentifierAllocator(taken []string) identifierAllocator {
	a := make(identifierAllocator, len(taken))
	for _, s := range taken {
		a[s] = struct{}{}
	}
	return a
}

func (o identifierAllocator) allocateIdentifier(want string) string {
	id := want
	for i := 2; ; i++ {
		if _, ok := o[id]; !ok {
			o[id] = struct{}{}
			return id
		}
		id = want + "_" + strconv.Itoa(i)
	}
}

// Output returns the generator's output, formatted in the standard Go style.
func (g *generator) Output() []byte {
	src, err := format.Source(g.buf.Bytes())
	if err != nil {
		log.Fatalf("Failed to format generated source code: %s\n%s", err, g.buf.String())
	}
	return src
}

// createPackageMap returns a map of import path to package name
// for specified importPaths.
func createPackageMap(importPaths []string) map[string]string {
	var pkg struct {
		Name       string
		ImportPath string
	}
	pkgMap := make(map[string]string)
	b := bytes.NewBuffer(nil)
	args := []string{"list", "-json"}
	args = append(args, importPaths...)
	cmd := exec.Command("go", args...)
	cmd.Stdout = b
	cmd.Run()
	dec := json.NewDecoder(b)
	for dec.More() {
		err := dec.Decode(&pkg)
		if err != nil {
			log.Printf("failed to decode 'go list' output: %v", err)
			continue
		}
		pkgMap[pkg.ImportPath] = pkg.Name
	}
	return pkgMap
}

func printVersion() {
	if version != "" {
		fmt.Printf("v%s\nCommit: %s\nDate: %s\n", version, commit, date)
	} else {
		printModuleVersion()
	}
}

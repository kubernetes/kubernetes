// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Unconvert removes redundant type conversions from Go packages.
package unconvert

import (
	"bytes"
	"flag"
	"fmt"
	"go/ast"
	"go/build"
	"go/format"
	"go/parser"
	"go/token"
	"go/types"
	"io/ioutil"
	"log"
	"os"
	"reflect"
	"runtime/pprof"
	"sort"
	"sync"
	"unicode"

	"github.com/kisielk/gotool"
	"golang.org/x/text/width"
	"golang.org/x/tools/go/loader"
)

// Unnecessary conversions are identified by the position
// of their left parenthesis within a source file.

type editSet map[token.Position]struct{}

type fileToEditSet map[string]editSet

func apply(file string, edits editSet) {
	if len(edits) == 0 {
		return
	}

	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, file, nil, parser.ParseComments)
	if err != nil {
		log.Fatal(err)
	}

	// Note: We modify edits during the walk.
	v := editor{edits: edits, file: fset.File(f.Package)}
	ast.Walk(&v, f)
	if len(edits) != 0 {
		log.Printf("%s: missing edits %s", file, edits)
	}

	// TODO(mdempsky): Write to temporary file and rename.
	var buf bytes.Buffer
	err = format.Node(&buf, fset, f)
	if err != nil {
		log.Fatal(err)
	}

	err = ioutil.WriteFile(file, buf.Bytes(), 0)
	if err != nil {
		log.Fatal(err)
	}
}

type editor struct {
	edits editSet
	file  *token.File
}

func (e *editor) Visit(n ast.Node) ast.Visitor {
	if n == nil {
		return nil
	}
	v := reflect.ValueOf(n).Elem()
	for i, n := 0, v.NumField(); i < n; i++ {
		switch f := v.Field(i).Addr().Interface().(type) {
		case *ast.Expr:
			e.rewrite(f)
		case *[]ast.Expr:
			for i := range *f {
				e.rewrite(&(*f)[i])
			}
		}
	}
	return e
}

func (e *editor) rewrite(f *ast.Expr) {
	call, ok := (*f).(*ast.CallExpr)
	if !ok {
		return
	}

	pos := e.file.Position(call.Lparen)
	if _, ok := e.edits[pos]; !ok {
		return
	}
	*f = call.Args[0]
	delete(e.edits, pos)
}

var (
	cr = []byte{'\r'}
	nl = []byte{'\n'}
)

func print(conversions []token.Position) {
	var file string
	var lines [][]byte

	for _, pos := range conversions {
		fmt.Printf("%s:%d:%d: unnecessary conversion\n", pos.Filename, pos.Line, pos.Column)
		if *flagV {
			if pos.Filename != file {
				buf, err := ioutil.ReadFile(pos.Filename)
				if err != nil {
					log.Fatal(err)
				}
				file = pos.Filename
				lines = bytes.Split(buf, nl)
			}

			line := bytes.TrimSuffix(lines[pos.Line-1], cr)
			fmt.Printf("%s\n", line)

			// For files processed by cgo, Column is the
			// column location after cgo processing, which
			// may be different than the source column
			// that we want here. In lieu of a better
			// heuristic for detecting this case, at least
			// avoid panicking if column is out of bounds.
			if pos.Column <= len(line) {
				fmt.Printf("%s^\n", rub(line[:pos.Column-1]))
			}
		}
	}
}

// Rub returns a copy of buf with all non-whitespace characters replaced
// by spaces (like rubbing them out with white out).
func rub(buf []byte) []byte {
	// TODO(mdempsky): Handle combining characters?
	var res bytes.Buffer
	for _, r := range string(buf) {
		if unicode.IsSpace(r) {
			res.WriteRune(r)
			continue
		}
		switch width.LookupRune(r).Kind() {
		case width.EastAsianWide, width.EastAsianFullwidth:
			res.WriteString("  ")
		default:
			res.WriteByte(' ')
		}
	}
	return res.Bytes()
}

var (
	flagAll        = flag.Bool("unconvert.all", false, "type check all GOOS and GOARCH combinations")
	flagApply      = flag.Bool("unconvert.apply", false, "apply edits to source files")
	flagCPUProfile = flag.String("unconvert.cpuprofile", "", "write CPU profile to file")
	// TODO(mdempsky): Better description and maybe flag name.
	flagSafe = flag.Bool("unconvert.safe", false, "be more conservative (experimental)")
	flagV    = flag.Bool("unconvert.v", false, "verbose output")
)

func usage() {
	fmt.Fprintf(os.Stderr, "usage: unconvert [flags] [package ...]\n")
	flag.PrintDefaults()
}

func nomain() {
	flag.Usage = usage
	flag.Parse()

	if *flagCPUProfile != "" {
		f, err := os.Create(*flagCPUProfile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	importPaths := gotool.ImportPaths(flag.Args())
	if len(importPaths) == 0 {
		return
	}

	var m fileToEditSet
	if *flagAll {
		m = mergeEdits(importPaths)
	} else {
		m = computeEdits(importPaths, build.Default.GOOS, build.Default.GOARCH, build.Default.CgoEnabled)
	}

	if *flagApply {
		var wg sync.WaitGroup
		for f, e := range m {
			wg.Add(1)
			f, e := f, e
			go func() {
				defer wg.Done()
				apply(f, e)
			}()
		}
		wg.Wait()
	} else {
		var conversions []token.Position
		for _, positions := range m {
			for pos := range positions {
				conversions = append(conversions, pos)
			}
		}
		sort.Sort(byPosition(conversions))
		print(conversions)
		if len(conversions) > 0 {
			os.Exit(1)
		}
	}
}

func Run(prog *loader.Program) []token.Position {
	m := computeEditsFromProg(prog)
	var conversions []token.Position
	for _, positions := range m {
		for pos := range positions {
			conversions = append(conversions, pos)
		}
	}
	return conversions
}

var plats = [...]struct {
	goos, goarch string
}{
	// TODO(mdempsky): buildall.bash also builds linux-386-387 and linux-arm-arm5.
	{"android", "386"},
	{"android", "amd64"},
	{"android", "arm"},
	{"android", "arm64"},
	{"darwin", "386"},
	{"darwin", "amd64"},
	{"darwin", "arm"},
	{"darwin", "arm64"},
	{"dragonfly", "amd64"},
	{"freebsd", "386"},
	{"freebsd", "amd64"},
	{"freebsd", "arm"},
	{"linux", "386"},
	{"linux", "amd64"},
	{"linux", "arm"},
	{"linux", "arm64"},
	{"linux", "mips64"},
	{"linux", "mips64le"},
	{"linux", "ppc64"},
	{"linux", "ppc64le"},
	{"linux", "s390x"},
	{"nacl", "386"},
	{"nacl", "amd64p32"},
	{"nacl", "arm"},
	{"netbsd", "386"},
	{"netbsd", "amd64"},
	{"netbsd", "arm"},
	{"openbsd", "386"},
	{"openbsd", "amd64"},
	{"openbsd", "arm"},
	{"plan9", "386"},
	{"plan9", "amd64"},
	{"plan9", "arm"},
	{"solaris", "amd64"},
	{"windows", "386"},
	{"windows", "amd64"},
}

func mergeEdits(importPaths []string) fileToEditSet {
	m := make(fileToEditSet)
	for _, plat := range plats {
		for f, e := range computeEdits(importPaths, plat.goos, plat.goarch, false) {
			if e0, ok := m[f]; ok {
				for k := range e0 {
					if _, ok := e[k]; !ok {
						delete(e0, k)
					}
				}
			} else {
				m[f] = e
			}
		}
	}
	return m
}

type noImporter struct{}

func (noImporter) Import(path string) (*types.Package, error) {
	panic("golang.org/x/tools/go/loader said this wouldn't be called")
}

func computeEdits(importPaths []string, os, arch string, cgoEnabled bool) fileToEditSet {
	ctxt := build.Default
	ctxt.GOOS = os
	ctxt.GOARCH = arch
	ctxt.CgoEnabled = cgoEnabled

	var conf loader.Config
	conf.Build = &ctxt
	conf.TypeChecker.Importer = noImporter{}
	for _, importPath := range importPaths {
		conf.Import(importPath)
	}
	prog, err := conf.Load()
	if err != nil {
		log.Fatal(err)
	}

	return computeEditsFromProg(prog)
}

func computeEditsFromProg(prog *loader.Program) fileToEditSet {
	type res struct {
		file  string
		edits editSet
	}
	ch := make(chan res)
	var wg sync.WaitGroup
	for _, pkg := range prog.InitialPackages() {
		for _, file := range pkg.Files {
			pkg, file := pkg, file
			wg.Add(1)
			go func() {
				defer wg.Done()
				v := visitor{pkg: pkg, file: prog.Fset.File(file.Package), edits: make(editSet)}
				ast.Walk(&v, file)
				ch <- res{v.file.Name(), v.edits}
			}()
		}
	}
	go func() {
		wg.Wait()
		close(ch)
	}()

	m := make(fileToEditSet)
	for r := range ch {
		m[r.file] = r.edits
	}
	return m
}

type step struct {
	n ast.Node
	i int
}

type visitor struct {
	pkg   *loader.PackageInfo
	file  *token.File
	edits editSet
	path  []step
}

func (v *visitor) Visit(node ast.Node) ast.Visitor {
	if node != nil {
		v.path = append(v.path, step{n: node})
	} else {
		n := len(v.path)
		v.path = v.path[:n-1]
		if n >= 2 {
			v.path[n-2].i++
		}
	}

	if call, ok := node.(*ast.CallExpr); ok {
		v.unconvert(call)
	}
	return v
}

func (v *visitor) unconvert(call *ast.CallExpr) {
	// TODO(mdempsky): Handle useless multi-conversions.

	// Conversions have exactly one argument.
	if len(call.Args) != 1 || call.Ellipsis != token.NoPos {
		return
	}
	ft, ok := v.pkg.Types[call.Fun]
	if !ok {
		fmt.Println("Missing type for function")
		return
	}
	if !ft.IsType() {
		// Function call; not a conversion.
		return
	}
	at, ok := v.pkg.Types[call.Args[0]]
	if !ok {
		fmt.Println("Missing type for argument")
		return
	}
	if !types.Identical(ft.Type, at.Type) {
		// A real conversion.
		return
	}
	if isUntypedValue(call.Args[0], &v.pkg.Info) {
		// Workaround golang.org/issue/13061.
		return
	}
	if *flagSafe && !v.isSafeContext(at.Type) {
		// TODO(mdempsky): Remove this message.
		fmt.Println("Skipped a possible type conversion because of -safe at", v.file.Position(call.Pos()))
		return
	}
	if v.isCgoCheckPointerContext() {
		// cmd/cgo generates explicit type conversions that
		// are often redundant when introducing
		// _cgoCheckPointer calls (issue #16).  Users can't do
		// anything about these, so skip over them.
		return
	}

	v.edits[v.file.Position(call.Lparen)] = struct{}{}
}

func (v *visitor) isCgoCheckPointerContext() bool {
	ctxt := &v.path[len(v.path)-2]
	if ctxt.i != 1 {
		return false
	}
	call, ok := ctxt.n.(*ast.CallExpr)
	if !ok {
		return false
	}
	ident, ok := call.Fun.(*ast.Ident)
	if !ok {
		return false
	}
	return ident.Name == "_cgoCheckPointer"
}

// isSafeContext reports whether the current context requires
// an expression of type t.
//
// TODO(mdempsky): That's a bad explanation.
func (v *visitor) isSafeContext(t types.Type) bool {
	ctxt := &v.path[len(v.path)-2]
	switch n := ctxt.n.(type) {
	case *ast.AssignStmt:
		pos := ctxt.i - len(n.Lhs)
		if pos < 0 {
			fmt.Println("Type conversion on LHS of assignment?")
			return false
		}
		if n.Tok == token.DEFINE {
			// Skip := assignments.
			return true
		}
		// We're a conversion in the pos'th element of n.Rhs.
		// Check that the corresponding element of n.Lhs is of type t.
		lt, ok := v.pkg.Types[n.Lhs[pos]]
		if !ok {
			fmt.Println("Missing type for LHS expression")
			return false
		}
		return types.Identical(t, lt.Type)
	case *ast.BinaryExpr:
		if n.Op == token.SHL || n.Op == token.SHR {
			if ctxt.i == 1 {
				// RHS of a shift is always safe.
				return true
			}
			// For the LHS, we should inspect up another level.
			fmt.Println("TODO(mdempsky): Handle LHS of shift expressions")
			return true
		}
		var other ast.Expr
		if ctxt.i == 0 {
			other = n.Y
		} else {
			other = n.X
		}
		ot, ok := v.pkg.Types[other]
		if !ok {
			fmt.Println("Missing type for other binop subexpr")
			return false
		}
		return types.Identical(t, ot.Type)
	case *ast.CallExpr:
		pos := ctxt.i - 1
		if pos < 0 {
			// Type conversion in the function subexpr is okay.
			return true
		}
		ft, ok := v.pkg.Types[n.Fun]
		if !ok {
			fmt.Println("Missing type for function expression")
			return false
		}
		sig, ok := ft.Type.(*types.Signature)
		if !ok {
			// "Function" is either a type conversion (ok) or a builtin (ok?).
			return true
		}
		params := sig.Params()
		var pt types.Type
		if sig.Variadic() && n.Ellipsis == token.NoPos && pos >= params.Len()-1 {
			pt = params.At(params.Len() - 1).Type().(*types.Slice).Elem()
		} else {
			pt = params.At(pos).Type()
		}
		return types.Identical(t, pt)
	case *ast.CompositeLit, *ast.KeyValueExpr:
		fmt.Println("TODO(mdempsky): Compare against value type of composite literal type at", v.file.Position(n.Pos()))
		return true
	case *ast.ReturnStmt:
		// TODO(mdempsky): Is there a better way to get the corresponding
		// return parameter type?
		var funcType *ast.FuncType
		for i := len(v.path) - 1; funcType == nil && i >= 0; i-- {
			switch f := v.path[i].n.(type) {
			case *ast.FuncDecl:
				funcType = f.Type
			case *ast.FuncLit:
				funcType = f.Type
			}
		}
		var typeExpr ast.Expr
		for i, j := ctxt.i, 0; j < len(funcType.Results.List); j++ {
			f := funcType.Results.List[j]
			if len(f.Names) == 0 {
				if i >= 1 {
					i--
					continue
				}
			} else {
				if i >= len(f.Names) {
					i -= len(f.Names)
					continue
				}
			}
			typeExpr = f.Type
			break
		}
		if typeExpr == nil {
			fmt.Println(ctxt)
		}
		pt, ok := v.pkg.Types[typeExpr]
		if !ok {
			fmt.Println("Missing type for return parameter at", v.file.Position(n.Pos()))
			return false
		}
		return types.Identical(t, pt.Type)
	case *ast.StarExpr, *ast.UnaryExpr:
		// TODO(mdempsky): I think these are always safe.
		return true
	case *ast.SwitchStmt:
		// TODO(mdempsky): I think this is always safe?
		return true
	default:
		// TODO(mdempsky): When can this happen?
		fmt.Printf("... huh, %T at %v\n", n, v.file.Position(n.Pos()))
		return true
	}
}

func isUntypedValue(n ast.Expr, info *types.Info) (res bool) {
	switch n := n.(type) {
	case *ast.BinaryExpr:
		switch n.Op {
		case token.SHL, token.SHR:
			// Shifts yield an untyped value if their LHS is untyped.
			return isUntypedValue(n.X, info)
		case token.EQL, token.NEQ, token.LSS, token.GTR, token.LEQ, token.GEQ:
			// Comparisons yield an untyped boolean value.
			return true
		case token.ADD, token.SUB, token.MUL, token.QUO, token.REM,
			token.AND, token.OR, token.XOR, token.AND_NOT,
			token.LAND, token.LOR:
			return isUntypedValue(n.X, info) && isUntypedValue(n.Y, info)
		}
	case *ast.UnaryExpr:
		switch n.Op {
		case token.ADD, token.SUB, token.NOT, token.XOR:
			return isUntypedValue(n.X, info)
		}
	case *ast.BasicLit:
		// Basic literals are always untyped.
		return true
	case *ast.ParenExpr:
		return isUntypedValue(n.X, info)
	case *ast.SelectorExpr:
		return isUntypedValue(n.Sel, info)
	case *ast.Ident:
		if obj, ok := info.Uses[n]; ok {
			if obj.Pkg() == nil && obj.Name() == "nil" {
				// The universal untyped zero value.
				return true
			}
			if b, ok := obj.Type().(*types.Basic); ok && b.Info()&types.IsUntyped != 0 {
				// Reference to an untyped constant.
				return true
			}
		}
	case *ast.CallExpr:
		if b, ok := asBuiltin(n.Fun, info); ok {
			switch b.Name() {
			case "real", "imag":
				return isUntypedValue(n.Args[0], info)
			case "complex":
				return isUntypedValue(n.Args[0], info) && isUntypedValue(n.Args[1], info)
			}
		}
	}

	return false
}

func asBuiltin(n ast.Expr, info *types.Info) (*types.Builtin, bool) {
	for {
		paren, ok := n.(*ast.ParenExpr)
		if !ok {
			break
		}
		n = paren.X
	}

	ident, ok := n.(*ast.Ident)
	if !ok {
		return nil, false
	}

	obj, ok := info.Uses[ident]
	if !ok {
		return nil, false
	}

	b, ok := obj.(*types.Builtin)
	return b, ok
}

type byPosition []token.Position

func (p byPosition) Len() int {
	return len(p)
}

func (p byPosition) Less(i, j int) bool {
	if p[i].Filename != p[j].Filename {
		return p[i].Filename < p[j].Filename
	}
	if p[i].Line != p[j].Line {
		return p[i].Line < p[j].Line
	}
	return p[i].Column < p[j].Column
}

func (p byPosition) Swap(i, j int) {
	p[i], p[j] = p[j], p[i]
}

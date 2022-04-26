package deadcode

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"golang.org/x/tools/go/loader"
)

var exitCode int

var (
	withTestFiles bool
)

type Issue struct {
	Pos             token.Position
	UnusedIdentName string
}

func Run(program *loader.Program) ([]Issue, error) {
	ctx := &Context{
		program: program,
	}
	report := ctx.Process()
	var issues []Issue
	for _, obj := range report {
		issues = append(issues, Issue{
			Pos:             program.Fset.Position(obj.Pos()),
			UnusedIdentName: obj.Name(),
		})
	}

	return issues, nil
}

func fatalf(format string, args ...interface{}) {
	panic(fmt.Errorf(format, args...))
}

type Context struct {
	cwd       string
	withTests bool

	program *loader.Program
}

// pos resolves a compact position encoding into a verbose one
func (ctx *Context) pos(pos token.Pos) token.Position {
	if ctx.cwd == "" {
		ctx.cwd, _ = os.Getwd()
	}
	p := ctx.program.Fset.Position(pos)
	f, err := filepath.Rel(ctx.cwd, p.Filename)
	if err == nil {
		p.Filename = f
	}
	return p
}

// error formats the error to standard error, adding program
// identification and a newline
func (ctx *Context) errorf(pos token.Pos, format string, args ...interface{}) {
	p := ctx.pos(pos)
	fmt.Fprintf(os.Stderr, p.String()+": "+format+"\n", args...)
	exitCode = 2
}

func (ctx *Context) Load(args ...string) {
	// TODO
}

func (ctx *Context) Process() []types.Object {
	prog := ctx.program
	var allUnused []types.Object
	for _, pkg := range prog.Imported {
		unused := ctx.doPackage(prog, pkg)
		allUnused = append(allUnused, unused...)
	}
	for _, pkg := range prog.Created {
		unused := ctx.doPackage(prog, pkg)
		allUnused = append(allUnused, unused...)
	}
	sort.Sort(objects(allUnused))
	return allUnused
}

func isTestFuncByName(name string) bool {
	return strings.HasPrefix(name, "Test") || strings.HasPrefix(name, "Benchmark") || strings.HasPrefix(name, "Example")
}

func (ctx *Context) doPackage(prog *loader.Program, pkg *loader.PackageInfo) []types.Object {
	used := make(map[types.Object]bool)
	for _, file := range pkg.Files {
		ast.Inspect(file, func(n ast.Node) bool {
			id, ok := n.(*ast.Ident)
			if !ok {
				return true
			}
			obj := pkg.Info.Uses[id]
			if obj != nil {
				used[obj] = true
			}
			return false
		})
	}

	global := pkg.Pkg.Scope()
	var unused []types.Object
	for _, name := range global.Names() {
		if pkg.Pkg.Name() == "main" && name == "main" {
			continue
		}
		obj := global.Lookup(name)
		_, isSig := obj.Type().(*types.Signature)
		pos := ctx.pos(obj.Pos())
		isTestMethod := isSig && isTestFuncByName(obj.Name()) && strings.HasSuffix(pos.Filename, "_test.go")
		if !used[obj] && ((pkg.Pkg.Name() == "main" && !isTestMethod) || !ast.IsExported(name)) {
			unused = append(unused, obj)
		}
	}
	return unused
}

type objects []types.Object

func (s objects) Len() int           { return len(s) }
func (s objects) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }
func (s objects) Less(i, j int) bool { return s[i].Pos() < s[j].Pos() }

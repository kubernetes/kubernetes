// Command dedupimport finds and removes duplicate imports that have the
// same import path but different import names. See 'dedupimport -h' for
// usage.
//
// When resolving duplicate imports, by default, it keeps the unnamed import
// and removes the named imports. This behavior can be customized with the
// '-keep' flag (described below). After resolving duplicates it updates the
// rest of the code in the file that may be using the old, removed import
// identifier to use the new import identifier.
//
// As a special case, the tool never removes side-effect imports ("_") and
// dot imports ("."); these imports are allowed to coexist with regular
// imports, even if the import paths are duplicated.
//
// The command exits with exit code 2 if the command was invoked incorrectly;
// 1 if there was an error while opening, parsing, or rewriting files; and
// 0 otherwise.
//
// The typical usage is:
//
//   dedupimport file1.go dir1 dir2 # prints updated versions to stdout
//   dedupimport -w file.go         # overwrite original source file
//   dedupimport -d file.go         # display diff
//   dedupimport -l file.go dir     # list the filenames that have duplicate imports
//
// Example
//
// Given the file
//
// 	package pkg
//
// 	import (
// 		"code.org/frontend"
// 		fe "code.org/frontend"
// 	)
//
// 	var client frontend.Client
// 	var server fe.Server
//
// running dedupimport with default options will produce
//
// 	package pkg
//
// 	import (
// 		"code.org/frontend"
// 	)
//
// 	var client frontend.Client
// 	var server frontend.Server
//
// Strategy to use when resolving duplicates
//
// The '-keep' flag allows you to choose which import to keep and which ones to
// remove when resolving duplicates in a file, aka the strategy to use:
//
//   - the "unnamed" strategy keeps the unnamed import if one exists, or the
//     first import otherwise;
//   - the "named" strategy keeps the first-occurring shortest named import if
//     one exists, or the first import otherwise;
//   - the "comment" strategy keeps the first-occurring import with either a
//     doc or a line comment if one exists, or the first import otherwise; and
//   - the "first" strategy keeps the first import.
//
// Inability to rewrite
//
// Sometimes rewriting a file to use the updated import declaration can be
// unsafe. In the following example, it is not possible to safely change "u"
// -> "url" inside fetch because the identifier, url, already exists in the
// scope and does not refer to the import.
//
// Such contrived scenarios rarely happen in practice.  But if they do, the
// command prints a warning and skips the file.
//
//   import u "net/url"
//   import "net/url"
//
//   var google = url.QueryEscape("https://google.com/?q=something")
//
//   func fetch(url string) {
//      u.Parse(url)
//      ...
//   }
//
// Package name guessing
//
// For unnamed imports, the command has to guess the import's package name by
// looking at the import path. The package name is, in most cases, the
// basename of the import path. The command automatically handles patterns
// such as these:
//
//   Import path                            Package name    Notes
//   -----------------                      ------------    ---------------
//   github.com/foo/bar                     bar             Standard naming
//   gopkg.in/yaml.v2                       yaml            Remove version
//   github.com/nishanths/go-xkcd           xkcd            Remove 'go-' prefix
//   github.com/nishanths/lyft-go           lyft            Remove '-go' suffix
//
// To instruct the command on how to handle more complicated patterns, the
// '-m' flag can be used. The format for the flag is:
//   importpath=packagename
// The flag can be repeated multiple times to specify multiple mappings. For
// example:
//
//   dedupimport -m github.com/proj/serverimpl=server \
//     -m github.com/priarie/go-k8s-client=clientk8s
package main

import (
	"bytes"
	"flag"
	"fmt"
	"go/ast"
	"go/build"
	"go/format"
	"go/parser"
	"go/scanner"
	"go/token"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"unicode"
)

const help = `usage: dedupimport [flags] [path ...]
See 'go doc github.com/nishanths/dedupimport' for details.`

func usage() {
	fmt.Fprintf(os.Stderr, "%s\n\n", help)
	flagSet.PrintDefaults()
	os.Exit(2)
}

type MultiFlag struct {
	name string
	m    map[string]string
}

func (m MultiFlag) String() string {
	if len(m.m) == 0 {
		return ""
	}
	return fmt.Sprint(m.m)
}

func (m MultiFlag) Set(val string) error {
	c := strings.Split(val, "=")
	if len(c) != 2 {
		return fmt.Errorf("wrong format for -%s: %s", m.name, val)
	}
	if m.m == nil {
		m.m = make(map[string]string)
	}
	m.m[c[0]] = c[1]
	return nil
}

var (
	flagSet    = flag.NewFlagSet("dedupimport", flag.ExitOnError)
	diff       = flagSet.Bool("d", false, "display diff instead of rewriting files")
	allErrors  = flagSet.Bool("e", false, "report all parse errors, not just the first 10 on different lines")
	list       = flagSet.Bool("l", false, "list files with duplicate imports")
	overwrite  = flagSet.Bool("w", false, "write result to source file instead of stdout")
	importOnly = flagSet.Bool("i", false, "only modify imports; don't adjust rest of the file")
	strategy   = flagSet.String("keep", "unnamed", "which import to keep: first, comment, named, or unnamed")
	pkgNames   = MultiFlag{name: "m"}
)

var exitCode = 0

func setExitCode(c int) {
	if c > exitCode {
		exitCode = c
	}
}

func main() {
	flagSet.Var(&pkgNames, "m", "`mapping` from import path to package name; can be repeated")
	flagSet.Usage = usage
	flagSet.Parse(os.Args[1:])

	switch *strategy {
	case "first", "comment", "named", "unnamed":
	default:
		fmt.Fprintf(os.Stderr, "unknown value for -keep: %s\n", *strategy)
		os.Exit(2)
	}

	// fset is the FileSet for the entire command invocation.
	var fset = token.NewFileSet()

	if flagSet.NArg() == 0 {
		if *overwrite {
			fmt.Fprint(os.Stderr, "cannot use -w with stdin\n")
			os.Exit(2)
		} else {
			handleFile(fset, true, "<standard input>", os.Stdout) // use the same filename that gofmt uses
		}
	} else {
		for i := 0; i < flagSet.NArg(); i++ {
			path := flagSet.Arg(i)
			info, err := os.Stat(path)
			if err != nil {
				fmt.Fprintln(os.Stderr, err)
				setExitCode(1)
			} else if info.IsDir() {
				handleDir(fset, path)
			} else {
				handleFile(fset, false, path, os.Stdout)
			}
		}
	}

	if exitCode != 0 {
		os.Exit(exitCode)
	}
}

func parserMode() parser.Mode {
	if *allErrors {
		return parser.ParseComments | parser.AllErrors
	}
	return parser.ParseComments
}

type posSpan struct {
	Start token.Pos
	End   token.Pos
}

func processFile(fset *token.FileSet, src []byte, filename string) (*ast.File, error) {
	file, err := parser.ParseFile(fset, filename, src, parserMode())
	if err != nil {
		return nil, err
	}

	// Record positions for specs.
	// Need to do this before updating file.Imports.
	pos := make([]posSpan, len(file.Imports))
	for i, s := range file.Imports {
		pos[i] = posSpan{s.Pos(), s.End()}
	}

	// Find duplicate imports.
	imports := markDuplicates(file.Imports)

	var keep, remove []*ast.ImportSpec
	for _, im := range imports {
		if im.remove {
			remove = append(remove, im.spec)
		} else {
			keep = append(keep, im.spec)
		}
	}
	if len(remove) == 0 {
		// nothing to do
		return nil, nil
	}

	// Record comments.
	cmap := ast.NewCommentMap(fset, file, file.Comments)

	file.Imports = keep   // update the file's imports.
	trimImportDecls(file) // update the file's AST.

	// Get rid of comments that no longer belong.
	file.Comments = cmap.Filter(file).Comments()

	if !*importOnly {
		srcDir := filepath.Dir(filename)

		// Get the identifiers in scopes.
		// We need it to check if rewriting selector exprs is safe.
		scope := walkFile(file)

		// Build up the selector expr rewrite rules.
		rules := make(map[string]string)
		for _, im := range imports {
			if !im.remove {
				continue
			}
			from := packageNameForImport(im.spec, srcDir)
			to := packageNameForImport(im.subsumedBy, srcDir)
			rules[from] = to
		}

		// Rewrite.
		err := rewriteSelectorExprs(fset, rules, scope, file.Name.Name)
		if err != nil {
			return nil, err
		}
	}

	// If an import is removed, merge the next line into it.
	for _, im := range imports {
		if im.remove {
			pos := im.spec.Pos()
			line := fset.Position(pos).Line
			fp := fset.File(pos)
			if line >= fp.LineCount() {
				// don't do merging at end of file
				continue
			}
			fp.MergeLine(line)
		}
	}
	// Update the positions we recorded earlier.
	// Don't have to worry about fixing up comments here
	// because comments for removed imports would have already been removed
	// by the commentMap work earlier.
	for i, im := range imports {
		s := im.spec
		if s.Name != nil {
			s.Name.NamePos = pos[i].Start
		}
		s.Path.ValuePos = pos[i].Start
		s.EndPos = pos[i].End
	}

	return file, nil
}

type scopeStack struct {
	list []*Scope
}

func (s *scopeStack) push(sc *Scope) {
	s.list = append(s.list, sc)
}

func (s *scopeStack) pop() *Scope {
	if len(s.list) == 0 {
		panic("pop of zero-length stack")
	}
	res := s.list[len(s.list)-1]
	s.list = s.list[:len(s.list)-1]
	return res
}

// latest returns the latest non-nil entry in the stack
// or nil if there is no such entry.
func (s *scopeStack) latest() *Scope {
	for i := len(s.list) - 1; i >= 0; i-- {
		if s.list[i] != nil {
			return s.list[i]
		}
	}
	return nil
}

// rewriteSelectorExprs rewrites selector exprs in the supplied scope based
// on the rewrite rules. If a rewrite could not be performed, it will be
// described in the returned error. The returned error will be of type
// MultiError (even if there was only a single error).
func rewriteSelectorExprs(fset *token.FileSet, rules map[string]string, root *Scope, pkgName string) error {
	// first, map nodes to their scopes.
	scopeByNode := make(map[ast.Node]*Scope)
	root.each(func(s *Scope) bool {
		scopeByNode[s.node] = s
		return true
	})

	var errs MultiError
	addError := func(e error) {
		errs = append(errs, e)
	}

	// NOTE: this doesn't protect against package scope variables fully.
	// For instance, 'var fe int' could be in a different file and visible
	// across the package, but we would not warn about a "frontend" -> "fe"
	// selector rewrite. This is okay for the most part, because
	// the code would have had a compile error before anyway.
	var stack scopeStack
	ast.Inspect(root.node, func(node ast.Node) bool {
		sc := scopeByNode[node]
		if node != nil {
			// enter a deeper level.  sc may be nil (because the node
			// wasn't a scope creating node).
			// the latest non-nil sc is the scope we want to track,
			// and this is the scope that is returned by calling `latest`.
			stack.push(sc)
		}

		switch x := node.(type) {
		case *ast.SelectorExpr:
			// we only care about package selector exprs,
			// which should always have X be of type *ast.Ident.
			ident, ok := x.X.(*ast.Ident)
			if !ok {
				// don't care
				break
			}
			from := ident.Name
			to, ok := rules[from]
			if !ok {
				// this selector expr is not one we want to rewrite
				break
			}
			latest := stack.latest()
			if latest == nil {
				panicf("[code bug] selector expr should be in a scope, but unaware of any such scope")
			}
			if isGoKeyword(to) {
				// source code must already have a parse or build error.
				addError(&GoKeywordError{fset.Position(x.X.Pos()), from, to})
				break
			}
			if !isValidIdent(to) {
				// source code must already have a parse/build error.
				addError(&InvalidIdentError{fset.Position(x.X.Pos()), from, to})
				break
			}
			if id, ok := latest.available(to); ok && id.NamePos <= ident.NamePos { // exists && declared before
				addError(&ScopeError{fset.Position(x.X.Pos()), from, to})
				break
			}
			ident.Name = to // rewrite
		}

		if node == nil {
			// depth-first unraveling call by ast.Inspect.  pop an entry.
			// the entry popped may be nil (see comment at `push`).
			stack.pop()
		}

		return true
	})

	if len(errs) == 0 {
		return nil
	}
	return errs
}

func isValidIdent(w string) bool {
	// https://golang.org/ref/spec#identifier
	if len(w) == 0 {
		return false
	}
	isLetter := func(r rune) bool {
		return unicode.In(r, unicode.Lu, unicode.Ll, unicode.Lt, unicode.Lm, unicode.Lo)
	}
	isNumber := func(r rune) bool {
		return unicode.In(r, unicode.Nd)
	}
	for i, r := range w {
		switch i {
		case 0:
			if !(isLetter(r) || r == '_') {
				return false
			}
		default:
			if !(isLetter(r) || r == '_' || isNumber(r)) {
				return false
			}
		}
	}
	return true
}

func isGoKeyword(w string) bool {
	switch w {
	case "break", "default", "func", "interface", "select",
		"case", "defer", "go", "map", "struct",
		"chan", "else", "goto", "package", "switch",
		"const", "fallthrough", "if", "range", "type",
		"continue", "for", "import", "return", "var":
		return true
	default:
		return false
	}
}

type InvalidIdentError struct {
	position token.Position
	from, to string
}

var _ error = (*InvalidIdentError)(nil)

func (s *InvalidIdentError) Error() string {
	return fmt.Sprintf("%s: cannot rewrite %s -> %s: identifier %[3]s is not a valid identifier; "+
		"specify a mapping for the import using '-m'", s.position, s.from, s.to)
}

type GoKeywordError struct {
	position token.Position
	from, to string
}

var _ error = (*GoKeywordError)(nil)

func (s *GoKeywordError) Error() string {
	return fmt.Sprintf("%s: cannot rewrite %s -> %s: identifier %[3]s is a go keyword; "+
		"specify a mapping for the import using '-m'", s.position, s.from, s.to)
}

type ScopeError struct {
	position token.Position
	from, to string
}

var _ error = (*ScopeError)(nil)

func (s *ScopeError) Error() string {
	return fmt.Sprintf("%s: cannot rewrite %s -> %s: identifier %[3]s in scope might not be referring to the import",
		s.position, s.from, s.to)
}

type MultiError []error

var _ error = (MultiError)(nil)

func (m MultiError) Error() string {
	if len(m) == 0 {
		panic("[code bug] MultiError has zero errors") // don't make such a MultiError in the first place.
	}
	var buf bytes.Buffer
	for i, e := range m {
		buf.WriteString(e.Error())
		if i != len(m)-1 {
			buf.WriteString("\n")
		}
	}
	return buf.String()
}

// trimImportDecls trims the file's import declarations based on the import
// specs present in file.Imports.
func trimImportDecls(file *ast.File) {
	lookup := make(map[*ast.ImportSpec]struct{}, len(file.Imports))
	for _, im := range file.Imports {
		lookup[im] = struct{}{}
	}

	for i := range file.Decls {
		genDecl, ok := file.Decls[i].(*ast.GenDecl)
		if !ok || genDecl.Tok != token.IMPORT {
			continue
		}
		var keep []ast.Spec // type is generic so that we can use in assignment below.
		for _, spec := range genDecl.Specs {
			im, ok := spec.(*ast.ImportSpec)
			if !ok {
				// WTF, doesn't match godoc
				panicf("expected ImportSpec")
			}
			if _, ok := lookup[im]; ok {
				// was not removed during deduping,
				// so append it to our list of imports to keep.
				keep = append(keep, spec)
			}
		}
		genDecl.Specs = keep
		file.Decls[i] = genDecl
	}

	var nonEmptyDecls []ast.Decl
	for _, decl := range file.Decls {
		genDecl, ok := decl.(*ast.GenDecl)
		if !ok || genDecl.Tok != token.IMPORT {
			nonEmptyDecls = append(nonEmptyDecls, decl)
			continue
		}
		if len(genDecl.Specs) != 0 {
			nonEmptyDecls = append(nonEmptyDecls, decl)
		}
	}
	file.Decls = nonEmptyDecls
}

// markDuplicates returns the import specs with a removal status marked.
// Neither the input slice nor its elements are modified.
func markDuplicates(input []*ast.ImportSpec) []*ImportSpec {
	imports := make([]*ImportSpec, len(input))
	for i := range input {
		imports[i] = &ImportSpec{input[i], false, nil}
	}

	importPaths := make(map[string][]*ImportSpec)
	for _, im := range imports {
		spec := im.spec
		// NOTE: The panics below indicate conditions that should have been
		// caught already by the parser.
		if spec.Path.Kind != token.STRING {
			panicf("import path %s is not a string", spec.Path.Value)
		}
		// skip dot and side effect imports. for now, let's assume it's okay
		// to have both these coexist with regular imports. In fact, it looks
		// like it's necessary to not remove _ imports; that's the only way both _
		// and regular import can be used together in a file.
		if spec.Name != nil && (spec.Name.Name == "." || spec.Name.Name == "_") {
			continue
		}
		// normalize `fmt` vs. "fmt", for instance
		path, err := normalizeImportPath(spec.Path.Value)
		if err != nil {
			// wasn't a valid string?
			panicf("unquoting path: %s", err)
		}
		importPaths[path] = append(importPaths[path], im)
	}

	duplicateImportPaths := make(map[string][]*ImportSpec)
	for p, v := range importPaths {
		if len(v) > 1 {
			duplicateImportPaths[p] = v
		}
	}

	for _, v := range duplicateImportPaths {
		var keepIdx int

		switch *strategy {
		case "unnamed":
			// Find the index of the first unnamed import.
			// That's the one we will keep.
			idx := -1
			for i := range v {
				if v[i].spec.Name == nil {
					idx = i
					break
				}
			}
			keepIdx = idx
			if keepIdx == -1 {
				// no unnamed import exists. fall back to keeping
				// the first one.
				keepIdx = 0
			}
		case "first":
			keepIdx = 0
		case "comment":
			// Find the index of the first import with either a doc comment
			// or line comment.
			idx := -1
			for i := range v {
				if v[i].spec.Comment != nil || v[i].spec.Doc != nil {
					idx = i
					break
				}
			}
			keepIdx = idx
			if keepIdx == -1 {
				// use first one.
				keepIdx = 0
			}
		case "named":
			// Find the shortest named import.
			// If multiple exist with the same shortest length, we keep the
			// first of those.
			idx := -1
			length := -1
			for i := range v {
				if v[i].spec.Name != nil && (len(v[i].spec.Name.Name) < length || length == -1) {
					idx = i
					length = len(v[i].spec.Name.Name)
				}
			}
			keepIdx = idx
			if keepIdx == -1 {
				// no named import existed at all.
				// fall back to keeping the first one.
				keepIdx = 0
			}
		}

		// mark imports for removal
		for i := 0; i < len(v); i++ {
			if i != keepIdx {
				v[i].remove = true
				v[i].subsumedBy = v[keepIdx].spec
			}
		}
	}

	return imports
}

func normalizeImportPath(p string) (string, error) {
	return strconv.Unquote(p)
}

func packageNameForImport(spec *ast.ImportSpec, srcDir string) string {
	if spec.Name != nil {
		// named import
		return spec.Name.Name
	}
	path, err := normalizeImportPath(spec.Path.Value)
	if err != nil {
		// wasn't a valid string?
		panicf("unquoting path: %s", err)
	}
	return packageNameForPath(path, srcDir)
}

func packageNameForPath(p string, srcDir string) string {
	// Use the mapping first.
	if name, ok := pkgNames.m[p]; ok {
		return name
	}
	// Try build.Import. Ignore the error; pkg could be non-nil
	// with sufficient information we care about regardless of the error.
	pkg, _ := build.Import(p, srcDir, build.AllowBinary|build.ImportComment)
	if pkg != nil && pkg.Name != "" {
		return pkg.Name
	}
	// Guess it.
	return guessPackageName(p)
}

// Guesses the package name based on the import path.
// The returned string may not be a valid identifier (and hence not a valid
// package name).
func guessPackageName(p string) string {
	// at its most complicated, this can do:
	// "foo.org/blah/go-yaml.v2" -> "yaml"
	return guessPackageName_(p, true)
}

var (
	dotvn = regexp.MustCompile(`\.v\d+$`)
)

func guessPackageName_(p string, again bool) string {
	sidx := strings.LastIndex(p, "/")
	if sidx == -1 {
		return p
	}

	last := p[sidx+1:]

	// Order matters. For instance, the .dotvn check should happen before the
	// "go-" prefix check.
	switch {
	case again && dotvn.MatchString(last):
		// foo.org/blah/go-yaml.v2
		// need to use (a cleaned up version of) "go-yaml"
		didx := strings.LastIndex(p, ".")
		if didx == -1 {
			panicf("[code bug] should have '.' in string: %s", p)
		}
		return guessPackageName_(p[:didx], false)
	case strings.HasPrefix(last, "go-"):
		// foo.org/go-yaml
		return strings.TrimPrefix(last, "go-")
	case strings.HasSuffix(last, "-go"):
		// foo.org/yaml-go
		return strings.TrimSuffix(last, "-go")
	default:
		return last
	}
}

type ImportSpec struct {
	spec       *ast.ImportSpec // this spec
	remove     bool            // indicator for removal
	subsumedBy *ast.ImportSpec // the spec replacing this spec; nil if remove==false
}

func panicf(format string, v ...interface{}) {
	s := fmt.Sprintf(format, v...)
	panic(s)
}

func handleFile(fset *token.FileSet, stdin bool, filename string, out io.Writer) {
	var src []byte
	var err error
	if stdin {
		src, err = ioutil.ReadAll(os.Stdin)
	} else {
		src, err = ioutil.ReadFile(filename)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		setExitCode(1)
		return
	}

	// Keep the following in sync with test code.
	changedFile, err := processFile(fset, src, filename)
	if err != nil {
		scanner.PrintError(os.Stderr, err)
		setExitCode(1)
		return
	}
	res := src
	if changedFile != nil {
		var buf bytes.Buffer
		err := format.Node(&buf, fset, changedFile)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			setExitCode(1)
			return
		}
		res = buf.Bytes()
	}
	err = writeOutput(out, src, res, filename)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		setExitCode(1)
		return
	}
}

func handleDir(fset *token.FileSet, p string) {
	if err := filepath.Walk(p, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !isGoFile(info) {
			return nil
		}
		handleFile(fset, false, path, os.Stdout)
		return nil
	}); err != nil {
		fmt.Fprintln(os.Stderr, err)
		setExitCode(1)
	}
}

func writeOutput(out io.Writer, src, res []byte, filename string) error {
	// Copied from processFile in cmd/gofmt.
	if !bytes.Equal(res, src) {
		if *list {
			fmt.Fprintln(out, filename)
		}
		// TODO: filename can be gibberish like "<stdin>" here, but -w is not
		// allowed for stdin in main, hence why this doesn't blow up. clean this
		// up.
		if *overwrite {
			fi, err := os.Stat(filename)
			if err != nil {
				return err
			}
			perm := fi.Mode().Perm()
			// make a temporary backup before overwriting original
			bakname, err := backupFile(filename+".", src, perm)
			if err != nil {
				return err
			}
			err = ioutil.WriteFile(filename, res, perm)
			if err != nil {
				os.Rename(bakname, filename)
				return err
			}
			err = os.Remove(bakname)
			if err != nil {
				return err
			}
		}
		if *diff {
			data, err := cmdDiff(src, res, filename)
			if err != nil {
				return fmt.Errorf("computing diff: %s", err)
			}
			fmt.Printf("diff -u %s %s\n", filepath.ToSlash(filename+".orig"), filepath.ToSlash(filename))
			out.Write(data)
		}
	}

	if !*list && !*overwrite && !*diff {
		_, err := out.Write(res)
		if err != nil {
			return nil
		}
	}

	return nil
}

func isGoFile(f os.FileInfo) bool {
	// ignore non-Go files
	name := f.Name()
	return !f.IsDir() && !strings.HasPrefix(name, ".") && !strings.HasPrefix(name, "_") && strings.HasSuffix(name, ".go")
}

// ----------------------------------------------------------------------------
// Copied from cmd/gofmt.
// https://github.com/golang/go/commit/e86168430f0aab8f971763e4b00c2aae7bec55f0

func writeTempFile(dir, prefix string, data []byte) (string, error) {
	file, err := ioutil.TempFile(dir, prefix)
	if err != nil {
		return "", err
	}
	_, err = file.Write(data)
	if err1 := file.Close(); err == nil {
		err = err1
	}
	if err != nil {
		os.Remove(file.Name())
		return "", err
	}
	return file.Name(), nil
}

func cmdDiff(b1, b2 []byte, filename string) (data []byte, err error) {
	f1, err := writeTempFile("", "dedupimport", b1)
	if err != nil {
		return
	}
	defer os.Remove(f1)

	f2, err := writeTempFile("", "dedupimport", b2)
	if err != nil {
		return
	}
	defer os.Remove(f2)

	cmd := "diff"
	if runtime.GOOS == "plan9" {
		cmd = "/bin/ape/diff"
	}

	data, err = exec.Command(cmd, "-u", f1, f2).CombinedOutput()
	if len(data) > 0 {
		// diff exits with a non-zero status when the files don't match.
		// Ignore that failure as long as we get output.
		return replaceTempFilename(data, filename)
	}
	return
}

// replaceTempFilename replaces temporary filenames in diff with actual one.
//
// --- /tmp/dedupimport316145376	2017-02-03 19:13:00.280468375 -0500
// +++ /tmp/dedupimport617882815	2017-02-03 19:13:00.280468375 -0500
// ...
// ->
// --- path/to/file.go.orig	2017-02-03 19:13:00.280468375 -0500
// +++ path/to/file.go	2017-02-03 19:13:00.280468375 -0500
// ...
func replaceTempFilename(diff []byte, filename string) ([]byte, error) {
	bs := bytes.SplitN(diff, []byte{'\n'}, 3)
	if len(bs) < 3 {
		return nil, fmt.Errorf("got unexpected diff for %s", filename)
	}
	// Preserve timestamps.
	var t0, t1 []byte
	if i := bytes.LastIndexByte(bs[0], '\t'); i != -1 {
		t0 = bs[0][i:]
	}
	if i := bytes.LastIndexByte(bs[1], '\t'); i != -1 {
		t1 = bs[1][i:]
	}
	// Always print filepath with slash separator.
	f := filepath.ToSlash(filename)
	bs[0] = []byte(fmt.Sprintf("--- %s%s", f+".orig", t0))
	bs[1] = []byte(fmt.Sprintf("+++ %s%s", f, t1))
	return bytes.Join(bs, []byte{'\n'}), nil
}

const chmodSupported = runtime.GOOS != "windows"

// backupFile writes data to a new file named filename<number> with permissions perm,
// with <number randomly chosen such that the file name is unique. backupFile returns
// the chosen file name.
func backupFile(filename string, data []byte, perm os.FileMode) (string, error) {
	// create backup file
	f, err := ioutil.TempFile(filepath.Dir(filename), filepath.Base(filename))
	if err != nil {
		return "", err
	}
	bakname := f.Name()
	if chmodSupported {
		err = f.Chmod(perm)
		if err != nil {
			f.Close()
			os.Remove(bakname)
			return bakname, err
		}
	}

	// write data to backup file
	n, err := f.Write(data)
	if err == nil && n < len(data) {
		err = io.ErrShortWrite
	}
	if err1 := f.Close(); err == nil {
		err = err1
	}

	return bakname, err
}

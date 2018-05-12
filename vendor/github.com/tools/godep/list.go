package main

import (
	"errors"
	"fmt"
	"go/build"
	"go/parser"
	"go/token"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"unicode"

	pathpkg "path"
)

var (
	gorootSrc            = filepath.Join(build.Default.GOROOT, "src")
	ignoreTags           = []string{"appengine", "ignore"} //TODO: appengine is a special case for now: https://github.com/tools/godep/issues/353
	versionMatch         = regexp.MustCompile(`\Ago\d+\.\d+\z`)
	versionNegativeMatch = regexp.MustCompile(`\A\!go\d+\.\d+\z`)
)

type errorMissingDep struct {
	i, dir string // import, dir
}

func (e errorMissingDep) Error() string {
	return "Unable to find dependent package " + e.i + " in context of " + e.dir
}

// packageContext is used to track an import and which package imported it.
type packageContext struct {
	pkg *build.Package // package that imports the import
	imp string         // import
}

// depScanner tracks the processed and to be processed packageContexts
type depScanner struct {
	processed []packageContext
	todo      []packageContext
}

// Next package and import to process
func (ds *depScanner) Next() (*build.Package, string) {
	c := ds.todo[0]
	ds.processed = append(ds.processed, c)
	ds.todo = ds.todo[1:]
	return c.pkg, c.imp
}

// Continue looping?
func (ds *depScanner) Continue() bool {
	return len(ds.todo) > 0
}

// Add a package and imports to the depScanner. Skips already processed/pending package/import combos
func (ds *depScanner) Add(pkg *build.Package, imports ...string) {
NextImport:
	for _, i := range imports {
		if i == "C" {
			i = "runtime/cgo"
		}
		for _, epc := range ds.processed {
			if pkg.Dir == epc.pkg.Dir && i == epc.imp {
				debugln("ctxts epc.pkg.Dir == pkg.Dir && i == epc.imp, skipping", epc.pkg.Dir, i)
				continue NextImport
			}
		}
		for _, epc := range ds.todo {
			if pkg.Dir == epc.pkg.Dir && i == epc.imp {
				debugln("ctxts epc.pkg.Dir == pkg.Dir && i == epc.imp, skipping", epc.pkg.Dir, i)
				continue NextImport
			}
		}
		pc := packageContext{pkg, i}
		debugln("Adding pc:", pc.pkg.Dir, pc.imp)
		ds.todo = append(ds.todo, pc)
	}
}

var (
	pkgCache = make(map[string]*build.Package) // dir => *build.Package
)

// returns the package in dir either from a cache or by importing it and then caching it
func fullPackageInDir(dir string) (*build.Package, error) {
	var err error
	pkg, ok := pkgCache[dir]
	if !ok {
		pkg, _ = build.ImportDir(dir, build.FindOnly)
		if pkg.Goroot {
			pkg, err = build.ImportDir(pkg.Dir, 0)
		} else {
			err = fillPackage(pkg)
		}
		if err == nil {
			pkgCache[dir] = pkg
		}
	}
	return pkg, err
}

// listPackage specified by path
func listPackage(path string) (*Package, error) {
	debugln("listPackage", path)
	var lp *build.Package
	dir, err := findDirForPath(path, nil)
	if err != nil {
		return nil, err
	}
	lp, err = fullPackageInDir(dir)
	p := &Package{
		Dir:            lp.Dir,
		Root:           lp.Root,
		ImportPath:     lp.ImportPath,
		XTestImports:   lp.XTestImports,
		TestImports:    lp.TestImports,
		GoFiles:        lp.GoFiles,
		CgoFiles:       lp.CgoFiles,
		TestGoFiles:    lp.TestGoFiles,
		XTestGoFiles:   lp.XTestGoFiles,
		IgnoredGoFiles: lp.IgnoredGoFiles,
	}
	p.Standard = lp.Goroot && lp.ImportPath != "" && !strings.Contains(lp.ImportPath, ".")
	if err != nil || p.Standard {
		return p, err
	}
	debugln("Looking For Package:", path, "in", dir)
	ppln(lp)

	ds := depScanner{}
	ds.Add(lp, lp.Imports...)
	for ds.Continue() {
		ip, i := ds.Next()

		debugf("Processing import %s for %s\n", i, ip.Dir)
		pdir, err := findDirForPath(i, ip)
		if err != nil {
			return nil, err
		}
		dp, err := fullPackageInDir(pdir)
		if err != nil { // This really should happen in this context though
			ppln(err)
			return nil, errorMissingDep{i: i, dir: ip.Dir}
		}
		ppln(dp)
		if !dp.Goroot {
			// Don't bother adding packages in GOROOT to the dependency scanner, they don't import things from outside of it.
			ds.Add(dp, dp.Imports...)
		}
		debugln("lp:")
		ppln(lp)
		debugln("ip:")
		ppln(ip)
		if lp == ip {
			debugln("lp == ip")
			p.Imports = append(p.Imports, dp.ImportPath)
		}
		p.Deps = append(p.Deps, dp.ImportPath)
		p.Dependencies = addDependency(p.Dependencies, dp)
	}
	p.Imports = uniq(p.Imports)
	p.Deps = uniq(p.Deps)
	debugln("Done Looking For Package:", path, "in", dir)
	ppln(p)
	return p, nil
}

func addDependency(deps []build.Package, d *build.Package) []build.Package {
	for i := range deps {
		if deps[i].Dir == d.Dir {
			return deps
		}
	}
	return append(deps, *d)
}

// finds the directory for the given import path in the context of the provided build.Package (if provided)
func findDirForPath(path string, ip *build.Package) (string, error) {
	debugln("findDirForPath", path, ip)
	var search []string

	if build.IsLocalImport(path) {
		dir := path
		if !filepath.IsAbs(dir) {
			if abs, err := filepath.Abs(dir); err == nil {
				// interpret relative to current directory
				dir = abs
			}
		}
		return dir, nil
	}

	// We need to check to see if the import exists in vendor/ folders up the hierarchy of the importing package
	if VendorExperiment && ip != nil {
		debugln("resolving vendor posibilities:", ip.Dir, ip.Root)
		cr := cleanPath(ip.Root)

		for base := cleanPath(ip.Dir); !pathEqual(base, cr); base = cleanPath(filepath.Dir(base)) {
			s := filepath.Join(base, "vendor", path)
			debugln("Adding search dir:", s)
			search = append(search, s)
		}
	}

	for _, base := range build.Default.SrcDirs() {
		search = append(search, filepath.Join(base, path))
	}

	for _, dir := range search {
		debugln("searching", dir)
		fi, err := stat(dir)
		if err == nil && fi.IsDir() {
			return dir, nil
		}
	}

	return "", errPackageNotFound{path}
}

type statEntry struct {
	fi  os.FileInfo
	err error
}

var (
	statCache = make(map[string]statEntry)
)

func clearStatCache() {
	statCache = make(map[string]statEntry)
}

func stat(p string) (os.FileInfo, error) {
	if e, ok := statCache[p]; ok {
		return e.fi, e.err
	}
	fi, err := os.Stat(p)
	statCache[p] = statEntry{fi, err}
	return fi, err
}

// fillPackage full of info. Assumes p.Dir is set at a minimum
func fillPackage(p *build.Package) error {
	if p.Goroot {
		return nil
	}

	if p.SrcRoot == "" {
		for _, base := range build.Default.SrcDirs() {
			if strings.HasPrefix(p.Dir, base) {
				p.SrcRoot = base
			}
		}
	}

	if p.SrcRoot == "" {
		return errors.New("Unable to find SrcRoot for package " + p.ImportPath)
	}

	if p.Root == "" {
		p.Root = filepath.Dir(p.SrcRoot)
	}

	var buildMatch = "+build "
	var buildFieldSplit = func(r rune) bool {
		return unicode.IsSpace(r) || r == ','
	}

	debugln("Filling package:", p.ImportPath, "from", p.Dir)
	gofiles, err := filepath.Glob(filepath.Join(p.Dir, "*.go"))
	if err != nil {
		debugln("Error globbing", err)
		return err
	}

	if len(gofiles) == 0 {
		return &build.NoGoError{Dir: p.Dir}
	}

	var testImports []string
	var imports []string
NextFile:
	for _, file := range gofiles {
		debugln(file)
		pf, err := parser.ParseFile(token.NewFileSet(), file, nil, parser.ImportsOnly|parser.ParseComments)
		if err != nil {
			return err
		}
		testFile := strings.HasSuffix(file, "_test.go")
		fname := filepath.Base(file)
		for _, c := range pf.Comments {
			ct := c.Text()
			if i := strings.Index(ct, buildMatch); i != -1 {
				for _, t := range strings.FieldsFunc(ct[i+len(buildMatch):], buildFieldSplit) {
					for _, tag := range ignoreTags {
						if t == tag {
							p.IgnoredGoFiles = append(p.IgnoredGoFiles, fname)
							continue NextFile
						}
					}

					if versionMatch.MatchString(t) && !isSameOrNewer(t, majorGoVersion) {
						debugln("Adding", fname, "to ignored list because of version tag", t)
						p.IgnoredGoFiles = append(p.IgnoredGoFiles, fname)
						continue NextFile
					}
					if versionNegativeMatch.MatchString(t) && isSameOrNewer(t[1:], majorGoVersion) {
						debugln("Adding", fname, "to ignored list because of version tag", t)
						p.IgnoredGoFiles = append(p.IgnoredGoFiles, fname)
						continue NextFile
					}
				}
			}
		}
		if testFile {
			p.TestGoFiles = append(p.TestGoFiles, fname)
		} else {
			p.GoFiles = append(p.GoFiles, fname)
		}
		for _, is := range pf.Imports {
			name, err := strconv.Unquote(is.Path.Value)
			if err != nil {
				return err // can't happen?
			}
			if testFile {
				testImports = append(testImports, name)
			} else {
				imports = append(imports, name)
			}
		}
	}
	imports = uniq(imports)
	testImports = uniq(testImports)
	p.Imports = imports
	p.TestImports = testImports
	return nil
}

// All of the following functions were vendored from go proper. Locations are noted in comments, but may change in future Go versions.

// importPaths returns the import paths to use for the given command line.
// $GOROOT/src/cmd/main.go:366
func importPaths(args []string) []string {
	debugf("importPathsNoDotExpansion(%q) == ", args)
	args = importPathsNoDotExpansion(args)
	debugf("%q\n", args)
	var out []string
	for _, a := range args {
		if strings.Contains(a, "...") {
			if build.IsLocalImport(a) {
				debugf("build.IsLocalImport(%q) == true\n", a)
				pkgs := allPackagesInFS(a)
				debugf("allPackagesInFS(%q) == %q\n", a, pkgs)
				out = append(out, pkgs...)
			} else {
				debugf("build.IsLocalImport(%q) == false\n", a)
				pkgs := allPackages(a)
				debugf("allPackages(%q) == %q\n", a, pkgs)
				out = append(out, allPackages(a)...)
			}
			continue
		}
		out = append(out, a)
	}
	return out
}

// importPathsNoDotExpansion returns the import paths to use for the given
// command line, but it does no ... expansion.
// $GOROOT/src/cmd/main.go:332
func importPathsNoDotExpansion(args []string) []string {
	if len(args) == 0 {
		return []string{"."}
	}
	var out []string
	for _, a := range args {
		// Arguments are supposed to be import paths, but
		// as a courtesy to Windows developers, rewrite \ to /
		// in command-line arguments.  Handles .\... and so on.
		if filepath.Separator == '\\' {
			a = strings.Replace(a, `\`, `/`, -1)
		}

		// Put argument in canonical form, but preserve leading ./.
		if strings.HasPrefix(a, "./") {
			a = "./" + pathpkg.Clean(a)
			if a == "./." {
				a = "."
			}
		} else {
			a = pathpkg.Clean(a)
		}
		if a == "all" || a == "std" || a == "cmd" {
			out = append(out, allPackages(a)...)
			continue
		}
		out = append(out, a)
	}
	return out
}

// allPackagesInFS is like allPackages but is passed a pattern
// beginning ./ or ../, meaning it should scan the tree rooted
// at the given directory.  There are ... in the pattern too.
// $GOROOT/src/cmd/main.go:620
func allPackagesInFS(pattern string) []string {
	pkgs := matchPackagesInFS(pattern)
	if len(pkgs) == 0 {
		fmt.Fprintf(os.Stderr, "warning: %q matched no packages\n", pattern)
	}
	return pkgs
}

// allPackages returns all the packages that can be found
// under the $GOPATH directories and $GOROOT matching pattern.
// The pattern is either "all" (all packages), "std" (standard packages),
// "cmd" (standard commands), or a path including "...".
// $GOROOT/src/cmd/main.go:542
func allPackages(pattern string) []string {
	pkgs := matchPackages(pattern)
	if len(pkgs) == 0 {
		fmt.Fprintf(os.Stderr, "warning: %q matched no packages\n", pattern)
	}
	return pkgs
}

// $GOROOT/src/cmd/main.go:554
// This has been changed to not use build.ImportDir
func matchPackages(pattern string) []string {
	match := func(string) bool { return true }
	treeCanMatch := func(string) bool { return true }
	if pattern != "all" && pattern != "std" && pattern != "cmd" {
		match = matchPattern(pattern)
		treeCanMatch = treeCanMatchPattern(pattern)
	}

	have := map[string]bool{
		"builtin": true, // ignore pseudo-package that exists only for documentation
	}
	if !build.Default.CgoEnabled {
		have["runtime/cgo"] = true // ignore during walk
	}
	var pkgs []string

	for _, src := range build.Default.SrcDirs() {
		if (pattern == "std" || pattern == "cmd") && src != gorootSrc {
			continue
		}
		src = filepath.Clean(src) + string(filepath.Separator)
		root := src
		if pattern == "cmd" {
			root += "cmd" + string(filepath.Separator)
		}
		filepath.Walk(root, func(path string, fi os.FileInfo, err error) error {
			if err != nil || !fi.IsDir() || path == src {
				return nil
			}

			// Avoid .foo, _foo, and testdata directory trees.
			_, elem := filepath.Split(path)
			if strings.HasPrefix(elem, ".") || strings.HasPrefix(elem, "_") || elem == "testdata" {
				return filepath.SkipDir
			}

			name := filepath.ToSlash(path[len(src):])
			if pattern == "std" && (strings.Contains(name, ".") || name == "cmd") {
				// The name "std" is only the standard library.
				// If the name has a dot, assume it's a domain name for go get,
				// and if the name is cmd, it's the root of the command tree.
				return filepath.SkipDir
			}
			if !treeCanMatch(name) {
				return filepath.SkipDir
			}
			if have[name] {
				return nil
			}
			have[name] = true
			if !match(name) {
				return nil
			}

			ap, err := filepath.Abs(path)
			if err != nil {
				return nil
			}
			if _, err = fullPackageInDir(ap); err != nil {
				debugf("matchPackage(%q) ap=%q Error: %q\n", ap, pattern, err)
				if _, noGo := err.(*build.NoGoError); noGo {
					return nil
				}
			}
			pkgs = append(pkgs, name)
			return nil
		})
	}
	return pkgs
}

// treeCanMatchPattern(pattern)(name) reports whether
// name or children of name can possibly match pattern.
// Pattern is the same limited glob accepted by matchPattern.
// $GOROOT/src/cmd/main.go:527
func treeCanMatchPattern(pattern string) func(name string) bool {
	wildCard := false
	if i := strings.Index(pattern, "..."); i >= 0 {
		wildCard = true
		pattern = pattern[:i]
	}
	return func(name string) bool {
		return len(name) <= len(pattern) && hasPathPrefix(pattern, name) ||
			wildCard && strings.HasPrefix(name, pattern)
	}
}

// hasPathPrefix reports whether the path s begins with the
// elements in prefix.
// $GOROOT/src/cmd/main.go:489
func hasPathPrefix(s, prefix string) bool {
	switch {
	default:
		return false
	case len(s) == len(prefix):
		return s == prefix
	case len(s) > len(prefix):
		if prefix != "" && prefix[len(prefix)-1] == '/' {
			return strings.HasPrefix(s, prefix)
		}
		return s[len(prefix)] == '/' && s[:len(prefix)] == prefix
	}
}

// $GOROOT/src/cmd/go/main.go:631
// This has been changed to not use build.ImportDir
func matchPackagesInFS(pattern string) []string {
	// Find directory to begin the scan.
	// Could be smarter but this one optimization
	// is enough for now, since ... is usually at the
	// end of a path.
	i := strings.Index(pattern, "...")
	dir, _ := pathpkg.Split(pattern[:i])

	// pattern begins with ./ or ../.
	// path.Clean will discard the ./ but not the ../.
	// We need to preserve the ./ for pattern matching
	// and in the returned import paths.
	prefix := ""
	if strings.HasPrefix(pattern, "./") {
		prefix = "./"
	}
	match := matchPattern(pattern)

	var pkgs []string
	filepath.Walk(dir, func(path string, fi os.FileInfo, err error) error {
		if err != nil || !fi.IsDir() {
			return nil
		}
		if path == dir {
			// filepath.Walk starts at dir and recurses. For the recursive case,
			// the path is the result of filepath.Join, which calls filepath.Clean.
			// The initial case is not Cleaned, though, so we do this explicitly.
			//
			// This converts a path like "./io/" to "io". Without this step, running
			// "cd $GOROOT/src; go list ./io/..." would incorrectly skip the io
			// package, because prepending the prefix "./" to the unclean path would
			// result in "././io", and match("././io") returns false.
			path = filepath.Clean(path)
		}

		// Avoid .foo, _foo, and testdata directory trees, but do not avoid "." or "..".
		_, elem := filepath.Split(path)
		dot := strings.HasPrefix(elem, ".") && elem != "." && elem != ".."
		if dot || strings.HasPrefix(elem, "_") || elem == "testdata" {
			return filepath.SkipDir
		}

		name := prefix + filepath.ToSlash(path)
		if !match(name) {
			return nil
		}
		ap, err := filepath.Abs(path)
		if err != nil {
			return nil
		}
		if _, err = fullPackageInDir(ap); err != nil {
			debugf("matchPackageInFS(%q) ap=%q Error: %q\n", ap, pattern, err)
			if _, noGo := err.(*build.NoGoError); !noGo {
				log.Print(err)
			}
			return nil
		}
		pkgs = append(pkgs, name)
		return nil
	})
	return pkgs
}

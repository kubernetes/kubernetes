package main

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"go/build"
	"io"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/kr/fs"
)

var cmdSave = &Command{
	Name:  "save",
	Args:  "[-r] [-t] [packages]",
	Short: "list and copy dependencies into Godeps",
	Long: `

Save writes a list of the named packages and their dependencies along
with the exact source control revision of each package, and copies
their source code into a subdirectory. Packages inside "." are excluded
from the list to be copied.

The list is written to Godeps/Godeps.json, and source code for all
dependencies is copied into either Godeps/_workspace or, if the vendor
experiment is turned on, vendor/.

The dependency list is a JSON document with the following structure:

	type Godeps struct {
		ImportPath string
		GoVersion  string   // Abridged output of 'go version'.
		Packages   []string // Arguments to godep save, if any.
		Deps       []struct {
			ImportPath string
			Comment    string // Tag or description of commit.
			Rev        string // VCS-specific commit ID.
		}
	}

Any packages already present in the list will be left unchanged.
To update a dependency to a newer revision, use 'godep update'.

If -r is given, import statements will be rewritten to refer directly
to the copied source code. This is not compatible with the vendor
experiment. Note that this will not rewrite the statements in the
files outside the project.

If -t is given, test files (*_test.go files + testdata directories) are
also saved.

For more about specifying packages, see 'go help packages'.
`,
	Run:          runSave,
	OnlyInGOPATH: true,
}

var (
	saveR, saveT bool
)

func init() {
	cmdSave.Flag.BoolVar(&saveR, "r", false, "rewrite import paths")
	cmdSave.Flag.BoolVar(&saveT, "t", false, "save test files")

}

func runSave(cmd *Command, args []string) {
	if VendorExperiment && saveR {
		log.Println("flag -r is incompatible with the vendoring experiment")
		cmd.UsageExit()
	}
	err := save(args)
	if err != nil {
		log.Fatalln(err)
	}
}

func dotPackage() (*build.Package, error) {
	dir, err := filepath.Abs(".")
	if err != nil {
		return nil, err
	}
	return build.ImportDir(dir, build.FindOnly)
}

func projectPackages(dDir string, a []*Package) []*Package {
	var projPkgs []*Package
	dotDir := fmt.Sprintf("%s%c", dDir, filepath.Separator)
	for _, p := range a {
		pkgDir := fmt.Sprintf("%s%c", p.Dir, filepath.Separator)
		if strings.HasPrefix(pkgDir, dotDir) {
			projPkgs = append(projPkgs, p)
		}
	}
	return projPkgs
}

func save(pkgs []string) error {
	var err error
	dp, err := dotPackage()
	if err != nil {
		return err
	}
	debugln("dotPackageImportPath:", dp.ImportPath)
	debugln("dotPackageDir:", dp.Dir)

	cv, err := goVersion()
	if err != nil {
		return err
	}
	verboseln("Go Version:", cv)

	gold, err := loadDefaultGodepsFile()
	if err != nil {
		if !os.IsNotExist(err) {
			return err
		}
		verboseln("No old Godeps.json found.")
		gold.GoVersion = cv
	}

	printVersionWarnings(gold.GoVersion)
	if len(gold.GoVersion) == 0 {
		gold.GoVersion = majorGoVersion
	} else {
		majorGoVersion, err = trimGoVersion(gold.GoVersion)
		if err != nil {
			log.Fatalf("Unable to determine go major version from value specified in %s: %s\n", gold.file(), gold.GoVersion)
		}
	}

	gnew := &Godeps{
		ImportPath: dp.ImportPath,
		GoVersion:  gold.GoVersion,
	}

	switch len(pkgs) {
	case 0:
		pkgs = []string{"."}
	default:
		gnew.Packages = pkgs
	}

	verboseln("Finding dependencies for", pkgs)
	a, err := LoadPackages(pkgs...)
	if err != nil {
		return err
	}

	for _, p := range a {
		verboseln("Found package:", p.ImportPath)
		verboseln("\tDeps:", strings.Join(p.Deps, " "))
	}
	ppln(a)

	projA := projectPackages(dp.Dir, a)
	debugln("Filtered projectPackages")
	ppln(projA)

	verboseln("Computing new Godeps.json file")
	err = gnew.fill(a, dp.ImportPath)
	if err != nil {
		return err
	}
	debugln("New Godeps Filled")
	ppln(gnew)

	if gnew.Deps == nil {
		gnew.Deps = make([]Dependency, 0) // produce json [], not null
	}
	gdisk := gnew.copy()
	err = carryVersions(&gold, gnew)
	if err != nil {
		return err
	}

	if gold.isOldFile {
		// If we are migrating from an old format file,
		// we require that the listed version of every
		// dependency must be installed in GOPATH, so it's
		// available to copy.
		if !eqDeps(gnew.Deps, gdisk.Deps) {
			return errors.New(strings.TrimSpace(needRestore))
		}
		gold = Godeps{}
	}
	os.Remove("Godeps") // remove regular file if present; ignore error
	readme := filepath.Join("Godeps", "Readme")
	err = writeFile(readme, strings.TrimSpace(Readme)+"\n")
	if err != nil {
		log.Println(err)
	}
	_, err = gnew.save()
	if err != nil {
		return err
	}

	verboseln("Computing diff between old and new deps")
	// We use a name starting with "_" so the go tool
	// ignores this directory when traversing packages
	// starting at the project's root. For example,
	//   godep go list ./...
	srcdir := filepath.FromSlash(strings.Trim(sep, "/"))
	rem := subDeps(gold.Deps, gnew.Deps)
	ppln(rem)
	add := subDeps(gnew.Deps, gold.Deps)
	ppln(add)
	if len(rem) > 0 {
		verboseln("Deps to remove:")
		for _, r := range rem {
			verboseln("\t", r.ImportPath)
		}
		verboseln("Removing unused dependencies")
		err = removeSrc(srcdir, rem)
		if err != nil {
			return err
		}
	}
	if len(add) > 0 {
		verboseln("Deps to add:")
		for _, a := range add {
			verboseln("\t", a.ImportPath)
		}
		verboseln("Adding new dependencies")
		err = copySrc(srcdir, add)
		if err != nil {
			return err
		}
	}
	if !VendorExperiment {
		f, _ := filepath.Split(srcdir)
		writeVCSIgnore(f)
	}
	var rewritePaths []string
	if saveR {
		for _, dep := range gnew.Deps {
			rewritePaths = append(rewritePaths, dep.ImportPath)
		}
	}
	verboseln("Rewriting paths (if necessary)")
	ppln(rewritePaths)
	return rewrite(projA, dp.ImportPath, rewritePaths)
}

func printVersionWarnings(ov string) {
	var warning bool
	cv, err := goVersion()
	if err != nil {
		return
	}
	// Trim the old version because we may have saved it w/o trimming it
	// cv is already trimmed by goVersion()
	tov, err := trimGoVersion(ov)
	if err != nil {
		return
	}

	if tov != ov {
		log.Printf("WARNING: Recorded go version (%s) with minor version string found.\n", ov)
		warning = true
	}
	if cv != tov {
		log.Printf("WARNING: Recorded major go version (%s) and in-use major go version (%s) differ.\n", tov, cv)
		warning = true
	}
	if warning {
		log.Println("To record current major go version run `godep update -goversion`.")
	}
}

type revError struct {
	ImportPath string
	WantRev    string
	HavePath   string
	HaveRev    string
}

func (v *revError) Error() string {
	return fmt.Sprintf("cannot save %s at revision %s: already have %s at revision %s.\n"+
		"Run `godep update %s' first.", v.ImportPath, v.WantRev, v.HavePath, v.HaveRev, v.HavePath)
}

// carryVersions copies Rev and Comment from a to b for
// each dependency with an identical ImportPath. For any
// dependency in b that appears to be from the same repo
// as one in a (for example, a parent or child directory),
// the Rev must already match - otherwise it is an error.
func carryVersions(a, b *Godeps) error {
	for i := range b.Deps {
		err := carryVersion(a, &b.Deps[i])
		if err != nil {
			return err
		}
	}
	return nil
}

func carryVersion(a *Godeps, db *Dependency) error {
	// First see if this exact package is already in the list.
	for _, da := range a.Deps {
		if db.ImportPath == da.ImportPath {
			db.Rev = da.Rev
			db.Comment = da.Comment
			return nil
		}
	}
	// No exact match, check for child or sibling package.
	// We can't handle mismatched versions for packages in
	// the same repo, so report that as an error.
	for _, da := range a.Deps {
		if strings.HasPrefix(db.ImportPath, da.ImportPath+"/") ||
			strings.HasPrefix(da.ImportPath, db.root+"/") {
			if da.Rev != db.Rev {
				return &revError{
					ImportPath: db.ImportPath,
					WantRev:    db.Rev,
					HavePath:   da.ImportPath,
					HaveRev:    da.Rev,
				}
			}
		}
	}
	// No related package in the list, must be a new repo.
	return nil
}

// subDeps returns a - b, using ImportPath for equality.
func subDeps(a, b []Dependency) (diff []Dependency) {
Diff:
	for _, da := range a {
		for _, db := range b {
			if da.ImportPath == db.ImportPath {
				continue Diff
			}
		}
		diff = append(diff, da)
	}
	return diff
}

func removeSrc(srcdir string, deps []Dependency) error {
	for _, dep := range deps {
		path := filepath.FromSlash(dep.ImportPath)
		err := os.RemoveAll(filepath.Join(srcdir, path))
		if err != nil {
			return err
		}
	}
	return nil
}

func copySrc(dir string, deps []Dependency) error {
	// mapping to see if we visited a parent directory already
	visited := make(map[string]bool)
	ok := true
	for _, dep := range deps {
		debugln("copySrc for", dep.ImportPath)
		srcdir := filepath.Join(dep.ws, "src")
		rel, err := filepath.Rel(srcdir, dep.dir)
		debugln("srcdir", srcdir)
		debugln("rel", rel)
		debugln("err", err)
		if err != nil { // this should never happen
			return err
		}
		dstpkgroot := filepath.Join(dir, rel)
		err = os.RemoveAll(dstpkgroot)
		if err != nil {
			log.Println(err)
			ok = false
		}

		// copy actual dependency
		vf := dep.vcs.listFiles(dep.dir)
		debugln("vf", vf)
		w := fs.Walk(dep.dir)
		for w.Step() {
			err = copyPkgFile(vf, dir, srcdir, w)
			if err != nil {
				log.Println(err)
				ok = false
			}
		}

		// Look for legal files in root
		//  some packages are imports as a sub-package but license info
		//  is at root:  exampleorg/common has license file in exampleorg
		//
		if dep.ImportPath == dep.root {
			// we are already at root
			continue
		}

		// prevent copying twice This could happen if we have
		//   two subpackages listed someorg/common and
		//   someorg/anotherpack which has their license in
		//   the parent dir of someorg
		rootdir := filepath.Join(srcdir, filepath.FromSlash(dep.root))
		if visited[rootdir] {
			continue
		}
		visited[rootdir] = true
		vf = dep.vcs.listFiles(rootdir)
		w = fs.Walk(rootdir)
		for w.Step() {
			fname := filepath.Base(w.Path())
			if IsLegalFile(fname) && !strings.Contains(w.Path(), sep) {
				err = copyPkgFile(vf, dir, srcdir, w)
				if err != nil {
					log.Println(err)
					ok = false
				}
			}
		}
	}

	if !ok {
		return errorCopyingSourceCode
	}

	return nil
}

func copyPkgFile(vf vcsFiles, dstroot, srcroot string, w *fs.Walker) error {
	if w.Err() != nil {
		return w.Err()
	}
	name := w.Stat().Name()
	if w.Stat().IsDir() {
		if name[0] == '.' || name[0] == '_' || (!saveT && name == "testdata") {
			// Skip directories starting with '.' or '_' or
			// 'testdata' (last is only skipped if saveT is false)
			w.SkipDir()
		}
		return nil
	}
	rel, err := filepath.Rel(srcroot, w.Path())
	if err != nil { // this should never happen
		return err
	}
	if !saveT && strings.HasSuffix(name, "_test.go") {
		if verbose {
			log.Printf("save: skipping test file: %s", w.Path())
		}
		return nil
	}
	if !vf.Contains(w.Path()) {
		if verbose {
			log.Printf("save: skipping untracked file: %s", w.Path())
		}
		return nil
	}
	return copyFile(filepath.Join(dstroot, rel), w.Path())
}

// copyFile copies a regular file from src to dst.
// dst is opened with os.Create.
// If the file name ends with .go,
// copyFile strips canonical import path annotations.
// These are comments of the form:
//   package foo // import "bar/foo"
//   package foo /* import "bar/foo" */
func copyFile(dst, src string) error {
	err := os.MkdirAll(filepath.Dir(dst), 0777)
	if err != nil {
		return err
	}

	linkDst, err := os.Readlink(src)
	if err == nil {
		return os.Symlink(linkDst, dst)
	}

	si, err := stat(src)
	if err != nil {
		return err
	}

	r, err := os.Open(src)
	if err != nil {
		return err
	}
	defer r.Close()

	w, err := os.Create(dst)
	if err != nil {
		return err
	}
	if err := os.Chmod(dst, si.Mode()); err != nil {
		return err
	}

	if strings.HasSuffix(dst, ".go") {
		debugln("Copy Without Import Comment", w, r)
		err = copyWithoutImportComment(w, r)
	} else {
		debugln("Copy (plain)", w, r)
		_, err = io.Copy(w, r)
	}
	err1 := w.Close()
	if err == nil {
		err = err1
	}

	return err
}

func copyWithoutImportComment(w io.Writer, r io.Reader) error {
	b := bufio.NewReader(r)
	for {
		l, err := b.ReadBytes('\n')
		eof := err == io.EOF
		if err != nil && err != io.EOF {
			return err
		}

		// If we have data then write it out...
		if len(l) > 0 {
			// Strip off \n if it exists because stripImportComment
			_, err := w.Write(append(stripImportComment(bytes.TrimRight(l, "\n")), '\n'))
			if err != nil {
				return err
			}
		}

		if eof {
			return nil
		}
	}
}

const (
	importAnnotation = `import\s+(?:"[^"]*"|` + "`[^`]*`" + `)`
	importComment    = `(?://\s*` + importAnnotation + `\s*$|/\*\s*` + importAnnotation + `\s*\*/)`
)

var (
	importCommentRE = regexp.MustCompile(`^\s*(package\s+\w+)\s+` + importComment + `(.*)`)
	pkgPrefix       = []byte("package ")
)

// stripImportComment returns line with its import comment removed.
// If s is not a package statement containing an import comment,
// it is returned unaltered.
// FIXME: expects lines w/o a \n at the end
// See also http://golang.org/s/go14customimport.
func stripImportComment(line []byte) []byte {
	if !bytes.HasPrefix(line, pkgPrefix) {
		// Fast path; this will skip all but one line in the file.
		// This assumes there is no whitespace before the keyword.
		return line
	}
	if m := importCommentRE.FindSubmatch(line); m != nil {
		return append(m[1], m[2]...)
	}
	return line
}

// Func writeVCSIgnore writes "ignore" files inside dir for known VCSs,
// so that dir/pkg and dir/bin don't accidentally get committed.
// It logs any errors it encounters.
func writeVCSIgnore(dir string) {
	// Currently git is the only VCS for which we know how to do this.
	// Mercurial and Bazaar have similar mechanisms, but they apparently
	// require writing files outside of dir.
	const ignore = "/pkg\n/bin\n"
	name := filepath.Join(dir, ".gitignore")
	err := writeFile(name, ignore)
	if err != nil {
		log.Println(err)
	}
}

// writeFile is like ioutil.WriteFile but it creates
// intermediate directories with os.MkdirAll.
func writeFile(name, body string) error {
	err := os.MkdirAll(filepath.Dir(name), 0777)
	if err != nil {
		return err
	}
	return ioutil.WriteFile(name, []byte(body), 0666)
}

const (
	// Readme contains the README text.
	Readme = `
This directory tree is generated automatically by godep.

Please do not edit.

See https://github.com/tools/godep for more information.
`
	needRestore = `
mismatched versions while migrating

It looks like you are switching from the old Godeps format
(from flag -copy=false). The old format is just a file; it
doesn't contain source code. For this migration, godep needs
the appropriate version of each dependency to be installed in
GOPATH, so that the source code is available to copy.

To fix this, run 'godep restore'.
`
)

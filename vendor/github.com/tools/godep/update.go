package main

import (
	"go/parser"
	"go/token"
	"log"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
)

var cmdUpdate = &Command{
	Name:  "update",
	Args:  "[-goversion] [packages]",
	Short: "update selected packages or the go version",
	Long: `
Update changes the named dependency packages to use the
revision of each currently installed in GOPATH. New code will
be copied into the Godeps workspace or vendor folder and the
new revision will be written to the manifest.

If -goversion is specified, update the recorded go version.

For more about specifying packages, see 'go help packages'.
`,
	Run:          runUpdate,
	OnlyInGOPATH: true,
}

var (
	updateGoVer bool
)

func init() {
	cmdUpdate.Flag.BoolVar(&saveT, "t", false, "save test files during update")
	cmdUpdate.Flag.BoolVar(&updateGoVer, "goversion", false, "update the recorded go version")
}

func runUpdate(cmd *Command, args []string) {
	if updateGoVer {
		err := updateGoVersion()
		if err != nil {
			log.Fatalln(err)
		}
	}
	if len(args) > 0 {
		err := update(args)
		if err != nil {
			log.Fatalln(err)
		}
	}
}

func updateGoVersion() error {
	gold, err := loadDefaultGodepsFile()
	if err != nil {
		if !os.IsNotExist(err) {
			return err
		}
	}
	cv, err := goVersion()
	if err != nil {
		return err
	}

	gv := gold.GoVersion
	gold.GoVersion = cv
	_, err = gold.save()
	if err != nil {
		return err
	}

	if gv != cv {
		log.Println("Updated major go version to", cv)
	}
	return nil

}

func update(args []string) error {
	if len(args) == 0 {
		args = []string{"."}
	}
	g, err := loadDefaultGodepsFile()
	if err != nil {
		return err
	}
	for _, arg := range args {
		arg := path.Clean(arg)
		any := markMatches(arg, g.Deps)
		if !any {
			log.Println("not in manifest:", arg)
		}
	}
	deps, rdeps, err := LoadVCSAndUpdate(g.Deps)
	if err != nil {
		return err
	}
	if len(deps) == 0 {
		return errorNoPackagesUpdatable
	}
	g.addOrUpdateDeps(deps)
	g.removeDeps(rdeps)
	if _, err = g.save(); err != nil {
		return err
	}

	srcdir := relativeVendorTarget(VendorExperiment)
	if err := removeSrc(filepath.FromSlash(strings.Trim(sep, "/")), rdeps); err != nil {
		return err
	}
	copySrc(srcdir, deps)

	ok, err := needRewrite(g.Packages)
	if err != nil {
		return err
	}
	var rewritePaths []string
	if ok {
		for _, dep := range g.Deps {
			rewritePaths = append(rewritePaths, dep.ImportPath)
		}
	}
	return rewrite(nil, g.ImportPath, rewritePaths)
}

func needRewrite(importPaths []string) (bool, error) {
	if len(importPaths) == 0 {
		importPaths = []string{"."}
	}
	a, err := LoadPackages(importPaths...)
	if err != nil {
		return false, err
	}
	for _, p := range a {
		for _, name := range p.allGoFiles() {
			path := filepath.Join(p.Dir, name)
			hasSep, err := hasRewrittenImportStatement(path)
			if err != nil {
				return false, err
			}
			if hasSep {
				return true, nil
			}
		}
	}
	return false, nil
}

func hasRewrittenImportStatement(path string) (bool, error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, path, nil, 0)
	if err != nil {
		return false, err
	}
	for _, s := range f.Imports {
		name, _ := strconv.Unquote(s.Path.Value)
		if strings.Contains(name, sep) {
			return true, nil
		}
	}
	return false, nil
}

// markMatches marks each entry in deps with an import path that
// matches pat. It returns whether any matches occurred.
func markMatches(pat string, deps []Dependency) (matched bool) {
	f := matchPattern(pat)
	for i, dep := range deps {
		if f(dep.ImportPath) {
			deps[i].matched = true
			matched = true
		}
	}
	return matched
}

func fillDeps(deps []Dependency) ([]Dependency, error) {
	for i := range deps {
		if deps[i].pkg != nil {
			continue
		}
		ps, err := LoadPackages(deps[i].ImportPath)
		if err != nil {
			if _, ok := err.(errPackageNotFound); ok {
				deps[i].missing = true
				continue
			}
			return nil, err
		}
		if len(ps) > 1 {
			panic("More than one package found for " + deps[i].ImportPath)
		}
		p := ps[0]
		deps[i].pkg = p
		deps[i].dir = p.Dir
		deps[i].ws = p.Root

		vcs, reporoot, err := VCSFromDir(p.Dir, filepath.Join(p.Root, "src"))
		if err != nil {
			return nil, errorLoadingDeps
		}
		deps[i].root = filepath.ToSlash(reporoot)
		deps[i].vcs = vcs
	}

	return deps, nil
}

// LoadVCSAndUpdate loads and updates a set of dependencies.
func LoadVCSAndUpdate(deps []Dependency) ([]Dependency, []Dependency, error) {
	var err1 error

	deps, err := fillDeps(deps)
	if err != nil {
		return nil, nil, err
	}

	repoMask := make(map[string]bool)
	for i := range deps {
		if !deps[i].matched {
			repoMask[deps[i].root] = true
		}
	}

	// Determine if we need any new packages because of new transitive imports
	for _, dep := range deps {
		if !dep.matched || dep.missing {
			continue
		}
		for _, dp := range dep.pkg.Dependencies {
			if dp.Goroot {
				continue
			}
			var have bool
			for _, d := range deps {
				if d.ImportPath == dp.ImportPath {
					have = true
					break
				}
			}
			if !have {
				deps = append(deps, Dependency{ImportPath: dp.ImportPath, matched: true})
			}
		}
	}

	deps, err = fillDeps(deps)
	if err != nil {
		return nil, nil, err
	}

	var toUpdate, toRemove []Dependency
	for _, d := range deps {
		if !d.matched || repoMask[d.root] {
			continue
		}
		if d.missing {
			toRemove = append(toRemove, d)
			continue
		}
		toUpdate = append(toUpdate, d)
	}

	debugln("toUpdate")
	ppln(toUpdate)

	var toCopy []Dependency
	for _, d := range toUpdate {
		id, err := d.vcs.identify(d.dir)
		if err != nil {
			log.Println(err)
			err1 = errorLoadingDeps
			continue
		}
		if d.vcs.isDirty(d.dir, id) {
			log.Println("dirty working tree (please commit changes):", d.dir)
		}
		d.Rev = id
		d.Comment = d.vcs.describe(d.dir, id)
		toCopy = append(toCopy, d)
	}
	debugln("toCopy")
	ppln(toCopy)

	if err1 != nil {
		return nil, nil, err1
	}
	return toCopy, toRemove, nil
}

package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
)

var (
	godepsFile    = filepath.Join("Godeps", "Godeps.json")
	oldGodepsFile = filepath.Join("Godeps")
)

// Godeps describes what a package needs to be rebuilt reproducibly.
// It's the same information stored in file Godeps.
type Godeps struct {
	ImportPath   string
	GoVersion    string
	GodepVersion string
	Packages     []string `json:",omitempty"` // Arguments to save, if any.
	Deps         []Dependency
	isOldFile    bool
}

func createGodepsFile() (*os.File, error) {
	return os.Create(godepsFile)
}

func loadGodepsFile(path string) (Godeps, error) {
	var g Godeps
	f, err := os.Open(path)
	if err != nil {
		return g, err
	}
	defer f.Close()
	err = json.NewDecoder(f).Decode(&g)
	if err != nil {
		err = fmt.Errorf("Unable to parse %s: %s", path, err.Error())
	}
	return g, err
}

func loadDefaultGodepsFile() (Godeps, error) {
	var g Godeps
	var err error
	g, err = loadGodepsFile(godepsFile)
	if err != nil {
		if os.IsNotExist(err) {
			var err1 error
			g, err1 = loadGodepsFile(oldGodepsFile)
			if err1 != nil {
				if os.IsNotExist(err1) {
					return g, err
				}
				return g, err1
			}
			g.isOldFile = true
			return g, nil
		}
	}
	return g, err
}

// pkgs is the list of packages to read dependencies for
func (g *Godeps) fill(pkgs []*Package, destImportPath string) error {
	debugln("fill", destImportPath)
	ppln(pkgs)
	var err1 error
	var path, testImports []string
	dipp := []string{destImportPath}
	for _, p := range pkgs {
		if p.Standard {
			log.Println("ignoring stdlib package:", p.ImportPath)
			continue
		}
		if p.Error.Err != "" {
			log.Println(p.Error.Err)
			err1 = errorLoadingPackages
			continue
		}
		path = append(path, p.ImportPath)
		path = append(path, p.Deps...)
		testImports = append(testImports, p.TestImports...)
		testImports = append(testImports, p.XTestImports...)
	}
	ps, err := LoadPackages(testImports...)
	if err != nil {
		return err
	}
	for _, p := range ps {
		if p.Standard {
			continue
		}
		if p.Error.Err != "" {
			log.Println(p.Error.Err)
			err1 = errorLoadingPackages
			continue
		}
		path = append(path, p.ImportPath)
		path = append(path, p.Deps...)
	}
	debugln("path", path)
	for i, p := range path {
		path[i] = unqualify(p)
	}
	path = uniq(path)
	debugln("uniq, unqualify'd path", path)
	ps, err = LoadPackages(path...)
	if err != nil {
		return err
	}
	for _, pkg := range ps {
		if pkg.Error.Err != "" {
			log.Println(pkg.Error.Err)
			err1 = errorLoadingDeps
			continue
		}
		if pkg.Standard || containsPathPrefix(dipp, pkg.ImportPath) {
			debugln("standard or dest skipping", pkg.ImportPath)
			continue
		}
		vcs, reporoot, err := VCSFromDir(pkg.Dir, filepath.Join(pkg.Root, "src"))
		if err != nil {
			log.Println(err)
			err1 = errorLoadingDeps
			continue
		}
		id, err := vcs.identify(pkg.Dir)
		if err != nil {
			log.Println(err)
			err1 = errorLoadingDeps
			continue
		}
		if vcs.isDirty(pkg.Dir, id) {
			log.Println("dirty working tree (please commit changes):", pkg.Dir)
			err1 = errorLoadingDeps
			continue
		}
		comment := vcs.describe(pkg.Dir, id)
		g.Deps = append(g.Deps, Dependency{
			ImportPath: pkg.ImportPath,
			Rev:        id,
			Comment:    comment,
			dir:        pkg.Dir,
			ws:         pkg.Root,
			root:       filepath.ToSlash(reporoot),
			vcs:        vcs,
		})
	}
	return err1
}

func (g *Godeps) copy() *Godeps {
	h := *g
	h.Deps = make([]Dependency, len(g.Deps))
	copy(h.Deps, g.Deps)
	return &h
}

func (g *Godeps) file() string {
	if g.isOldFile {
		return oldGodepsFile
	}
	return godepsFile
}

func (g *Godeps) save() (int64, error) {
	f, err := os.Create(g.file())
	if err != nil {
		return 0, err
	}
	defer f.Close()
	return g.writeTo(f)
}

func (g *Godeps) writeTo(w io.Writer) (int64, error) {
	g.GodepVersion = fmt.Sprintf("v%d", version) // godep always writes its current version.
	b, err := json.MarshalIndent(g, "", "\t")
	if err != nil {
		return 0, err
	}
	n, err := w.Write(append(b, '\n'))
	return int64(n), err
}

func (g *Godeps) addOrUpdateDeps(deps []Dependency) {
	var missing []Dependency
	for _, d := range deps {
		var found bool
		for i := range g.Deps {
			if g.Deps[i].ImportPath == d.ImportPath {
				g.Deps[i] = d
				found = true
				break
			}
		}
		if !found {
			missing = append(missing, d)
		}
	}
	for _, d := range missing {
		g.Deps = append(g.Deps, d)
	}
}

func (g *Godeps) removeDeps(deps []Dependency) {
	var f []Dependency
	for i := range g.Deps {
		var found bool
		for _, d := range deps {
			if g.Deps[i].ImportPath == d.ImportPath {
				found = true
				break
			}
		}
		if !found {
			f = append(f, g.Deps[i])
		}
	}
	g.Deps = f
}

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dep

import (
	"log"
	"os"
	"path/filepath"
	"runtime"

	"github.com/golang/dep/gps"
	"github.com/golang/dep/internal/fs"
	"github.com/pkg/errors"
)

// Ctx defines the supporting context of dep.
//
// A properly initialized Ctx has a GOPATH containing the project root and non-nil Loggers.
//
//	ctx := &dep.Ctx{
//		WorkingDir: GOPATH + "/src/project/root",
//		GOPATH: GOPATH,
//		Out: log.New(os.Stdout, "", 0),
//		Err: log.New(os.Stderr, "", 0),
//	}
//
// Ctx.DetectProjectGOPATH() helps with setting the containing GOPATH.
//
//	ctx.GOPATH, err := Ctx.DetectProjectGOPATH(project)
//	if err != nil {
//		// Could not determine which GOPATH to use for the project.
//	}
//
type Ctx struct {
	WorkingDir     string      // Where to execute.
	GOPATH         string      // Selected Go path, containing WorkingDir.
	GOPATHs        []string    // Other Go paths.
	Out, Err       *log.Logger // Required loggers.
	Verbose        bool        // Enables more verbose logging.
	DisableLocking bool        // When set, no lock file will be created to protect against simultaneous dep processes.
	Cachedir       string      // Cache directory loaded from environment.
}

// SetPaths sets the WorkingDir and GOPATHs fields. If GOPATHs is empty, then
// the GOPATH environment variable (or the default GOPATH) is used instead.
func (c *Ctx) SetPaths(wd string, GOPATHs ...string) error {
	if wd == "" {
		return errors.New("cannot set Ctx.WorkingDir to an empty path")
	}
	c.WorkingDir = wd

	if len(GOPATHs) == 0 {
		GOPATH := os.Getenv("GOPATH")
		if GOPATH == "" {
			GOPATH = defaultGOPATH()
		}
		GOPATHs = filepath.SplitList(GOPATH)
	}

	c.GOPATHs = append(c.GOPATHs, GOPATHs...)

	return nil
}

// defaultGOPATH gets the default GOPATH that was added in 1.8
// copied from go/build/build.go
func defaultGOPATH() string {
	env := "HOME"
	if runtime.GOOS == "windows" {
		env = "USERPROFILE"
	} else if runtime.GOOS == "plan9" {
		env = "home"
	}
	if home := os.Getenv(env); home != "" {
		def := filepath.Join(home, "go")
		if def == runtime.GOROOT() {
			// Don't set the default GOPATH to GOROOT,
			// as that will trigger warnings from the go tool.
			return ""
		}
		return def
	}
	return ""
}

// SourceManager produces an instance of gps's built-in SourceManager
// initialized to log to the receiver's logger.
func (c *Ctx) SourceManager() (*gps.SourceMgr, error) {
	cachedir := c.Cachedir
	if cachedir == "" {
		// When `DEPCACHEDIR` isn't set in the env, use the default - `$GOPATH/pkg/dep`.
		cachedir = filepath.Join(c.GOPATH, "pkg", "dep")
		// Create the default cachedir if it does not exist.
		if err := os.MkdirAll(cachedir, 0777); err != nil {
			return nil, errors.Wrap(err, "failed to create default cache directory")
		}
	}

	return gps.NewSourceManager(gps.SourceManagerConfig{
		Cachedir:       cachedir,
		Logger:         c.Out,
		DisableLocking: c.DisableLocking,
	})
}

// LoadProject starts from the current working directory and searches up the
// directory tree for a project root.  The search stops when a file with the name
// ManifestName (Gopkg.toml, by default) is located.
//
// The Project contains the parsed manifest as well as a parsed lock file, if
// present.  The import path is calculated as the remaining path segment
// below Ctx.GOPATH/src.
func (c *Ctx) LoadProject() (*Project, error) {
	root, err := findProjectRoot(c.WorkingDir)
	if err != nil {
		return nil, err
	}

	err = checkGopkgFilenames(root)
	if err != nil {
		return nil, err
	}

	p := new(Project)

	if err = p.SetRoot(root); err != nil {
		return nil, err
	}

	c.GOPATH, err = c.DetectProjectGOPATH(p)
	if err != nil {
		return nil, err
	}

	ip, err := c.ImportForAbs(p.AbsRoot)
	if err != nil {
		return nil, errors.Wrap(err, "root project import")
	}
	p.ImportRoot = gps.ProjectRoot(ip)

	mp := filepath.Join(p.AbsRoot, ManifestName)
	mf, err := os.Open(mp)
	if err != nil {
		if os.IsNotExist(err) {
			// TODO: list possible solutions? (dep init, cd $project)
			return nil, errors.Errorf("no %v found in project root %v", ManifestName, p.AbsRoot)
		}
		// Unable to read the manifest file
		return nil, err
	}
	defer mf.Close()

	var warns []error
	p.Manifest, warns, err = readManifest(mf)
	for _, warn := range warns {
		c.Err.Printf("dep: WARNING: %v\n", warn)
	}
	if err != nil {
		return nil, errors.Wrapf(err, "error while parsing %s", mp)
	}

	lp := filepath.Join(p.AbsRoot, LockName)
	lf, err := os.Open(lp)
	if err != nil {
		if os.IsNotExist(err) {
			// It's fine for the lock not to exist
			return p, nil
		}
		// But if a lock does exist and we can't open it, that's a problem
		return nil, errors.Wrapf(err, "could not open %s", lp)
	}
	defer lf.Close()

	p.Lock, err = readLock(lf)
	if err != nil {
		return nil, errors.Wrapf(err, "error while parsing %s", lp)
	}

	return p, nil
}

// DetectProjectGOPATH attempt to find the GOPATH containing the project.
//
//  If p.AbsRoot is not a symlink and is within a GOPATH, the GOPATH containing p.AbsRoot is returned.
//  If p.AbsRoot is a symlink and is not within any known GOPATH, the GOPATH containing p.ResolvedAbsRoot is returned.
//
// p.AbsRoot is assumed to be a symlink if it is not the same as p.ResolvedAbsRoot.
//
// DetectProjectGOPATH will return an error in the following cases:
//
//  If p.AbsRoot is not a symlink and is not within any known GOPATH.
//  If neither p.AbsRoot nor p.ResolvedAbsRoot are within a known GOPATH.
//  If both p.AbsRoot and p.ResolvedAbsRoot are within the same GOPATH.
//  If p.AbsRoot and p.ResolvedAbsRoot are each within a different GOPATH.
func (c *Ctx) DetectProjectGOPATH(p *Project) (string, error) {
	if p.AbsRoot == "" || p.ResolvedAbsRoot == "" {
		return "", errors.New("project AbsRoot and ResolvedAbsRoot must be set to detect GOPATH")
	}

	pGOPATH, perr := c.detectGOPATH(p.AbsRoot)

	// If p.AbsRoot is a not symlink, attempt to detect GOPATH for p.AbsRoot only.
	if equal, _ := fs.EquivalentPaths(p.AbsRoot, p.ResolvedAbsRoot); equal {
		return pGOPATH, perr
	}

	rGOPATH, rerr := c.detectGOPATH(p.ResolvedAbsRoot)

	// If detectGOPATH() failed for both p.AbsRoot and p.ResolvedAbsRoot, then both are not within any known GOPATHs.
	if perr != nil && rerr != nil {
		return "", errors.Errorf("both %s and %s are not within any known GOPATH", p.AbsRoot, p.ResolvedAbsRoot)
	}

	// If pGOPATH equals rGOPATH, then both are within the same GOPATH.
	if equal, _ := fs.EquivalentPaths(pGOPATH, rGOPATH); equal {
		return "", errors.Errorf("both %s and %s are in the same GOPATH %s", p.AbsRoot, p.ResolvedAbsRoot, pGOPATH)
	}

	if pGOPATH != "" && rGOPATH != "" {
		return "", errors.Errorf("%s and %s are both in different GOPATHs", p.AbsRoot, p.ResolvedAbsRoot)
	}

	// Otherwise, either the p.AbsRoot or p.ResolvedAbsRoot is within a GOPATH.
	if pGOPATH == "" {
		return rGOPATH, nil
	}

	return pGOPATH, nil
}

// detectGOPATH detects the GOPATH for a given path from ctx.GOPATHs.
func (c *Ctx) detectGOPATH(path string) (string, error) {
	for _, gp := range c.GOPATHs {
		isPrefix, err := fs.HasFilepathPrefix(path, gp)
		if err != nil {
			return "", errors.Wrap(err, "failed to detect GOPATH")
		}
		if isPrefix {
			return gp, nil
		}
	}
	return "", errors.Errorf("%s is not within a known GOPATH/src", path)
}

// ImportForAbs returns the import path for an absolute project path by trimming the
// `$GOPATH/src/` prefix.  Returns an error for paths equal to, or without this prefix.
func (c *Ctx) ImportForAbs(path string) (string, error) {
	srcprefix := filepath.Join(c.GOPATH, "src") + string(filepath.Separator)
	isPrefix, err := fs.HasFilepathPrefix(path, srcprefix)
	if err != nil {
		return "", errors.Wrap(err, "failed to find import path")
	}
	if isPrefix {
		if len(path) <= len(srcprefix) {
			return "", errors.New("dep does not currently support using GOPATH/src as the project root")
		}

		// filepath.ToSlash because we're dealing with an import path now,
		// not an fs path
		return filepath.ToSlash(path[len(srcprefix):]), nil
	}

	return "", errors.Errorf("%s is not within any GOPATH/src", path)
}

// AbsForImport returns the absolute path for the project root
// including the $GOPATH. This will not work with stdlib packages and the
// package directory needs to exist.
func (c *Ctx) AbsForImport(path string) (string, error) {
	posspath := filepath.Join(c.GOPATH, "src", path)
	dirOK, err := fs.IsDir(posspath)
	if err != nil {
		return "", errors.Wrapf(err, "checking if %s is a directory", posspath)
	}
	if !dirOK {
		return "", errors.Errorf("%s does not exist", posspath)
	}
	return posspath, nil
}

// ValidateParams ensure that solving can be completed with the specified params.
func (c *Ctx) ValidateParams(sm gps.SourceManager, params gps.SolveParameters) error {
	err := gps.ValidateParams(params, sm)
	if err != nil {
		if deduceErrs, ok := err.(gps.DeductionErrs); ok {
			c.Err.Println("The following errors occurred while deducing packages:")
			for ip, dErr := range deduceErrs {
				c.Err.Printf("  * \"%s\": %s", ip, dErr)
			}
			c.Err.Println()
		}
	}

	return errors.Wrap(err, "validateParams")
}

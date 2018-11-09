package main

import (
	"errors"
	"go/build"
	"log"
	"os"
	"path/filepath"

	"golang.org/x/tools/go/vcs"
)

var cmdRestore = &Command{
	Name:  "restore",
	Short: "check out listed dependency versions in GOPATH",
	Long: `
Restore checks out the Godeps-specified version of each package in GOPATH.

NOTE: restore leaves git repositories in a detached state. go1.6+ no longer
checks out the master branch when doing a "go get", see:
https://github.com/golang/go/commit/42206598671a44111c8f726ad33dc7b265bdf669.

`,
	Run:          runRestore,
	OnlyInGOPATH: true,
}

// Three phases:
// 1. Download all deps
// 2. Restore all deps (checkout the recorded rev)
// 3. Attempt to load all deps as a simple consistency check
func runRestore(cmd *Command, args []string) {
	if len(build.Default.GOPATH) == 0 {
		log.Println("Error restore requires GOPATH but it is empty.")
		os.Exit(1)
	}

	var hadError bool
	checkErr := func(s string) {
		if hadError {
			log.Println(s)
			os.Exit(1)
		}
	}

	g, err := loadDefaultGodepsFile()
	if err != nil {
		log.Fatalln(err)
	}
	for i, dep := range g.Deps {
		verboseln("Downloading dependency (if needed):", dep.ImportPath)
		err := download(&dep)
		if err != nil {
			log.Printf("error downloading dep (%s): %s\n", dep.ImportPath, err)
			hadError = true
		}
		g.Deps[i] = dep
	}
	checkErr("Error downloading some deps. Aborting restore and check.")
	for _, dep := range g.Deps {
		verboseln("Restoring dependency (if needed):", dep.ImportPath)
		err := restore(dep)
		if err != nil {
			log.Printf("error restoring dep (%s): %s\n", dep.ImportPath, err)
			hadError = true
		}
	}
	checkErr("Error restoring some deps. Aborting check.")
	for _, dep := range g.Deps {
		verboseln("Checking dependency:", dep.ImportPath)
		_, err := LoadPackages(dep.ImportPath)
		if err != nil {
			log.Printf("Dep (%s) restored, but was unable to load it with error:\n\t%s\n", dep.ImportPath, err)
			if me, ok := err.(errorMissingDep); ok {
				log.Println("\tThis may be because the dependencies were saved with an older version of godep (< v33).")
				log.Printf("\tTry `go get %s`. Then `godep save` to update deps.\n", me.i)
			}
			hadError = true
		}
	}
	checkErr("Error checking some deps.")
}

var downloaded = make(map[string]bool)

// download the given dependency.
// 2 Passes: 1) go get -d <pkg>, 2) git pull (if necessary)
func download(dep *Dependency) error {

	rr, err := vcs.RepoRootForImportPath(dep.ImportPath, debug)
	if err != nil {
		debugln("Error determining repo root for", dep.ImportPath)
		return err
	}
	ppln("rr", rr)

	dep.vcs = cmd[rr.VCS]

	// try to find an existing directory in the GOPATHs
	for _, gp := range filepath.SplitList(build.Default.GOPATH) {
		t := filepath.Join(gp, "src", rr.Root)
		fi, err := os.Stat(t)
		if err != nil {
			continue
		}
		if fi.IsDir() {
			dep.root = t
			break
		}
	}

	// If none found, just pick the first GOPATH entry (AFAICT that's what go get does)
	if dep.root == "" {
		dep.root = filepath.Join(filepath.SplitList(build.Default.GOPATH)[0], "src", rr.Root)
	}
	ppln("dep", dep)

	if downloaded[rr.Repo] {
		verboseln("Skipping already downloaded repo", rr.Repo)
		return nil
	}

	fi, err := os.Stat(dep.root)
	if err != nil {
		if os.IsNotExist(err) {
			if err := os.MkdirAll(filepath.Dir(dep.root), os.ModePerm); err != nil {
				debugln("Error creating base dir of", dep.root)
				return err
			}
			err := rr.VCS.CreateAtRev(dep.root, rr.Repo, dep.Rev)
			debugln("CreatedAtRev", dep.root, rr.Repo, dep.Rev)
			if err != nil {
				debugln("CreateAtRev error", err)
				return err
			}
			downloaded[rr.Repo] = true
			return nil
		}
		debugln("Error checking repo root for", dep.ImportPath, "at", dep.root, ":", err)
		return err
	}

	if !fi.IsDir() {
		return errors.New("repo root src dir exists, but isn't a directory for " + dep.ImportPath + " at " + dep.root)
	}

	if !dep.vcs.exists(dep.root, dep.Rev) {
		debugln("Updating existing", dep.root)
		if dep.vcs == vcsGit {
			detached, err := gitDetached(dep.root)
			if err != nil {
				return err
			}
			if detached {
				db, err := gitDefaultBranch(dep.root)
				if err != nil {
					return err
				}
				if err := gitCheckout(dep.root, db); err != nil {
					return err
				}
			}
		}

		dep.vcs.vcs.Download(dep.root)
		downloaded[rr.Repo] = true
	}

	debugln("Nothing to download")
	return nil
}

var restored = make(map[string]string) // dep.root -> dep.Rev

// restore checks out the given revision.
func restore(dep Dependency) error {
	rev, ok := restored[dep.root]
	debugln(rev)
	debugln(ok)
	debugln(dep.root)
	if ok {
		if rev != dep.Rev {
			return errors.New("Wanted to restore rev " + dep.Rev + ", already restored rev " + rev + " for another package in the repo")
		}
		verboseln("Skipping already restored repo")
		return nil
	}

	debugln("Restoring:", dep.ImportPath, dep.Rev)
	err := dep.vcs.RevSync(dep.root, dep.Rev)
	if err == nil {
		restored[dep.root] = dep.Rev
	}
	return err
}

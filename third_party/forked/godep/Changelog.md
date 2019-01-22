#v80 (2018/01/26)

* Address lin/vet feedback.

#v79 (2017/02/01)

* Fixes #531: fullPackageInDir didn't capture the error from fillPackage()

#v78 (2017/01/19)

* Don't use build.ImportDir when discovering packages for the package spec. Fixes #529

#v77 (2017/01/13)

* Don't include quotes around hg revisions

#v76 (2017/01/10)

* Default to vendor being on unless older go versions.

#v75 (2016/11/02)

* Add "AUTHORS" and "CONTRIBUTORS" to legal files list: https://github.com/tools/godep/pull/522

#v74 (2016/06/01)

* Enable vendor/ on go1.7
* No longer use a godep workspace, use vendor/ (yay!)
* Notify that support for Godep workspaces will be removed once go1.8 ships

#v73 (2016/05/31)

* Fix permission changes on Windows via @alexbrand. Closes #481.

#v72 (2016/05/27)

* Improve handling of git remote show origin. Should help in cases where remote HEAD is ambiguous.
* Add ISSUE_TEMPLATE

#v71 (2016/05/24)

* Preserve permissions on copied files.

#v70 (2016/05/20)

* Fix the May changelog dates
* No need to call build.Import, we already have the root of the dependency. Fixes an additional comment on #365

#v69 (2016/05/16)

* Make sure `devel-<short sha>` enabled `vendor/` unless there is a classic Godep _workspace already.

#v68 (2016/05/16)

* `devel-<short sha>` is always considered newer than any released go version

#v67 (2016/05/13)

* Attempt to handle missing deps a little better.

#v66 (2016/05/10)

* Use `git remote show origin` to find the default branch when restoring a git based package repository that is in detached head state

#v65 (2016/05/09)

* Rewrite update so that it considers new transitive dependencies, both in the same repo and outside of it.

#v64 (2016/05/09)

* godep update golang.org/x/tools/go/vcs

#v63 (2016/05/03)

* Support recording devel-<short sha> so development versions of Go can be matched

#v62 (2016/04/07)

* Note new go1.6+ behavior of not checking out master in README / restore help text.

#v61 (2016/04/06)

* Obey go version build tags based on recorded major go version. Fixes #448.

#v60 (2016/03/18)

* Make the $GOPATH check a warning.

#v59 (2016/03/18)

* Enforce requirement to be inside of a go src directory. A lot of time is usually spent
tracking down bug reports where people are doign stuff from outside of their $GOPATH. This
should help with that, at least until there it time to properly test godep use outside of a
$GOPATH and fix the issues.

#v58 (2016/03/15)

* Add GodepVersion to Godeps.json file so that as godep changes / adds features / fixes bugs we can know which version of godep most recently wrote out the file.

#v57 (2016/03/07)

* Don't use `git rev-parse --show-toplevel` to determine git repo roots as it resolves symlinks: https://github.com/tools/godep/pull/418

# v56 (2016/02/26)

* replace path comparisons with case insensitive pathEqual()
* add versionString() to debug output
* Send log output to Stderr

# v55 2016/02/22

* re-saved deps to clean out extra stuff (see v54; godep restore; godep save -r=false; rm -rf Godeps; godep save -r). We're still using a workspace with rewrites so users of older go version can still go get this tool.
* Replace simple == with strings.EqualFold in listFiles to avoid problems with case insensitive filesystems ("Code" != "code" when doing a byte by byte comparison)

# v54 2016/02/22

* Update some docs around vendor/
* More precise recording of dependencies. Removed recursive copying of sub directories of a package (precise vendoring). This should allow using `./...` with the go tool for compilation of project using `vendor/`. See https://github.com/tools/godep/pull/415

# v53 2016/02/11

* Disable VendorExperiment if a godep workspace already exists.

# v52 2016/01/27

* Trim 'rc' out of go version strings when determining major version.

# v51 2016/01/21

* Trim 'beta' out of go version strings when determining major version.

# v50 2016/01/19

* More verbose output on save -v.

# v49 2016/01/13

* Add UK spelling license/licence to the pile + fix up a bunch of typos
* Clarify tag handling in docs

# v48 2016/01/13

* Abort restore if there is no $GOPATH set.

# v47 2016/01/12

* Dev versions of go should honor the current meaning of GO15VENDOREXPERIMENT

# v46 2016/01/03

* Record "devel" when the release is a devel release of go (compiled from git).

# v45 2015/12/28

* Upcase windows drive letters before comparing. Fixes #383.

# v44 2015/12/23

* Clean package roots when attempting to find a vendor directory so we don't loop forever.
    * Fixes 382

# v43 2015/12/22

* Better error messages when parsing Godeps.json: Fixes #372

# v42 2015/12/22

* Fix a bunch of GO15VENDOREXPERIMENT issues
    * Find package directories better. Previously we used build.FindOnly which didn't work the way I expected it to (any dir would work w/o error).
    * Set the VendorExperiment bool based on go version as 1.6 defaults to on.
    * A bunch of extra debugging for use while sanity checking myself.
    * vendor flag for test structs.
    * Some tests for vendor/ stuff:
        * Basic Test
        * Transitive
        * Transitive, across GOPATHs + collapse vendor/ directories.
* Should Fix #358

# v41 2015/12/17

* Don't rewrite packages outside of the project. This would happen if you specified
  an external package for vendoring when you ran `goodep save -r ./... github.com/some/other/package`

# v40 2015/12/17

* When downloading a dependency, create the base directory if needed.

# v39 2015/12/16

* Record only the major go version (ex. go1.5) instead of the complete string.

# v38 2015/12/16

* Replace `go get`, further fix up restore error handling/reporting.
    * Fixes #186
    * Don't bother restoring/downloading if already done.

# v37 2015/12/15

* Change up how download/restore works a little
    * Try to load the package after downloading/restoring. Previously
      that was done too early in the process.
    * make previous verbose output debug output
    * report a typed error instead of a string from listPackage so it can
      be asserted to provide a nicer error.
    * Catch go get errors that say there are no go files found. See code
      comment as to why.
    * do *all* downloading during download phase.

# v36 2015/12/14

* Fixes #358: Using wrong variable. Will add test after release.

# v35 2015/12/11

* Fixes #356: Major performance regressions in v34
    * Enable cpu profiling via flag on save.
    * Cache packages by dir
    * Don't do a full import pass on deps for packages in the GOROOT
    * create a bit less garbage at times
* Generalize -v & -d flags

# v34 2015/12/08

* We now use build.Context to help locate packages only and do our own parsing (via go/ast).
* Fixes reported issues caused by v33 (Removal of `go list`):
    * #345: Bug in godep restore
    * #346: Fix loading a dot package
    * #348: Godep save issue when importing lib/pq
    * #350: undefined: build.MultiplePackageError
    * #351: stow away helper files
    * #353: cannot find package "appengine"
        * Don't process imports of `.go` files tagged with the `appengine` build tag.

# v33 2015/12/07

* Replace the use of `go list`. This is a large change although all existing tests pass.
    * Don't process the imports of `.go` files with the `ignore` build tag.

# v32 2015/12/02

* Eval Symlinks in Contains() check.

# v31 2015/12/02

* In restore, mention which package had the problem -- @shurcool

# v30 2015/11/25

* Add `-t` flag to the `godep get` command.

# v29 2015/11/17

* Temp work around to fix issue with LICENSE files.

# v28 2015/11/09

* Make `version` an actual command.

# v27 2015/11/06

* run command once during restore -v

# v26 2015/11/05

* Better fix for the issue fixed in v25: All update paths are now path.Clean()'d

# v25 2015/11/05

* `godep update package/` == `godep update package`. Fixes #313

# v24 2015/11/05

* Honor -t during update. Fixes #312

# v23 2015/11/05

* Do not use --debug to find full revision name for mercurial repositories

# v22 2015/11/14

* s/GOVENDOREXPERIMENT/GO15VENDOREXPERIMENT :-(

# v21 2015/11/13

* Fix #310: Case insensitive fs issue

# v20 2015/11/13

* Attempt to include license files when vendoring. (@client9)

# v19 2015/11/3

* Fix conflict error message. Revisions were swapped. Also better selection of package that needs update.

# v18 2015/10/16

* Improve error message when trying to save a conflicting revision.

# v17 2015/10/15

* Fix for v16 bug. All vcs list commands now produce paths relative to the root of the vcs.

# v16 2015/10/15

* Determine repo root using vcs commands and use that instead of dep.dir

# v15 2015/10/14

* Update .travis.yml file to do releases to github

# v14 2015/10/08

* Don't print out a workspace path when GO15VENDOREXPERIMENT is active. The vendor/ directory is not a valid workspace, so can't be added to your $GOPATH.

# v13 2015/10/07

* Do restores in 2 separate steps, first download all deps and then check out the recorded revisions.
* Update Changelog date format

# v12 2015/09/22

* Extract errors into separate file.

# v11 2015/08/22

* Amend code to pass golint.

# v10 2015/09/21

* Analyse vendored package test dependencies.
* Update documentation.

# v9 2015/09/17

* Don't save test dependencies by default.

# v8 2015/09/17

* Reorganize code.

# v7 2015/09/09

* Add verbose flag.
* Skip untracked files.
* Add VCS list command.

# v6 2015/09/04

*  Revert ignoring testdata directories and instead ignore it while
processing Go files and copy the whole directory unmodified.

# v5 2015/09/04

* Fix vcs selection in restore command to work as go get does

# v4 2015/09/03

* Remove the deprecated copy option.

# v3 2015/08/26

* Ignore testdata directories

# v2 2015/08/11

* Include command line packages in the set to copy

This is a simplification to how we define the behavior
of the save command. Now it has two distinct package
parameters, the "root set" and the "destination", and
they have clearer roles. The packages listed on the
command line form the root set; they and all their
dependencies will be copied into the Godeps directory.
Additionally, the destination (always ".") will form the
initial list of "seen" import paths to exclude from
copying.

In the common case, the root set is equal to the
destination, so the effective behavior doesn't change.
This is primarily just a simpler definition. However, if
the user specifies a package on the command line that
lives outside of . then that package will be copied.

As a side effect, there's a simplification to the way we
add packages to the initial "seen" set. Formerly, to
avoid copying dependencies unnecessarily, we would try
to find the root of the VCS repo for each package in the
root set, and mark the import path of the entire repo as
seen. This meant for a repo at path C, if destination
C/S imports C/T, we would not copy C/T into C/S/Godeps.
Now we don't treat the repo root specially, and as
mentioned above, the destination alone is considered
seen.

This also means we don't require listed packages to be
in VCS unless they're outside of the destination.

# v1 2015/07/20

* godep version command

Output the version as well as some godep runtime information that is
useful for debugging user's issues.

The version const would be bumped each time a PR is merged into master
to ensure that we'll be able to tell which version someone got when they
did a `go get github.com/tools/godep`.

# Older changes

Many and more, see `git log -p`

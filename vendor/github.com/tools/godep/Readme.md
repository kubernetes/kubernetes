# Godep - Archived

Please use [dep](https://github.com/golang/dep) or another tool instead.

The rest of this readme is preserved for those that may still need its contents.

[![Build Status](https://travis-ci.org/tools/godep.svg)](https://travis-ci.org/tools/godep)

[![GoDoc](https://godoc.org/github.com/tools/godep?status.svg)](https://godoc.org/github.com/tools/godep)

godep helps build packages reproducibly by fixing their dependencies.

This tool assumes you are working in a standard Go workspace, as described [here](http://golang.org/doc/code.html). We
expect godep to build on Go 1.4* or newer, but you can use it on any project that works with Go 1 or newer.

Please check the [FAQ](FAQ.md) if you have a question.

## Golang Dep

The Go community now has the [dep](https://github.com/golang/dep) project to
manage dependencies. Please consider trying to migrate from Godep to dep. If there
is an issue preventing you from migrating please file an issue with dep so the
problem can be corrected. Godep will continue to be supported for some time but
is considered to be in a state of support rather than active feature development.

## Install

```console
go get github.com/tools/godep
```

## How to use godep with a new project

Assuming you've got everything working already, so you can build your project
with `go install` and test it with `go test`, it's one command to start using:

```console
godep save
```

This will save a list of dependencies to the file `Godeps/Godeps.json` and copy
their source code into `vendor/` (or `Godeps/_workspace/` when using older
versions of Go). Godep does **not copy**:

- files from source repositories that are not tracked in version control.
- `*_test.go` files.
- `testdata` directories.
- files outside of the go packages.

Godep does not process the imports of `.go` files with either the `ignore`
or `appengine` build tags.

Test files and testdata directories can be saved by adding `-t`.

Read over the contents of `vendor/` and make sure it looks reasonable. Then
commit the `Godeps/` and `vendor/` directories to version control.

## The deprecated `-r` flag

For older versions of Go, the `-r` flag tells save to automatically rewrite
package import paths. This allows your code to refer directly to the copied
dependencies in `Godeps/_workspace`. So, a package C that depends on package
D will actually import `C/Godeps/_workspace/src/D`. This makes C's repo
self-contained and causes `go get` to build C with the right version of all
dependencies.

If you don't use `-r`, when using older version of Go, then in order to use the
fixed dependencies and get reproducible builds, you must make sure that **every
time** you run a Go-related command, you wrap it in one of these two ways:

- If the command you are running is just `go`, run it as `godep go ...`, e.g.
  `godep go install -v ./...`
- When using a different command, set your `$GOPATH` using `godep path` as
  described below.

`-r` isn't necessary with go1.6+ and isn't allowed.

## Additional Operations

### Restore

The `godep restore` installs the
package versions specified in `Godeps/Godeps.json` to your `$GOPATH`. This
modifies the state of packages in your `$GOPATH`. NOTE: `godep restore` leaves
git repositories in a detached state. `go1.6`+ no longer checks out the master
branch when doing a `go get`, see [here](https://github.com/golang/go/commit/42206598671a44111c8f726ad33dc7b265bdf669).

> If you run `godep restore` in your main `$GOPATH` `go get -u` will fail on packages that are behind master.

Please see the [FAQ](https://github.com/tools/godep/blob/master/FAQ.md#should-i-use-godep-restore) section about restore.

### Edit-test Cycle

1. Edit code
1. Run `godep go test`
1. (repeat)

### Add a Dependency

To add a new package foo/bar, do this:

1. Run `go get foo/bar`
1. Edit your code to import foo/bar.
1. Run `godep save` (or `godep save ./...`).

### Update a Dependency

To update a package from your `$GOPATH`, do this:

1. Run `go get -u foo/bar`
1. Run `godep update foo/bar`.

You can use the `...` wildcard, for example `godep update foo/...`. Before comitting the change, you'll probably want to
inspect the changes to Godeps, for example with `git diff`, and make sure it looks reasonable.

## Multiple Packages

If your repository has more than one package, you're probably accustomed to
running commands like `go test ./...`, `go install ./...`, and `go fmt ./...`.
Similarly, you should run `godep save ./...` to capture the dependencies of all
packages in your application.

## File Format

Godeps is a json file with the following structure:

```go
type Godeps struct {
  ImportPath   string
  GoVersion    string   // Abridged output of 'go version'.
  GodepVersion string   // Abridged output of 'godep version'
  Packages     []string // Arguments to godep save, if any.
  Deps         []struct {
    ImportPath string
    Comment    string // Description of commit, if present.
    Rev        string // VCS-specific commit ID.
  }
}
```

Example Godeps:

```json
{
  "ImportPath": "github.com/kr/hk",
  "GoVersion": "go1.6",
  "Deps": [
    {
      "ImportPath": "code.google.com/p/go-netrc/netrc",
      "Rev": "28676070ab99"
    },
    {
      "ImportPath": "github.com/kr/binarydist",
      "Rev": "3380ade90f8b0dfa3e363fd7d7e941fa857d0d13"
    }
  ]
}
```

## Migrating to vendor/

Godep supports the Go 1.5+ vendor/
[experiment](https://github.com/golang/go/commit/183cc0cd41f06f83cb7a2490a499e3f9101befff)
utilizing the same environment variable that the go tooling itself supports
(`GO15VENDOREXPERIMENT`).

godep mostly works the same way as the `go` command line tool. If you have go
1.5.X and set `GO15VENDOREXPERIMENT=1` or have go1.6.X (or devel) `vendor/`
is enabled. **Unless** you already have a `Godeps/_workspace`. This is a safety
feature and godep warns you about this.

When `vendor/` is enabled godep will write the vendored code into the top level
`./vendor/` directory. A `./Godeps/Godeps.json` file is created to track
the dependencies and revisions. `vendor/` is not compatible with rewrites.

There is currently no automated migration between the old Godeps workspace and
the vendor directory, but the following steps should work:

```term
# just to be safe
$ unset GO15VENDOREXPERIMENT

# restore currently vendored deps to the $GOPATH
$ godep restore

# The next line is only needed to automatically undo rewritten imports that were
# created with godep save -r.
$ godep save -r=false <pkg spec>

# Remove the old Godeps folder
$ rm -rf Godeps

# If on go1.5.X to enable `vendor/`
$ export GO15VENDOREXPERIMENT=1

# re-analyze deps and save to `vendor/`.
$ godep save <pkg spec>

# Add the changes to your VCS
$ git add -A . ; git commit -am "Godep workspace -> vendor/"

# You should see your Godeps/_workspace/src files "moved" to vendor/.
```

## Releasing

1. Increment the version in `version.go`.
1. Tag the commit with the same version number.
1. Update `Changelog.md`.

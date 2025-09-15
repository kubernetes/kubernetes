// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package packages loads Go packages for inspection and analysis.

The [Load] function takes as input a list of patterns and returns a
list of [Package] values describing individual packages matched by those
patterns.
A [Config] specifies configuration options, the most important of which is
the [LoadMode], which controls the amount of detail in the loaded packages.

Load passes most patterns directly to the underlying build tool.
The default build tool is the go command.
Its supported patterns are described at
https://pkg.go.dev/cmd/go#hdr-Package_lists_and_patterns.
Other build systems may be supported by providing a "driver";
see [The driver protocol].

All patterns with the prefix "query=", where query is a
non-empty string of letters from [a-z], are reserved and may be
interpreted as query operators.

Two query operators are currently supported: "file" and "pattern".

The query "file=path/to/file.go" matches the package or packages enclosing
the Go source file path/to/file.go.  For example "file=~/go/src/fmt/print.go"
might return the packages "fmt" and "fmt [fmt.test]".

The query "pattern=string" causes "string" to be passed directly to
the underlying build tool. In most cases this is unnecessary,
but an application can use Load("pattern=" + x) as an escaping mechanism
to ensure that x is not interpreted as a query operator if it contains '='.

All other query operators are reserved for future use and currently
cause Load to report an error.

The Package struct provides basic information about the package, including

  - ID, a unique identifier for the package in the returned set;
  - GoFiles, the names of the package's Go source files;
  - Imports, a map from source import strings to the Packages they name;
  - Types, the type information for the package's exported symbols;
  - Syntax, the parsed syntax trees for the package's source code; and
  - TypesInfo, the result of a complete type-check of the package syntax trees.

(See the documentation for type Package for the complete list of fields
and more detailed descriptions.)

For example,

	Load(nil, "bytes", "unicode...")

returns four Package structs describing the standard library packages
bytes, unicode, unicode/utf16, and unicode/utf8. Note that one pattern
can match multiple packages and that a package might be matched by
multiple patterns: in general it is not possible to determine which
packages correspond to which patterns.

Note that the list returned by Load contains only the packages matched
by the patterns. Their dependencies can be found by walking the import
graph using the Imports fields.

The Load function can be configured by passing a pointer to a Config as
the first argument. A nil Config is equivalent to the zero Config, which
causes Load to run in [LoadFiles] mode, collecting minimal information.
See the documentation for type Config for details.

As noted earlier, the Config.Mode controls the amount of detail
reported about the loaded packages. See the documentation for type LoadMode
for details.

Most tools should pass their command-line arguments (after any flags)
uninterpreted to Load, so that it can interpret them
according to the conventions of the underlying build system.

See the Example function for typical usage.

# The driver protocol

Load may be used to load Go packages even in Go projects that use
alternative build systems, by installing an appropriate "driver"
program for the build system and specifying its location in the
GOPACKAGESDRIVER environment variable.
For example,
https://github.com/bazelbuild/rules_go/wiki/Editor-and-tool-integration
explains how to use the driver for Bazel.

The driver program is responsible for interpreting patterns in its
preferred notation and reporting information about the packages that
those patterns identify. Drivers must also support the special "file="
and "pattern=" patterns described above.

The patterns are provided as positional command-line arguments. A
JSON-encoded [DriverRequest] message providing additional information
is written to the driver's standard input. The driver must write a
JSON-encoded [DriverResponse] message to its standard output. (This
message differs from the JSON schema produced by 'go list'.)

The value of the PWD environment variable seen by the driver process
is the preferred name of its working directory. (The working directory
may have other aliases due to symbolic links; see the comment on the
Dir field of [exec.Cmd] for related information.)
When the driver process emits in its response the name of a file
that is a descendant of this directory, it must use an absolute path
that has the value of PWD as a prefix, to ensure that the returned
filenames satisfy the original query.
*/
package packages // import "golang.org/x/tools/go/packages"

/*

Motivation and design considerations

The new package's design solves problems addressed by two existing
packages: go/build, which locates and describes packages, and
golang.org/x/tools/go/loader, which loads, parses and type-checks them.
The go/build.Package structure encodes too much of the 'go build' way
of organizing projects, leaving us in need of a data type that describes a
package of Go source code independent of the underlying build system.
We wanted something that works equally well with go build and vgo, and
also other build systems such as Bazel and Blaze, making it possible to
construct analysis tools that work in all these environments.
Tools such as errcheck and staticcheck were essentially unavailable to
the Go community at Google, and some of Google's internal tools for Go
are unavailable externally.
This new package provides a uniform way to obtain package metadata by
querying each of these build systems, optionally supporting their
preferred command-line notations for packages, so that tools integrate
neatly with users' build environments. The Metadata query function
executes an external query tool appropriate to the current workspace.

Loading packages always returns the complete import graph "all the way down",
even if all you want is information about a single package, because the query
mechanisms of all the build systems we currently support ({go,vgo} list, and
blaze/bazel aspect-based query) cannot provide detailed information
about one package without visiting all its dependencies too, so there is
no additional asymptotic cost to providing transitive information.
(This property might not be true of a hypothetical 5th build system.)

In calls to TypeCheck, all initial packages, and any package that
transitively depends on one of them, must be loaded from source.
Consider A->B->C->D->E: if A,C are initial, A,B,C must be loaded from
source; D may be loaded from export data, and E may not be loaded at all
(though it's possible that D's export data mentions it, so a
types.Package may be created for it and exposed.)

The old loader had a feature to suppress type-checking of function
bodies on a per-package basis, primarily intended to reduce the work of
obtaining type information for imported packages. Now that imports are
satisfied by export data, the optimization no longer seems necessary.

Despite some early attempts, the old loader did not exploit export data,
instead always using the equivalent of WholeProgram mode. This was due
to the complexity of mixing source and export data packages (now
resolved by the upward traversal mentioned above), and because export data
files were nearly always missing or stale. Now that 'go build' supports
caching, all the underlying build systems can guarantee to produce
export data in a reasonable (amortized) time.

Test "main" packages synthesized by the build system are now reported as
first-class packages, avoiding the need for clients (such as go/ssa) to
reinvent this generation logic.

One way in which go/packages is simpler than the old loader is in its
treatment of in-package tests. In-package tests are packages that
consist of all the files of the library under test, plus the test files.
The old loader constructed in-package tests by a two-phase process of
mutation called "augmentation": first it would construct and type check
all the ordinary library packages and type-check the packages that
depend on them; then it would add more (test) files to the package and
type-check again. This two-phase approach had four major problems:
1) in processing the tests, the loader modified the library package,
   leaving no way for a client application to see both the test
   package and the library package; one would mutate into the other.
2) because test files can declare additional methods on types defined in
   the library portion of the package, the dispatch of method calls in
   the library portion was affected by the presence of the test files.
   This should have been a clue that the packages were logically
   different.
3) this model of "augmentation" assumed at most one in-package test
   per library package, which is true of projects using 'go build',
   but not other build systems.
4) because of the two-phase nature of test processing, all packages that
   import the library package had to be processed before augmentation,
   forcing a "one-shot" API and preventing the client from calling Load
   in several times in sequence as is now possible in WholeProgram mode.
   (TypeCheck mode has a similar one-shot restriction for a different reason.)

Early drafts of this package supported "multi-shot" operation.
Although it allowed clients to make a sequence of calls (or concurrent
calls) to Load, building up the graph of Packages incrementally,
it was of marginal value: it complicated the API
(since it allowed some options to vary across calls but not others),
it complicated the implementation,
it cannot be made to work in Types mode, as explained above,
and it was less efficient than making one combined call (when this is possible).
Among the clients we have inspected, none made multiple calls to load
but could not be easily and satisfactorily modified to make only a single call.
However, applications changes may be required.
For example, the ssadump command loads the user-specified packages
and in addition the runtime package.  It is tempting to simply append
"runtime" to the user-provided list, but that does not work if the user
specified an ad-hoc package such as [a.go b.go].
Instead, ssadump no longer requests the runtime package,
but seeks it among the dependencies of the user-specified packages,
and emits an error if it is not found.

Questions & Tasks

- Add GOARCH/GOOS?
  They are not portable concepts, but could be made portable.
  Our goal has been to allow users to express themselves using the conventions
  of the underlying build system: if the build system honors GOARCH
  during a build and during a metadata query, then so should
  applications built atop that query mechanism.
  Conversely, if the target architecture of the build is determined by
  command-line flags, the application can pass the relevant
  flags through to the build system using a command such as:
    myapp -query_flag="--cpu=amd64" -query_flag="--os=darwin"
  However, this approach is low-level, unwieldy, and non-portable.
  GOOS and GOARCH seem important enough to warrant a dedicated option.

- How should we handle partial failures such as a mixture of good and
  malformed patterns, existing and non-existent packages, successful and
  failed builds, import failures, import cycles, and so on, in a call to
  Load?

- Support bazel, blaze, and go1.10 list, not just go1.11 list.

- Handle (and test) various partial success cases, e.g.
  a mixture of good packages and:
  invalid patterns
  nonexistent packages
  empty packages
  packages with malformed package or import declarations
  unreadable files
  import cycles
  other parse errors
  type errors
  Make sure we record errors at the correct place in the graph.

- Missing packages among initial arguments are not reported.
  Return bogus packages for them, like golist does.

- "undeclared name" errors (for example) are reported out of source file
  order. I suspect this is due to the breadth-first resolution now used
  by go/types. Is that a bug? Discuss with gri.

*/

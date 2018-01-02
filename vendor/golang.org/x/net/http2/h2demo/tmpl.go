// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build h2demo

package main

import "html/template"

var pushTmpl = template.Must(template.New("serverpush").Parse(`

<!DOCTYPE html>
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#375EAB">

  <title>HTTP/2 Server Push Demo</title>

<link type="text/css" rel="stylesheet" href="/serverpush/static/style.css?{{.CacheBust}}">
<script>
window.initFuncs = [];
</script>

<script>
function showtimes() {
	var times = 'DOM loaded: ' + (window.performance.timing.domContentLoadedEventEnd - window.performance.timing.navigationStart) + 'ms, '
	times += 'DOM complete (all loaded): ' + (window.performance.timing.domComplete - window.performance.timing.navigationStart) + 'ms, '
	times += 'Load event fired: ' + (window.performance.timing.loadEventStart - window.performance.timing.navigationStart) + 'ms'
	document.getElementById('loadtimes').innerHTML = times
}
</script>

</head>
<body onload="showtimes()">

<div style="background:#fff9a4;padding:10px">
Note: This page exists for demonstration purposes. For the actual cmd/go docs, go to <a href="golang.org/cmd/go">golang.org/cmd/go</a>.
</div>

<div style="padding:20px">


<a href="https://{{.HTTPSHost}}/serverpush">HTTP/2 with Server Push</a> | <a href="http://{{.HTTPHost}}/serverpush">HTTP only</a>
<div id="loadtimes"></div>

</div>

<div id='lowframe' style="position: fixed; bottom: 0; left: 0; height: 0; width: 100%; border-top: thin solid grey; background-color: white; overflow: auto;">
...
</div><!-- #lowframe -->

<div id="topbar" class="wide"><div class="container">
<div class="top-heading" id="heading-wide"><a href="/">The Go Programming Language</a></div>
<div class="top-heading" id="heading-narrow"><a href="/">Go</a></div>
<a href="#" id="menu-button"><span id="menu-button-arrow">&#9661;</span></a>
<form method="GET" action="/search">
<div id="menu">
<a href="/doc/">Documents</a>
<a href="/pkg/">Packages</a>
<a href="/project/">The Project</a>
<a href="/help/">Help</a>
<a href="/blog/">Blog</a>

<a id="playgroundButton" href="http://play.golang.org/" title="Show Go Playground">Play</a>

<input type="text" id="search" name="q" class="inactive" value="Search" placeholder="Search">
</div>
</form>

</div></div>


<div id="playground" class="play">
	<div class="input"><textarea class="code" spellcheck="false">package main

import "fmt"

func main() {
	fmt.Println("Hello, 世界")
}</textarea></div>
	<div class="output"></div>
	<div class="buttons">
		<a class="run" title="Run this code [shift-enter]">Run</a>
		<a class="fmt" title="Format this code">Format</a>
		
		<a class="share" title="Share this code">Share</a>
		
	</div>
</div>


<div id="page" class="wide">
<div class="container">


  <h1>Command go</h1>




<div id="nav"></div>


<!--
	Copyright 2009 The Go Authors. All rights reserved.
	Use of this source code is governed by a BSD-style
	license that can be found in the LICENSE file.
-->
<!--
	Note: Static (i.e., not template-generated) href and id
	attributes start with "pkg-" to make it impossible for
	them to conflict with generated attributes (some of which
	correspond to Go identifiers).
-->

	<script type='text/javascript'>
	document.ANALYSIS_DATA = null;
	document.CALLGRAPH = null;
	</script>

	
		
		<p>
Go is a tool for managing Go source code.
</p>
<p>
Usage:
</p>
<pre>go command [arguments]
</pre>
<p>
The commands are:
</p>
<pre>build       compile packages and dependencies
clean       remove object files
doc         show documentation for package or symbol
env         print Go environment information
bug         start a bug report
fix         run go tool fix on packages
fmt         run gofmt on package sources
generate    generate Go files by processing source
get         download and install packages and dependencies
install     compile and install packages and dependencies
list        list packages
run         compile and run Go program
test        test packages
tool        run specified go tool
version     print Go version
vet         run go tool vet on packages
</pre>
<p>
Use &#34;go help [command]&#34; for more information about a command.
</p>
<p>
Additional help topics:
</p>
<pre>c           calling between Go and C
buildmode   description of build modes
filetype    file types
gopath      GOPATH environment variable
environment environment variables
importpath  import path syntax
packages    description of package lists
testflag    description of testing flags
testfunc    description of testing functions
</pre>
<p>
Use &#34;go help [topic]&#34; for more information about that topic.
</p>
<h3 id="hdr-Compile_packages_and_dependencies">Compile packages and dependencies</h3>
<p>
Usage:
</p>
<pre>go build [-o output] [-i] [build flags] [packages]
</pre>
<p>
Build compiles the packages named by the import paths,
along with their dependencies, but it does not install the results.
</p>
<p>
If the arguments to build are a list of .go files, build treats
them as a list of source files specifying a single package.
</p>
<p>
When compiling a single main package, build writes
the resulting executable to an output file named after
the first source file (&#39;go build ed.go rx.go&#39; writes &#39;ed&#39; or &#39;ed.exe&#39;)
or the source code directory (&#39;go build unix/sam&#39; writes &#39;sam&#39; or &#39;sam.exe&#39;).
The &#39;.exe&#39; suffix is added when writing a Windows executable.
</p>
<p>
When compiling multiple packages or a single non-main package,
build compiles the packages but discards the resulting object,
serving only as a check that the packages can be built.
</p>
<p>
When compiling packages, build ignores files that end in &#39;_test.go&#39;.
</p>
<p>
The -o flag, only allowed when compiling a single package,
forces build to write the resulting executable or object
to the named output file, instead of the default behavior described
in the last two paragraphs.
</p>
<p>
The -i flag installs the packages that are dependencies of the target.
</p>
<p>
The build flags are shared by the build, clean, get, install, list, run,
and test commands:
</p>
<pre>-a
	force rebuilding of packages that are already up-to-date.
-n
	print the commands but do not run them.
-p n
	the number of programs, such as build commands or
	test binaries, that can be run in parallel.
	The default is the number of CPUs available.
-race
	enable data race detection.
	Supported only on linux/amd64, freebsd/amd64, darwin/amd64 and windows/amd64.
-msan
	enable interoperation with memory sanitizer.
	Supported only on linux/amd64,
	and only with Clang/LLVM as the host C compiler.
-v
	print the names of packages as they are compiled.
-work
	print the name of the temporary work directory and
	do not delete it when exiting.
-x
	print the commands.

-asmflags &#39;flag list&#39;
	arguments to pass on each go tool asm invocation.
-buildmode mode
	build mode to use. See &#39;go help buildmode&#39; for more.
-compiler name
	name of compiler to use, as in runtime.Compiler (gccgo or gc).
-gccgoflags &#39;arg list&#39;
	arguments to pass on each gccgo compiler/linker invocation.
-gcflags &#39;arg list&#39;
	arguments to pass on each go tool compile invocation.
-installsuffix suffix
	a suffix to use in the name of the package installation directory,
	in order to keep output separate from default builds.
	If using the -race flag, the install suffix is automatically set to race
	or, if set explicitly, has _race appended to it.  Likewise for the -msan
	flag.  Using a -buildmode option that requires non-default compile flags
	has a similar effect.
-ldflags &#39;flag list&#39;
	arguments to pass on each go tool link invocation.
-linkshared
	link against shared libraries previously created with
	-buildmode=shared.
-pkgdir dir
	install and load all packages from dir instead of the usual locations.
	For example, when building with a non-standard configuration,
	use -pkgdir to keep generated packages in a separate location.
-tags &#39;tag list&#39;
	a list of build tags to consider satisfied during the build.
	For more information about build tags, see the description of
	build constraints in the documentation for the go/build package.
-toolexec &#39;cmd args&#39;
	a program to use to invoke toolchain programs like vet and asm.
	For example, instead of running asm, the go command will run
	&#39;cmd args /path/to/asm &lt;arguments for asm&gt;&#39;.
</pre>
<p>
The list flags accept a space-separated list of strings. To embed spaces
in an element in the list, surround it with either single or double quotes.
</p>
<p>
For more about specifying packages, see &#39;go help packages&#39;.
For more about where packages and binaries are installed,
run &#39;go help gopath&#39;.
For more about calling between Go and C/C++, run &#39;go help c&#39;.
</p>
<p>
Note: Build adheres to certain conventions such as those described
by &#39;go help gopath&#39;. Not all projects can follow these conventions,
however. Installations that have their own conventions or that use
a separate software build system may choose to use lower-level
invocations such as &#39;go tool compile&#39; and &#39;go tool link&#39; to avoid
some of the overheads and design decisions of the build tool.
</p>
<p>
See also: go install, go get, go clean.
</p>
<h3 id="hdr-Remove_object_files">Remove object files</h3>
<p>
Usage:
</p>
<pre>go clean [-i] [-r] [-n] [-x] [build flags] [packages]
</pre>
<p>
Clean removes object files from package source directories.
The go command builds most objects in a temporary directory,
so go clean is mainly concerned with object files left by other
tools or by manual invocations of go build.
</p>
<p>
Specifically, clean removes the following files from each of the
source directories corresponding to the import paths:
</p>
<pre>_obj/            old object directory, left from Makefiles
_test/           old test directory, left from Makefiles
_testmain.go     old gotest file, left from Makefiles
test.out         old test log, left from Makefiles
build.out        old test log, left from Makefiles
*.[568ao]        object files, left from Makefiles

DIR(.exe)        from go build
DIR.test(.exe)   from go test -c
MAINFILE(.exe)   from go build MAINFILE.go
*.so             from SWIG
</pre>
<p>
In the list, DIR represents the final path element of the
directory, and MAINFILE is the base name of any Go source
file in the directory that is not included when building
the package.
</p>
<p>
The -i flag causes clean to remove the corresponding installed
archive or binary (what &#39;go install&#39; would create).
</p>
<p>
The -n flag causes clean to print the remove commands it would execute,
but not run them.
</p>
<p>
The -r flag causes clean to be applied recursively to all the
dependencies of the packages named by the import paths.
</p>
<p>
The -x flag causes clean to print remove commands as it executes them.
</p>
<p>
For more about build flags, see &#39;go help build&#39;.
</p>
<p>
For more about specifying packages, see &#39;go help packages&#39;.
</p>
<h3 id="hdr-Show_documentation_for_package_or_symbol">Show documentation for package or symbol</h3>
<p>
Usage:
</p>
<pre>go doc [-u] [-c] [package|[package.]symbol[.method]]
</pre>
<p>
Doc prints the documentation comments associated with the item identified by its
arguments (a package, const, func, type, var, or method) followed by a one-line
summary of each of the first-level items &#34;under&#34; that item (package-level
declarations for a package, methods for a type, etc.).
</p>
<p>
Doc accepts zero, one, or two arguments.
</p>
<p>
Given no arguments, that is, when run as
</p>
<pre>go doc
</pre>
<p>
it prints the package documentation for the package in the current directory.
If the package is a command (package main), the exported symbols of the package
are elided from the presentation unless the -cmd flag is provided.
</p>
<p>
When run with one argument, the argument is treated as a Go-syntax-like
representation of the item to be documented. What the argument selects depends
on what is installed in GOROOT and GOPATH, as well as the form of the argument,
which is schematically one of these:
</p>
<pre>go doc &lt;pkg&gt;
go doc &lt;sym&gt;[.&lt;method&gt;]
go doc [&lt;pkg&gt;.]&lt;sym&gt;[.&lt;method&gt;]
go doc [&lt;pkg&gt;.][&lt;sym&gt;.]&lt;method&gt;
</pre>
<p>
The first item in this list matched by the argument is the one whose documentation
is printed. (See the examples below.) However, if the argument starts with a capital
letter it is assumed to identify a symbol or method in the current directory.
</p>
<p>
For packages, the order of scanning is determined lexically in breadth-first order.
That is, the package presented is the one that matches the search and is nearest
the root and lexically first at its level of the hierarchy.  The GOROOT tree is
always scanned in its entirety before GOPATH.
</p>
<p>
If there is no package specified or matched, the package in the current
directory is selected, so &#34;go doc Foo&#34; shows the documentation for symbol Foo in
the current package.
</p>
<p>
The package path must be either a qualified path or a proper suffix of a
path. The go tool&#39;s usual package mechanism does not apply: package path
elements like . and ... are not implemented by go doc.
</p>
<p>
When run with two arguments, the first must be a full package path (not just a
suffix), and the second is a symbol or symbol and method; this is similar to the
syntax accepted by godoc:
</p>
<pre>go doc &lt;pkg&gt; &lt;sym&gt;[.&lt;method&gt;]
</pre>
<p>
In all forms, when matching symbols, lower-case letters in the argument match
either case but upper-case letters match exactly. This means that there may be
multiple matches of a lower-case argument in a package if different symbols have
different cases. If this occurs, documentation for all matches is printed.
</p>
<p>
Examples:
</p>
<pre>go doc
	Show documentation for current package.
go doc Foo
	Show documentation for Foo in the current package.
	(Foo starts with a capital letter so it cannot match
	a package path.)
go doc encoding/json
	Show documentation for the encoding/json package.
go doc json
	Shorthand for encoding/json.
go doc json.Number (or go doc json.number)
	Show documentation and method summary for json.Number.
go doc json.Number.Int64 (or go doc json.number.int64)
	Show documentation for json.Number&#39;s Int64 method.
go doc cmd/doc
	Show package docs for the doc command.
go doc -cmd cmd/doc
	Show package docs and exported symbols within the doc command.
go doc template.new
	Show documentation for html/template&#39;s New function.
	(html/template is lexically before text/template)
go doc text/template.new # One argument
	Show documentation for text/template&#39;s New function.
go doc text/template new # Two arguments
	Show documentation for text/template&#39;s New function.

At least in the current tree, these invocations all print the
documentation for json.Decoder&#39;s Decode method:

go doc json.Decoder.Decode
go doc json.decoder.decode
go doc json.decode
cd go/src/encoding/json; go doc decode
</pre>
<p>
Flags:
</p>
<pre>-c
	Respect case when matching symbols.
-cmd
	Treat a command (package main) like a regular package.
	Otherwise package main&#39;s exported symbols are hidden
	when showing the package&#39;s top-level documentation.
-u
	Show documentation for unexported as well as exported
	symbols and methods.
</pre>
<h3 id="hdr-Print_Go_environment_information">Print Go environment information</h3>
<p>
Usage:
</p>
<pre>go env [var ...]
</pre>
<p>
Env prints Go environment information.
</p>
<p>
By default env prints information as a shell script
(on Windows, a batch file).  If one or more variable
names is given as arguments,  env prints the value of
each named variable on its own line.
</p>
<h3 id="hdr-Start_a_bug_report">Start a bug report</h3>
<p>
Usage:
</p>
<pre>go bug
</pre>
<p>
Bug opens the default browser and starts a new bug report.
The report includes useful system information.
</p>
<h3 id="hdr-Run_go_tool_fix_on_packages">Run go tool fix on packages</h3>
<p>
Usage:
</p>
<pre>go fix [packages]
</pre>
<p>
Fix runs the Go fix command on the packages named by the import paths.
</p>
<p>
For more about fix, see &#39;go doc cmd/fix&#39;.
For more about specifying packages, see &#39;go help packages&#39;.
</p>
<p>
To run fix with specific options, run &#39;go tool fix&#39;.
</p>
<p>
See also: go fmt, go vet.
</p>
<h3 id="hdr-Run_gofmt_on_package_sources">Run gofmt on package sources</h3>
<p>
Usage:
</p>
<pre>go fmt [-n] [-x] [packages]
</pre>
<p>
Fmt runs the command &#39;gofmt -l -w&#39; on the packages named
by the import paths.  It prints the names of the files that are modified.
</p>
<p>
For more about gofmt, see &#39;go doc cmd/gofmt&#39;.
For more about specifying packages, see &#39;go help packages&#39;.
</p>
<p>
The -n flag prints commands that would be executed.
The -x flag prints commands as they are executed.
</p>
<p>
To run gofmt with specific options, run gofmt itself.
</p>
<p>
See also: go fix, go vet.
</p>
<h3 id="hdr-Generate_Go_files_by_processing_source">Generate Go files by processing source</h3>
<p>
Usage:
</p>
<pre>go generate [-run regexp] [-n] [-v] [-x] [build flags] [file.go... | packages]
</pre>
<p>
Generate runs commands described by directives within existing
files. Those commands can run any process but the intent is to
create or update Go source files.
</p>
<p>
Go generate is never run automatically by go build, go get, go test,
and so on. It must be run explicitly.
</p>
<p>
Go generate scans the file for directives, which are lines of
the form,
</p>
<pre>//go:generate command argument...
</pre>
<p>
(note: no leading spaces and no space in &#34;//go&#34;) where command
is the generator to be run, corresponding to an executable file
that can be run locally. It must either be in the shell path
(gofmt), a fully qualified path (/usr/you/bin/mytool), or a
command alias, described below.
</p>
<p>
Note that go generate does not parse the file, so lines that look
like directives in comments or multiline strings will be treated
as directives.
</p>
<p>
The arguments to the directive are space-separated tokens or
double-quoted strings passed to the generator as individual
arguments when it is run.
</p>
<p>
Quoted strings use Go syntax and are evaluated before execution; a
quoted string appears as a single argument to the generator.
</p>
<p>
Go generate sets several variables when it runs the generator:
</p>
<pre>$GOARCH
	The execution architecture (arm, amd64, etc.)
$GOOS
	The execution operating system (linux, windows, etc.)
$GOFILE
	The base name of the file.
$GOLINE
	The line number of the directive in the source file.
$GOPACKAGE
	The name of the package of the file containing the directive.
$DOLLAR
	A dollar sign.
</pre>
<p>
Other than variable substitution and quoted-string evaluation, no
special processing such as &#34;globbing&#34; is performed on the command
line.
</p>
<p>
As a last step before running the command, any invocations of any
environment variables with alphanumeric names, such as $GOFILE or
$HOME, are expanded throughout the command line. The syntax for
variable expansion is $NAME on all operating systems.  Due to the
order of evaluation, variables are expanded even inside quoted
strings. If the variable NAME is not set, $NAME expands to the
empty string.
</p>
<p>
A directive of the form,
</p>
<pre>//go:generate -command xxx args...
</pre>
<p>
specifies, for the remainder of this source file only, that the
string xxx represents the command identified by the arguments. This
can be used to create aliases or to handle multiword generators.
For example,
</p>
<pre>//go:generate -command foo go tool foo
</pre>
<p>
specifies that the command &#34;foo&#34; represents the generator
&#34;go tool foo&#34;.
</p>
<p>
Generate processes packages in the order given on the command line,
one at a time. If the command line lists .go files, they are treated
as a single package. Within a package, generate processes the
source files in a package in file name order, one at a time. Within
a source file, generate runs generators in the order they appear
in the file, one at a time.
</p>
<p>
If any generator returns an error exit status, &#34;go generate&#34; skips
all further processing for that package.
</p>
<p>
The generator is run in the package&#39;s source directory.
</p>
<p>
Go generate accepts one specific flag:
</p>
<pre>-run=&#34;&#34;
	if non-empty, specifies a regular expression to select
	directives whose full original source text (excluding
	any trailing spaces and final newline) matches the
	expression.
</pre>
<p>
It also accepts the standard build flags including -v, -n, and -x.
The -v flag prints the names of packages and files as they are
processed.
The -n flag prints commands that would be executed.
The -x flag prints commands as they are executed.
</p>
<p>
For more about build flags, see &#39;go help build&#39;.
</p>
<p>
For more about specifying packages, see &#39;go help packages&#39;.
</p>
<h3 id="hdr-Download_and_install_packages_and_dependencies">Download and install packages and dependencies</h3>
<p>
Usage:
</p>
<pre>go get [-d] [-f] [-fix] [-insecure] [-t] [-u] [build flags] [packages]
</pre>
<p>
Get downloads the packages named by the import paths, along with their
dependencies. It then installs the named packages, like &#39;go install&#39;.
</p>
<p>
The -d flag instructs get to stop after downloading the packages; that is,
it instructs get not to install the packages.
</p>
<p>
The -f flag, valid only when -u is set, forces get -u not to verify that
each package has been checked out from the source control repository
implied by its import path. This can be useful if the source is a local fork
of the original.
</p>
<p>
The -fix flag instructs get to run the fix tool on the downloaded packages
before resolving dependencies or building the code.
</p>
<p>
The -insecure flag permits fetching from repositories and resolving
custom domains using insecure schemes such as HTTP. Use with caution.
</p>
<p>
The -t flag instructs get to also download the packages required to build
the tests for the specified packages.
</p>
<p>
The -u flag instructs get to use the network to update the named packages
and their dependencies.  By default, get uses the network to check out
missing packages but does not use it to look for updates to existing packages.
</p>
<p>
The -v flag enables verbose progress and debug output.
</p>
<p>
Get also accepts build flags to control the installation. See &#39;go help build&#39;.
</p>
<p>
When checking out a new package, get creates the target directory
GOPATH/src/&lt;import-path&gt;. If the GOPATH contains multiple entries,
get uses the first one. For more details see: &#39;go help gopath&#39;.
</p>
<p>
When checking out or updating a package, get looks for a branch or tag
that matches the locally installed version of Go. The most important
rule is that if the local installation is running version &#34;go1&#34;, get
searches for a branch or tag named &#34;go1&#34;. If no such version exists it
retrieves the most recent version of the package.
</p>
<p>
When go get checks out or updates a Git repository,
it also updates any git submodules referenced by the repository.
</p>
<p>
Get never checks out or updates code stored in vendor directories.
</p>
<p>
For more about specifying packages, see &#39;go help packages&#39;.
</p>
<p>
For more about how &#39;go get&#39; finds source code to
download, see &#39;go help importpath&#39;.
</p>
<p>
See also: go build, go install, go clean.
</p>
<h3 id="hdr-Compile_and_install_packages_and_dependencies">Compile and install packages and dependencies</h3>
<p>
Usage:
</p>
<pre>go install [build flags] [packages]
</pre>
<p>
Install compiles and installs the packages named by the import paths,
along with their dependencies.
</p>
<p>
For more about the build flags, see &#39;go help build&#39;.
For more about specifying packages, see &#39;go help packages&#39;.
</p>
<p>
See also: go build, go get, go clean.
</p>
<h3 id="hdr-List_packages">List packages</h3>
<p>
Usage:
</p>
<pre>go list [-e] [-f format] [-json] [build flags] [packages]
</pre>
<p>
List lists the packages named by the import paths, one per line.
</p>
<p>
The default output shows the package import path:
</p>
<pre>bytes
encoding/json
github.com/gorilla/mux
golang.org/x/net/html
</pre>
<p>
The -f flag specifies an alternate format for the list, using the
syntax of package template.  The default output is equivalent to -f
&#39;&#39;. The struct being passed to the template is:
</p>
<pre>type Package struct {
    Dir           string // directory containing package sources
    ImportPath    string // import path of package in dir
    ImportComment string // path in import comment on package statement
    Name          string // package name
    Doc           string // package documentation string
    Target        string // install path
    Shlib         string // the shared library that contains this package (only set when -linkshared)
    Goroot        bool   // is this package in the Go root?
    Standard      bool   // is this package part of the standard Go library?
    Stale         bool   // would &#39;go install&#39; do anything for this package?
    StaleReason   string // explanation for Stale==true
    Root          string // Go root or Go path dir containing this package
    ConflictDir   string // this directory shadows Dir in $GOPATH
    BinaryOnly    bool   // binary-only package: cannot be recompiled from sources

    // Source files
    GoFiles        []string // .go source files (excluding CgoFiles, TestGoFiles, XTestGoFiles)
    CgoFiles       []string // .go sources files that import &#34;C&#34;
    IgnoredGoFiles []string // .go sources ignored due to build constraints
    CFiles         []string // .c source files
    CXXFiles       []string // .cc, .cxx and .cpp source files
    MFiles         []string // .m source files
    HFiles         []string // .h, .hh, .hpp and .hxx source files
    FFiles         []string // .f, .F, .for and .f90 Fortran source files
    SFiles         []string // .s source files
    SwigFiles      []string // .swig files
    SwigCXXFiles   []string // .swigcxx files
    SysoFiles      []string // .syso object files to add to archive
    TestGoFiles    []string // _test.go files in package
    XTestGoFiles   []string // _test.go files outside package

    // Cgo directives
    CgoCFLAGS    []string // cgo: flags for C compiler
    CgoCPPFLAGS  []string // cgo: flags for C preprocessor
    CgoCXXFLAGS  []string // cgo: flags for C++ compiler
    CgoFFLAGS    []string // cgo: flags for Fortran compiler
    CgoLDFLAGS   []string // cgo: flags for linker
    CgoPkgConfig []string // cgo: pkg-config names

    // Dependency information
    Imports      []string // import paths used by this package
    Deps         []string // all (recursively) imported dependencies
    TestImports  []string // imports from TestGoFiles
    XTestImports []string // imports from XTestGoFiles

    // Error information
    Incomplete bool            // this package or a dependency has an error
    Error      *PackageError   // error loading package
    DepsErrors []*PackageError // errors loading dependencies
}
</pre>
<p>
Packages stored in vendor directories report an ImportPath that includes the
path to the vendor directory (for example, &#34;d/vendor/p&#34; instead of &#34;p&#34;),
so that the ImportPath uniquely identifies a given copy of a package.
The Imports, Deps, TestImports, and XTestImports lists also contain these
expanded imports paths. See golang.org/s/go15vendor for more about vendoring.
</p>
<p>
The error information, if any, is
</p>
<pre>type PackageError struct {
    ImportStack   []string // shortest path from package named on command line to this one
    Pos           string   // position of error (if present, file:line:col)
    Err           string   // the error itself
}
</pre>
<p>
The template function &#34;join&#34; calls strings.Join.
</p>
<p>
The template function &#34;context&#34; returns the build context, defined as:
</p>
<pre>type Context struct {
	GOARCH        string   // target architecture
	GOOS          string   // target operating system
	GOROOT        string   // Go root
	GOPATH        string   // Go path
	CgoEnabled    bool     // whether cgo can be used
	UseAllFiles   bool     // use files regardless of +build lines, file names
	Compiler      string   // compiler to assume when computing target paths
	BuildTags     []string // build constraints to match in +build lines
	ReleaseTags   []string // releases the current release is compatible with
	InstallSuffix string   // suffix to use in the name of the install dir
}
</pre>
<p>
For more information about the meaning of these fields see the documentation
for the go/build package&#39;s Context type.
</p>
<p>
The -json flag causes the package data to be printed in JSON format
instead of using the template format.
</p>
<p>
The -e flag changes the handling of erroneous packages, those that
cannot be found or are malformed.  By default, the list command
prints an error to standard error for each erroneous package and
omits the packages from consideration during the usual printing.
With the -e flag, the list command never prints errors to standard
error and instead processes the erroneous packages with the usual
printing.  Erroneous packages will have a non-empty ImportPath and
a non-nil Error field; other information may or may not be missing
(zeroed).
</p>
<p>
For more about build flags, see &#39;go help build&#39;.
</p>
<p>
For more about specifying packages, see &#39;go help packages&#39;.
</p>
<h3 id="hdr-Compile_and_run_Go_program">Compile and run Go program</h3>
<p>
Usage:
</p>
<pre>go run [build flags] [-exec xprog] gofiles... [arguments...]
</pre>
<p>
Run compiles and runs the main package comprising the named Go source files.
A Go source file is defined to be a file ending in a literal &#34;.go&#34; suffix.
</p>
<p>
By default, &#39;go run&#39; runs the compiled binary directly: &#39;a.out arguments...&#39;.
If the -exec flag is given, &#39;go run&#39; invokes the binary using xprog:
</p>
<pre>&#39;xprog a.out arguments...&#39;.
</pre>
<p>
If the -exec flag is not given, GOOS or GOARCH is different from the system
default, and a program named go_$GOOS_$GOARCH_exec can be found
on the current search path, &#39;go run&#39; invokes the binary using that program,
for example &#39;go_nacl_386_exec a.out arguments...&#39;. This allows execution of
cross-compiled programs when a simulator or other execution method is
available.
</p>
<p>
For more about build flags, see &#39;go help build&#39;.
</p>
<p>
See also: go build.
</p>
<h3 id="hdr-Test_packages">Test packages</h3>
<p>
Usage:
</p>
<pre>go test [build/test flags] [packages] [build/test flags &amp; test binary flags]
</pre>
<p>
&#39;Go test&#39; automates testing the packages named by the import paths.
It prints a summary of the test results in the format:
</p>
<pre>ok   archive/tar   0.011s
FAIL archive/zip   0.022s
ok   compress/gzip 0.033s
...
</pre>
<p>
followed by detailed output for each failed package.
</p>
<p>
&#39;Go test&#39; recompiles each package along with any files with names matching
the file pattern &#34;*_test.go&#34;.
Files whose names begin with &#34;_&#34; (including &#34;_test.go&#34;) or &#34;.&#34; are ignored.
These additional files can contain test functions, benchmark functions, and
example functions.  See &#39;go help testfunc&#39; for more.
Each listed package causes the execution of a separate test binary.
</p>
<p>
Test files that declare a package with the suffix &#34;_test&#34; will be compiled as a
separate package, and then linked and run with the main test binary.
</p>
<p>
The go tool will ignore a directory named &#34;testdata&#34;, making it available
to hold ancillary data needed by the tests.
</p>
<p>
By default, go test needs no arguments.  It compiles and tests the package
with source in the current directory, including tests, and runs the tests.
</p>
<p>
The package is built in a temporary directory so it does not interfere with the
non-test installation.
</p>
<p>
In addition to the build flags, the flags handled by &#39;go test&#39; itself are:
</p>
<pre>-args
    Pass the remainder of the command line (everything after -args)
    to the test binary, uninterpreted and unchanged.
    Because this flag consumes the remainder of the command line,
    the package list (if present) must appear before this flag.

-c
    Compile the test binary to pkg.test but do not run it
    (where pkg is the last element of the package&#39;s import path).
    The file name can be changed with the -o flag.

-exec xprog
    Run the test binary using xprog. The behavior is the same as
    in &#39;go run&#39;. See &#39;go help run&#39; for details.

-i
    Install packages that are dependencies of the test.
    Do not run the test.

-o file
    Compile the test binary to the named file.
    The test still runs (unless -c or -i is specified).
</pre>
<p>
The test binary also accepts flags that control execution of the test; these
flags are also accessible by &#39;go test&#39;. See &#39;go help testflag&#39; for details.
</p>
<p>
For more about build flags, see &#39;go help build&#39;.
For more about specifying packages, see &#39;go help packages&#39;.
</p>
<p>
See also: go build, go vet.
</p>
<h3 id="hdr-Run_specified_go_tool">Run specified go tool</h3>
<p>
Usage:
</p>
<pre>go tool [-n] command [args...]
</pre>
<p>
Tool runs the go tool command identified by the arguments.
With no arguments it prints the list of known tools.
</p>
<p>
The -n flag causes tool to print the command that would be
executed but not execute it.
</p>
<p>
For more about each tool command, see &#39;go tool command -h&#39;.
</p>
<h3 id="hdr-Print_Go_version">Print Go version</h3>
<p>
Usage:
</p>
<pre>go version
</pre>
<p>
Version prints the Go version, as reported by runtime.Version.
</p>
<h3 id="hdr-Run_go_tool_vet_on_packages">Run go tool vet on packages</h3>
<p>
Usage:
</p>
<pre>go vet [-n] [-x] [build flags] [packages]
</pre>
<p>
Vet runs the Go vet command on the packages named by the import paths.
</p>
<p>
For more about vet, see &#39;go doc cmd/vet&#39;.
For more about specifying packages, see &#39;go help packages&#39;.
</p>
<p>
To run the vet tool with specific options, run &#39;go tool vet&#39;.
</p>
<p>
The -n flag prints commands that would be executed.
The -x flag prints commands as they are executed.
</p>
<p>
For more about build flags, see &#39;go help build&#39;.
</p>
<p>
See also: go fmt, go fix.
</p>
<h3 id="hdr-Calling_between_Go_and_C">Calling between Go and C</h3>
<p>
There are two different ways to call between Go and C/C++ code.
</p>
<p>
The first is the cgo tool, which is part of the Go distribution.  For
information on how to use it see the cgo documentation (go doc cmd/cgo).
</p>
<p>
The second is the SWIG program, which is a general tool for
interfacing between languages.  For information on SWIG see
<a href="http://swig.org/">http://swig.org/</a>.  When running go build, any file with a .swig
extension will be passed to SWIG.  Any file with a .swigcxx extension
will be passed to SWIG with the -c++ option.
</p>
<p>
When either cgo or SWIG is used, go build will pass any .c, .m, .s,
or .S files to the C compiler, and any .cc, .cpp, .cxx files to the C++
compiler.  The CC or CXX environment variables may be set to determine
the C or C++ compiler, respectively, to use.
</p>
<h3 id="hdr-Description_of_build_modes">Description of build modes</h3>
<p>
The &#39;go build&#39; and &#39;go install&#39; commands take a -buildmode argument which
indicates which kind of object file is to be built. Currently supported values
are:
</p>
<pre>-buildmode=archive
	Build the listed non-main packages into .a files. Packages named
	main are ignored.

-buildmode=c-archive
	Build the listed main package, plus all packages it imports,
	into a C archive file. The only callable symbols will be those
	functions exported using a cgo //export comment. Requires
	exactly one main package to be listed.

-buildmode=c-shared
	Build the listed main packages, plus all packages that they
	import, into C shared libraries. The only callable symbols will
	be those functions exported using a cgo //export comment.
	Non-main packages are ignored.

-buildmode=default
	Listed main packages are built into executables and listed
	non-main packages are built into .a files (the default
	behavior).

-buildmode=shared
	Combine all the listed non-main packages into a single shared
	library that will be used when building with the -linkshared
	option. Packages named main are ignored.

-buildmode=exe
	Build the listed main packages and everything they import into
	executables. Packages not named main are ignored.

-buildmode=pie
	Build the listed main packages and everything they import into
	position independent executables (PIE). Packages not named
	main are ignored.

-buildmode=plugin
	Build the listed main packages, plus all packages that they
	import, into a Go plugin. Packages not named main are ignored.
</pre>
<h3 id="hdr-File_types">File types</h3>
<p>
The go command examines the contents of a restricted set of files
in each directory. It identifies which files to examine based on
the extension of the file name. These extensions are:
</p>
<pre>.go
	Go source files.
.c, .h
	C source files.
	If the package uses cgo or SWIG, these will be compiled with the
	OS-native compiler (typically gcc); otherwise they will
	trigger an error.
.cc, .cpp, .cxx, .hh, .hpp, .hxx
	C++ source files. Only useful with cgo or SWIG, and always
	compiled with the OS-native compiler.
.m
	Objective-C source files. Only useful with cgo, and always
	compiled with the OS-native compiler.
.s, .S
	Assembler source files.
	If the package uses cgo or SWIG, these will be assembled with the
	OS-native assembler (typically gcc (sic)); otherwise they
	will be assembled with the Go assembler.
.swig, .swigcxx
	SWIG definition files.
.syso
	System object files.
</pre>
<p>
Files of each of these types except .syso may contain build
constraints, but the go command stops scanning for build constraints
at the first item in the file that is not a blank line or //-style
line comment. See the go/build package documentation for
more details.
</p>
<p>
Non-test Go source files can also include a //go:binary-only-package
comment, indicating that the package sources are included
for documentation only and must not be used to build the
package binary. This enables distribution of Go packages in
their compiled form alone. See the go/build package documentation
for more details.
</p>
<h3 id="hdr-GOPATH_environment_variable">GOPATH environment variable</h3>
<p>
The Go path is used to resolve import statements.
It is implemented by and documented in the go/build package.
</p>
<p>
The GOPATH environment variable lists places to look for Go code.
On Unix, the value is a colon-separated string.
On Windows, the value is a semicolon-separated string.
On Plan 9, the value is a list.
</p>
<p>
If the environment variable is unset, GOPATH defaults
to a subdirectory named &#34;go&#34; in the user&#39;s home directory
($HOME/go on Unix, %USERPROFILE%\go on Windows),
unless that directory holds a Go distribution.
Run &#34;go env GOPATH&#34; to see the current GOPATH.
</p>
<p>
See <a href="https://golang.org/wiki/SettingGOPATH">https://golang.org/wiki/SettingGOPATH</a> to set a custom GOPATH.
</p>
<p>
Each directory listed in GOPATH must have a prescribed structure:
</p>
<p>
The src directory holds source code.  The path below src
determines the import path or executable name.
</p>
<p>
The pkg directory holds installed package objects.
As in the Go tree, each target operating system and
architecture pair has its own subdirectory of pkg
(pkg/GOOS_GOARCH).
</p>
<p>
If DIR is a directory listed in the GOPATH, a package with
source in DIR/src/foo/bar can be imported as &#34;foo/bar&#34; and
has its compiled form installed to &#34;DIR/pkg/GOOS_GOARCH/foo/bar.a&#34;.
</p>
<p>
The bin directory holds compiled commands.
Each command is named for its source directory, but only
the final element, not the entire path.  That is, the
command with source in DIR/src/foo/quux is installed into
DIR/bin/quux, not DIR/bin/foo/quux.  The &#34;foo/&#34; prefix is stripped
so that you can add DIR/bin to your PATH to get at the
installed commands.  If the GOBIN environment variable is
set, commands are installed to the directory it names instead
of DIR/bin. GOBIN must be an absolute path.
</p>
<p>
Here&#39;s an example directory layout:
</p>
<pre>GOPATH=/home/user/go

/home/user/go/
    src/
        foo/
            bar/               (go code in package bar)
                x.go
            quux/              (go code in package main)
                y.go
    bin/
        quux                   (installed command)
    pkg/
        linux_amd64/
            foo/
                bar.a          (installed package object)
</pre>
<p>
Go searches each directory listed in GOPATH to find source code,
but new packages are always downloaded into the first directory
in the list.
</p>
<p>
See <a href="https://golang.org/doc/code.html">https://golang.org/doc/code.html</a> for an example.
</p>
<h3 id="hdr-Internal_Directories">Internal Directories</h3>
<p>
Code in or below a directory named &#34;internal&#34; is importable only
by code in the directory tree rooted at the parent of &#34;internal&#34;.
Here&#39;s an extended version of the directory layout above:
</p>
<pre>/home/user/go/
    src/
        crash/
            bang/              (go code in package bang)
                b.go
        foo/                   (go code in package foo)
            f.go
            bar/               (go code in package bar)
                x.go
            internal/
                baz/           (go code in package baz)
                    z.go
            quux/              (go code in package main)
                y.go
</pre>
<p>
The code in z.go is imported as &#34;foo/internal/baz&#34;, but that
import statement can only appear in source files in the subtree
rooted at foo. The source files foo/f.go, foo/bar/x.go, and
foo/quux/y.go can all import &#34;foo/internal/baz&#34;, but the source file
crash/bang/b.go cannot.
</p>
<p>
See <a href="https://golang.org/s/go14internal">https://golang.org/s/go14internal</a> for details.
</p>
<h3 id="hdr-Vendor_Directories">Vendor Directories</h3>
<p>
Go 1.6 includes support for using local copies of external dependencies
to satisfy imports of those dependencies, often referred to as vendoring.
</p>
<p>
Code below a directory named &#34;vendor&#34; is importable only
by code in the directory tree rooted at the parent of &#34;vendor&#34;,
and only using an import path that omits the prefix up to and
including the vendor element.
</p>
<p>
Here&#39;s the example from the previous section,
but with the &#34;internal&#34; directory renamed to &#34;vendor&#34;
and a new foo/vendor/crash/bang directory added:
</p>
<pre>/home/user/go/
    src/
        crash/
            bang/              (go code in package bang)
                b.go
        foo/                   (go code in package foo)
            f.go
            bar/               (go code in package bar)
                x.go
            vendor/
                crash/
                    bang/      (go code in package bang)
                        b.go
                baz/           (go code in package baz)
                    z.go
            quux/              (go code in package main)
                y.go
</pre>
<p>
The same visibility rules apply as for internal, but the code
in z.go is imported as &#34;baz&#34;, not as &#34;foo/vendor/baz&#34;.
</p>
<p>
Code in vendor directories deeper in the source tree shadows
code in higher directories. Within the subtree rooted at foo, an import
of &#34;crash/bang&#34; resolves to &#34;foo/vendor/crash/bang&#34;, not the
top-level &#34;crash/bang&#34;.
</p>
<p>
Code in vendor directories is not subject to import path
checking (see &#39;go help importpath&#39;).
</p>
<p>
When &#39;go get&#39; checks out or updates a git repository, it now also
updates submodules.
</p>
<p>
Vendor directories do not affect the placement of new repositories
being checked out for the first time by &#39;go get&#39;: those are always
placed in the main GOPATH, never in a vendor subtree.
</p>
<p>
See <a href="https://golang.org/s/go15vendor">https://golang.org/s/go15vendor</a> for details.
</p>
<h3 id="hdr-Environment_variables">Environment variables</h3>
<p>
The go command, and the tools it invokes, examine a few different
environment variables. For many of these, you can see the default
value of on your system by running &#39;go env NAME&#39;, where NAME is the
name of the variable.
</p>
<p>
General-purpose environment variables:
</p>
<pre>GCCGO
	The gccgo command to run for &#39;go build -compiler=gccgo&#39;.
GOARCH
	The architecture, or processor, for which to compile code.
	Examples are amd64, 386, arm, ppc64.
GOBIN
	The directory where &#39;go install&#39; will install a command.
GOOS
	The operating system for which to compile code.
	Examples are linux, darwin, windows, netbsd.
GOPATH
	For more details see: &#39;go help gopath&#39;.
GORACE
	Options for the race detector.
	See <a href="https://golang.org/doc/articles/race_detector.html">https://golang.org/doc/articles/race_detector.html</a>.
GOROOT
	The root of the go tree.
</pre>
<p>
Environment variables for use with cgo:
</p>
<pre>CC
	The command to use to compile C code.
CGO_ENABLED
	Whether the cgo command is supported.  Either 0 or 1.
CGO_CFLAGS
	Flags that cgo will pass to the compiler when compiling
	C code.
CGO_CPPFLAGS
	Flags that cgo will pass to the compiler when compiling
	C or C++ code.
CGO_CXXFLAGS
	Flags that cgo will pass to the compiler when compiling
	C++ code.
CGO_FFLAGS
	Flags that cgo will pass to the compiler when compiling
	Fortran code.
CGO_LDFLAGS
	Flags that cgo will pass to the compiler when linking.
CXX
	The command to use to compile C++ code.
PKG_CONFIG
	Path to pkg-config tool.
</pre>
<p>
Architecture-specific environment variables:
</p>
<pre>GOARM
	For GOARCH=arm, the ARM architecture for which to compile.
	Valid values are 5, 6, 7.
GO386
	For GOARCH=386, the floating point instruction set.
	Valid values are 387, sse2.
</pre>
<p>
Special-purpose environment variables:
</p>
<pre>GOROOT_FINAL
	The root of the installed Go tree, when it is
	installed in a location other than where it is built.
	File names in stack traces are rewritten from GOROOT to
	GOROOT_FINAL.
GO_EXTLINK_ENABLED
	Whether the linker should use external linking mode
	when using -linkmode=auto with code that uses cgo.
	Set to 0 to disable external linking mode, 1 to enable it.
GIT_ALLOW_PROTOCOL
	Defined by Git. A colon-separated list of schemes that are allowed to be used
	with git fetch/clone. If set, any scheme not explicitly mentioned will be
	considered insecure by &#39;go get&#39;.
</pre>
<h3 id="hdr-Import_path_syntax">Import path syntax</h3>
<p>
An import path (see &#39;go help packages&#39;) denotes a package stored in the local
file system.  In general, an import path denotes either a standard package (such
as &#34;unicode/utf8&#34;) or a package found in one of the work spaces (For more
details see: &#39;go help gopath&#39;).
</p>
<h3 id="hdr-Relative_import_paths">Relative import paths</h3>
<p>
An import path beginning with ./ or ../ is called a relative path.
The toolchain supports relative import paths as a shortcut in two ways.
</p>
<p>
First, a relative path can be used as a shorthand on the command line.
If you are working in the directory containing the code imported as
&#34;unicode&#34; and want to run the tests for &#34;unicode/utf8&#34;, you can type
&#34;go test ./utf8&#34; instead of needing to specify the full path.
Similarly, in the reverse situation, &#34;go test ..&#34; will test &#34;unicode&#34; from
the &#34;unicode/utf8&#34; directory. Relative patterns are also allowed, like
&#34;go test ./...&#34; to test all subdirectories. See &#39;go help packages&#39; for details
on the pattern syntax.
</p>
<p>
Second, if you are compiling a Go program not in a work space,
you can use a relative path in an import statement in that program
to refer to nearby code also not in a work space.
This makes it easy to experiment with small multipackage programs
outside of the usual work spaces, but such programs cannot be
installed with &#34;go install&#34; (there is no work space in which to install them),
so they are rebuilt from scratch each time they are built.
To avoid ambiguity, Go programs cannot use relative import paths
within a work space.
</p>
<h3 id="hdr-Remote_import_paths">Remote import paths</h3>
<p>
Certain import paths also
describe how to obtain the source code for the package using
a revision control system.
</p>
<p>
A few common code hosting sites have special syntax:
</p>
<pre>Bitbucket (Git, Mercurial)

	import &#34;bitbucket.org/user/project&#34;
	import &#34;bitbucket.org/user/project/sub/directory&#34;

GitHub (Git)

	import &#34;github.com/user/project&#34;
	import &#34;github.com/user/project/sub/directory&#34;

Launchpad (Bazaar)

	import &#34;launchpad.net/project&#34;
	import &#34;launchpad.net/project/series&#34;
	import &#34;launchpad.net/project/series/sub/directory&#34;

	import &#34;launchpad.net/~user/project/branch&#34;
	import &#34;launchpad.net/~user/project/branch/sub/directory&#34;

IBM DevOps Services (Git)

	import &#34;hub.jazz.net/git/user/project&#34;
	import &#34;hub.jazz.net/git/user/project/sub/directory&#34;
</pre>
<p>
For code hosted on other servers, import paths may either be qualified
with the version control type, or the go tool can dynamically fetch
the import path over https/http and discover where the code resides
from a &lt;meta&gt; tag in the HTML.
</p>
<p>
To declare the code location, an import path of the form
</p>
<pre>repository.vcs/path
</pre>
<p>
specifies the given repository, with or without the .vcs suffix,
using the named version control system, and then the path inside
that repository.  The supported version control systems are:
</p>
<pre>Bazaar      .bzr
Git         .git
Mercurial   .hg
Subversion  .svn
</pre>
<p>
For example,
</p>
<pre>import &#34;example.org/user/foo.hg&#34;
</pre>
<p>
denotes the root directory of the Mercurial repository at
example.org/user/foo or foo.hg, and
</p>
<pre>import &#34;example.org/repo.git/foo/bar&#34;
</pre>
<p>
denotes the foo/bar directory of the Git repository at
example.org/repo or repo.git.
</p>
<p>
When a version control system supports multiple protocols,
each is tried in turn when downloading.  For example, a Git
download tries https://, then git+ssh://.
</p>
<p>
By default, downloads are restricted to known secure protocols
(e.g. https, ssh). To override this setting for Git downloads, the
GIT_ALLOW_PROTOCOL environment variable can be set (For more details see:
&#39;go help environment&#39;).
</p>
<p>
If the import path is not a known code hosting site and also lacks a
version control qualifier, the go tool attempts to fetch the import
over https/http and looks for a &lt;meta&gt; tag in the document&#39;s HTML
&lt;head&gt;.
</p>
<p>
The meta tag has the form:
</p>
<pre>&lt;meta name=&#34;go-import&#34; content=&#34;import-prefix vcs repo-root&#34;&gt;
</pre>
<p>
The import-prefix is the import path corresponding to the repository
root. It must be a prefix or an exact match of the package being
fetched with &#34;go get&#34;. If it&#39;s not an exact match, another http
request is made at the prefix to verify the &lt;meta&gt; tags match.
</p>
<p>
The meta tag should appear as early in the file as possible.
In particular, it should appear before any raw JavaScript or CSS,
to avoid confusing the go command&#39;s restricted parser.
</p>
<p>
The vcs is one of &#34;git&#34;, &#34;hg&#34;, &#34;svn&#34;, etc,
</p>
<p>
The repo-root is the root of the version control system
containing a scheme and not containing a .vcs qualifier.
</p>
<p>
For example,
</p>
<pre>import &#34;example.org/pkg/foo&#34;
</pre>
<p>
will result in the following requests:
</p>
<pre><a href="https://example.org/pkg/foo?go-get=1">https://example.org/pkg/foo?go-get=1</a> (preferred)
<a href="http://example.org/pkg/foo?go-get=1">http://example.org/pkg/foo?go-get=1</a>  (fallback, only with -insecure)
</pre>
<p>
If that page contains the meta tag
</p>
<pre>&lt;meta name=&#34;go-import&#34; content=&#34;example.org git <a href="https://code.org/r/p/exproj">https://code.org/r/p/exproj</a>&#34;&gt;
</pre>
<p>
the go tool will verify that <a href="https://example.org/?go-get=1">https://example.org/?go-get=1</a> contains the
same meta tag and then git clone <a href="https://code.org/r/p/exproj">https://code.org/r/p/exproj</a> into
GOPATH/src/example.org.
</p>
<p>
New downloaded packages are written to the first directory listed in the GOPATH
environment variable (For more details see: &#39;go help gopath&#39;).
</p>
<p>
The go command attempts to download the version of the
package appropriate for the Go release being used.
Run &#39;go help get&#39; for more.
</p>
<h3 id="hdr-Import_path_checking">Import path checking</h3>
<p>
When the custom import path feature described above redirects to a
known code hosting site, each of the resulting packages has two possible
import paths, using the custom domain or the known hosting site.
</p>
<p>
A package statement is said to have an &#34;import comment&#34; if it is immediately
followed (before the next newline) by a comment of one of these two forms:
</p>
<pre>package math // import &#34;path&#34;
package math /* import &#34;path&#34; */
</pre>
<p>
The go command will refuse to install a package with an import comment
unless it is being referred to by that import path. In this way, import comments
let package authors make sure the custom import path is used and not a
direct path to the underlying code hosting site.
</p>
<p>
Import path checking is disabled for code found within vendor trees.
This makes it possible to copy code into alternate locations in vendor trees
without needing to update import comments.
</p>
<p>
See <a href="https://golang.org/s/go14customimport">https://golang.org/s/go14customimport</a> for details.
</p>
<h3 id="hdr-Description_of_package_lists">Description of package lists</h3>
<p>
Many commands apply to a set of packages:
</p>
<pre>go action [packages]
</pre>
<p>
Usually, [packages] is a list of import paths.
</p>
<p>
An import path that is a rooted path or that begins with
a . or .. element is interpreted as a file system path and
denotes the package in that directory.
</p>
<p>
Otherwise, the import path P denotes the package found in
the directory DIR/src/P for some DIR listed in the GOPATH
environment variable (For more details see: &#39;go help gopath&#39;).
</p>
<p>
If no import paths are given, the action applies to the
package in the current directory.
</p>
<p>
There are four reserved names for paths that should not be used
for packages to be built with the go tool:
</p>
<p>
- &#34;main&#34; denotes the top-level package in a stand-alone executable.
</p>
<p>
- &#34;all&#34; expands to all package directories found in all the GOPATH
trees. For example, &#39;go list all&#39; lists all the packages on the local
system.
</p>
<p>
- &#34;std&#34; is like all but expands to just the packages in the standard
Go library.
</p>
<p>
- &#34;cmd&#34; expands to the Go repository&#39;s commands and their
internal libraries.
</p>
<p>
Import paths beginning with &#34;cmd/&#34; only match source code in
the Go repository.
</p>
<p>
An import path is a pattern if it includes one or more &#34;...&#34; wildcards,
each of which can match any string, including the empty string and
strings containing slashes.  Such a pattern expands to all package
directories found in the GOPATH trees with names matching the
patterns.  As a special case, x/... matches x as well as x&#39;s subdirectories.
For example, net/... expands to net and packages in its subdirectories.
</p>
<p>
An import path can also name a package to be downloaded from
a remote repository.  Run &#39;go help importpath&#39; for details.
</p>
<p>
Every package in a program must have a unique import path.
By convention, this is arranged by starting each path with a
unique prefix that belongs to you.  For example, paths used
internally at Google all begin with &#39;google&#39;, and paths
denoting remote repositories begin with the path to the code,
such as &#39;github.com/user/repo&#39;.
</p>
<p>
Packages in a program need not have unique package names,
but there are two reserved package names with special meaning.
The name main indicates a command, not a library.
Commands are built into binaries and cannot be imported.
The name documentation indicates documentation for
a non-Go program in the directory. Files in package documentation
are ignored by the go command.
</p>
<p>
As a special case, if the package list is a list of .go files from a
single directory, the command is applied to a single synthesized
package made up of exactly those files, ignoring any build constraints
in those files and ignoring any other files in the directory.
</p>
<p>
Directory and file names that begin with &#34;.&#34; or &#34;_&#34; are ignored
by the go tool, as are directories named &#34;testdata&#34;.
</p>
<h3 id="hdr-Description_of_testing_flags">Description of testing flags</h3>
<p>
The &#39;go test&#39; command takes both flags that apply to &#39;go test&#39; itself
and flags that apply to the resulting test binary.
</p>
<p>
Several of the flags control profiling and write an execution profile
suitable for &#34;go tool pprof&#34;; run &#34;go tool pprof -h&#34; for more
information.  The --alloc_space, --alloc_objects, and --show_bytes
options of pprof control how the information is presented.
</p>
<p>
The following flags are recognized by the &#39;go test&#39; command and
control the execution of any test:
</p>
<pre>-bench regexp
    Run (sub)benchmarks matching a regular expression.
    The given regular expression is split into smaller ones by
    top-level &#39;/&#39;, where each must match the corresponding part of a
    benchmark&#39;s identifier.
    By default, no benchmarks run. To run all benchmarks,
    use &#39;-bench .&#39; or &#39;-bench=.&#39;.

-benchtime t
    Run enough iterations of each benchmark to take t, specified
    as a time.Duration (for example, -benchtime 1h30s).
    The default is 1 second (1s).

-count n
    Run each test and benchmark n times (default 1).
    If -cpu is set, run n times for each GOMAXPROCS value.
    Examples are always run once.

-cover
    Enable coverage analysis.

-covermode set,count,atomic
    Set the mode for coverage analysis for the package[s]
    being tested. The default is &#34;set&#34; unless -race is enabled,
    in which case it is &#34;atomic&#34;.
    The values:
	set: bool: does this statement run?
	count: int: how many times does this statement run?
	atomic: int: count, but correct in multithreaded tests;
		significantly more expensive.
    Sets -cover.

-coverpkg pkg1,pkg2,pkg3
    Apply coverage analysis in each test to the given list of packages.
    The default is for each test to analyze only the package being tested.
    Packages are specified as import paths.
    Sets -cover.

-cpu 1,2,4
    Specify a list of GOMAXPROCS values for which the tests or
    benchmarks should be executed.  The default is the current value
    of GOMAXPROCS.

-parallel n
    Allow parallel execution of test functions that call t.Parallel.
    The value of this flag is the maximum number of tests to run
    simultaneously; by default, it is set to the value of GOMAXPROCS.
    Note that -parallel only applies within a single test binary.
    The &#39;go test&#39; command may run tests for different packages
    in parallel as well, according to the setting of the -p flag
    (see &#39;go help build&#39;).

-run regexp
    Run only those tests and examples matching the regular expression.
    For tests the regular expression is split into smaller ones by
    top-level &#39;/&#39;, where each must match the corresponding part of a
    test&#39;s identifier.

-short
    Tell long-running tests to shorten their run time.
    It is off by default but set during all.bash so that installing
    the Go tree can run a sanity check but not spend time running
    exhaustive tests.

-timeout t
    If a test runs longer than t, panic.
    The default is 10 minutes (10m).

-v
    Verbose output: log all tests as they are run. Also print all
    text from Log and Logf calls even if the test succeeds.
</pre>
<p>
The following flags are also recognized by &#39;go test&#39; and can be used to
profile the tests during execution:
</p>
<pre>-benchmem
    Print memory allocation statistics for benchmarks.

-blockprofile block.out
    Write a goroutine blocking profile to the specified file
    when all tests are complete.
    Writes test binary as -c would.

-blockprofilerate n
    Control the detail provided in goroutine blocking profiles by
    calling runtime.SetBlockProfileRate with n.
    See &#39;go doc runtime.SetBlockProfileRate&#39;.
    The profiler aims to sample, on average, one blocking event every
    n nanoseconds the program spends blocked.  By default,
    if -test.blockprofile is set without this flag, all blocking events
    are recorded, equivalent to -test.blockprofilerate=1.

-coverprofile cover.out
    Write a coverage profile to the file after all tests have passed.
    Sets -cover.

-cpuprofile cpu.out
    Write a CPU profile to the specified file before exiting.
    Writes test binary as -c would.

-memprofile mem.out
    Write a memory profile to the file after all tests have passed.
    Writes test binary as -c would.

-memprofilerate n
    Enable more precise (and expensive) memory profiles by setting
    runtime.MemProfileRate.  See &#39;go doc runtime.MemProfileRate&#39;.
    To profile all memory allocations, use -test.memprofilerate=1
    and pass --alloc_space flag to the pprof tool.

-mutexprofile mutex.out
    Write a mutex contention profile to the specified file
    when all tests are complete.
    Writes test binary as -c would.

-mutexprofilefraction n
    Sample 1 in n stack traces of goroutines holding a
    contended mutex.

-outputdir directory
    Place output files from profiling in the specified directory,
    by default the directory in which &#34;go test&#34; is running.

-trace trace.out
    Write an execution trace to the specified file before exiting.
</pre>
<p>
Each of these flags is also recognized with an optional &#39;test.&#39; prefix,
as in -test.v. When invoking the generated test binary (the result of
&#39;go test -c&#39;) directly, however, the prefix is mandatory.
</p>
<p>
The &#39;go test&#39; command rewrites or removes recognized flags,
as appropriate, both before and after the optional package list,
before invoking the test binary.
</p>
<p>
For instance, the command
</p>
<pre>go test -v -myflag testdata -cpuprofile=prof.out -x
</pre>
<p>
will compile the test binary and then run it as
</p>
<pre>pkg.test -test.v -myflag testdata -test.cpuprofile=prof.out
</pre>
<p>
(The -x flag is removed because it applies only to the go command&#39;s
execution, not to the test itself.)
</p>
<p>
The test flags that generate profiles (other than for coverage) also
leave the test binary in pkg.test for use when analyzing the profiles.
</p>
<p>
When &#39;go test&#39; runs a test binary, it does so from within the
corresponding package&#39;s source code directory. Depending on the test,
it may be necessary to do the same when invoking a generated test
binary directly.
</p>
<p>
The command-line package list, if present, must appear before any
flag not known to the go test command. Continuing the example above,
the package list would have to appear before -myflag, but could appear
on either side of -v.
</p>
<p>
To keep an argument for a test binary from being interpreted as a
known flag or a package name, use -args (see &#39;go help test&#39;) which
passes the remainder of the command line through to the test binary
uninterpreted and unaltered.
</p>
<p>
For instance, the command
</p>
<pre>go test -v -args -x -v
</pre>
<p>
will compile the test binary and then run it as
</p>
<pre>pkg.test -test.v -x -v
</pre>
<p>
Similarly,
</p>
<pre>go test -args math
</pre>
<p>
will compile the test binary and then run it as
</p>
<pre>pkg.test math
</pre>
<p>
In the first example, the -x and the second -v are passed through to the
test binary unchanged and with no effect on the go command itself.
In the second example, the argument math is passed through to the test
binary, instead of being interpreted as the package list.
</p>
<h3 id="hdr-Description_of_testing_functions">Description of testing functions</h3>
<p>
The &#39;go test&#39; command expects to find test, benchmark, and example functions
in the &#34;*_test.go&#34; files corresponding to the package under test.
</p>
<p>
A test function is one named TestXXX (where XXX is any alphanumeric string
not starting with a lower case letter) and should have the signature,
</p>
<pre>func TestXXX(t *testing.T) { ... }
</pre>
<p>
A benchmark function is one named BenchmarkXXX and should have the signature,
</p>
<pre>func BenchmarkXXX(b *testing.B) { ... }
</pre>
<p>
An example function is similar to a test function but, instead of using
*testing.T to report success or failure, prints output to os.Stdout.
If the last comment in the function starts with &#34;Output:&#34; then the output
is compared exactly against the comment (see examples below). If the last
comment begins with &#34;Unordered output:&#34; then the output is compared to the
comment, however the order of the lines is ignored. An example with no such
comment is compiled but not executed. An example with no text after
&#34;Output:&#34; is compiled, executed, and expected to produce no output.
</p>
<p>
Godoc displays the body of ExampleXXX to demonstrate the use
of the function, constant, or variable XXX.  An example of a method M with
receiver type T or *T is named ExampleT_M.  There may be multiple examples
for a given function, constant, or variable, distinguished by a trailing _xxx,
where xxx is a suffix not beginning with an upper case letter.
</p>
<p>
Here is an example of an example:
</p>
<pre>func ExamplePrintln() {
	Println(&#34;The output of\nthis example.&#34;)
	// Output: The output of
	// this example.
}
</pre>
<p>
Here is another example where the ordering of the output is ignored:
</p>
<pre>func ExamplePerm() {
	for _, value := range Perm(4) {
		fmt.Println(value)
	}

	// Unordered output: 4
	// 2
	// 1
	// 3
	// 0
}
</pre>
<p>
The entire test file is presented as the example when it contains a single
example function, at least one other function, type, variable, or constant
declaration, and no test or benchmark functions.
</p>
<p>
See the documentation of the testing package for more information.
</p>

<div id="footer">
Build version go1.8.<br>
Except as <a href="https://developers.google.com/site-policies#restrictions">noted</a>,
the content of this page is licensed under the
Creative Commons Attribution 3.0 License,
and code is licensed under a <a href="/LICENSE">BSD license</a>.<br>
<a href="/doc/tos.html">Terms of Service</a> | 
<a href="http://www.google.com/intl/en/policies/privacy/">Privacy Policy</a>
</div>

</div><!-- .container -->
</div><!-- #page -->

<!-- TODO(adonovan): load these from <head> using "defer" attribute? -->
<script type="text/javascript" src="/serverpush/static/jquery.min.js?{{.CacheBust}}"></script>
<script type="text/javascript" src="/serverpush/static/playground.js?{{.CacheBust}}"></script>
<script>var goVersion = "go1.8";</script>
<script type="text/javascript" src="/serverpush/static/godocs.js?{{.CacheBust}}"></script>
</body>
</html>
`))

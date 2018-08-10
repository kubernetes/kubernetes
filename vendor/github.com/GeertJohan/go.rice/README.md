## go.rice

[![Build Status](https://travis-ci.org/GeertJohan/go.rice.png)](https://travis-ci.org/GeertJohan/go.rice)
[![Godoc](https://img.shields.io/badge/godoc-go.rice-blue.svg?style=flat-square)](https://godoc.org/github.com/GeertJohan/go.rice)

go.rice is a [Go](http://golang.org) package that makes working with resources such as html,js,css,images and templates very easy. During development `go.rice` will load required files directly from disk. Upon deployment it is easy to add all resource files to a executable using the `rice` tool, without changing the source code for your package. go.rice provides several methods to add resources to a binary.

### What does it do?
The first thing go.rice does is finding the correct absolute path for your resource files. Say you are executing go binary in your home directory, but your `html-files` are located in `$GOPATH/src/yourApplication/html-files`. `go.rice` will lookup the correct path for that directory (relative to the location of yourApplication). The only thing you have to do is include the resources using `rice.FindBox("html-files")`.

This only works when the source is available to the machine executing the binary. Which is always the case when the binary was installed with `go get` or `go install`. It might occur that you wish to simply provide a binary, without source. The `rice` tool analyses source code and finds call's to `rice.FindBox(..)` and adds the required directories to the executable binary. There are several methods to add these resources. You can 'embed' by generating go source code, or append the resource to the executable as zip file. In both cases `go.rice` will detect the embedded or appended resources and load those, instead of looking up files from disk.

### Installation

Use `go get` to install the package the `rice` tool.
```
go get github.com/GeertJohan/go.rice
go get github.com/GeertJohan/go.rice/rice
```

### Package usage

Import the package: `import "github.com/GeertJohan/go.rice"`

**Serving a static content folder over HTTP with a rice Box**
```go
http.Handle("/", http.FileServer(rice.MustFindBox("http-files").HTTPBox()))
http.ListenAndServe(":8080", nil)
```

**Service a static content folder over HTTP at a non-root location**
```go
box := rice.MustFindBox("cssfiles")
cssFileServer := http.StripPrefix("/css/", http.FileServer(box.HTTPBox()))
http.Handle("/css/", cssFileServer)
http.ListenAndServe(":8080", nil)
```

Note the *trailing slash* in `/css/` in both the call to
`http.StripPrefix` and `http.Handle`.

**Loading a template**
```go
// find a rice.Box
templateBox, err := rice.FindBox("example-templates")
if err != nil {
	log.Fatal(err)
}
// get file contents as string
templateString, err := templateBox.String("message.tmpl")
if err != nil {
	log.Fatal(err)
}
// parse and execute the template
tmplMessage, err := template.New("message").Parse(templateString)
if err != nil {
	log.Fatal(err)
}
tmplMessage.Execute(os.Stdout, map[string]string{"Message": "Hello, world!"})

```

Never call `FindBox()` or `MustFindBox()` from an `init()` function, as the boxes might have not been loaded at that time.

### Tool usage
The `rice` tool lets you add the resources to a binary executable so the files are not loaded from the filesystem anymore. This creates a 'standalone' executable. There are several ways to add the resources to a binary, each has pro's and con's but all will work without requiring changes to the way you load the resources.

#### embed-go
**Embed resources by generating Go source code**

This method must be executed before building. It generates a single Go source file called *rice-box.go* for each package, that is compiled by the go compiler into the binary.

The downside with this option is that the generated go source files can become very large, which will slow down compilation and require lots of memory to compile.

Execute the following commands:
```
rice embed-go
go build
```

*A Note on Symbolic Links*: `embed-go` uses the `os.Walk` function
from the standard library.  The `os.Walk` function does **not** follow
symbolic links.  So, when creating a box, be aware that any symbolic
links inside your box's directory will not be followed.  **However**,
if the box itself is a symbolic link, its actual location will be
resolved first and then walked.  In summary, if your box location is a
symbolic link, it will be followed but none of the symbolic links in
the box will be followed.

#### embed-syso
**Embed resources by generating a coff .syso file and some .go source code**

** This method is experimental and should not be used for production systems just yet **

This method must be executed before building. It generates a COFF .syso file and Go source file that are compiled by the go compiler into the binary.

Execute the following commands:
```
rice embed-syso
go build
```

#### append
**Append resources to executable as zip file**

This method changes an already built executable. It appends the resources as zip file to the binary. It makes compilation a lot faster and can be used with large resource files.

Downsides for appending are that it requires `zip` to be installed and does not provide a working Seek method.

Run the following commands to create a standalone executable.
```
go build -o example
rice append --exec example
```

**Note: requires zip command to be installed**

On windows, install zip from http://gnuwin32.sourceforge.net/packages/zip.htm or cygwin/msys toolsets.

#### Help information
Run `rice -h` for information about all options.

You can run the -h option for each sub-command, e.g. `rice append -h`.

### Order of precedence
When opening a new box, the rice package tries to locate the resources in the following order:

 - embedded in generated go source
 - appended as zip
 - 'live' from filesystem


### License
This project is licensed under a Simplified BSD license. Please read the [LICENSE file][license].

### TODO & Development
This package is not completed yet. Though it already provides working embedding, some important featuers are still missing.
 - implement Readdir() correctly on virtualDir
 - in-code TODO's
 - find boxes in imported packages

Less important stuff:
 - idea, os/arch dependent embeds. rice checks if embedding file has _os_arch or build flags. If box is not requested by file without buildflags, then the buildflags are applied to the embed file.

### Package documentation

You will find package documentation at [godoc.org/github.com/GeertJohan/go.rice][godoc].


 [license]: https://github.com/GeertJohan/go.rice/blob/master/LICENSE
 [godoc]: http://godoc.org/github.com/GeertJohan/go.rice

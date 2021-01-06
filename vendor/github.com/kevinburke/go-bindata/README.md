## bindata

*This fork is maintained by Kevin Burke, and is the version trusted by Homebrew.
Changes made include:*

- Atomic writes; generated file cannot be read while partially complete.

- Better encoding of files that contain characters in the Unicode format range.

- Generated file reports file sizes.

- Generated code is run through go fmt.

- SHA256 hashes are computed for all files and stored in the binary. You can
use this to detect in-memory corruption and to provide easy cache-busting
mechanisms.

- Added AssetString and MustAssetString functions.

- ByName is not public.

- Some errors in file writes were unchecked, but are now checked.

- File modes are stored in octal (0644) instead of nonsensical decimal (420)

This package converts any file into manageable Go source code. Useful for
embedding binary data into a go program. The file data is optionally gzip
compressed before being converted to a raw byte slice.

It comes with a command line tool in the `go-bindata` subdirectory. This tool
offers a set of command line options, used to customize the output being
generated.


### Installation

On Macs, you can install the binary using [Homebrew](https://brew.sh):

```
brew install go-bindata
```

You can also download a binary from the [releases page][releases]. Switch in
your GOOS for the word "linux" below, and the latest version for the version
listed below:

```
curl --silent --location --output /usr/local/bin/go-bindata https://github.com/kevinburke/go-bindata/releases/download/v3.11.0/go-bindata-linux-amd64
chmod 755 /usr/local/bin/go-bindata
```

Alternatively, you can download the source code, if you have a working Go
installation:

```
go get -u github.com/kevinburke/go-bindata/...
```

[releases]: https://github.com/kevinburke/go-bindata/releases

### Usage

Conversion is done on one or more sets of files. They are all embedded in a new
Go source file, along with a table of contents and an `Asset` function,
which allows quick access to the asset, based on its name.

The simplest invocation generates a `bindata.go` file in the current
working directory. It includes all assets from the `data` directory.

	$ go-bindata data/

To include all input sub-directories recursively, use the ellipsis postfix
as defined for Go import paths. Otherwise it will only consider assets in the
input directory itself.

	$ go-bindata data/...

To specify the name of the output file being generated, use the `-o` option:

	$ go-bindata -o myfile.go data/

Multiple input directories can be specified if necessary.

	$ go-bindata dir1/... /path/to/dir2/... dir3


The following paragraphs detail some of the command line options which can be
supplied to `go-bindata`. Refer to the `testdata/out` directory for various
output examples from the assets in `testdata/in`. Each example uses different
command line options.

To ignore files, pass in regexes using -ignore, for example:

    $ go-bindata -ignore=\\.gitignore data/...

### Accessing an asset

To access asset data, we use the `Asset(string) ([]byte, error)` function which
is included in the generated output.

	data, err := Asset("pub/style/foo.css")
	if err != nil {
		// Asset was not found.
	}

	// use asset data


### Debug vs Release builds

When invoking the program with the `-debug` flag, the generated code does
not actually include the asset data. Instead, it generates function stubs
which load the data from the original file on disk. The asset API remains
identical between debug and release builds, so your code will not have to
change.

This is useful during development when you expect the assets to change often.
The host application using these assets uses the same API in both cases and
will not have to care where the actual data comes from.

An example is a Go webserver with some embedded, static web content like
HTML, JS and CSS files. While developing it, you do not want to rebuild the
whole server and restart it every time you make a change to a bit of
javascript. You just want to build and launch the server once. Then just press
refresh in the browser to see those changes. Embedding the assets with the
`debug` flag allows you to do just that. When you are finished developing and
ready for deployment, just re-invoke `go-bindata` without the `-debug` flag.
It will now embed the latest version of the assets.


### Lower memory footprint

Using the `-nomemcopy` flag, will alter the way the output file is generated.
It will employ a hack that allows us to read the file data directly from
the compiled program's `.rodata` section. This ensures that when we
call our generated function, we omit unnecessary memcopies.

The downside of this, is that it requires dependencies on the `reflect` and
`unsafe` packages. These may be restricted on platforms like AppEngine and
thus prevent you from using this mode.

Another disadvantage is that the byte slice we create, is strictly read-only.
For most use-cases this is not a problem, but if you ever try to alter the
returned byte slice, a runtime panic is thrown. Use this mode only on target
platforms where memory constraints are an issue.

The default behavior is to use the old code generation method. This prevents the
two previously mentioned issues, but will employ at least one extra memcopy and
thus increase memory requirements.

For instance, consider the following two examples:

This would be the default mode, using an extra memcopy but gives a safe
implementation without dependencies on `reflect` and `unsafe`:

```go
func myfile() []byte {
    return []byte{0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a}
}
```

Here is the same functionality, but uses the `.rodata` hack.
The byte slice returned from this example can not be written to without
generating a runtime error.

```go
var _myfile = "\x89\x50\x4e\x47\x0d\x0a\x1a"

func myfile() []byte {
    var empty [0]byte
    sx := (*reflect.StringHeader)(unsafe.Pointer(&_myfile))
    b := empty[:]
    bx := (*reflect.SliceHeader)(unsafe.Pointer(&b))
    bx.Data = sx.Data
    bx.Len = len(_myfile)
    bx.Cap = bx.Len
    return b
}
```


### Optional compression

When the `-nocompress` flag is given, the supplied resource is *not* GZIP
compressed before being turned into Go code. The data should still be accessed
through a function call, so nothing changes in the usage of the generated file.

This feature is useful if you do not care for compression, or the supplied
resource is already compressed. Doing it again would not add any value and may
even increase the size of the data.

The default behavior of the program is to use compression.


### Path prefix stripping

The keys used in the `_bindata` map, are the same as the input file name
passed to `go-bindata`. This includes the path. In most cases, this is not
desireable, as it puts potentially sensitive information in your code base.
For this purpose, the tool supplies another command line flag `-prefix`.
This accepts a portion of a path name, which should be stripped off from
the map keys and function names.

For example, running without the `-prefix` flag, we get:

	$ go-bindata /path/to/templates/

	_bindata["/path/to/templates/foo.html"] = path_to_templates_foo_html

Running with the `-prefix` flag, we get:

	$ go-bindata -prefix "/path/to/" /path/to/templates/

	_bindata["templates/foo.html"] = templates_foo_html

### Build tags

With the optional `-tags` flag, you can specify any go build tags that
must be fulfilled for the output file to be included in a build. This
is useful when including binary data in multiple formats, where the desired
format is specified at build time with the appropriate tags.

The tags are appended to a `// +build` line in the beginning of the output file
and must follow the build tags syntax specified by the go tool.

### Related projects

[go-bindata-assetfs](https://github.com/elazarl/go-bindata-assetfs#readme) -
implements `http.FileSystem` interface. Allows you to serve assets with `net/http`.

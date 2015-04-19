# tar-fs

filesystem bindings for [tar-stream](https://github.com/mafintosh/tar-stream).

```
npm install tar-fs
```

[![build status](https://secure.travis-ci.org/mafintosh/tar-fs.png)](http://travis-ci.org/mafintosh/tar-fs)

## Usage

tar-fs allows you to pack directories into tarballs and extract tarballs into directories.

``` js
var tar = require('tar-fs')
var fs = require('fs')

// packing a directory
tar.pack('./my-directory').pipe(fs.createWriteStream('my-tarball.tar'))

// extracting a directory
fs.createReadStream('my-other-tarball.tar').pipe(tar.extract('./my-other-directory'))
```

To ignore various files when packing or extracting add a ignore function to the options

``` js
var pack = tar.pack('./my-directory', {
  ignore: function(name) {
    return path.extname(name) === '.bin' // ignore .bin files when packing
  }
})

var extract = tar.extract('./my-other-directory', {
  ignore: function(name) {
    return path.extname(name) === '.bin' // ignore .bin files inside the tarball when extracing
  }
})
```

You can also specify which entries to pack using the `entries` option

```js
var pack = tar.pack('./my-directory', {
  entries: ['file1', 'subdir/file2'] // only the specific entries will be packed
})
```

If you want to modify the headers when packing/extracting add a map function to the options

``` js
var pack = tar.pack('./my-directory', {
  map: function(header) {
    header.name = 'prefixed/'+header.name
    return header
  }
})

var extract = tar.extract('./my-directory', {
  map: function(header) {
    header.name = 'another-prefix/'+header.name
    return header
  }
})
```

Similarly you can use `mapStream` incase you wanna modify the input/output file streams

``` js
var pack = tar.pack('./my-directory', {
  mapStream: function(fileStream, header) {
    if (path.extname(header.file) === '.js') {
      return fileStream.pipe(someTransform)
    }
    return fileStream;
  }
})

var extract = tar.extract('./my-directory', {
  mapStream: function(fileStream, header) {
    if (path.extname(header.file) === '.js') {
      return fileStream.pipe(someTransform)
    }
    return fileStream;
  }
})
```

Set `options.fmode` and `options.dmode` to ensure that files/directories extracted have the corresponding modes

``` js
var extract = tar.extract('./my-directory', {
  dmode: 0555, // all dirs and files should be readable
  fmode: 0444
})
```

It can be useful to use `dmode` and `fmode` if you are packing/unpacking tarballs between *nix/windows to ensure that all files/directories unpacked are readable.

Set `options.strict` to `false` if you want to ignore errors due to unsupported entry types (like device files)

To dereference symlinks (pack the contents of the symlink instead of the link itself) set `options.dereference` to `true`.

## Copy a directory

Copying a directory with permissions and mtime intact is as simple as

``` js
tar.pack('source-directory').pipe(tar.extract('dest-directory'))
```

## Performance

Packing and extracting a 6.1 GB with 2496 directories and 2398 files yields the following results on my Macbook Air.
[See the benchmark here](https://gist.github.com/mafintosh/8102201)

* tar-fs: 34.261 ms
* [node-tar](https://github.com/isaacs/node-tar): 366.123 ms (or 10x slower)

## License

MIT

# tar-stream

tar-stream is a streaming tar parser and generator and nothing else. It is streams2 and operates purely using streams which means you can easily extract/parse tarballs without ever hitting the file system.

```
npm install tar-stream
```

[![build status](https://secure.travis-ci.org/mafintosh/tar-stream.png)](http://travis-ci.org/mafintosh/tar-stream)

## Usage

tar-stream exposes two streams, [pack](https://github.com/mafintosh/tar-stream#packing) which creates tarballs and [extract](https://github.com/mafintosh/tar-stream#extracting) which extracts tarballs. To [modify an existing tarball](https://github.com/mafintosh/tar-stream#modifying-existing-tarballs) use both.


It implementes USTAR with additional support for pax extended headers. It should be compatible with all popular tar distributions out there (gnutar, bsdtar etc)

## Related

If you want to pack/unpack directories on the file system check out [tar-fs](https://github.com/mafintosh/tar-fs) which provides file system bindings to this module.

## Packing

To create a pack stream use `tar.pack()` and call `pack.entry(header, [callback])` to add tar entries.

``` js
var tar = require('tar-stream')
var pack = tar.pack() // pack is a streams2 stream

// add a file called my-test.txt with the content "Hello World!"
pack.entry({ name: 'my-test.txt' }, 'Hello World!')

// add a file called my-stream-test.txt from a stream
var entry = pack.entry({ name: 'my-stream-test.txt', size: 11 }, function(err) {
  // the stream was added
  // no more entries
  pack.finalize()
})

entry.write('hello')
entry.write(' ')
entry.write('world')
entry.end()

// pipe the pack stream somewhere
pack.pipe(process.stdout)
```

## Extracting

To extract a stream use `tar.extract()` and listen for `extract.on('entry', header, stream, callback)`

``` js
var extract = tar.extract()

extract.on('entry', function(header, stream, callback) {
  // header is the tar header
  // stream is the content body (might be an empty stream)
  // call next when you are done with this entry

  stream.on('end', function() {
    callback() // ready for next entry
  })

  stream.resume() // just auto drain the stream
})

extract.on('finish', function() {
  // all entries read
})

pack.pipe(extract)
```

## Headers

The header object using in `entry` should contain the following properties.
Most of these values can be found by stat'ing a file.

``` js
{
  name: 'path/to/this/entry.txt',
  size: 1314,        // entry size. defaults to 0
  mode: 0644,        // entry mode. defaults to to 0755 for dirs and 0644 otherwise
  mtime: new Date(), // last modified date for entry. defaults to now.
  type: 'file',      // type of entry. defaults to file. can be:
                     // file | link | symlink | directory | block-device
                     // character-device | fifo | contigious-file
  linkname: 'path',  // linked file name
  uid: 0,            // uid of entry owner. defaults to 0
  gid: 0,            // gid of entry owner. defaults to 0
  uname: 'maf',      // uname of entry owner. defaults to null
  gname: 'staff',    // gname of entry owner. defaults to null
  devmajor: 0,       // device major version. defaults to 0
  devminor: 0        // device minor version. defaults to 0
}
```

## Modifying existing tarballs

Using tar-stream it is easy to rewrite paths / change modes etc in an existing tarball.

``` js
var extract = tar.extract()
var pack = tar.pack()
var path = require('path')

extract.on('entry', function(header, stream, callback) {
  // let's prefix all names with 'tmp'
  header.name = path.join('tmp', header.name)
  // write the new entry to the pack stream
  stream.pipe(pack.entry(header, callback))
})

extract.on('finish', function() {
  // all entries done - lets finalize it
  pack.finalize()
})

// pipe the old tarball to the extractor
oldTarballStream.pipe(extract)

// pipe the new tarball the another stream
pack.pipe(newTarballStream)
```

## Performance

[See tar-fs for a performance comparison with node-tar](https://github.com/mafintosh/tar-fs/blob/master/README.md#performance)

# License

MIT

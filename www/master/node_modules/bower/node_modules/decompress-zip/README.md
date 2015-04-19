# decompress-zip [![Build Status](https://travis-ci.org/bower/decompress-zip.svg?branch=master)](https://travis-ci.org/bower/decompress-zip)

> Extract files from a ZIP archive


## Usage

### .extract(options)

Extracts the contents of the ZIP archive `file`.

Returns an EventEmitter with two possible events - `error` on an error, and `extract` when the extraction has completed. The value passed to the `extract` event is a basic log of each file and how it was compressed.

**Options**
- **path** *String* - Path to extract into (default `.`)
- **follow** *Boolean* - If true, rather than create stored symlinks as symlinks make a shallow copy of the target instead (default `false`)
- **filter** *Function* - A function that will be called once for each file in the archive. It takes one argument which is an object containing details of the file. Return true for any file that you want to extract, and false otherwise. (default `null`)
- **strip** *Number* - Remove leading folders in the path structure. Equivalent to `--strip-components` for tar.

```js
var DecompressZip = require('decompress-zip');
var unzipper = new DecompressZip(filename)

unzipper.on('error', function (err) {
    console.log('Caught an error');
});

unzipper.on('extract', function (log) {
    console.log('Finished extracting');
});

unzipper.on('progress', function (fileIndex, fileCount) {
    console.log('Extracted file ' + (fileIndex + 1) + ' of ' + fileCount);
});

unzipper.extract({
    path: 'some/path',
    filter: function (file) {
        return file.type !== "SymbolicLink";
    }
});
```

If `path` does not exist, decompress-zip will attempt to create it first.

### .list()

Much like extract, except:
- the success event is `list`
- the data for the event is an array of paths
- no files are actually extracted
- there are no options

```js
var DecompressZip = require('decompress-zip');
var unzipper = new DecompressZip(filename)

unzipper.on('error', function (err) {
    console.log('Caught an error');
});

unzipper.on('list', function (files) {
    console.log('The archive contains:');
    console.log(files);
});

unzipper.list();
```


## License

MIT © Bower team

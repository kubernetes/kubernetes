# Split (matcher)

[![build status](https://secure.travis-ci.org/dominictarr/split.png)](http://travis-ci.org/dominictarr/split)

Break up a stream and reassemble it so that each line is a chunk. matcher may be a `String`, or a `RegExp` 

Example, read every line in a file ...

``` js
  fs.createReadStream(file)
    .pipe(split())
    .on('data', function (line) {
      //each chunk now is a seperate line!
    })

```

`split` takes the same arguments as `string.split` except it defaults to '/\r?\n/' instead of ',', and the optional `limit` paremeter is ignored.
[String#split](https://developer.mozilla.org/en/JavaScript/Reference/Global_Objects/String/split)

# NDJ - Newline Delimited Json

`split` accepts a function which transforms each line.

``` js
fs.createReadStream(file)
  .pipe(split(JSON.parse))
  .on('data', function (obj) {
    //each chunk now is a a js object
  })
  .on('error', function (err) {
    //syntax errors will land here
    //note, this ends the stream.
  })
```

# License

MIT

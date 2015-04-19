var touch = require("../touch")
  , fs = require("fs")
  , path = require("path")
  , nopt = require("nopt")
  , types = { atime: Boolean
            , mtime: Boolean
            , time: Date
            , ref: path
            , nocreate: Boolean
            , force: Boolean }
  , shorthands = { a: "--atime"
                 , m: "--mtime"
                 , r: "--ref"
                 , t: "--time"
                 , c: "--nocreate"
                 , f: "--force" }

var options = nopt(types, shorthands)

var files = options.argv.remain
delete options.argv

files.forEach(function (file) {
  touch(file, options, function (er) {
    if (er) {
      console.error("bad touch!")
      throw er
    }
    console.error(file, fs.statSync(file))
  })
})

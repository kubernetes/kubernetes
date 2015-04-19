var IgnoreFile = require("../")
, fs = require('fs')

// set the ignores just for this test
var c = require("./common.js")
c.ignores({ ".gitignore": ["a/b/c/abc"] })
c.ignores({ ".ignore": ["*", "!a/b/c/abc"] })

// the only files we expect to see
var expected =
  [ "/a"
  , "/a/b"
  , "/a/b/c"
  , "/a/b/c/abc" ]

var originalReadFile = fs.readFile
, parallelCount = 0
, firstCall

// Overwrite fs.readFile so that when .gitignore and .ignore are read in
// parallel, .ignore will always be read first.
fs.readFile = function (filename, options, callback) {
  if (typeof options === 'function') {
    callback = options
    options = false
  }

  parallelCount++

  process.nextTick(function () {
    if (parallelCount > 1) {
      if (!firstCall) {
        return firstCall = function (cb) {
          originalReadFile(filename, options, function (err, data) {
            callback(err, data)
            if (cb) cb()
          })
        }
      }

      if (filename.indexOf('.gitignore') !== -1) {
        firstCall(function () {
          originalReadFile(filename, options, callback)
        })
      } else {
        originalReadFile(filename, options, function (err, data) {
          callback(err, data)
          firstCall()
        })
      }
    } else {
      originalReadFile(filename, options, callback)
      parallelCount = 0
    }
  })
}

require("tap").test("read file order", function (t) {
  t.pass("start")

  IgnoreFile({ path: __dirname + "/fixtures"
             , ignoreFiles: [".gitignore", ".ignore"] })
    .on("ignoreFile", function (e) {
      console.error("ignore file!", e)
    })
    .on("child", function (e) {
      var p = e.path.substr(e.root.path.length)
      var i = expected.indexOf(p)
      if (i === -1) {
        t.fail("unexpected file found", {f: p})
      } else {
        t.pass(p)
        expected.splice(i, 1)
      }
    })
    .on("close", function () {
      fs.readFile = originalReadFile
      t.notOk(expected.length, "all expected files should be seen")
      t.end()
    })
})

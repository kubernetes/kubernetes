// ignore most things
var IgnoreFile = require("../")

// set the ignores just for this test
var c = require("./common.js")
c.ignores({ ".ignore": ["*", "!a/b/c/.abc", "!/c/b/a/cba"] })

// the only files we expect to see
var expected =
  [ "/a/b/c/.abc"
  , "/a"
  , "/a/b"
  , "/a/b/c"
  , "/c/b/a/cba"
  , "/c"
  , "/c/b"
  , "/c/b/a" ]

require("tap").test("basic ignore rules", function (t) {
  t.pass("start")

  IgnoreFile({ path: __dirname + "/fixtures"
             , ignoreFiles: [".ignore"] })
    .on("ignoreFile", function (e) {
      console.error("ignore file!", e)
    })
    .on("child", function (e) {
      var p = e.path.substr(e.root.path.length)
      var i = expected.indexOf(p)
      if (i === -1) {
        t.fail("unexpected file found", {file: p})
      } else {
        t.pass(p)
        expected.splice(i, 1)
      }
    })
    .on("close", function () {
      t.notOk(expected.length, "all expected files should be seen")
      t.end()
    })
})

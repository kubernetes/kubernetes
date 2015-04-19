// ignore most things
var IgnoreFile = require("../")

// set the ignores just for this test
var c = require("./common.js")
c.ignores(
  { ".ignore": ["*", "a", "c", "!a/b/c/.abc", "!/c/b/a/cba"]
  , "a/.ignore": [ "!*", ".ignore" ] // unignore everything
  , "a/a/.ignore": [ "*" ] // re-ignore everything
  , "a/b/.ignore": [ "*", "!/c/.abc" ] // original unignore
  , "a/c/.ignore": [ "*" ] // ignore everything again
  , "c/b/a/.ignore": [ "!cba", "!.cba", "!/a{bc,cb}" ]
  })

// the only files we expect to see
var expected =
  [ "/a"
  , "/a/a"
  , "/a/b"
  , "/a/b/c"
  , "/a/b/c/.abc"
  , "/a/c"
  , "/c"
  , "/c/b"
  , "/c/b/a"
  , "/c/b/a/cba"
  , "/c/b/a/.cba"
  , "/c/b/a/abc"
  , "/c/b/a/acb" ]

require("tap").test("basic ignore rules", function (t) {
  t.pass("start")

  IgnoreFile({ path: __dirname + "/fixtures"
             , ignoreFiles: [".ignore"] })
    .on("child", function (e) {
      var p = e.path.substr(e.root.path.length)
      var i = expected.indexOf(p)
      if (i === -1) {
        console.log("not ok "+p)
        t.fail("unexpected file found", {found: p})
      } else {
        t.pass(p)
        expected.splice(i, 1)
      }
    })
    .on("close", function () {
      t.deepEqual(expected, [], "all expected files should be seen")
      t.end()
    })
})

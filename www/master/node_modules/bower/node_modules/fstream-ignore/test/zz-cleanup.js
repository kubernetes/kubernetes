var tap = require("tap")
, rimraf = require("rimraf")
, path = require("path")

tap.test("remove fixtures", function (t) {
  rimraf(path.resolve(__dirname, "fixtures"), function (er) {
    t.ifError(er, "remove fixtures")
    t.end()
  })
})

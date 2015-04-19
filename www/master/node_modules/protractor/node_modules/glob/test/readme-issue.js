var test = require("tap").test
var glob = require("../")

var mkdirp = require("mkdirp")
var fs = require("fs")
var rimraf = require("rimraf")
var dir = __dirname + "/package"

test("setup", function (t) {
  mkdirp.sync(dir)
  fs.writeFileSync(dir + "/package.json", "{}", "ascii")
  fs.writeFileSync(dir + "/README", "x", "ascii")
  t.pass("setup done")
  t.end()
})

test("glob", function (t) {
  var opt = {
    cwd: dir,
    nocase: true,
    mark: true
  }

  glob("README?(.*)", opt, function (er, files) {
    if (er)
      throw er
    t.same(files, ["README"])
    t.end()
  })
})

test("cleanup", function (t) {
  rimraf.sync(dir)
  t.pass("clean")
  t.end()
})

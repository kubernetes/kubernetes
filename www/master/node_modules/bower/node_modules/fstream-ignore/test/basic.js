var IgnoreFile = require("../")

// set the ignores just for this test
var c = require("./common.js")
c.ignores({ "a/.basic-ignore": ["b/", "aca"] })

// the files that we expect to not see
var notAllowed =
  [ /^\/a\/b\/.*/
  , /^\/a\/.*\/aca$/ ]


require("tap").test("basic ignore rules", function (t) {
  t.pass("start")

  IgnoreFile({ path: __dirname + "/fixtures"
             , ignoreFiles: [".basic-ignore"] })
    .on("ignoreFile", function (e) {
      console.error("ignore file!", e)
    })
    .on("child", function (e) {
      var p = e.path.substr(e.root.path.length)
      notAllowed.forEach(function (na) {
        t.dissimilar(p, na)
      })
    })
    .on("close", t.end.bind(t))
})

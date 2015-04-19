var rimraf = require("../rimraf")
  , path = require("path")
rimraf(path.join(__dirname, "target"), function (er) {
  if (er) throw er
})

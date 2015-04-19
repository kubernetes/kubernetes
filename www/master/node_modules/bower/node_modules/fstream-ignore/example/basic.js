var Ignore = require("../")
Ignore({ path: __dirname
       , ignoreFiles: [".ignore", ".gitignore"]
       })
  .on("child", function (c) {
    console.error(c.path.substr(c.root.path.length + 1))
    c.on("ignoreFile", onIgnoreFile)
  })
  .on("ignoreFile", onIgnoreFile)

function onIgnoreFile (e) {
  console.error("adding ignore file", e.path)
}

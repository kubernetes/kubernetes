if (require.main === module) {
  console.log("0..1")
  console.log("ok 1 trivial pass")
  return
}

var fs = require("fs")
var path = require("path")
var rimraf = require("rimraf")

exports.ignores = ignores
exports.writeIgnoreFile = writeIgnoreFile
exports.writeIgnores = writeIgnores
exports.clearIgnores = clearIgnores

function writeIgnoreFile (file, rules) {
  file = path.resolve(__dirname, "fixtures", file)
  if (Array.isArray(rules)) {
    rules = rules.join("\n")
  }
  fs.writeFileSync(file, rules)
  console.error(file, rules)
}

function writeIgnores (set) {
  Object.keys(set).forEach(function (f) {
    writeIgnoreFile(f, set[f])
  })
}

function clearIgnores (set) {
  Object.keys(set).forEach(function (file) {
    fs.unlinkSync(path.resolve(__dirname, "fixtures", file))
  })
}

function ignores (set) {
  writeIgnores(set)
  process.on("exit", clearIgnores.bind(null, set))
}

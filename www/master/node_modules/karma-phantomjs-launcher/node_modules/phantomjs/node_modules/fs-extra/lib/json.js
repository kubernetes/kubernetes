var fs = require('graceful-fs')
var path = require('path')
var jsonFile = require('jsonfile')
var mkdir = require('./mkdir')

function outputJsonSync(file, data) {
  var dir = path.dirname(file)

  if (!fs.existsSync(dir))
    mkdir.mkdirsSync(dir)

  jsonFile.writeFileSync(file, data)
}

function outputJson(file, data, callback) {
  var dir = path.dirname(file)

  fs.exists(dir, function(itDoes) {
    if (itDoes) return jsonFile.writeFile(file, data, callback)

    mkdir.mkdirs(dir, function(err) {
      if (err) return callback(err)
      jsonFile.writeFile(file, data, callback)
    })
  })
}

module.exports = {
  outputJsonSync: outputJsonSync,
  outputJson: outputJson
}

var path = require('path')
var fs = require('graceful-fs')
var mkdir = require('./mkdir')

function createFile (file, callback) {
  function makeFile() {
    fs.writeFile(file, '', function(err) {
      if (err) return callback(err)
      callback()
    })
  }

  fs.exists(file, function(fileExists) {
    if (fileExists) return callback()
    var dir = path.dirname(file)
    fs.exists(dir, function(dirExists) {
      if (dirExists) return makeFile()
      mkdir.mkdirs(dir, function(err) {
        if (err) return callback(err)
        makeFile()
      })
    })
  })
}

function createFileSync (file) {
  if (fs.existsSync(file)) return

  var dir = path.dirname(file)
  if (!fs.existsSync(dir))
    mkdir.mkdirsSync(dir)

  fs.writeFileSync(file, '')
}

module.exports = {
  createFile: createFile,
  createFileSync: createFileSync
}

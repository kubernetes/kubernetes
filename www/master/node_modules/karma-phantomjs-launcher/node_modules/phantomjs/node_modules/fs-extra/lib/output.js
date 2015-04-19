var path = require('path')
var fs = require('graceful-fs')
var mkdir = require('./mkdir')

function outputFile (file, data, encoding, callback) {
  if (typeof encoding === 'function') {
    callback = encoding
    encoding = 'utf8'
  }

  var dir = path.dirname(file)
  fs.exists(dir, function(itDoes) {
    if (itDoes) return fs.writeFile(file, data, encoding, callback)
    
    mkdir.mkdirs(dir, function(err) {
      if (err) return callback(err)
      
      fs.writeFile(file, data, encoding, callback)
    })
  })
}

function outputFileSync (file, data, encoding) {
  var dir = path.dirname(file)
  if (fs.existsSync(dir)) 
    return fs.writeFileSync.apply(fs, arguments)
  mkdir.mkdirsSync(dir)
  fs.writeFileSync.apply(fs, arguments)
}

module.exports = {
  outputFile: outputFile,
  outputFileSync: outputFileSync
}

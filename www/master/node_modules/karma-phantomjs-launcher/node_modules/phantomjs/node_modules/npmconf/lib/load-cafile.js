module.exports = loadCAFile

var fs = require('fs')

function loadCAFile(cafilePath, cb) {
  if (!cafilePath)
    return process.nextTick(cb)

  fs.readFile(cafilePath, 'utf8', afterCARead.bind(this))

  function afterCARead(er, cadata) {
    if (er)
      return cb(er)

    var delim = '-----END CERTIFICATE-----'
    var output

    output = cadata
      .split(delim)
      .filter(function(xs) {
        return !!xs.trim()
      })
      .map(function(xs) {
        return xs.trimLeft() + delim
      })

    this.set('ca', output)
    cb(null)
  }

}

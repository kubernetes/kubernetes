// most of this code was written by Andrew Kelley
// licensed under the BSD license: see
// https://github.com/andrewrk/node-mv/blob/master/package.json

// this needs a cleanup

var fs = require('graceful-fs')
var ncp = require('./_copy').ncp
var path = require('path')
var rimraf = require('rimraf')
var mkdirp = require('./mkdir').mkdirs

function mv(source, dest, options, callback){
  if (typeof options === 'function') {
    callback = options
    options = {}
  }

  var shouldMkdirp = !!options.mkdirp
  var clobber = options.clobber !== false
  var limit = options.limit || 16

  if (shouldMkdirp) {
    mkdirs()
  } else {
    doRename()
  }

  function mkdirs() {
    mkdirp(path.dirname(dest), function(err) {
      if (err) return callback(err)
      doRename()
    })
  }

  function doRename() {
    if (clobber) {
      fs.rename(source, dest, function(err) {
        if (!err) return callback()

        if (err.code === 'ENOTEMPTY') {
          rimraf(dest, function(err) {
            if (err) return callback(err)
            options.clobber = false // just clobbered it, no need to do it again
            mv(source, dest, options, callback)
          })
          return
        }

        if (err.code !== 'EXDEV') return callback(err)
        moveFileAcrossDevice(source, dest, clobber, limit, callback)
      })
    } else {
      fs.link(source, dest, function(err) {
        if (err) {
          if (err.code === 'EXDEV') {
            moveFileAcrossDevice(source, dest, clobber, limit, callback)
            return
          }
          if (err.code === 'EISDIR' || err.code === 'EPERM') {
            moveDirAcrossDevice(source, dest, clobber, limit, callback)
            return
          }
          callback(err)
          return
        }
        fs.unlink(source, callback)
      })
    }
  }
}

function moveFileAcrossDevice(source, dest, clobber, limit, callback) {
  var outFlags = clobber ? 'w' : 'wx'
  var ins = fs.createReadStream(source)
  var outs = fs.createWriteStream(dest, {flags: outFlags})

  ins.on('error', function(err) {
    ins.destroy()
    outs.destroy()
    outs.removeListener('close', onClose)

    // may want to create a directory but `out` line above
    // creates an empty file for us: See #108
    // don't care about error here
    fs.unlink(dest, function() {
      // note: `err` here is from the input stream errror
      if (err.code === 'EISDIR' || err.code === 'EPERM') {
        moveDirAcrossDevice(source, dest, clobber, limit, callback)
      } else {
        callback(err)
      }
    })
  })

  outs.on('error', function(err) {
    ins.destroy()
    outs.destroy()
    outs.removeListener('close', onClose)
    callback(err)
  })

  outs.once('close', onClose)
  ins.pipe(outs)

  function onClose() {
    fs.unlink(source, callback)
  }
}

function moveDirAcrossDevice(source, dest, clobber, limit, callback) {
  var options = {
    stopOnErr: true,
    clobber: false,
    limit: limit,
  }

  function startNcp() {
    ncp(source, dest, options, function(errList) {
      if (errList) return callback(errList[0])
      rimraf(source, callback)
    })
  }

  if (clobber) {
    rimraf(dest, function(err) {
      if (err) return callback(err)
      startNcp()
    })
  } else {
    startNcp()
  }
}

module.exports = mv


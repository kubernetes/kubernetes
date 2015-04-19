var fs = require('graceful-fs')
var path = require('path')

var octal_0777 = parseInt('0777', 8)

function mkdirs(p, opts, f, made) {
  if (typeof opts === 'function') {
    f = opts
    opts = {}
  }
  else if (!opts || typeof opts !== 'object') {
    opts = { mode: opts }
  }

  var mode = opts.mode
  var xfs = opts.fs || fs

  if (mode === undefined) {
    mode = octal_0777 & (~process.umask())
  }
  if (!made) made = null

  var cb = f || function () {}
  p = path.resolve(p)

  xfs.mkdir(p, mode, function (er) {
    if (!er) {
      made = made || p
      return cb(null, made)
    }
    switch (er.code) {
      case 'ENOENT':
        if (path.dirname(p) == p) return cb(er)
        mkdirs(path.dirname(p), opts, function (er, made) {
          if (er) cb(er, made)
          else mkdirs(p, opts, cb, made)
        })
        break

      // In the case of any other error, just see if there's a dir
      // there already.  If so, then hooray!  If not, then something
      // is borked.
      default:
        xfs.stat(p, function (er2, stat) {
          // if the stat fails, then that's super weird.
          // let the original error be the failure reason.
          if (er2 || !stat.isDirectory()) cb(er, made)
          else cb(null, made)
        })
        break
    }
  })
}

function mkdirsSync (p, opts, made) {
  if (!opts || typeof opts !== 'object') {
    opts = { mode: opts }
  }

  var mode = opts.mode
  var xfs = opts.fs || fs

  if (mode === undefined) {
    mode = octal_0777 & (~process.umask())
  }
  if (!made) made = null

  p = path.resolve(p)

  try {
    xfs.mkdirSync(p, mode)
    made = made || p
  }
  catch (err0) {
    switch (err0.code) {
      case 'ENOENT' :
        made = mkdirsSync(path.dirname(p), opts, made)
        mkdirsSync(p, opts, made)
        break

      // In the case of any other error, just see if there's a dir
      // there already.  If so, then hooray!  If not, then something
      // is borked.
      default:
        var stat
        try {
          stat = xfs.statSync(p)
        }
        catch (err1) {
          throw err0
        }
        if (!stat.isDirectory()) throw err0
        break
    }
  }

  return made
}

module.exports = {
  mkdirs: mkdirs,
  mkdirsSync: mkdirsSync
}

var fs = require("fs")
  , cons = require("constants")

module.exports = touch
touch.touchSync = touch.sync = function (f, options) {
  return touch(f, options)
}

touch.ftouch = ftouch
touch.ftouchSync = function (fd, options) {
  return ftouch(fd, options)
}

function validOpts (options) {
  options = Object.create(options || {})

  // {mtime: true}, {ctime: true}
  // If set to something else, then treat as epoch ms value
  var now = new Date(options.time || Date.now())
  if (!options.atime && !options.mtime) {
    options.atime = options.mtime = now
  } else if (true === options.atime) {
    options.atime = now
  } else if (true === options.mtime) {
    options.mtime = now
  }

  var oflags = 0
  if (!options.force) {
    oflags = oflags | cons.O_RDWR
  }
  if (!options.nocreate) {
    oflags = oflags | cons.O_CREAT
  }

  options.oflags = oflags
  return options
}

function optionsRef (then, arg, options, cb) {
  if (!options.ref) return then(arg, options, cb)

  return cb
       ? fs.stat(options.ref, optionsRefcb(then, arg, options, cb))
       : optionsRefcb(then, arg, options)(null, fs.statSync(options.ref))
}

function optionsRefcb (then, arg, options, cb) { return function (er, s) {
  if (er) {
    er.path = er.file = options.ref
    return cb(er)
  }
  options.atime = options.atime && s.atime.getTime()
  options.mtime = options.mtime && s.mtime.getTime()

  // so we don't keep doing this.
  options.ref = null

  return then(arg, options, cb)
}}

function touch (f, options, cb) {
  if (typeof options === "function") cb = options, options = null
  options = validOpts(options)
  return optionsRef(touch_, f, validOpts(options), cb)
}

function touch_ (f, options, cb) {
  return openThenF(f, options, cb)
}

function openThenF (f, options, cb) {
  options.closeAfter = true
  return cb
       ? fs.open(f, options.oflags, openThenFcb(options, cb))
       : openThenFcb(options)(null, fs.openSync(f, options.oflags))
}

function openThenFcb (options, cb) { return function (er, fd) {
  if (er) {
    if (fd && options.closeAfter) fs.close(fd, function () {})
    return cb(er)
  }
  return ftouch(fd, options, cb)
}}

function ftouch (fd, options, cb) {
  if (typeof options === "function") cb = options, options = null
  return optionsRef(ftouch_, fd, validOpts(options), cb)
}

function ftouch_ (fd, options, cb) {
  // still not set.  leave as what the file already has.
  return fstatThenFutimes(fd, options, cb)
}

function fstatThenFutimes (fd, options, cb) {
  if (options.atime && options.mtime) return thenFutimes(fd, options, cb)

  return cb
       ? fs.fstat(fd, fstatThenFutimescb(fd, options, cb))
       : fstatThenFutimescb(fd, options)(null, fs.fstatSync(fd))
}

function fstatThenFutimescb (fd, options, cb) { return function (er, s) {
  if (er) {
    if (options.closeAfter) fs.close(fd, function () {})
    return cb(er)
  }
  options.atime = options.atime || s.atime.getTime()
  options.mtime = options.mtime || s.mtime.getTime()
  return thenFutimes(fd, options, cb)
}}

function thenFutimes (fd, options, cb) {
  if (typeof options.atime === "object") {
    options.atime = options.atime.getTime()
  }
  if (typeof options.mtime === "object") {
    options.mtime = options.mtime.getTime()
  }

  var a = parseInt(options.atime / 1000, 10)
    , m = parseInt(options.mtime / 1000, 10)
  return cb
       ? fs.futimes(fd, a, m, thenFutimescb(fd, options, cb))
       : thenFutimescb(fd, options)(null, fs.futimesSync(fd, a, m))
}

function thenFutimescb (fd, options, cb) { return function (er, res) {
  if (er) {
    if (options.closeAfter) fs.close(fd, function () {})
    return cb(er)
  }
  return finish(fd, options, res, cb)
}}

function finish (fd, options, res, cb) {
  return options.closeAfter ? finishClose(fd, options, res, cb)
       : cb ? cb(null, res)
       : res
}

function finishClose (fd, options, res, cb) {
  return cb
       ? fs.close(fd, finishClosecb(res, options, cb))
       : finishClosecb(res, options)(null, fs.closeSync(fd))
}

function finishClosecb (res, options, cb) { return function (er) {
  if (er) return cb(er)
  options.closeAfter = null
  return finish(null, options, res, cb)
}}

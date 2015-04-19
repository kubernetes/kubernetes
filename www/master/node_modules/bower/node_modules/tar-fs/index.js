var tar = require('tar-stream')
var pump = require('pump')
var mkdirp = require('mkdirp')
var fs = require('fs')
var path = require('path')
var os = require('os')

var win32 = os.platform() === 'win32'

var noop = function() {}

var echo = function(name) {
  return name
}

var normalize = !win32 ? echo : function(name) {
  return name.replace(/\\/g, '/')
}

var statAll = function(fs, stat, cwd, ignore, entries) {
  var queue = entries || ['.']

  return function loop(callback) {
    if (!queue.length) return callback()
    var next = queue.shift()
    var nextAbs = path.join(cwd, next)

    stat(nextAbs, function(err, stat) {
      if (err) return callback(err)

      if (!stat.isDirectory()) return callback(null, next, stat)

      fs.readdir(nextAbs, function(err, files) {
        if (err) return callback(err)

        for (var i = 0; i < files.length; i++) {
          if (!ignore(path.join(cwd, next, files[i]))) queue.push(path.join(next, files[i]))
        }

        callback(null, next, stat)
      })
    })
  }
}

var strip = function(map, level) {
  return function(header) {
    header.name = header.name.split('/').slice(level).join('/')
    if (header.linkname) header.linkname = header.linkname.split('/').slice(level).join('/')
    return map(header)
  }
}

exports.pack = function(cwd, opts) {
  if (!cwd) cwd = '.'
  if (!opts) opts = {}

  var xfs = opts.fs || fs
  var ignore = opts.ignore || opts.filter || noop
  var map = opts.map || noop
  var mapStream = opts.mapStream || echo
  var statNext = statAll(xfs, opts.dereference ? xfs.stat : xfs.lstat, cwd, ignore, opts.entries)
  var strict = opts.strict !== false
  var pack = tar.pack()

  if (opts.strip) map = strip(map, opts.strip)

  var onlink = function(filename, header) {
    xfs.readlink(path.join(cwd, filename), function(err, linkname) {
      if (err) return pack.destroy(err)
      header.linkname = normalize(linkname)
      pack.entry(header, onnextentry)
    })
  }

  var onstat = function(err, filename, stat) {
    if (err) return pack.destroy(err)
    if (!filename) return pack.finalize()

    if (stat.isSocket()) return onnextentry() // tar does not support sockets...

    var header = {
      name: normalize(filename),
      mode: stat.mode,
      mtime: stat.mtime,
      size: stat.size,
      type: 'file',
      uid: stat.uid,
      gid: stat.gid
    }

    header = map(header) || header

    if (stat.isDirectory()) {
      header.size = 0
      header.type = 'directory'
      return pack.entry(header, onnextentry)
    }

    if (stat.isSymbolicLink()) {
      header.size = 0
      header.type = 'symlink'
      return onlink(filename, header)
    }

    // TODO: add fifo etc...

    if (!stat.isFile()) {
      if (strict) return pack.destroy(new Error('unsupported type for '+filename))
      return onnextentry()
    }

    var entry = pack.entry(header, onnextentry)
    if (!entry) return
    var rs = xfs.createReadStream(path.join(cwd, filename))

    pump(mapStream(rs, header), entry)
  }

  var onnextentry = function(err) {
    if (err) return pack.destroy(err)
    statNext(onstat)
  }

  onnextentry()

  return pack
}

var head = function(list) {
  return list.length ? list[list.length-1] : null
}

var processGetuid = function() {
  return process.getuid ? process.getuid() : -1
}

var processUmask = function() {
  return process.umask ? process.umask() : 0
}

exports.extract = function(cwd, opts) {
  if (!cwd) cwd = '.'
  if (!opts) opts = {}

  var xfs = opts.fs || fs
  var ignore = opts.ignore || opts.filter || noop
  var map = opts.map || noop
  var mapStream = opts.mapStream || echo
  var own = opts.chown !== false && !win32 && processGetuid() === 0
  var extract = tar.extract()
  var stack = []
  var now = new Date()
  var umask = typeof opts.umask === 'number' ? ~opts.umask : ~processUmask()
  var dmode = typeof opts.dmode === 'number' ? opts.dmode : 0
  var fmode = typeof opts.fmode === 'number' ? opts.fmode : 0
  var strict = opts.strict !== false

  if (opts.strip) map = strip(map, opts.strip)

  if (opts.readable) {
    dmode |= 0555
    fmode |= 0444
  }
  if (opts.writable) {
    dmode |= 0333
    fmode |= 0222
  }

  var utimesParent = function(name, cb) { // we just set the mtime on the parent dir again everytime we write an entry
    var top
    while ((top = head(stack)) && name.slice(0, top[0].length) !== top[0]) stack.pop()
    if (!top) return cb()
    xfs.utimes(top[0], now, top[1], cb)
  }

  var utimes = function(name, header, cb) {
    if (opts.utimes === false) return cb()

    if (header.type === 'directory') return xfs.utimes(name, now, header.mtime, cb)
    if (header.type === 'symlink') return utimesParent(name, cb) // TODO: how to set mtime on link?

    xfs.utimes(name, now, header.mtime, function(err) {
      if (err) return cb(err)
      utimesParent(name, cb)
    })
  }

  var chperm = function(name, header, cb) {
    var link = header.type === 'symlink'
    var chmod = link ? xfs.lchmod : xfs.chmod
    var chown = link ? xfs.lchown : xfs.chown

    if (!chmod) return cb()
    chmod(name, (header.mode | (header.type === 'directory' ? dmode : fmode)) & umask, function(err) {
      if (err) return cb(err)
      if (!own) return cb()
      if (!chown) return cb()
      chown(name, header.uid, header.gid, cb)
    })
  }

  extract.on('entry', function(header, stream, next) {
    header = map(header) || header
    var name = path.join(cwd, path.join('/', header.name))

    if (ignore(name)) {
      stream.resume()
      return next()
    }

    var stat = function(err) {
      if (err) return next(err)
      if (win32) return next()
      utimes(name, header, function(err) {
        if (err) return next(err)
        chperm(name, header, next)
      })
    }

    var onlink = function() {
      if (win32) return next() // skip symlinks on win for now before it can be tested
      xfs.unlink(name, function() {
        xfs.symlink(header.linkname, name, stat)
      })
    }

    var onfile = function() {
      var ws = xfs.createWriteStream(name)

      pump(mapStream(stream, header), ws, function(err) {
        if (err) return next(err)
        ws.on('close', stat)
      })
    }

    if (header.type === 'directory') {
      stack.push([name, header.mtime])
      return mkdirp(name, {fs:xfs}, stat)
    }

    mkdirp(path.dirname(name), {fs:xfs}, function(err) {
      if (err) return next(err)
      if (header.type === 'symlink') return onlink()

      if (header.type !== 'file') {
        if (strict) return next(new Error('unsupported type for '+name+' ('+header.type+')'))
        stream.resume()
        return next()
      }

      onfile()
    })
  })

  return extract
}

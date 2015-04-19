// Basically just a wrapper around an fs.ReadStream

module.exports = FileReader

var fs = require("graceful-fs")
  , fstream = require("../fstream.js")
  , Reader = fstream.Reader
  , inherits = require("inherits")
  , mkdir = require("mkdirp")
  , Reader = require("./reader.js")
  , EOF = {EOF: true}
  , CLOSE = {CLOSE: true}

inherits(FileReader, Reader)

function FileReader (props) {
  // console.error("    FR create", props.path, props.size, new Error().stack)
  var me = this
  if (!(me instanceof FileReader)) throw new Error(
    "FileReader must be called as constructor.")

  // should already be established as a File type
  // XXX Todo: preserve hardlinks by tracking dev+inode+nlink,
  // with a HardLinkReader class.
  if (!((props.type === "Link" && props.Link) ||
        (props.type === "File" && props.File))) {
    throw new Error("Non-file type "+ props.type)
  }

  me._buffer = []
  me._bytesEmitted = 0
  Reader.call(me, props)
}

FileReader.prototype._getStream = function () {
  var me = this
    , stream = me._stream = fs.createReadStream(me._path, me.props)

  if (me.props.blksize) {
    stream.bufferSize = me.props.blksize
  }

  stream.on("open", me.emit.bind(me, "open"))

  stream.on("data", function (c) {
    // console.error("\t\t%d %s", c.length, me.basename)
    me._bytesEmitted += c.length
    // no point saving empty chunks
    if (!c.length) return
    else if (me._paused || me._buffer.length) {
      me._buffer.push(c)
      me._read()
    } else me.emit("data", c)
  })

  stream.on("end", function () {
    if (me._paused || me._buffer.length) {
      // console.error("FR Buffering End", me._path)
      me._buffer.push(EOF)
      me._read()
    } else {
      me.emit("end")
    }

    if (me._bytesEmitted !== me.props.size) {
      me.error("Didn't get expected byte count\n"+
               "expect: "+me.props.size + "\n" +
               "actual: "+me._bytesEmitted)
    }
  })

  stream.on("close", function () {
    if (me._paused || me._buffer.length) {
      // console.error("FR Buffering Close", me._path)
      me._buffer.push(CLOSE)
      me._read()
    } else {
      // console.error("FR close 1", me._path)
      me.emit("close")
    }
  })

  stream.on("error", function (e) {
    me.emit("error", e);
  });

  me._read()
}

FileReader.prototype._read = function () {
  var me = this
  // console.error("FR _read", me._path)
  if (me._paused) {
    // console.error("FR _read paused", me._path)
    return
  }

  if (!me._stream) {
    // console.error("FR _getStream calling", me._path)
    return me._getStream()
  }

  // clear out the buffer, if there is one.
  if (me._buffer.length) {
    // console.error("FR _read has buffer", me._buffer.length, me._path)
    var buf = me._buffer
    for (var i = 0, l = buf.length; i < l; i ++) {
      var c = buf[i]
      if (c === EOF) {
        // console.error("FR Read emitting buffered end", me._path)
        me.emit("end")
      } else if (c === CLOSE) {
        // console.error("FR Read emitting buffered close", me._path)
        me.emit("close")
      } else {
        // console.error("FR Read emitting buffered data", me._path)
        me.emit("data", c)
      }

      if (me._paused) {
        // console.error("FR Read Re-pausing at "+i, me._path)
        me._buffer = buf.slice(i)
        return
      }
    }
    me._buffer.length = 0
  }
  // console.error("FR _read done")
  // that's about all there is to it.
}

FileReader.prototype.pause = function (who) {
  var me = this
  // console.error("FR Pause", me._path)
  if (me._paused) return
  who = who || me
  me._paused = true
  if (me._stream) me._stream.pause()
  me.emit("pause", who)
}

FileReader.prototype.resume = function (who) {
  var me = this
  // console.error("FR Resume", me._path)
  if (!me._paused) return
  who = who || me
  me.emit("resume", who)
  me._paused = false
  if (me._stream) me._stream.resume()
  me._read()
}

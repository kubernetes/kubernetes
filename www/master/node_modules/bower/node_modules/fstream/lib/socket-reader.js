// Just get the stats, and then don't do anything.
// You can't really "read" from a socket.  You "connect" to it.
// Mostly, this is here so that reading a dir with a socket in it
// doesn't blow up.

module.exports = SocketReader

var fs = require("graceful-fs")
  , fstream = require("../fstream.js")
  , inherits = require("inherits")
  , mkdir = require("mkdirp")
  , Reader = require("./reader.js")

inherits(SocketReader, Reader)

function SocketReader (props) {
  var me = this
  if (!(me instanceof SocketReader)) throw new Error(
    "SocketReader must be called as constructor.")

  if (!(props.type === "Socket" && props.Socket)) {
    throw new Error("Non-socket type "+ props.type)
  }

  Reader.call(me, props)
}

SocketReader.prototype._read = function () {
  var me = this
  if (me._paused) return
  // basically just a no-op, since we got all the info we have
  // from the _stat method
  if (!me._ended) {
    me.emit("end")
    me.emit("close")
    me._ended = true
  }
}

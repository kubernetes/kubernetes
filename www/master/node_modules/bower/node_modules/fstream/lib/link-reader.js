// Basically just a wrapper around an fs.readlink
//
// XXX: Enhance this to support the Link type, by keeping
// a lookup table of {<dev+inode>:<path>}, so that hardlinks
// can be preserved in tarballs.

module.exports = LinkReader

var fs = require("graceful-fs")
  , fstream = require("../fstream.js")
  , inherits = require("inherits")
  , mkdir = require("mkdirp")
  , Reader = require("./reader.js")

inherits(LinkReader, Reader)

function LinkReader (props) {
  var me = this
  if (!(me instanceof LinkReader)) throw new Error(
    "LinkReader must be called as constructor.")

  if (!((props.type === "Link" && props.Link) ||
        (props.type === "SymbolicLink" && props.SymbolicLink))) {
    throw new Error("Non-link type "+ props.type)
  }

  Reader.call(me, props)
}

// When piping a LinkReader into a LinkWriter, we have to
// already have the linkpath property set, so that has to
// happen *before* the "ready" event, which means we need to
// override the _stat method.
LinkReader.prototype._stat = function (currentStat) {
  var me = this
  fs.readlink(me._path, function (er, linkpath) {
    if (er) return me.error(er)
    me.linkpath = me.props.linkpath = linkpath
    me.emit("linkpath", linkpath)
    Reader.prototype._stat.call(me, currentStat)
  })
}

LinkReader.prototype._read = function () {
  var me = this
  if (me._paused) return
  // basically just a no-op, since we got all the info we need
  // from the _stat method
  if (!me._ended) {
    me.emit("end")
    me.emit("close")
    me._ended = true
  }
}

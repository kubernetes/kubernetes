// It is expected that, when .add() returns false, the consumer
// of the DirWriter will pause until a "drain" event occurs. Note
// that this is *almost always going to be the case*, unless the
// thing being written is some sort of unsupported type, and thus
// skipped over.

module.exports = DirWriter

var fs = require("graceful-fs")
  , fstream = require("../fstream.js")
  , Writer = require("./writer.js")
  , inherits = require("inherits")
  , mkdir = require("mkdirp")
  , path = require("path")
  , collect = require("./collect.js")

inherits(DirWriter, Writer)

function DirWriter (props) {
  var me = this
  if (!(me instanceof DirWriter)) me.error(
    "DirWriter must be called as constructor.", null, true)

  // should already be established as a Directory type
  if (props.type !== "Directory" || !props.Directory) {
    me.error("Non-directory type "+ props.type + " " +
                    JSON.stringify(props), null, true)
  }

  Writer.call(this, props)
}

DirWriter.prototype._create = function () {
  var me = this
  mkdir(me._path, Writer.dirmode, function (er) {
    if (er) return me.error(er)
    // ready to start getting entries!
    me.ready = true
    me.emit("ready")
    me._process()
  })
}

// a DirWriter has an add(entry) method, but its .write() doesn't
// do anything.  Why a no-op rather than a throw?  Because this
// leaves open the door for writing directory metadata for
// gnu/solaris style dumpdirs.
DirWriter.prototype.write = function () {
  return true
}

DirWriter.prototype.end = function () {
  this._ended = true
  this._process()
}

DirWriter.prototype.add = function (entry) {
  var me = this

  // console.error("\tadd", entry._path, "->", me._path)
  collect(entry)
  if (!me.ready || me._currentEntry) {
    me._buffer.push(entry)
    return false
  }

  // create a new writer, and pipe the incoming entry into it.
  if (me._ended) {
    return me.error("add after end")
  }

  me._buffer.push(entry)
  me._process()

  return 0 === this._buffer.length
}

DirWriter.prototype._process = function () {
  var me = this

  // console.error("DW Process p=%j", me._processing, me.basename)

  if (me._processing) return

  var entry = me._buffer.shift()
  if (!entry) {
    // console.error("DW Drain")
    me.emit("drain")
    if (me._ended) me._finish()
    return
  }

  me._processing = true
  // console.error("DW Entry", entry._path)

  me.emit("entry", entry)

  // ok, add this entry
  //
  // don't allow recursive copying
  var p = entry
  do {
    var pp = p._path || p.path
    if (pp === me.root._path || pp === me._path ||
        (pp && pp.indexOf(me._path) === 0)) {
      // console.error("DW Exit (recursive)", entry.basename, me._path)
      me._processing = false
      if (entry._collected) entry.pipe()
      return me._process()
    }
  } while (p = p.parent)

  // console.error("DW not recursive")

  // chop off the entry's root dir, replace with ours
  var props = { parent: me
              , root: me.root || me
              , type: entry.type
              , depth: me.depth + 1 }

  var p = entry._path || entry.path || entry.props.path
  if (entry.parent) {
    p = p.substr(entry.parent._path.length + 1)
  }
  // get rid of any ../../ shenanigans
  props.path = path.join(me.path, path.join("/", p))

  // if i have a filter, the child should inherit it.
  props.filter = me.filter

  // all the rest of the stuff, copy over from the source.
  Object.keys(entry.props).forEach(function (k) {
    if (!props.hasOwnProperty(k)) {
      props[k] = entry.props[k]
    }
  })

  // not sure at this point what kind of writer this is.
  var child = me._currentChild = new Writer(props)
  child.on("ready", function () {
    // console.error("DW Child Ready", child.type, child._path)
    // console.error("  resuming", entry._path)
    entry.pipe(child)
    entry.resume()
  })

  // XXX Make this work in node.
  // Long filenames should not break stuff.
  child.on("error", function (er) {
    if (child._swallowErrors) {
      me.warn(er)
      child.emit("end")
      child.emit("close")
    } else {
      me.emit("error", er)
    }
  })

  // we fire _end internally *after* end, so that we don't move on
  // until any "end" listeners have had their chance to do stuff.
  child.on("close", onend)
  var ended = false
  function onend () {
    if (ended) return
    ended = true
    // console.error("* DW Child end", child.basename)
    me._currentChild = null
    me._processing = false
    me._process()
  }
}

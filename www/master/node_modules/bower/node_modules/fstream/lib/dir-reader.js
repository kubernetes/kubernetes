// A thing that emits "entry" events with Reader objects
// Pausing it causes it to stop emitting entry events, and also
// pauses the current entry if there is one.

module.exports = DirReader

var fs = require("graceful-fs")
  , fstream = require("../fstream.js")
  , Reader = fstream.Reader
  , inherits = require("inherits")
  , mkdir = require("mkdirp")
  , path = require("path")
  , Reader = require("./reader.js")
  , assert = require("assert").ok

inherits(DirReader, Reader)

function DirReader (props) {
  var me = this
  if (!(me instanceof DirReader)) throw new Error(
    "DirReader must be called as constructor.")

  // should already be established as a Directory type
  if (props.type !== "Directory" || !props.Directory) {
    throw new Error("Non-directory type "+ props.type)
  }

  me.entries = null
  me._index = -1
  me._paused = false
  me._length = -1

  if (props.sort) {
    this.sort = props.sort
  }

  Reader.call(this, props)
}

DirReader.prototype._getEntries = function () {
  var me = this

  // race condition.  might pause() before calling _getEntries,
  // and then resume, and try to get them a second time.
  if (me._gotEntries) return
  me._gotEntries = true

  fs.readdir(me._path, function (er, entries) {
    if (er) return me.error(er)

    me.entries = entries

    me.emit("entries", entries)
    if (me._paused) me.once("resume", processEntries)
    else processEntries()

    function processEntries () {
      me._length = me.entries.length
      if (typeof me.sort === "function") {
        me.entries = me.entries.sort(me.sort.bind(me))
      }
      me._read()
    }
  })
}

// start walking the dir, and emit an "entry" event for each one.
DirReader.prototype._read = function () {
  var me = this

  if (!me.entries) return me._getEntries()

  if (me._paused || me._currentEntry || me._aborted) {
    // console.error("DR paused=%j, current=%j, aborted=%j", me._paused, !!me._currentEntry, me._aborted)
    return
  }

  me._index ++
  if (me._index >= me.entries.length) {
    if (!me._ended) {
      me._ended = true
      me.emit("end")
      me.emit("close")
    }
    return
  }

  // ok, handle this one, then.

  // save creating a proxy, by stat'ing the thing now.
  var p = path.resolve(me._path, me.entries[me._index])
  assert(p !== me._path)
  assert(me.entries[me._index])

  // set this to prevent trying to _read() again in the stat time.
  me._currentEntry = p
  fs[ me.props.follow ? "stat" : "lstat" ](p, function (er, stat) {
    if (er) return me.error(er)

    var who = me._proxy || me

    stat.path = p
    stat.basename = path.basename(p)
    stat.dirname = path.dirname(p)
    var childProps = me.getChildProps.call(who, stat)
    childProps.path = p
    childProps.basename = path.basename(p)
    childProps.dirname = path.dirname(p)

    var entry = Reader(childProps, stat)

    // console.error("DR Entry", p, stat.size)

    me._currentEntry = entry

    // "entry" events are for direct entries in a specific dir.
    // "child" events are for any and all children at all levels.
    // This nomenclature is not completely final.

    entry.on("pause", function (who) {
      if (!me._paused && !entry._disowned) {
        me.pause(who)
      }
    })

    entry.on("resume", function (who) {
      if (me._paused && !entry._disowned) {
        me.resume(who)
      }
    })

    entry.on("stat", function (props) {
      me.emit("_entryStat", entry, props)
      if (entry._aborted) return
      if (entry._paused) entry.once("resume", function () {
        me.emit("entryStat", entry, props)
      })
      else me.emit("entryStat", entry, props)
    })

    entry.on("ready", function EMITCHILD () {
      // console.error("DR emit child", entry._path)
      if (me._paused) {
        // console.error("  DR emit child - try again later")
        // pause the child, and emit the "entry" event once we drain.
        // console.error("DR pausing child entry")
        entry.pause(me)
        return me.once("resume", EMITCHILD)
      }

      // skip over sockets.  they can't be piped around properly,
      // so there's really no sense even acknowledging them.
      // if someone really wants to see them, they can listen to
      // the "socket" events.
      if (entry.type === "Socket") {
        me.emit("socket", entry)
      } else {
        me.emitEntry(entry)
      }
    })

    var ended = false
    entry.on("close", onend)
    entry.on("disown", onend)
    function onend () {
      if (ended) return
      ended = true
      me.emit("childEnd", entry)
      me.emit("entryEnd", entry)
      me._currentEntry = null
      if (!me._paused) {
        me._read()
      }
    }

    // XXX Remove this.  Works in node as of 0.6.2 or so.
    // Long filenames should not break stuff.
    entry.on("error", function (er) {
      if (entry._swallowErrors) {
        me.warn(er)
        entry.emit("end")
        entry.emit("close")
      } else {
        me.emit("error", er)
      }
    })

    // proxy up some events.
    ; [ "child"
      , "childEnd"
      , "warn"
      ].forEach(function (ev) {
        entry.on(ev, me.emit.bind(me, ev))
      })
  })
}

DirReader.prototype.disown = function (entry) {
  entry.emit("beforeDisown")
  entry._disowned = true
  entry.parent = entry.root = null
  if (entry === this._currentEntry) {
    this._currentEntry = null
  }
  entry.emit("disown")
}

DirReader.prototype.getChildProps = function (stat) {
  return { depth: this.depth + 1
         , root: this.root || this
         , parent: this
         , follow: this.follow
         , filter: this.filter
         , sort: this.props.sort
         , hardlinks: this.props.hardlinks
         }
}

DirReader.prototype.pause = function (who) {
  var me = this
  if (me._paused) return
  who = who || me
  me._paused = true
  if (me._currentEntry && me._currentEntry.pause) {
    me._currentEntry.pause(who)
  }
  me.emit("pause", who)
}

DirReader.prototype.resume = function (who) {
  var me = this
  if (!me._paused) return
  who = who || me

  me._paused = false
  // console.error("DR Emit Resume", me._path)
  me.emit("resume", who)
  if (me._paused) {
    // console.error("DR Re-paused", me._path)
    return
  }

  if (me._currentEntry) {
    if (me._currentEntry.resume) me._currentEntry.resume(who)
  } else me._read()
}

DirReader.prototype.emitEntry = function (entry) {
  this.emit("entry", entry)
  this.emit("child", entry)
}

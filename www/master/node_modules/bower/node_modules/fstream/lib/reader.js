
module.exports = Reader

var fs = require("graceful-fs")
  , Stream = require("stream").Stream
  , inherits = require("inherits")
  , path = require("path")
  , getType = require("./get-type.js")
  , hardLinks = Reader.hardLinks = {}
  , Abstract = require("./abstract.js")

// Must do this *before* loading the child classes
inherits(Reader, Abstract)

var DirReader = require("./dir-reader.js")
  , FileReader = require("./file-reader.js")
  , LinkReader = require("./link-reader.js")
  , SocketReader = require("./socket-reader.js")
  , ProxyReader = require("./proxy-reader.js")

function Reader (props, currentStat) {
  var me = this
  if (!(me instanceof Reader)) return new Reader(props, currentStat)

  if (typeof props === "string") {
    props = { path: props }
  }

  if (!props.path) {
    me.error("Must provide a path", null, true)
  }

  // polymorphism.
  // call fstream.Reader(dir) to get a DirReader object, etc.
  // Note that, unlike in the Writer case, ProxyReader is going
  // to be the *normal* state of affairs, since we rarely know
  // the type of a file prior to reading it.


  var type
    , ClassType

  if (props.type && typeof props.type === "function") {
    type = props.type
    ClassType = type
  } else {
    type = getType(props)
    ClassType = Reader
  }

  if (currentStat && !type) {
    type = getType(currentStat)
    props[type] = true
    props.type = type
  }

  switch (type) {
    case "Directory":
      ClassType = DirReader
      break

    case "Link":
      // XXX hard links are just files.
      // However, it would be good to keep track of files' dev+inode
      // and nlink values, and create a HardLinkReader that emits
      // a linkpath value of the original copy, so that the tar
      // writer can preserve them.
      // ClassType = HardLinkReader
      // break

    case "File":
      ClassType = FileReader
      break

    case "SymbolicLink":
      ClassType = LinkReader
      break

    case "Socket":
      ClassType = SocketReader
      break

    case null:
      ClassType = ProxyReader
      break
  }

  if (!(me instanceof ClassType)) {
    return new ClassType(props)
  }

  Abstract.call(me)

  me.readable = true
  me.writable = false

  me.type = type
  me.props = props
  me.depth = props.depth = props.depth || 0
  me.parent = props.parent || null
  me.root = props.root || (props.parent && props.parent.root) || me

  me._path = me.path = path.resolve(props.path)
  if (process.platform === "win32") {
    me.path = me._path = me.path.replace(/\?/g, "_")
    if (me._path.length >= 260) {
      // how DOES one create files on the moon?
      // if the path has spaces in it, then UNC will fail.
      me._swallowErrors = true
      //if (me._path.indexOf(" ") === -1) {
        me._path = "\\\\?\\" + me.path.replace(/\//g, "\\")
      //}
    }
  }
  me.basename = props.basename = path.basename(me.path)
  me.dirname = props.dirname = path.dirname(me.path)

  // these have served their purpose, and are now just noisy clutter
  props.parent = props.root = null

  // console.error("\n\n\n%s setting size to", props.path, props.size)
  me.size = props.size
  me.filter = typeof props.filter === "function" ? props.filter : null
  if (props.sort === "alpha") props.sort = alphasort

  // start the ball rolling.
  // this will stat the thing, and then call me._read()
  // to start reading whatever it is.
  // console.error("calling stat", props.path, currentStat)
  me._stat(currentStat)
}

function alphasort (a, b) {
  return a === b ? 0
       : a.toLowerCase() > b.toLowerCase() ? 1
       : a.toLowerCase() < b.toLowerCase() ? -1
       : a > b ? 1
       : -1
}

Reader.prototype._stat = function (currentStat) {
  var me = this
    , props = me.props
    , stat = props.follow ? "stat" : "lstat"
  // console.error("Reader._stat", me._path, currentStat)
  if (currentStat) process.nextTick(statCb.bind(null, null, currentStat))
  else fs[stat](me._path, statCb)


  function statCb (er, props_) {
    // console.error("Reader._stat, statCb", me._path, props_, props_.nlink)
    if (er) return me.error(er)

    Object.keys(props_).forEach(function (k) {
      props[k] = props_[k]
    })

    // if it's not the expected size, then abort here.
    if (undefined !== me.size && props.size !== me.size) {
      return me.error("incorrect size")
    }
    me.size = props.size

    var type = getType(props)
    var handleHardlinks = props.hardlinks !== false
    
    // special little thing for handling hardlinks.
    if (handleHardlinks && type !== "Directory" && props.nlink && props.nlink > 1) {
      var k = props.dev + ":" + props.ino
      // console.error("Reader has nlink", me._path, k)
      if (hardLinks[k] === me._path || !hardLinks[k]) hardLinks[k] = me._path
      else {
        // switch into hardlink mode.
        type = me.type = me.props.type = "Link"
        me.Link = me.props.Link = true
        me.linkpath = me.props.linkpath = hardLinks[k]
        // console.error("Hardlink detected, switching mode", me._path, me.linkpath)
        // Setting __proto__ would arguably be the "correct"
        // approach here, but that just seems too wrong.
        me._stat = me._read = LinkReader.prototype._read
      }
    }

    if (me.type && me.type !== type) {
      me.error("Unexpected type: " + type)
    }

    // if the filter doesn't pass, then just skip over this one.
    // still have to emit end so that dir-walking can move on.
    if (me.filter) {
      var who = me._proxy || me
      // special handling for ProxyReaders
      if (!me.filter.call(who, who, props)) {
        if (!me._disowned) {
          me.abort()
          me.emit("end")
          me.emit("close")
        }
        return
      }
    }

    // last chance to abort or disown before the flow starts!
    var events = ["_stat", "stat", "ready"]
    var e = 0
    ;(function go () {
      if (me._aborted) {
        me.emit("end")
        me.emit("close")
        return
      }

      if (me._paused && me.type !== "Directory") {
        me.once("resume", go)
        return
      }

      var ev = events[e ++]
      if (!ev) {
        return me._read()
      }
      me.emit(ev, props)
      go()
    })()
  }
}

Reader.prototype.pipe = function (dest, opts) {
  var me = this
  if (typeof dest.add === "function") {
    // piping to a multi-compatible, and we've got directory entries.
    me.on("entry", function (entry) {
      var ret = dest.add(entry)
      if (false === ret) {
        me.pause()
      }
    })
  }

  // console.error("R Pipe apply Stream Pipe")
  return Stream.prototype.pipe.apply(this, arguments)
}

Reader.prototype.pause = function (who) {
  this._paused = true
  who = who || this
  this.emit("pause", who)
  if (this._stream) this._stream.pause(who)
}

Reader.prototype.resume = function (who) {
  this._paused = false
  who = who || this
  this.emit("resume", who)
  if (this._stream) this._stream.resume(who)
  this._read()
}

Reader.prototype._read = function () {
  this.error("Cannot read unknown type: "+this.type)
}


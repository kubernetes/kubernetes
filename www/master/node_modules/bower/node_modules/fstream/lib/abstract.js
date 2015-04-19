// the parent class for all fstreams.

module.exports = Abstract

var Stream = require("stream").Stream
  , inherits = require("inherits")

function Abstract () {
  Stream.call(this)
}

inherits(Abstract, Stream)

Abstract.prototype.on = function (ev, fn) {
  if (ev === "ready" && this.ready) {
    process.nextTick(fn.bind(this))
  } else {
    Stream.prototype.on.call(this, ev, fn)
  }
  return this
}

Abstract.prototype.abort = function () {
  this._aborted = true
  this.emit("abort")
}

Abstract.prototype.destroy = function () {}

Abstract.prototype.warn = function (msg, code) {
  var me = this
    , er = decorate(msg, code, me)
  if (!me.listeners("warn")) {
    console.error("%s %s\n" +
                  "path = %s\n" +
                  "syscall = %s\n" +
                  "fstream_type = %s\n" +
                  "fstream_path = %s\n" +
                  "fstream_unc_path = %s\n" +
                  "fstream_class = %s\n" +
                  "fstream_stack =\n%s\n",
                  code || "UNKNOWN",
                  er.stack,
                  er.path,
                  er.syscall,
                  er.fstream_type,
                  er.fstream_path,
                  er.fstream_unc_path,
                  er.fstream_class,
                  er.fstream_stack.join("\n"))
  } else {
    me.emit("warn", er)
  }
}

Abstract.prototype.info = function (msg, code) {
  this.emit("info", msg, code)
}

Abstract.prototype.error = function (msg, code, th) {
  var er = decorate(msg, code, this)
  if (th) throw er
  else this.emit("error", er)
}

function decorate (er, code, me) {
  if (!(er instanceof Error)) er = new Error(er)
  er.code = er.code || code
  er.path = er.path || me.path
  er.fstream_type = er.fstream_type || me.type
  er.fstream_path = er.fstream_path || me.path
  if (me._path !== me.path) {
    er.fstream_unc_path = er.fstream_unc_path || me._path
  }
  if (me.linkpath) {
    er.fstream_linkpath = er.fstream_linkpath || me.linkpath
  }
  er.fstream_class = er.fstream_class || me.constructor.name
  er.fstream_stack = er.fstream_stack ||
    new Error().stack.split(/\n/).slice(3).map(function (s) {
      return s.replace(/^    at /, "")
    })

  return er
}

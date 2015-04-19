// Essentially, this is a fstream.DirReader class, but with a
// bit of special logic to read the specified sort of ignore files,
// and a filter that prevents it from picking up anything excluded
// by those files.

var Minimatch = require("minimatch").Minimatch
, fstream = require("fstream")
, DirReader = fstream.DirReader
, inherits = require("inherits")
, path = require("path")
, fs = require("fs")

module.exports = IgnoreReader

inherits(IgnoreReader, DirReader)

function IgnoreReader (props) {
  if (!(this instanceof IgnoreReader)) {
    return new IgnoreReader(props)
  }

  // must be a Directory type
  if (typeof props === "string") {
    props = { path: path.resolve(props) }
  }

  props.type = "Directory"
  props.Directory = true

  if (!props.ignoreFiles) props.ignoreFiles = [".ignore"]
  this.ignoreFiles = props.ignoreFiles

  this.ignoreRules = null

  // ensure that .ignore files always show up at the top of the list
  // that way, they can be read before proceeding to handle other
  // entries in that same folder
  if (props.sort) {
    this._sort = props.sort === "alpha" ? alphasort : props.sort
    props.sort = null
  }

  this.on("entries", function () {
    // if there are any ignore files in the list, then
    // pause and add them.
    // then, filter the list based on our ignoreRules

    var hasIg = this.entries.some(this.isIgnoreFile, this)

    if (!hasIg) return this.filterEntries()

    this.addIgnoreFiles()
  })

  // we filter entries before we know what they are.
  // however, directories have to be re-tested against
  // rules with a "/" appended, because "a/b/" will only
  // match if "a/b" is a dir, and not otherwise.
  this.on("_entryStat", function (entry, props) {
    var t = entry.basename
    if (!this.applyIgnores(entry.basename,
                           entry.type === "Directory",
                           entry)) {
      entry.abort()
    }
  }.bind(this))

  DirReader.call(this, props)
}


IgnoreReader.prototype.addIgnoreFiles = function () {
  if (this._paused) {
    this.once("resume", this.addIgnoreFiles)
    return
  }
  if (this._ignoreFilesAdded) return
  this._ignoreFilesAdded = true

  var newIg = this.entries.filter(this.isIgnoreFile, this)
  , count = newIg.length
  , errState = null

  if (!count) return

  this.pause()

  var then = function (er) {
    if (errState) return
    if (er) return this.emit("error", errState = er)
    if (-- count === 0) {
      this.filterEntries()
      this.resume()
    } else {
      this.addIgnoreFile(newIg[newIg.length - count], then)
    }
  }.bind(this)

  this.addIgnoreFile(newIg[0], then)
}


IgnoreReader.prototype.isIgnoreFile = function (e) {
  return e !== "." &&
         e !== ".." &&
         -1 !== this.ignoreFiles.indexOf(e)
}


IgnoreReader.prototype.getChildProps = function (stat) {
  var props = DirReader.prototype.getChildProps.call(this, stat)
  props.ignoreFiles = this.ignoreFiles

  // Directories have to be read as IgnoreReaders
  // otherwise fstream.Reader will create a DirReader instead.
  if (stat.isDirectory()) {
    props.type = this.constructor
  }
  return props
}


IgnoreReader.prototype.addIgnoreFile = function (e, cb) {
  // read the file, and then call addIgnoreRules
  // if there's an error, then tell the cb about it.

  var ig = path.resolve(this.path, e)
  fs.readFile(ig, function (er, data) {
    if (er) return cb(er)

    this.emit("ignoreFile", e, data)
    var rules = this.readRules(data, e)
    this.addIgnoreRules(rules, e)
    cb()
  }.bind(this))
}


IgnoreReader.prototype.readRules = function (buf, e) {
  return buf.toString().split(/\r?\n/)
}


// Override this to do fancier things, like read the
// "files" array from a package.json file or something.
IgnoreReader.prototype.addIgnoreRules = function (set, e) {
  // filter out anything obvious
  set = set.filter(function (s) {
    s = s.trim()
    return s && !s.match(/^#/)
  })

  // no rules to add!
  if (!set.length) return

  // now get a minimatch object for each one of these.
  // Note that we need to allow dot files by default, and
  // not switch the meaning of their exclusion
  var mmopt = { matchBase: true, dot: true, flipNegate: true }
  , mm = set.map(function (s) {
    var m = new Minimatch(s, mmopt)
    m.ignoreFile = e
    return m
  })

  if (!this.ignoreRules) this.ignoreRules = []
  this.ignoreRules.push.apply(this.ignoreRules, mm)
}


IgnoreReader.prototype.filterEntries = function () {
  // this exclusion is at the point where we know the list of
  // entries in the dir, but don't know what they are.  since
  // some of them *might* be directories, we have to run the
  // match in dir-mode as well, so that we'll pick up partials
  // of files that will be included later.  Anything included
  // at this point will be checked again later once we know
  // what it is.
  this.entries = this.entries.filter(function (entry) {
    // at this point, we don't know if it's a dir or not.
    return this.applyIgnores(entry) || this.applyIgnores(entry, true)
  }, this)
}


IgnoreReader.prototype.applyIgnores = function (entry, partial, obj) {
  var included = true

  // this = /a/b/c
  // entry = d
  // parent /a/b sees c/d
  if (this.parent && this.parent.applyIgnores) {
    var pt = this.basename + "/" + entry
    included = this.parent.applyIgnores(pt, partial)
  }

  // Negated Rules
  // Since we're *ignoring* things here, negating means that a file
  // is re-included, if it would have been excluded by a previous
  // rule.  So, negated rules are only relevant if the file
  // has been excluded.
  //
  // Similarly, if a file has been excluded, then there's no point
  // trying it against rules that have already been applied
  //
  // We're using the "flipnegate" flag here, which tells minimatch
  // to set the "negate" for our information, but still report
  // whether the core pattern was a hit or a miss.

  if (!this.ignoreRules) {
    return included
  }

  this.ignoreRules.forEach(function (rule) {
    // negation means inclusion
    if (rule.negate && included ||
        !rule.negate && !included) {
      // unnecessary
      return
    }

    // first, match against /foo/bar
    var match = rule.match("/" + entry)

    if (!match) {
      // try with the leading / trimmed off the test
      // eg: foo/bar instead of /foo/bar
      match = rule.match(entry)
    }

    // if the entry is a directory, then it will match
    // with a trailing slash. eg: /foo/bar/ or foo/bar/
    if (!match && partial) {
      match = rule.match("/" + entry + "/") ||
              rule.match(entry + "/")
    }

    // When including a file with a negated rule, it's
    // relevant if a directory partially matches, since
    // it may then match a file within it.
    // Eg, if you ignore /a, but !/a/b/c
    if (!match && rule.negate && partial) {
      match = rule.match("/" + entry, true) ||
              rule.match(entry, true)
    }

    if (match) {
      included = rule.negate
    }
  }, this)

  return included
}


IgnoreReader.prototype.sort = function (a, b) {
  var aig = this.ignoreFiles.indexOf(a) !== -1
  , big = this.ignoreFiles.indexOf(b) !== -1

  if (aig && !big) return -1
  if (big && !aig) return 1
  return this._sort(a, b)
}

IgnoreReader.prototype._sort = function (a, b) {
  return 0
}

function alphasort (a, b) {
  return a === b ? 0
       : a.toLowerCase() > b.toLowerCase() ? 1
       : a.toLowerCase() < b.toLowerCase() ? -1
       : a > b ? 1
       : -1
}

var test = require('tap').test

// simulated ulimit
// this is like graceful-fs, but in reverse
var fs_ = require('fs')
var fs = require('../graceful-fs.js')
var files = fs.readdirSync(__dirname)

// Ok, no more actual file reading!

var fds = 0
var nextFd = 60
var limit = 8
fs_.open = function (path, flags, mode, cb) {
  process.nextTick(function() {
    ++fds
    if (fds >= limit) {
      --fds
      var er = new Error('EMFILE Curses!')
      er.code = 'EMFILE'
      er.path = path
      return cb(er)
    } else {
      cb(null, nextFd++)
    }
  })
}

fs_.openSync = function (path, flags, mode) {
  if (fds >= limit) {
    var er = new Error('EMFILE Curses!')
    er.code = 'EMFILE'
    er.path = path
    throw er
  } else {
    ++fds
    return nextFd++
  }
}

fs_.close = function (fd, cb) {
  process.nextTick(function () {
    --fds
    cb()
  })
}

fs_.closeSync = function (fd) {
  --fds
}

fs_.readdir = function (path, cb) {
  process.nextTick(function() {
    if (fds >= limit) {
      var er = new Error('EMFILE Curses!')
      er.code = 'EMFILE'
      er.path = path
      return cb(er)
    } else {
      ++fds
      process.nextTick(function () {
        --fds
        cb(null, [__filename, "some-other-file.js"])
      })
    }
  })
}

fs_.readdirSync = function (path) {
  if (fds >= limit) {
    var er = new Error('EMFILE Curses!')
    er.code = 'EMFILE'
    er.path = path
    throw er
  } else {
    return [__filename, "some-other-file.js"]
  }
}


test('open emfile autoreduce', function (t) {
  fs.MIN_MAX_OPEN = 4
  t.equal(fs.MAX_OPEN, 1024)

  var max = 12
  for (var i = 0; i < max; i++) {
    fs.open(__filename, 'r', next(i))
  }

  var phase = 0

  var expect =
      [ [ 0, 60, null, 1024, 4, 12, 1 ],
        [ 1, 61, null, 1024, 4, 12, 2 ],
        [ 2, 62, null, 1024, 4, 12, 3 ],
        [ 3, 63, null, 1024, 4, 12, 4 ],
        [ 4, 64, null, 1024, 4, 12, 5 ],
        [ 5, 65, null, 1024, 4, 12, 6 ],
        [ 6, 66, null, 1024, 4, 12, 7 ],
        [ 7, 67, null, 6, 4, 5, 1 ],
        [ 8, 68, null, 6, 4, 5, 2 ],
        [ 9, 69, null, 6, 4, 5, 3 ],
        [ 10, 70, null, 6, 4, 5, 4 ],
        [ 11, 71, null, 6, 4, 5, 5 ] ]

  var actual = []

  function next (i) { return function (er, fd) {
    if (er)
      throw er
    actual.push([i, fd, er, fs.MAX_OPEN, fs.MIN_MAX_OPEN, fs._curOpen, fds])

    if (i === max - 1) {
      t.same(actual, expect)
      t.ok(fs.MAX_OPEN < limit)
      t.end()
    }

    fs.close(fd)
  } }
})

test('readdir emfile autoreduce', function (t) {
  fs.MAX_OPEN = 1024
  var max = 12
  for (var i = 0; i < max; i ++) {
    fs.readdir(__dirname, next(i))
  }

  var expect =
      [ [0,[__filename,"some-other-file.js"],null,7,4,7,7],
        [1,[__filename,"some-other-file.js"],null,7,4,7,6],
        [2,[__filename,"some-other-file.js"],null,7,4,7,5],
        [3,[__filename,"some-other-file.js"],null,7,4,7,4],
        [4,[__filename,"some-other-file.js"],null,7,4,7,3],
        [5,[__filename,"some-other-file.js"],null,7,4,6,2],
        [6,[__filename,"some-other-file.js"],null,7,4,5,1],
        [7,[__filename,"some-other-file.js"],null,7,4,4,0],
        [8,[__filename,"some-other-file.js"],null,7,4,3,3],
        [9,[__filename,"some-other-file.js"],null,7,4,2,2],
        [10,[__filename,"some-other-file.js"],null,7,4,1,1],
        [11,[__filename,"some-other-file.js"],null,7,4,0,0] ]

  var actual = []

  function next (i) { return function (er, files) {
    if (er)
      throw er
    var line = [i, files, er, fs.MAX_OPEN, fs.MIN_MAX_OPEN, fs._curOpen, fds ]
    actual.push(line)

    if (i === max - 1) {
      t.ok(fs.MAX_OPEN < limit)
      t.same(actual, expect)
      t.end()
    }
  } }
})

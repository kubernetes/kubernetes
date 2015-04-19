var test = require("tap").test
var glob = require('../')
process.chdir(__dirname)

// expose timing issues
var lag = 5
glob.Glob.prototype._stat = function(o) { return function(f, cb) {
  var args = arguments
  setTimeout(function() {
    o.call(this, f, cb)
  }.bind(this), lag += 5)
}}(glob.Glob.prototype._stat)


test("mark, with **", function (t) {
  glob("a/*b*/**", {mark: true}, function (er, results) {
    if (er)
      throw er
    var expect =
      [ 'a/abcdef/',
        'a/abcdef/g/',
        'a/abcdef/g/h',
        'a/abcfed/',
        'a/abcfed/g/',
        'a/abcfed/g/h',
        'a/b/',
        'a/b/c/',
        'a/b/c/d',
        'a/bc/',
        'a/bc/e/',
        'a/bc/e/f',
        'a/cb/',
        'a/cb/e/',
        'a/cb/e/f' ]

    t.same(results, expect)
    t.end()
  })
})

test("mark, no / on pattern", function (t) {
  glob("a/*", {mark: true}, function (er, results) {
    if (er)
      throw er
    var expect = [ 'a/abcdef/',
                   'a/abcfed/',
                   'a/b/',
                   'a/bc/',
                   'a/c/',
                   'a/cb/' ]

    if (process.platform !== "win32")
      expect.push('a/symlink/')

    t.same(results, expect)
    t.end()
  }).on('match', function(m) {
    t.similar(m, /\/$/)
  })
})

test("mark=false, no / on pattern", function (t) {
  glob("a/*", function (er, results) {
    if (er)
      throw er
    var expect = [ 'a/abcdef',
                   'a/abcfed',
                   'a/b',
                   'a/bc',
                   'a/c',
                   'a/cb' ]

    if (process.platform !== "win32")
      expect.push('a/symlink')
    t.same(results, expect)
    t.end()
  }).on('match', function(m) {
    t.similar(m, /[^\/]$/)
  })
})

test("mark=true, / on pattern", function (t) {
  glob("a/*/", {mark: true}, function (er, results) {
    if (er)
      throw er
    var expect = [ 'a/abcdef/',
                    'a/abcfed/',
                    'a/b/',
                    'a/bc/',
                    'a/c/',
                    'a/cb/' ]
    if (process.platform !== "win32")
      expect.push('a/symlink/')
    t.same(results, expect)
    t.end()
  }).on('match', function(m) {
    t.similar(m, /\/$/)
  })
})

test("mark=false, / on pattern", function (t) {
  glob("a/*/", function (er, results) {
    if (er)
      throw er
    var expect = [ 'a/abcdef/',
                   'a/abcfed/',
                   'a/b/',
                   'a/bc/',
                   'a/c/',
                   'a/cb/' ]
    if (process.platform !== "win32")
      expect.push('a/symlink/')
    t.same(results, expect)
    t.end()
  }).on('match', function(m) {
    t.similar(m, /\/$/)
  })
})

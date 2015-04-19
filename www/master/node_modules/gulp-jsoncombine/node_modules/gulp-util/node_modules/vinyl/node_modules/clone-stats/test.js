var test = require('tape')
var clone = require('./')
var fs = require('fs')

test('file', function(t) {
  compare(t, fs.statSync(__filename))
  t.end()
})

test('directory', function(t) {
  compare(t, fs.statSync(__dirname))
  t.end()
})

function compare(t, stat) {
  var copy = clone(stat)

  t.deepEqual(stat, copy, 'clone has equal properties')
  t.ok(stat instanceof fs.Stats, 'original is an fs.Stat')
  t.ok(copy instanceof fs.Stats, 'copy is an fs.Stat')

  ;['isDirectory'
  , 'isFile'
  , 'isBlockDevice'
  , 'isCharacterDevice'
  , 'isSymbolicLink'
  , 'isFIFO'
  , 'isSocket'
  ].forEach(function(method) {
    t.equal(
        stat[method].call(stat)
      , copy[method].call(copy)
      , 'equal value for stat.' + method + '()'
    )
  })
}

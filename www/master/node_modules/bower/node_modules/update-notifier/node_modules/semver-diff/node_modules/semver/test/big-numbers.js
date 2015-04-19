var test = require('tap').test
var semver = require('../')

test('long version is too long', function (t) {
  var v = '1.2.' + new Array(256).join('1')
  t.throws(function () {
    new semver.SemVer(v)
  })
  t.equal(semver.valid(v, false), null)
  t.equal(semver.valid(v, true), null)
  t.equal(semver.inc(v, 'patch'), null)
  t.end()
})

test('big number is like too long version', function (t) {
  var v = '1.2.' + new Array(100).join('1')
  t.throws(function () {
    new semver.SemVer(v)
  })
  t.equal(semver.valid(v, false), null)
  t.equal(semver.valid(v, true), null)
  t.equal(semver.inc(v, 'patch'), null)
  t.end()
})

test('parsing null does not throw', function (t) {
  t.equal(semver.parse(null), null)
  t.equal(semver.parse({}), null)
  t.equal(semver.parse(new semver.SemVer('1.2.3')).version, '1.2.3')
  t.end()
})

'use strict';
/*jshint asi: true */

var test = require('tap').test
  , rx = require('..')
  , fs = require('fs')
  , convert = require('..')

test('\nresolving a "/*# sourceMappingURL=map-file-comment.css.map*/" style comment inside a given css content', function (t) {
  var css = fs.readFileSync(__dirname + '/fixtures/map-file-comment.css', 'utf8')
  var conv = convert.fromMapFileSource(css, __dirname + '/fixtures');
  var sm = conv.toObject();

  t.deepEqual(
      sm.sources
    , [ './client/sass/core.scss',
        './client/sass/main.scss' ]
    , 'resolves paths of original sources'
  )

  t.equal(sm.file, 'map-file-comment.css', 'includes filename of generated file')
  t.equal(
      sm.mappings
    , 'AAAA,wBAAyB;EACvB,UAAU,EAAE,IAAI;EAChB,MAAM,EAAE,KAAK;EACb,OAAO,EAAE,IAAI;EACb,aAAa,EAAE,iBAAiB;EAChC,KAAK,EAAE,OAAkB;;AAG3B,wBAAyB;EACvB,OAAO,EAAE,IAAI;;ACTf,gBAAiB;EACf,UAAU,EAAE,IAAI;EAChB,KAAK,EAAE,MAAM;;AAGf,kBAAmB;EACjB,MAAM,EAAE,IAAI;EACZ,OAAO,EAAE,IAAI;EACb,UAAU,EAAE,KAAK;EACjB,aAAa,EAAE,GAAG;EAClB,KAAK,EAAE,KAAK;;AAEd,kBAAmB;EACjB,KAAK,EAAE,KAAK;;AAGd,mBAAoB;EAClB,KAAK,EAAE,KAAK;EACZ,MAAM,EAAE,IAAI;EACZ,OAAO,EAAE,IAAI;EACb,SAAS,EAAE,IAAI'
    , 'includes mappings'
  )
  t.end()
})

test('\nresolving a "//# sourceMappingURL=map-file-comment.css.map" style comment inside a given css content', function (t) {
  var css = fs.readFileSync(__dirname + '/fixtures/map-file-comment-double-slash.css', 'utf8')
  var conv = convert.fromMapFileSource(css, __dirname + '/fixtures');
  var sm = conv.toObject();

  t.deepEqual(
      sm.sources
    , [ './client/sass/core.scss',
        './client/sass/main.scss' ]
    , 'resolves paths of original sources'
  )

  t.equal(sm.file, 'map-file-comment.css', 'includes filename of generated file')
  t.equal(
      sm.mappings
    , 'AAAA,wBAAyB;EACvB,UAAU,EAAE,IAAI;EAChB,MAAM,EAAE,KAAK;EACb,OAAO,EAAE,IAAI;EACb,aAAa,EAAE,iBAAiB;EAChC,KAAK,EAAE,OAAkB;;AAG3B,wBAAyB;EACvB,OAAO,EAAE,IAAI;;ACTf,gBAAiB;EACf,UAAU,EAAE,IAAI;EAChB,KAAK,EAAE,MAAM;;AAGf,kBAAmB;EACjB,MAAM,EAAE,IAAI;EACZ,OAAO,EAAE,IAAI;EACb,UAAU,EAAE,KAAK;EACjB,aAAa,EAAE,GAAG;EAClB,KAAK,EAAE,KAAK;;AAEd,kBAAmB;EACjB,KAAK,EAAE,KAAK;;AAGd,mBAAoB;EAClB,KAAK,EAAE,KAAK;EACZ,MAAM,EAAE,IAAI;EACZ,OAAO,EAAE,IAAI;EACb,SAAS,EAAE,IAAI'
    , 'includes mappings'
  )
  t.end()
})

test('\nresolving a /*# sourceMappingURL=data:application/json;base64,... */ style comment inside a given css content', function(t) {
  var css = fs.readFileSync(__dirname + '/fixtures/map-file-comment-inline.css', 'utf8')
  var conv = convert.fromSource(css, __dirname + '/fixtures')
  var sm = conv.toObject()

  t.deepEqual(
      sm.sources
    , [ './client/sass/core.scss',
        './client/sass/main.scss' ]
    , 'resolves paths of original sources'
  )

  t.equal(sm.file, 'map-file-comment.css', 'includes filename of generated file')
  t.equal(
      sm.mappings
    , 'AAAA,wBAAyB;EACvB,UAAU,EAAE,IAAI;EAChB,MAAM,EAAE,KAAK;EACb,OAAO,EAAE,IAAI;EACb,aAAa,EAAE,iBAAiB;EAChC,KAAK,EAAE,OAAkB;;AAG3B,wBAAyB;EACvB,OAAO,EAAE,IAAI;;ACTf,gBAAiB;EACf,UAAU,EAAE,IAAI;EAChB,KAAK,EAAE,MAAM;;AAGf,kBAAmB;EACjB,MAAM,EAAE,IAAI;EACZ,OAAO,EAAE,IAAI;EACb,UAAU,EAAE,KAAK;EACjB,aAAa,EAAE,GAAG;EAClB,KAAK,EAAE,KAAK;;AAEd,kBAAmB;EACjB,KAAK,EAAE,KAAK;;AAGd,mBAAoB;EAClB,KAAK,EAAE,KAAK;EACZ,MAAM,EAAE,IAAI;EACZ,OAAO,EAAE,IAAI;EACb,SAAS,EAAE,IAAI'
    , 'includes mappings'
  )
  t.end()
})

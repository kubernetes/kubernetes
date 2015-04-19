/* -*- Mode: js; js-indent-level: 2; -*- */
/*
 * Copyright 2011 Mozilla Foundation and contributors
 * Licensed under the New BSD license. See LICENSE or:
 * http://opensource.org/licenses/BSD-3-Clause
 */
if (typeof define !== 'function') {
    var define = require('amdefine')(module, require);
}
define(function (require, exports, module) {

  var SourceMapConsumer = require('../../lib/source-map/source-map-consumer').SourceMapConsumer;
  var SourceMapGenerator = require('../../lib/source-map/source-map-generator').SourceMapGenerator;

  exports['test that we can instantiate with a string or an objects'] = function (assert, util) {
    assert.doesNotThrow(function () {
      var map = new SourceMapConsumer(util.testMap);
    });
    assert.doesNotThrow(function () {
      var map = new SourceMapConsumer(JSON.stringify(util.testMap));
    });
  };

  exports['test that the `sources` field has the original sources'] = function (assert, util) {
    var map = new SourceMapConsumer(util.testMap);
    var sources = map.sources;

    assert.equal(sources[0], '/the/root/one.js');
    assert.equal(sources[1], '/the/root/two.js');
    assert.equal(sources.length, 2);
  };

  exports['test that the source root is reflected in a mapping\'s source field'] = function (assert, util) {
    var map = new SourceMapConsumer(util.testMap);
    var mapping;

    mapping = map.originalPositionFor({
      line: 2,
      column: 1
    });
    assert.equal(mapping.source, '/the/root/two.js');

    mapping = map.originalPositionFor({
      line: 1,
      column: 1
    });
    assert.equal(mapping.source, '/the/root/one.js');
  };

  exports['test mapping tokens back exactly'] = function (assert, util) {
    var map = new SourceMapConsumer(util.testMap);

    util.assertMapping(1, 1, '/the/root/one.js', 1, 1, null, map, assert);
    util.assertMapping(1, 5, '/the/root/one.js', 1, 5, null, map, assert);
    util.assertMapping(1, 9, '/the/root/one.js', 1, 11, null, map, assert);
    util.assertMapping(1, 18, '/the/root/one.js', 1, 21, 'bar', map, assert);
    util.assertMapping(1, 21, '/the/root/one.js', 2, 3, null, map, assert);
    util.assertMapping(1, 28, '/the/root/one.js', 2, 10, 'baz', map, assert);
    util.assertMapping(1, 32, '/the/root/one.js', 2, 14, 'bar', map, assert);

    util.assertMapping(2, 1, '/the/root/two.js', 1, 1, null, map, assert);
    util.assertMapping(2, 5, '/the/root/two.js', 1, 5, null, map, assert);
    util.assertMapping(2, 9, '/the/root/two.js', 1, 11, null, map, assert);
    util.assertMapping(2, 18, '/the/root/two.js', 1, 21, 'n', map, assert);
    util.assertMapping(2, 21, '/the/root/two.js', 2, 3, null, map, assert);
    util.assertMapping(2, 28, '/the/root/two.js', 2, 10, 'n', map, assert);
  };

  exports['test mapping tokens fuzzy'] = function (assert, util) {
    var map = new SourceMapConsumer(util.testMap);

    // Finding original positions
    util.assertMapping(1, 20, '/the/root/one.js', 1, 21, 'bar', map, assert, true);
    util.assertMapping(1, 30, '/the/root/one.js', 2, 10, 'baz', map, assert, true);
    util.assertMapping(2, 12, '/the/root/two.js', 1, 11, null, map, assert, true);

    // Finding generated positions
    util.assertMapping(1, 18, '/the/root/one.js', 1, 22, 'bar', map, assert, null, true);
    util.assertMapping(1, 28, '/the/root/one.js', 2, 13, 'baz', map, assert, null, true);
    util.assertMapping(2, 9, '/the/root/two.js', 1, 16, null, map, assert, null, true);
  };

  exports['test creating source map consumers with )]}\' prefix'] = function (assert, util) {
    assert.doesNotThrow(function () {
      var map = new SourceMapConsumer(")]}'" + JSON.stringify(util.testMap));
    });
  };

  exports['test eachMapping'] = function (assert, util) {
    var map = new SourceMapConsumer(util.testMap);
    var previousLine = -Infinity;
    var previousColumn = -Infinity;
    map.eachMapping(function (mapping) {
      assert.ok(mapping.generatedLine >= previousLine);

      if (mapping.source) {
        assert.equal(mapping.source.indexOf(util.testMap.sourceRoot), 0);
      }

      if (mapping.generatedLine === previousLine) {
        assert.ok(mapping.generatedColumn >= previousColumn);
        previousColumn = mapping.generatedColumn;
      }
      else {
        previousLine = mapping.generatedLine;
        previousColumn = -Infinity;
      }
    });
  };

  exports['test iterating over mappings in a different order'] = function (assert, util) {
    var map = new SourceMapConsumer(util.testMap);
    var previousLine = -Infinity;
    var previousColumn = -Infinity;
    var previousSource = "";
    map.eachMapping(function (mapping) {
      assert.ok(mapping.source >= previousSource);

      if (mapping.source === previousSource) {
        assert.ok(mapping.originalLine >= previousLine);

        if (mapping.originalLine === previousLine) {
          assert.ok(mapping.originalColumn >= previousColumn);
          previousColumn = mapping.originalColumn;
        }
        else {
          previousLine = mapping.originalLine;
          previousColumn = -Infinity;
        }
      }
      else {
        previousSource = mapping.source;
        previousLine = -Infinity;
        previousColumn = -Infinity;
      }
    }, null, SourceMapConsumer.ORIGINAL_ORDER);
  };

  exports['test that we can set the context for `this` in eachMapping'] = function (assert, util) {
    var map = new SourceMapConsumer(util.testMap);
    var context = {};
    map.eachMapping(function () {
      assert.equal(this, context);
    }, context);
  };

  exports['test that the `sourcesContent` field has the original sources'] = function (assert, util) {
    var map = new SourceMapConsumer(util.testMapWithSourcesContent);
    var sourcesContent = map.sourcesContent;

    assert.equal(sourcesContent[0], ' ONE.foo = function (bar) {\n   return baz(bar);\n };');
    assert.equal(sourcesContent[1], ' TWO.inc = function (n) {\n   return n + 1;\n };');
    assert.equal(sourcesContent.length, 2);
  };

  exports['test that we can get the original sources for the sources'] = function (assert, util) {
    var map = new SourceMapConsumer(util.testMapWithSourcesContent);
    var sources = map.sources;

    assert.equal(map.sourceContentFor(sources[0]), ' ONE.foo = function (bar) {\n   return baz(bar);\n };');
    assert.equal(map.sourceContentFor(sources[1]), ' TWO.inc = function (n) {\n   return n + 1;\n };');
    assert.equal(map.sourceContentFor("one.js"), ' ONE.foo = function (bar) {\n   return baz(bar);\n };');
    assert.equal(map.sourceContentFor("two.js"), ' TWO.inc = function (n) {\n   return n + 1;\n };');
    assert.throws(function () {
      map.sourceContentFor("");
    }, Error);
    assert.throws(function () {
      map.sourceContentFor("/the/root/three.js");
    }, Error);
    assert.throws(function () {
      map.sourceContentFor("three.js");
    }, Error);
  };

  exports['test sourceRoot + generatedPositionFor'] = function (assert, util) {
    var map = new SourceMapGenerator({
      sourceRoot: 'foo/bar',
      file: 'baz.js'
    });
    map.addMapping({
      original: { line: 1, column: 1 },
      generated: { line: 2, column: 2 },
      source: 'bang.coffee'
    });
    map.addMapping({
      original: { line: 5, column: 5 },
      generated: { line: 6, column: 6 },
      source: 'bang.coffee'
    });
    map = new SourceMapConsumer(map.toString());

    // Should handle without sourceRoot.
    var pos = map.generatedPositionFor({
      line: 1,
      column: 1,
      source: 'bang.coffee'
    });

    assert.equal(pos.line, 2);
    assert.equal(pos.column, 2);

    // Should handle with sourceRoot.
    var pos = map.generatedPositionFor({
      line: 1,
      column: 1,
      source: 'foo/bar/bang.coffee'
    });

    assert.equal(pos.line, 2);
    assert.equal(pos.column, 2);
  };

  exports['test sourceRoot + originalPositionFor'] = function (assert, util) {
    var map = new SourceMapGenerator({
      sourceRoot: 'foo/bar',
      file: 'baz.js'
    });
    map.addMapping({
      original: { line: 1, column: 1 },
      generated: { line: 2, column: 2 },
      source: 'bang.coffee'
    });
    map = new SourceMapConsumer(map.toString());

    var pos = map.originalPositionFor({
      line: 2,
      column: 2,
    });

    // Should always have the prepended source root
    assert.equal(pos.source, 'foo/bar/bang.coffee');
    assert.equal(pos.line, 1);
    assert.equal(pos.column, 1);
  };

  exports['test github issue #56'] = function (assert, util) {
    var map = new SourceMapGenerator({
      sourceRoot: 'http://',
      file: 'www.example.com/foo.js'
    });
    map.addMapping({
      original: { line: 1, column: 1 },
      generated: { line: 2, column: 2 },
      source: 'www.example.com/original.js'
    });
    map = new SourceMapConsumer(map.toString());

    var sources = map.sources;
    assert.equal(sources.length, 1);
    assert.equal(sources[0], 'http://www.example.com/original.js');
  };

  exports['test github issue #43'] = function (assert, util) {
    var map = new SourceMapGenerator({
      sourceRoot: 'http://example.com',
      file: 'foo.js'
    });
    map.addMapping({
      original: { line: 1, column: 1 },
      generated: { line: 2, column: 2 },
      source: 'http://cdn.example.com/original.js'
    });
    map = new SourceMapConsumer(map.toString());

    var sources = map.sources;
    assert.equal(sources.length, 1,
                 'Should only be one source.');
    assert.equal(sources[0], 'http://cdn.example.com/original.js',
                 'Should not be joined with the sourceRoot.');
  };

  exports['test absolute path, but same host sources'] = function (assert, util) {
    var map = new SourceMapGenerator({
      sourceRoot: 'http://example.com/foo/bar',
      file: 'foo.js'
    });
    map.addMapping({
      original: { line: 1, column: 1 },
      generated: { line: 2, column: 2 },
      source: '/original.js'
    });
    map = new SourceMapConsumer(map.toString());

    var sources = map.sources;
    assert.equal(sources.length, 1,
                 'Should only be one source.');
    assert.equal(sources[0], 'http://example.com/original.js',
                 'Source should be relative the host of the source root.');
  };

  exports['test github issue #64'] = function (assert, util) {
    var map = new SourceMapConsumer({
      "version": 3,
      "file": "foo.js",
      "sourceRoot": "http://example.com/",
      "sources": ["/a"],
      "names": [],
      "mappings": "AACA",
      "sourcesContent": ["foo"]
    });

    assert.equal(map.sourceContentFor("a"), "foo");
    assert.equal(map.sourceContentFor("/a"), "foo");
  };

  exports['test bug 885597'] = function (assert, util) {
    var map = new SourceMapConsumer({
      "version": 3,
      "file": "foo.js",
      "sourceRoot": "file:///Users/AlGore/Invented/The/Internet/",
      "sources": ["/a"],
      "names": [],
      "mappings": "AACA",
      "sourcesContent": ["foo"]
    });

    var s = map.sources[0];
    assert.equal(map.sourceContentFor(s), "foo");
  };

  exports['test github issue #72, duplicate sources'] = function (assert, util) {
    var map = new SourceMapConsumer({
      "version": 3,
      "file": "foo.js",
      "sources": ["source1.js", "source1.js", "source3.js"],
      "names": [],
      "mappings": ";EAAC;;IAEE;;MEEE",
      "sourceRoot": "http://example.com"
    });

    var pos = map.originalPositionFor({
      line: 2,
      column: 2
    });
    assert.equal(pos.source, 'http://example.com/source1.js');
    assert.equal(pos.line, 1);
    assert.equal(pos.column, 1);

    var pos = map.originalPositionFor({
      line: 4,
      column: 4
    });
    assert.equal(pos.source, 'http://example.com/source1.js');
    assert.equal(pos.line, 3);
    assert.equal(pos.column, 3);

    var pos = map.originalPositionFor({
      line: 6,
      column: 6
    });
    assert.equal(pos.source, 'http://example.com/source3.js');
    assert.equal(pos.line, 5);
    assert.equal(pos.column, 5);
  };

  exports['test github issue #72, duplicate names'] = function (assert, util) {
    var map = new SourceMapConsumer({
      "version": 3,
      "file": "foo.js",
      "sources": ["source.js"],
      "names": ["name1", "name1", "name3"],
      "mappings": ";EAACA;;IAEEA;;MAEEE",
      "sourceRoot": "http://example.com"
    });

    var pos = map.originalPositionFor({
      line: 2,
      column: 2
    });
    assert.equal(pos.name, 'name1');
    assert.equal(pos.line, 1);
    assert.equal(pos.column, 1);

    var pos = map.originalPositionFor({
      line: 4,
      column: 4
    });
    assert.equal(pos.name, 'name1');
    assert.equal(pos.line, 3);
    assert.equal(pos.column, 3);

    var pos = map.originalPositionFor({
      line: 6,
      column: 6
    });
    assert.equal(pos.name, 'name3');
    assert.equal(pos.line, 5);
    assert.equal(pos.column, 5);
  };

  exports['test SourceMapConsumer.fromSourceMap'] = function (assert, util) {
    var smg = new SourceMapGenerator({
      sourceRoot: 'http://example.com/',
      file: 'foo.js'
    });
    smg.addMapping({
      original: { line: 1, column: 1 },
      generated: { line: 2, column: 2 },
      source: 'bar.js'
    });
    smg.addMapping({
      original: { line: 2, column: 2 },
      generated: { line: 4, column: 4 },
      source: 'baz.js',
      name: 'dirtMcGirt'
    });
    smg.setSourceContent('baz.js', 'baz.js content');

    var smc = SourceMapConsumer.fromSourceMap(smg);
    assert.equal(smc.file, 'foo.js');
    assert.equal(smc.sourceRoot, 'http://example.com/');
    assert.equal(smc.sources.length, 2);
    assert.equal(smc.sources[0], 'http://example.com/bar.js');
    assert.equal(smc.sources[1], 'http://example.com/baz.js');
    assert.equal(smc.sourceContentFor('baz.js'), 'baz.js content');

    var pos = smc.originalPositionFor({
      line: 2,
      column: 2
    });
    assert.equal(pos.line, 1);
    assert.equal(pos.column, 1);
    assert.equal(pos.source, 'http://example.com/bar.js');
    assert.equal(pos.name, null);

    pos = smc.generatedPositionFor({
      line: 1,
      column: 1,
      source: 'http://example.com/bar.js'
    });
    assert.equal(pos.line, 2);
    assert.equal(pos.column, 2);

    pos = smc.originalPositionFor({
      line: 4,
      column: 4
    });
    assert.equal(pos.line, 2);
    assert.equal(pos.column, 2);
    assert.equal(pos.source, 'http://example.com/baz.js');
    assert.equal(pos.name, 'dirtMcGirt');

    pos = smc.generatedPositionFor({
      line: 2,
      column: 2,
      source: 'http://example.com/baz.js'
    });
    assert.equal(pos.line, 4);
    assert.equal(pos.column, 4);
  };
});

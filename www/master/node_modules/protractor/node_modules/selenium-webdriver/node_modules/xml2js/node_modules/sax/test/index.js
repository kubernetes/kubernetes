var globalsBefore = JSON.stringify(Object.keys(global))
  , util = require("util")
  , assert = require("assert")
  , fs = require("fs")
  , path = require("path")
  , sax = require("../lib/sax")

exports.sax = sax

// handy way to do simple unit tests
// if the options contains an xml string, it'll be written and the parser closed.
// otherwise, it's assumed that the test will write and close.
exports.test = function test (options) {
  var xml = options.xml
    , parser = sax.parser(options.strict, options.opt)
    , expect = options.expect
    , e = 0
  sax.EVENTS.forEach(function (ev) {
    parser["on" + ev] = function (n) {
      if (process.env.DEBUG) {
        console.error({ expect: expect[e]
                      , actual: [ev, n] })
      }
      if (e >= expect.length && (ev === "end" || ev === "ready")) return
      assert.ok( e < expect.length,
        "expectation #"+e+" "+util.inspect(expect[e])+"\n"+
        "Unexpected event: "+ev+" "+(n ? util.inspect(n) : ""))
      var inspected = n instanceof Error ? "\n"+ n.message : util.inspect(n)
      assert.equal(ev, expect[e][0],
        "expectation #"+e+"\n"+
        "Didn't get expected event\n"+
        "expect: "+expect[e][0] + " " +util.inspect(expect[e][1])+"\n"+
        "actual: "+ev+" "+inspected+"\n")
      if (ev === "error") assert.equal(n.message, expect[e][1])
      else assert.deepEqual(n, expect[e][1],
        "expectation #"+e+"\n"+
        "Didn't get expected argument\n"+
        "expect: "+expect[e][0] + " " +util.inspect(expect[e][1])+"\n"+
        "actual: "+ev+" "+inspected+"\n")
      e++
      if (ev === "error") parser.resume()
    }
  })
  if (xml) parser.write(xml).close()
  return parser
}

if (module === require.main) {
  var running = true
    , failures = 0

  function fail (file, er) {
    util.error("Failed: "+file)
    util.error(er.stack || er.message)
    failures ++
  }

  fs.readdir(__dirname, function (error, files) {
    files = files.filter(function (file) {
      return (/\.js$/.exec(file) && file !== 'index.js')
    })
    var n = files.length
      , i = 0
    console.log("0.." + n)
    files.forEach(function (file) {
      // run this test.
      try {
        require(path.resolve(__dirname, file))
        var globalsAfter = JSON.stringify(Object.keys(global))
        if (globalsAfter !== globalsBefore) {
          var er = new Error("new globals introduced\n"+
                             "expected: "+globalsBefore+"\n"+
                             "actual:   "+globalsAfter)
          globalsBefore = globalsAfter
          throw er
        }
        console.log("ok " + (++i) + " - " + file)
      } catch (er) {
        console.log("not ok "+ (++i) + " - " + file)
        fail(file, er)
      }
    })
    if (!failures) return console.log("#all pass")
    else return console.error(failures + " failure" + (failures > 1 ? "s" : ""))
  })
}

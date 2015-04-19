var i = require("../")
  , tap = require("tap")
  , test = tap.test
  , fs = require("fs")
  , path = require("path")
  , fixture = path.resolve(__dirname, "./fixtures/foo.ini")
  , data = fs.readFileSync(fixture, "utf8")
  , d
  , expectE = 'o=p\n'
            + 'a with spaces=b  c\n'
            + '" xa  n          p "="\\"\\r\\nyoyoyo\\r\\r\\n"\n'
            + '"[disturbing]"=hey you never know\n'
            + 's=something\n'
            + 's1=\"something\'\n'
            + 's2=something else\n'
            + 'zr[]=deedee\n'
            + 'ar[]=one\n'
            + 'ar[]=three\n'
            + 'ar[]=this is included\n'
            + 'br=warm\n'
            + 'eq=\"eq=eq\"\n'
            + '\n'
            + '[a]\n'
            + 'av=a val\n'
            + 'e={ o: p, a: '
            + '{ av: a val, b: { c: { e: "this [value]" '
            + '} } } }\nj="\\"{ o: \\"p\\", a: { av:'
            + ' \\"a val\\", b: { c: { e: \\"this [value]'
            + '\\" } } } }\\""\n"[]"=a square?\n'
            + 'cr[]=four\ncr[]=eight\n\n'
            +'[a.b.c]\ne=1\n'
            + 'j=2\n\n[x\\.y\\.z]\nx.y.z=xyz\n\n'
            + '[x\\.y\\.z.a\\.b\\.c]\na.b.c=abc\n'
            + 'nocomment=this\\; this is not a comment\n'
            + 'noHashComment=this\\# this is not a comment\n'
  , expectD =
    { o: 'p',
      'a with spaces': 'b  c',
      " xa  n          p ":'"\r\nyoyoyo\r\r\n',
      '[disturbing]': 'hey you never know',
      's': 'something',
      's1' : '\"something\'',
      's2': 'something else',
      'zr': ['deedee'],
      'ar': ['one', 'three', 'this is included'],
      'br': 'warm',
      'eq': 'eq=eq',
      a:
       { av: 'a val',
         e: '{ o: p, a: { av: a val, b: { c: { e: "this [value]" } } } }',
         j: '"{ o: "p", a: { av: "a val", b: { c: { e: "this [value]" } } } }"',
         "[]": "a square?",
         cr: ['four', 'eight'],
         b: { c: { e: '1', j: '2' } } },
      'x.y.z': {
        'x.y.z': 'xyz',
        'a.b.c': {
          'a.b.c': 'abc',
          'nocomment': 'this\; this is not a comment',
          noHashComment: 'this\# this is not a comment'
        }
      }
    }
  , expectF = '[prefix.log]\n'
            + 'type=file\n\n'
            + '[prefix.log.level]\n'
            + 'label=debug\n'
            + 'value=10\n'
  , expectG = '[log]\n'
            + 'type = file\n\n'
            + '[log.level]\n'
            + 'label = debug\n'
            + 'value = 10\n'

test("decode from file", function (t) {
  var d = i.decode(data)
  t.deepEqual(d, expectD)
  t.end()
})

test("encode from data", function (t) {
  var e = i.encode(expectD)
  t.deepEqual(e, expectE)

  var obj = {log: { type:'file', level: {label:'debug', value:10} } }
  e = i.encode(obj)
  t.notEqual(e.slice(0, 1), '\n', 'Never a blank first line')
  t.notEqual(e.slice(-2), '\n\n', 'Never a blank final line')

  t.end()
})

test("encode with option", function (t) {
  var obj = {log: { type:'file', level: {label:'debug', value:10} } }
  e = i.encode(obj, {section: 'prefix'})

  t.equal(e, expectF)
  t.end()
})

test("encode with whitespace", function (t) {
  var obj = {log: { type:'file', level: {label:'debug', value:10} } }
  e = i.encode(obj, {whitespace: true})

  t.equal(e, expectG)
  t.end()
})

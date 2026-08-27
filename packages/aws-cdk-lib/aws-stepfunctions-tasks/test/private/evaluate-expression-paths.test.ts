import { pathsInsideStringLiterals } from '../../lib/private/evaluate-expression-paths';

describe('pathsInsideStringLiterals', () => {
  describe('returns nothing for paths in code position', () => {
    test.each<[string, string]>([
      ['arithmetic', '$.a + $.b'],
      ['comparison and ternary', '$.a > $.b ? $.a : $.b'],
      ['method call arguments', 'Math.max($.a, $.b)'],
      ['JSON.parse of a value (no quotes)', 'JSON.parse($.body)'],
      ['array literal element', '[$.a, $.b]'],
      ['object literal value', '({ x: $.a })'],
      ['template interpolation', '`year=${$.year}`'],
      ['nested interpolation with an object literal', '`v=${ { a: $.a }.a }`'],
      ['string concat with the path outside the quotes', "$.first + ' ' + $.last"],
      ['a plain string with no path', "'just a literal'"],
      ['division operator', '$.a / $.b'],
      ['regex literal at the start of an expression', '/re/.test($.a)'],
      ['regex literal passed as an argument', '$.a.match(/re/)'],
      ['a path inside a regex body is not flagged', '/x$.y/.test($.z)'],
      ['quotes and backticks inside a regex do not open a string or template', "/'\"`/.test($.a)"],
      ['an escaped `/` inside a regex does not close it', '/a\\/b/.test($.a)'],
      ['a nested template interpolation', '`a${`b ${$.c}`}`'],
      ['an object literal inside interpolation, path after it', '`${ {a:1}+$.b }`'],
      ['no referenced paths at all', '2 + 2'],
    ])('%s', (_name, expression) => {
      expect(pathsInsideStringLiterals(expression)).toEqual([]);
    });
  });

  describe('flags a path inside a plain string', () => {
    test.each<[string, string, string]>([
      ['single-quoted string', "'year=$.year'", '$.year'],
      ['double-quoted string', '"hello $.user"', '$.user'],
      ['JSON.parse of a quoted path', "JSON.parse('$.arr')", '$.arr'],
      ['single quotes literal inside a double-quoted string', '"key=\'$.a\'"', '$.a'],
      ['double quotes escaped inside a single-quoted string', "'a \\\"b\\\" c $.x'", '$.x'],
      ['a quoted path passed as an argument', 'JSON.parse("$.a")', '$.a'],
      ['an escaped quote does not close the string', "'it\\'s $.a'", '$.a'],
      ['a quoted path in an argument after a regex', "/re/.test('$.a')", '$.a'],
      ['a quoted path in the second argument to .replace', "$.s.replace(/pat/g, '$.rep')", '$.rep'],
    ])('%s', (_name, expression, path) => {
      expect(pathsInsideStringLiterals(expression)).toEqual([{ path, context: 'plainString' }]);
    });
  });

  describe('flags a path inside template text', () => {
    test.each<[string, string, string]>([
      ['template literal text (no ${})', '`total: $.n`', '$.n'],
      ['literal braces without interpolation', '`{$.x}`', '$.x'],
      ['an escaped ${ that stays template text', '`a\\${$.b}`', '$.b'],
      ['inner template literal text', '`a${`b $.c`}`', '$.c'],
      ['a single-quoted path inside a template is template text', "`x=${$.a} and '$.b'`", '$.b'],
      ['a path in a template argument passed after a regex', '/re/.test(`year=$.year`)', '$.year'],
    ])('%s', (_name, expression, path) => {
      expect(pathsInsideStringLiterals(expression)).toEqual([{ path, context: 'templateText' }]);
    });
  });

  test('returns each offending path once, even when repeated', () => {
    expect(pathsInsideStringLiterals("'$.a' + '$.a'")).toEqual([{ path: '$.a', context: 'plainString' }]);
  });

  test('reports multiple distinct offending paths', () => {
    expect(pathsInsideStringLiterals("JSON.parse('$.a').concat('$.b')")).toEqual([
      { path: '$.a', context: 'plainString' },
      { path: '$.b', context: 'plainString' },
    ]);
  });

  test('distinguishes a path in code from a path in a string in the same expression', () => {
    // $.a is a real argument (safe); $.b is literal text inside the string (offending)
    expect(pathsInsideStringLiterals("f($.a) + 'x=$.b'")).toEqual([{ path: '$.b', context: 'plainString' }]);
  });

  test('treats an interpolated path as safe even alongside a raw one in the same template', () => {
    // ${$.good} is code; the bare $.bad is template text
    expect(pathsInsideStringLiterals('`${$.good}-$.bad`')).toEqual([{ path: '$.bad', context: 'templateText' }]);
  });

  describe('an unterminated literal marks the remainder as that literal', () => {
    test('unterminated single-quoted string', () => {
      expect(pathsInsideStringLiterals("'x=$.a")).toEqual([{ path: '$.a', context: 'plainString' }]);
    });

    test('unterminated double-quoted string', () => {
      expect(pathsInsideStringLiterals('"x=$.a')).toEqual([{ path: '$.a', context: 'plainString' }]);
    });

    test('unterminated template literal', () => {
      expect(pathsInsideStringLiterals('`x=$.a')).toEqual([{ path: '$.a', context: 'templateText' }]);
    });

    test('unterminated regex swallows the remainder without flagging any path', () => {
      expect(pathsInsideStringLiterals("/pat + '$.a'")).toEqual([]);
    });
  });
});

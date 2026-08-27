import type { Event } from '../../lib/aws-stepfunctions-tasks/eval-nodejs-handler';
import { handler } from '../../lib/aws-stepfunctions-tasks/eval-nodejs-handler';

beforeAll(() => {
  jest.spyOn(console, 'log').mockImplementation();
});

afterAll(() => {
  jest.restoreAllMocks();
});

test('with numbers', async () => {
  // GIVEN
  const event: Event = {
    expression: '$.a + $.b',
    expressionAttributeValues: {
      '$.a': 4,
      '$.b': 5,
    },
  };

  // THEN
  const evaluated = await handler(event);
  expect(evaluated).toBe(9);
});

test('with strings', async () => {
  // GIVEN
  const event: Event = {
    expression: '`${$.a} ${$.b}`',
    expressionAttributeValues: {
      '$.a': 'Hello',
      '$.b': 'world!',
    },
  };

  // THEN
  const evaluated = await handler(event);
  expect(evaluated).toBe('Hello world!');
});

test('resolves a template interpolation embedded in string text', async () => {
  // GIVEN
  const event: Event = {
    expression: '`Hello ${$.user}, you waited ${$.seconds} seconds`',
    expressionAttributeValues: {
      '$.user': 'Bob',
      '$.seconds': 5,
    },
  };

  // THEN
  const evaluated = await handler(event);
  expect(evaluated).toBe('Hello Bob, you waited 5 seconds');
});

test('with lists', async () => {
  // GIVEN
  const event: Event = {
    expression: '$.a.map(x => x * 2)',
    expressionAttributeValues: {
      '$.a': [1, 2, 3],
    },
  };

  // THEN
  const evaluated = await handler(event);
  expect(evaluated).toEqual([2, 4, 6]);
});

test('with duplicated entries', async () => {
  // GIVEN
  const event: Event = {
    expression: '$.a + $.a',
    expressionAttributeValues: {
      '$.a': 1,
    },
  };

  // THEN
  const evaluated = await handler(event);
  expect(evaluated).toBe(2);
});

test('with dash and underscore in path', async () => {
  // GIVEN
  const event: Event = {
    expression: '$.a_b + $.c-d + $[_e]',
    expressionAttributeValues: {
      '$.a_b': 1,
      '$.c-d': 2,
      '$[_e]': 3,
    },
  };

  // THEN
  const evaluated = await handler(event);
  expect(evaluated).toBe(6);
});

test('with nested (dotted) paths', async () => {
  // GIVEN
  const event: Event = {
    expression: '$.a.b + $.a.c',
    expressionAttributeValues: {
      '$.a.b': 2,
      '$.a.c': 3,
    },
  };

  // THEN
  const evaluated = await handler(event);
  expect(evaluated).toBe(5);
});

test('with no referenced paths', async () => {
  // GIVEN
  const event: Event = {
    expression: '1 + 2',
    expressionAttributeValues: {},
  };

  // THEN
  expect(await handler(event)).toBe(3);
});

test('resolves arbitrary expressions referencing the values', async () => {
  // GIVEN
  const event: Event = {
    expression: '(new Date($.epoch)).toUTCString()',
    expressionAttributeValues: {
      '$.epoch': 0,
    },
  };

  // THEN
  expect(await handler(event)).toBe('Thu, 01 Jan 1970 00:00:00 GMT');
});

test('preserves value types (object and array members)', async () => {
  // GIVEN
  const event: Event = {
    expression: '$.obj.count + $.list.length',
    expressionAttributeValues: {
      '$.obj': { count: 10 },
      '$.list': [1, 2, 3],
    },
  };

  // THEN
  expect(await handler(event)).toBe(13);
});

describe('overlapping paths resolve to the correct value', () => {
  test.each([
    ['$.a + $.ab', { '$.a': 1, '$.ab': 2 }, 3],
    ['$.ab + $.a', { '$.a': 1, '$.ab': 2 }, 3],
    ['$.a + $.abc + $.ab', { '$.a': 1, '$.ab': 2, '$.abc': 4 }, 7],
    ['$.a + $.a.b', { '$.a': 1, '$.a.b': 5 }, 6],
  ])('%s', async (expression, expressionAttributeValues, expected) => {
    expect(await handler({ expression, expressionAttributeValues })).toBe(expected);
  });
});

describe('referenced values are treated as data, not code', () => {
  test('a value equal to another path is not re-substituted', async () => {
    const evaluated = await handler({
      expression: '$.a + $.b',
      expressionAttributeValues: { '$.a': '$.b', '$.b': '+(2*21)+' },
    });
    expect(evaluated).toBe('$.b+(2*21)+');
  });

  test('special regex replacement patterns in a value are inert', async () => {
    const evaluated = await handler({
      expression: '$.a',
      expressionAttributeValues: { '$.a': '$& $` $\' $1' },
    });
    expect(evaluated).toBe('$& $` $\' $1');
  });

  test('a value cannot introduce executable code via string concatenation', async () => {
    const evaluated = await handler({
      expression: '$.a + $.b',
      expressionAttributeValues: { '$.a': 'a', '$.b': '`${1+1}`' },
    });
    expect(evaluated).toBe('a`${1+1}`');
  });

  test('a value that reaches globals is returned as data, not executed', async () => {
    const evaluated = await handler({
      expression: '$.a',
      expressionAttributeValues: { '$.a': 'process.exit(1)' },
    });
    expect(evaluated).toBe('process.exit(1)');
  });

  test('a value that resembles a template placeholder is not interpolated', async () => {
    const evaluated = await handler({
      expression: '$.a',
      expressionAttributeValues: { '$.a': '${7 * 7}' },
    });
    expect(evaluated).toBe('${7 * 7}');
  });

  test('a value that resembles a statement is returned as data', async () => {
    const evaluated = await handler({
      expression: '$.a + $.b',
      expressionAttributeValues: { '$.a': '$.b', '$.b': '; 2 * 21;' },
    });
    expect(evaluated).toBe('$.b; 2 * 21;');
  });

  test('a value used with a regex literal is matched as data, not executed', async () => {
    const evaluated = await handler({
      expression: '/^[a-z]+$/.test($.a)',
      expressionAttributeValues: { '$.a': '/; 2 * 21; /' },
    });
    expect(evaluated).toBe(false);
  });

  test('values with prototype-polluting keys stay inert', async () => {
    const evaluated = await handler({
      expression: '$.a',
      expressionAttributeValues: JSON.parse('{ "$.a": "ok", "__proto__": { "x": 1 }, "constructor": "c" }'),
    });
    expect(evaluated).toBe('ok');
    expect(({} as any).x).toBeUndefined();
  });
});

test('rejects when the expression references a path that has no value', async () => {
  await expect(handler({
    expression: '$.a + $.b',
    expressionAttributeValues: { '$.a': 1 },
  })).rejects.toThrow(ReferenceError);
});

describe('a path embedded inside a string literal is rejected, not folded in', () => {
  // Rewriting a path inside a string literal yields invalid syntax, so it fails
  // rather than folding the value into the string - regardless of the value.
  test.each([
    ['"hello $.user"', { '$.user': 'World' }],
    ['"hello $.user"', { '$.user': '+(2 * 21)+' }],
  ])('%s', async (expression, expressionAttributeValues) => {
    await expect(handler({ expression, expressionAttributeValues })).rejects.toThrow();
  });
});

describe('supports multi-statement expressions', () => {
  test('a var declaration followed by an expression', async () => {
    const evaluated = await handler({
      expression: 'var tmp = $.a + $.b; tmp * 2',
      expressionAttributeValues: { '$.a': 3, '$.b': 4 },
    });
    expect(evaluated).toBe(14);
  });

  test('a const declaration referencing a value', async () => {
    const evaluated = await handler({
      expression: 'const x = $.a; x + 1',
      expressionAttributeValues: { '$.a': 10 },
    });
    expect(evaluated).toBe(11);
  });
});

describe('module loading is not available', () => {
  // `require` is not in scope and `import` is not an expression.
  test.each([
    ['a require call', "require('fs')"],
    ['a static import', "import * as fs from 'fs'"],
  ])('%s throws', async (_name, expression) => {
    await expect(handler({ expression, expressionAttributeValues: {} })).rejects.toThrow();
  });
});

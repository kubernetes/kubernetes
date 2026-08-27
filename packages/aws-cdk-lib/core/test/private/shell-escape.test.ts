import { posixShellEscape } from '../../lib/private/shell-escape';

describe('posixShellEscape', () => {
  test('wraps a plain value in single quotes', () => {
    expect(posixShellEscape('hello')).toEqual("'hello'");
  });

  test('preserves an empty string as an empty shell word', () => {
    expect(posixShellEscape('')).toEqual("''");
  });

  test('preserves whitespace as part of a single word', () => {
    expect(posixShellEscape('/asset-output/python lib')).toEqual("'/asset-output/python lib'");
  });

  // Single-quoting is unconditional, so metacharacters are passed through
  // verbatim and the shell treats them as literal characters.
  test.each([
    ['command separator', 'a; touch /tmp/pwn'],
    ['command substitution', 'a$(touch /tmp/pwn)'],
    ['backtick substitution', 'a`touch /tmp/pwn`'],
    ['pipe', 'a | touch /tmp/pwn'],
    ['background and chain', 'a & touch /tmp/pwn && b'],
    ['redirect', 'a > /tmp/pwn'],
    ['newline', 'a\ntouch /tmp/pwn'],
    ['variable expansion', 'a$HOME'],
    ['glob', 'build/**'],
  ])('quotes %s as literal shell data', (_name, arg) => {
    expect(posixShellEscape(arg)).toEqual(`'${arg}'`);
  });

  test('escapes an embedded single quote by closing, escaping and reopening', () => {
    expect(posixShellEscape("it's")).toEqual("'it'\\''s'");
  });

  test('escapes every single quote in a value', () => {
    expect(posixShellEscape("'a'b'")).toEqual("''\\''a'\\''b'\\'''");
  });

  // Payload that can turn an interpolated shell argument into a command
  // sequence when single quotes are not escaped
  test('escapes single-quote breakout payload', () => {
    expect(posixShellEscape("x'; touch /tmp/pwn; echo '"))
      .toEqual("'x'\\''; touch /tmp/pwn; echo '\\'''");
  });
});

/**
 * Escapes a single argument for safe inclusion in a POSIX shell command.
 *
 * Every argument is single-quoted unconditionally (like Python's `shlex.quote`),
 * so any character that would otherwise have meaning to the shell is preserved
 * as a literal part of the value.
 */
export function posixShellEscape(arg: string): string {
  return "'" + arg.replace(/'/g, "'\\''") + "'";
}

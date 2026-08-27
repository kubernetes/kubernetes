/**
 * Matches the JSONPath tokens referenced by an expression.
 *
 * Shared by `EvaluateExpression._renderTask` (which binds the tokens) and by the
 * string-literal scan below (which detects them), so the two never drift apart.
 */
export const PATH_PATTERN = /\$[.\[][.a-zA-Z[\]0-9-_]+/g;

/**
 * Is the `/` at `slashIndex` the start of a regex literal, or the division operator?
 *
 * JavaScript treats `/` as division when a value has just been produced (a variable,
 * a number, or a closing `)`/`]`), and as the start of a regex everywhere else. We
 * approximate that by looking at the character just before the `/`: if it could be
 * the last character of a value (letter, digit, `_`, `$`, `)`, `]`), it's division;
 * otherwise, it's a regex.
 */
function startsRegexLiteral(expression: string, slashIndex: number): boolean {
  // Step over any whitespace immediately before the `/`.
  let j = slashIndex - 1;
  while (j >= 0 && /\s/.test(expression[j])) j--;
  // Nothing before the `/`? Then it has to be a regex - division needs a left operand.
  if (j < 0) return true;
  // If the previous character can end a value, it's division; otherwise, regex.
  return !/[a-zA-Z0-9_$)\]]/.test(expression[j]);
}

/**
 * Finds referenced JSONPath tokens written as literal text inside a string or template
 * literal, tagged by the kind of literal they sit in.
 *
 * At runtime the eval Lambda only resolves a path used as code (e.g. inside a `${...}`
 * interpolation or as a function argument). A path written as plain literal text never
 * resolves and would silently evaluate to the wrong value, so surfacing these lets the
 * construct fail fast at synth. This is a convenience check, not a strict guarantee: the
 * Lambda binds every referenced path as data regardless of where it appears.
 *
 * Detection is a single state-machine pass over five contexts (code, single-quoted string,
 * double-quoted string, template text, and regex literal). A `${...}` interpolation is treated
 * as code (a `templateExpression` frame), with a brace-depth stack so nested braces and
 * templates are handled, and an escaped delimiter never closes its literal. Inside a regex
 * literal, the scanner just walks over each character until the closing `/` - a path or a
 * quote in there is not looked at, so it is neither flagged nor able to open a string.
 *
 * Each row shows the raw `expression` value; any quote or backtick shown is part of the
 * expression itself.
 *
 * ```text
 * $.a + $.b                       -> []            (code)
 * JSON.parse($.body)              -> []            (code)
 * `year=${$.year}`                -> []            (interpolation)
 * 'year=$.year'                   -> [$.year]      (plain string)
 * "hello $.user"                  -> [$.user]      (plain string)
 * `total: $.n`                    -> [$.n]         (template text)
 * JSON.parse('$.a').concat('$.b') -> [$.a, $.b]    (paths in quoted args)
 * "key='$.a'"                     -> [$.a]         (single quotes literal inside a double-quoted string)
 * ```
 */
export function pathsInsideStringLiterals(
  expression: string,
): Array<{ path: string; context: 'plainString' | 'templateText' }> {
  // Literal context of each index, or undefined when the index is in code position.
  const context: Array<'plainString' | 'templateText' | undefined> = new Array(expression.length).fill(undefined);

  // `code` is top-level code position; `templateExpression` is code inside a `${...}`
  // interpolation. They behave identically while scanning, but are kept distinct so the
  // matching `}` knows whether to close an interpolation (returning to the surrounding
  // template) or is just balancing braces in ordinary code.
  type Frame =
    | { kind: 'code'; braceDepth: number }
    | { kind: 'templateExpression'; braceDepth: number }
    | { kind: 'squote' }
    | { kind: 'dquote' }
    | { kind: 'template' }
    | { kind: 'regex' };

  // Stack top is the current context; a `${...}` interpolation pushes a `templateExpression`
  // frame whose brace-depth counter lets the matching `}` return to the surrounding template.
  // The base `code` frame is never popped, so the stack is always non-empty.
  const stack: Frame[] = [{ kind: 'code', braceDepth: 0 }];
  let i = 0;

  while (i < expression.length) {
    const top = stack[stack.length - 1];
    const c = expression[i];

    if (top.kind === 'squote' || top.kind === 'dquote') {
      const closer = top.kind === 'squote' ? '\'' : '"';
      context[i] = 'plainString';
      if (c === '\\') {
        // An escaped char stays inside the string and cannot close it.
        context[i + 1] = 'plainString';
        i += 2;
        continue;
      }
      if (c === closer) {
        stack.pop();
      }
      i++;
      continue;
    }

    if (top.kind === 'regex') {
      // Skip to the closing `/`. `\` starts a regex escape sequence, so `\/`
      // (a literal slash in the pattern) doesn't count as the close.
      if (c === '\\') {
        i += 2;
        continue;
      }
      if (c === '/') {
        stack.pop();
      }
      i++;
      continue;
    }

    if (top.kind === 'template') {
      if (c === '\\') {
        // An escaped char (including `\$`, so `\${` does not open an interpolation).
        context[i] = 'templateText';
        context[i + 1] = 'templateText';
        i += 2;
        continue;
      }
      if (c === '`') {
        stack.pop();
        i++;
        continue;
      }
      if (c === '$' && expression[i + 1] === '{') {
        stack.push({ kind: 'templateExpression', braceDepth: 0 });
        i += 2;
        continue;
      }
      context[i] = 'templateText';
      i++;
      continue;
    }

    // top.kind === 'code' or 'templateExpression' (both are code position)
    if (c === '\'') {
      stack.push({ kind: 'squote' });
      i++;
      continue;
    }
    if (c === '"') {
      stack.push({ kind: 'dquote' });
      i++;
      continue;
    }
    if (c === '`') {
      stack.push({ kind: 'template' });
      i++;
      continue;
    }
    if (c === '/' && startsRegexLiteral(expression, i)) {
      stack.push({ kind: 'regex' });
      i++;
      continue;
    }
    if (c === '{') {
      top.braceDepth++;
      i++;
      continue;
    }
    if (c === '}') {
      if (top.kind === 'templateExpression' && top.braceDepth === 0) {
        stack.pop();
      } else if (top.braceDepth > 0) {
        top.braceDepth--;
      }
      i++;
      continue;
    }
    i++;
  }

  // Collect offending paths into a `Set` per context to de-dupe (a path can appear
  // in both a plain string and template text).
  const plainString = new Set<string>();
  const templateText = new Set<string>();
  // Reset lastIndex: PATH_PATTERN is a shared global (`/g`) regex that retains state across calls.
  PATH_PATTERN.lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = PATH_PATTERN.exec(expression)) !== null) {
    const ctx = context[match.index];
    if (ctx === 'plainString') {
      plainString.add(match[0]);
    } else if (ctx === 'templateText') {
      templateText.add(match[0]);
    }
  }

  return [
    ...Array.from(plainString, (path) => ({ path, context: 'plainString' as const })),
    ...Array.from(templateText, (path) => ({ path, context: 'templateText' as const })),
  ];
}

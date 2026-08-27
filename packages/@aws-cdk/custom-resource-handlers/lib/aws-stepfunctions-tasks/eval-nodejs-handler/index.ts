
/**
 * The event received by the Lambda function
 *
 * @internal
 */
export interface Event {
  /**
   * The expression to evaluate
   */
  readonly expression: string;

  /**
   * The expression attribute values
   */
  readonly expressionAttributeValues: { [key: string]: any };
}

function escapeRegex(x: string) {
  return x.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&');
}

export async function handler(event: Event): Promise<any> {
  console.log('Event: %j', { ...event, ResponseURL: '...' });

  // The expression refers to state values by JSONPath (e.g. `$.a`), which are not
  // variables in scope when it runs. Bind the values to a lookup object and rewrite
  // each path to read from it, so the expression can use their values.
  const values: { [key: string]: any } = Object.create(null);
  Object.assign(values, event.expressionAttributeValues);

  // Order the paths longest-first so that when one path is a prefix of another
  // (e.g. `$.a` and `$.ab`), the longer path is matched rather than the shorter
  // one matching inside it.
  const paths = Object.keys(values).sort((a, b) => b.length - a.length);

  // Rewrite each referenced path to a lookup on the values object,
  // e.g. `$.a + $.b` becomes `values["$.a"] + values["$.b"]`.
  let expression = event.expression;
  if (paths.length > 0) {
    const pathPattern = new RegExp(paths.map(escapeRegex).join('|'), 'g');
    expression = expression.replace(pathPattern, (path) => `values[${JSON.stringify(path)}]`);
  }
  console.log(`Expression: ${expression}`);

  // Evaluate the expression with the values available in scope, so any valid
  // expression - including multiple statements - works as written.
  return new Function('values', '__expr', 'return eval(__expr);')(values, expression);
}

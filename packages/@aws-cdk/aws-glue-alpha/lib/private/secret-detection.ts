import { Annotations, Token } from 'aws-cdk-lib/core';
import type { IConstruct } from 'constructs';

/**
 * Key names that suggest a value holds a credential or other sensitive data.
 * Matched case-insensitively against the keys of free-form string maps that are
 * emitted verbatim into the synthesized template.
 */
const SECRET_KEY_PATTERN = /pass(word|wd)?|secret|token|credential|api[-_]?key|private[-_]?key|access[-_]?key/i;

/**
 * Warn when a free-form key/value map appears to carry a plaintext secret.
 *
 * Glue connection properties and job arguments are open-ended string maps that
 * are emitted verbatim into the CloudFormation template. A literal secret placed
 * there is stored in plaintext in the template, `cdk.out`, and source control.
 *
 * This emits a suppressible synthesis-time warning (never an error - the map is a
 * legitimate escape hatch) when a key looks secret-like AND its value is a plain
 * literal. Values that are tokens - `SecretValue` (`{{resolve:...}}`), `Fn::*`
 * intrinsics, or other unresolved references - are assumed to be handled securely
 * and are not flagged.
 *
 * @param scope the construct used as the annotation scope
 * @param values the key/value map to inspect (may be undefined)
 * @param warningCode the `addWarningV2` acknowledgement code
 * @param remediation guidance appended to the warning message
 */
export function warnOnPlaintextSecrets(
  scope: IConstruct,
  values: { [key: string]: string } | undefined,
  warningCode: string,
  remediation: string,
): void {
  if (!values) {
    return;
  }

  const flagged = Object.entries(values)
    .filter(([key, value]) => SECRET_KEY_PATTERN.test(key) && !Token.isUnresolved(value))
    .map(([key]) => key);

  if (flagged.length === 0) {
    return;
  }

  Annotations.of(scope).addWarningV2(
    warningCode,
    `the following ${flagged.length === 1 ? 'key appears' : 'keys appear'} to hold a plaintext secret and will be stored ` +
      `in plaintext in the synthesized template: ${flagged.map(k => JSON.stringify(k)).join(', ')}. ${remediation}`,
  );
}

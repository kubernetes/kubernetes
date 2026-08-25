import * as cdk from '../../../core';
import { lit } from '../../../core/lib/private/literal-string';

/**
 * Throws an error if a duration is defined and not an integer number of seconds at or above a minimum.
 *
 * There is deliberately no upper bound. The timeouts validated here are governed by adjustable
 * service quotas (`Response timeout per origin`, `Keep-alive timeout per origin`), so the effective
 * maximum depends on the target account and can only be enforced by the service at deploy time.
 * A hardcoded ceiling rejects values the service would accept.
 *
 * The integer check is a defensive guard. `Duration.toSeconds()` already rejects fractional values,
 * so no input reaches it today.
 */
export function validateMinimumSeconds(name: string, min: number, duration?: cdk.Duration) {
  if (duration === undefined) { return; }
  const value = duration.toSeconds();
  if (!Number.isInteger(value) || value < min) {
    throw new cdk.UnscopedValidationError(lit`InvalidDurationRange`, `${name}: Must be an int ${min} seconds or greater; received ${value}.`);
  }
}

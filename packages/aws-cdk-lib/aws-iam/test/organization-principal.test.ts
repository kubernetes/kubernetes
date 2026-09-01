import { Stack } from '../../core';
import * as cdk from '../../core';
import * as iam from '../lib';

describe('OrganizationPrincipal', () => {
  test('accepts valid organization ID', () => {
    // WHEN / THEN
    expect(() => {
      new iam.OrganizationPrincipal('o-123456789012');
    }).not.toThrow();
  });

  test.each([
    ['empty string', ''],
    ['invalid prefix', 'invalid-org-id'],
    ['too short', 'o-short'],
    ['too long', 'o-thisnameistoooooooooooooooooolong'],
  ])('throws error for non-compliant organization ID format: %s', (_, invalidId) => {
    // WHEN / THEN
    expect(() => {
      new iam.OrganizationPrincipal(invalidId);
    }).toThrow(`Expected Organization ID must match regex pattern ^o-[a-z0-9]{10,32}$, received ${invalidId}`);
  });

  test('allows token as organization ID without validation', () => {
    // GIVEN
    const orgIdToken = cdk.Token.asString({ Ref: 'OrgId' });

    // WHEN / THEN
    expect(() => {
      new iam.OrganizationPrincipal(orgIdToken);
    }).not.toThrow();
  });

  test('creates correct policy fragment', () => {
    // GIVEN
    const stack = new Stack();

    // WHEN
    const principal = new iam.OrganizationPrincipal('o-123456789012');

    // THEN
    expect(stack.resolve(principal.policyFragment.principalJson)).toEqual({
      AWS: ['*'],
    });

    expect(stack.resolve(principal.policyFragment.conditions)).toEqual({
      StringEquals: {
        'aws:PrincipalOrgID': 'o-123456789012',
      },
    });
  });
});

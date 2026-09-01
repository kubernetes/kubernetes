import { Annotations, Template, Match } from '../../assertions';
import * as iam from '../../aws-iam';
import { App, Stack } from '../../core';
import * as cxapi from '../../cx-api';
import { EventBus } from '../lib';

/**
 * Helper functions for common assertions in EventBus grant tests
 */

/**
 * Verifies that no resource policy (EventBusPolicy resource) was created.
 */
function assertNoResourcePolicy(stack: Stack) {
  Template.fromStack(stack).resourceCountIs('AWS::Events::EventBusPolicy', 0);
}

/**
 * Verifies that no identity-based policy was created.
 */
function assertNoIdentityPolicy(stack: Stack) {
  Template.fromStack(stack).resourceCountIs('AWS::IAM::Policy', 0);
}

/**
 * Verifies that only an identity-based policy was created (no resource policy)
 */
function assertOnlyIdentityPolicy(stack: Stack, grant: iam.Grant) {
  Template.fromStack(stack).resourceCountIs('AWS::Events::EventBusPolicy', 0);
  Template.fromStack(stack).resourceCountIs('AWS::IAM::Policy', 1);
  expect(grant.principalStatement).toBeDefined();
  expect(grant.resourceStatement).toBeUndefined();
}

/**
 * Verifies that only a resource policy was created (no identity policy)
 */
function assertOnlyResourcePolicy(stack: Stack, grant: iam.Grant) {
  Template.fromStack(stack).resourceCountIs('AWS::Events::EventBusPolicy', 1);
  Template.fromStack(stack).resourceCountIs('AWS::IAM::Policy', 0);
  expect(grant.principalStatement).toBeUndefined();
  expect(grant.resourceStatement).toBeDefined();
}

/**
 * Verifies that no warnings were generated in the stack
 */
function assertNoWarnings(stack: Stack) {
  const warnings = Annotations.fromStack(stack).findWarning('*', Match.anyValue());
  expect(warnings).toHaveLength(0);
}

/**
 * Verifies that a warning was generated for the given path and message
 */
function assertHasWarning(stack: Stack, path: string, warningMessage: string, exactMatch: boolean = true) {
  if (exactMatch) {
    Annotations.fromStack(stack).hasWarning(path, warningMessage);
  } else {
    Annotations.fromStack(stack).hasWarning(path, Match.stringLikeRegexp(warningMessage));
  }
}

/**
 * Verifies the structure of an identity-based policy with the given resource ARN
 */
function assertIdentityBasedPolicy(stack: Stack, resourceArn: any) {
  Template.fromStack(stack).hasResourceProperties('AWS::IAM::Policy', {
    PolicyDocument: {
      Statement: [{
        Action: 'events:PutEvents',
        Effect: 'Allow',
        Resource: resourceArn,
      }],
      Version: '2012-10-17',
    },
  });
}

/**
 * Verifies the structure of a service principal resource policy
 */
function assertServicePrincipalResourcePolicy(stack: Stack, serviceName: string, conditions?: any) {
  Template.fromStack(stack).hasResourceProperties('AWS::Events::EventBusPolicy', {
    Statement: Match.objectLike({
      Effect: 'Allow',
      Action: 'events:PutEvents',
      Principal: { Service: serviceName },
      Resource: Match.anyValue(),
      Condition: conditions ? Match.objectLike(conditions) : Match.absent(),
    }),
  });
}

/**
 * Verifies the structure of a cross-account resource policy
 */
function assertCrossAccountResourcePolicy(stack: Stack, accountId: string) {
  Template.fromStack(stack).hasResourceProperties('AWS::Events::EventBusPolicy', {
    Statement: Match.objectLike({
      Effect: 'Allow',
      Action: 'events:PutEvents',
      Principal: {
        AWS: {
          'Fn::Join': ['', [
            'arn:',
            { Ref: 'AWS::Partition' },
            ':iam::' + accountId + ':root',
          ]],
        },
      },
      Resource: Match.anyValue(),
    }),
  });
}

/**
 * Verifies the structure of an organization resource policy
 */
function assertOrganizationResourcePolicy(stack: Stack, orgId: string) {
  Template.fromStack(stack).hasResourceProperties('AWS::Events::EventBusPolicy', {
    Statement: Match.objectLike({
      Effect: 'Allow',
      Action: 'events:PutEvents',
      Principal: { AWS: '*' },
      Resource: Match.anyValue(),
      Condition: {
        StringEquals: {
          'aws:PrincipalOrgID': orgId,
        },
      },
    }),
  });
}

describe('EventBus grants', () => {
  describe('with feature flag EVENTBUS_POLICY_SID_REQUIRED enabled', () => {
    let app: App;
    let stack: Stack;
    let eventBus: EventBus;
    const defaultSid = 'DefaultSid';

    beforeEach(() => {
      app = new App({
        context: {
          [cxapi.EVENTBUS_POLICY_SID_REQUIRED]: true,
        },
      });
      stack = new Stack(app, 'Stack1');
      eventBus = new EventBus(stack, 'EventBus');
    });

    describe('same-account scenarios', () => {
      test('creates only identity-based policy for IAM role', () => {
        // GIVEN
        const role = new iam.Role(stack, 'Role', {
          assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
        });

        // WHEN
        const grant = eventBus.grantPutEventsTo(role);

        // THEN
        assertIdentityBasedPolicy(stack, {
          'Fn::GetAtt': ['EventBus7B8748AA', 'Arn'],
        });
        assertOnlyIdentityPolicy(stack, grant);
        expect(grant.success).toBeTruthy();
      });

      test('creates only resource policy for service principal with source account condition', () => {
        // GIVEN
        const servicePrincipal = new iam.ServicePrincipal('states.amazonaws.com', {
          conditions: {
            StringEquals: {
              'aws:SourceAccount': Stack.of(stack).account,
            },
          },
        });

        // WHEN
        const grant = eventBus.grantPutEventsTo(servicePrincipal, defaultSid);

        // THEN
        assertServicePrincipalResourcePolicy(stack, 'states.amazonaws.com', {
          StringEquals: {
            'aws:SourceAccount': { Ref: 'AWS::AccountId' },
          },
        });
        assertOnlyResourcePolicy(stack, grant);
        expect(grant.success).toBeTruthy();
      });
    });

    describe('cross-account scenarios', () => {
      test('creates resource policy for cross-account role principal', () => {
        // GIVEN
        const otherAccountRole = new iam.AccountPrincipal('123456789012');

        // WHEN
        const grant = eventBus.grantPutEventsTo(otherAccountRole, defaultSid);

        // THEN
        assertCrossAccountResourcePolicy(stack, '123456789012');
        assertOnlyResourcePolicy(stack, grant);
        expect(grant.success).toBeTruthy();
      });

      test('creates resource policy for organization principal with condition', () => {
        // GIVEN
        const organizationPrincipal = new iam.OrganizationPrincipal('o-12345abcdef');

        // WHEN
        const grant = eventBus.grantPutEventsTo(organizationPrincipal, defaultSid);

        // THEN
        assertOrganizationResourcePolicy(stack, 'o-12345abcdef');
        assertOnlyResourcePolicy(stack, grant);
        expect(grant.success).toBeTruthy();
      });
    });

    describe('imported event bus scenarios', () => {
      test('adds warning when granting to service principal on imported bus', () => {
        // GIVEN
        const importedEventBus = EventBus.fromEventBusName(stack, 'ImportedBus', 'external-bus');
        const servicePrincipal = new iam.ServicePrincipal('states.amazonaws.com');

        // WHEN
        const grant = importedEventBus.grantPutEventsTo(servicePrincipal);

        // THEN
        assertNoResourcePolicy(stack);
        expect(grant.success).toBeTruthy();
        assertHasWarning(stack, '/Stack1/ImportedBus',
          'Unable to add necessary permissions to imported target event bus: arn:\\${Token\\[AWS\\.Partition\\.[0-9]+\\]}:events:\\${Token\\[AWS\\.Region\\.[0-9]+\\]}:\\${Token\\[AWS\\.AccountId\\.[0-9]+\\]}:event-bus/external-bus \\[ack: @aws-cdk/aws-events:eventBusAddToResourcePolicy\\]',
          false);
      });

      test('creates identity-based policy for imported event bus from ARN without warnings', () => {
        // GIVEN
        const importedEventBus = EventBus.fromEventBusArn(stack, 'ImportedBus', 'arn:aws:events:us-east-1:123456789012:event-bus/event-bus-name');
        const role = new iam.Role(stack, 'Role', {
          assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
        });

        // WHEN
        const grant = importedEventBus.grantPutEventsTo(role);

        // THEN
        assertIdentityBasedPolicy(stack, 'arn:aws:events:us-east-1:123456789012:event-bus/event-bus-name');
        assertOnlyIdentityPolicy(stack, grant);
        assertNoWarnings(stack);
        expect(grant.success).toBeTruthy();
      });

      test('creates identity-based policy for imported event bus from name without warnings', () => {
        // GIVEN
        const importedEventBus = EventBus.fromEventBusName(stack, 'ImportedBus', 'externalEventBusName');
        const role = new iam.Role(stack, 'Role', {
          assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
        });

        // WHEN
        const grant = importedEventBus.grantPutEventsTo(role);

        // THEN
        assertIdentityBasedPolicy(stack, {
          'Fn::Join': ['', [
            'arn:',
            { Ref: 'AWS::Partition' },
            ':events:',
            { Ref: 'AWS::Region' },
            ':',
            { Ref: 'AWS::AccountId' },
            ':event-bus/externalEventBusName',
          ]],
        });
        assertOnlyIdentityPolicy(stack, grant);
        assertNoWarnings(stack);
        expect(grant.success).toBeTruthy();
      });
    });

    describe('cross-stack scenarios', () => {
      test('creates identity-based policy in different stack', () => {
        // GIVEN
        const stack2 = new Stack(app, 'Stack2');
        const role = new iam.Role(stack2, 'Role', {
          assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
        });

        // WHEN
        const grant = eventBus.grantPutEventsTo(role, defaultSid);

        // THEN
        Template.fromStack(stack).resourceCountIs('AWS::Events::EventBusPolicy', 0);
        assertIdentityBasedPolicy(stack2, {
          'Fn::ImportValue': Match.stringLikeRegexp('Stack1:ExportsOutput.*'),
        });
        assertOnlyIdentityPolicy(stack2, grant);
        expect(grant.success).toBeTruthy();
      });

      test('creates identity-based policy for imported event bus in different stack', () => {
        // GIVEN
        const stack2 = new Stack(app, 'Stack2');
        const importedBus = EventBus.fromEventBusArn(stack2, 'ImportedBus', eventBus.eventBusArn);
        const role = new iam.Role(stack2, 'Role', {
          assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
        });

        // WHEN
        const grant = importedBus.grantPutEventsTo(role);

        // THEN
        assertIdentityBasedPolicy(stack2, {
          'Fn::ImportValue': Match.stringLikeRegexp('Stack1:ExportsOutputFnGetAtt.*'),
        });
        assertOnlyIdentityPolicy(stack2, grant);
        expect(grant.success).toBeTruthy();
      });
    });

    test('creates identity-based policy for imported role by name', () => {
      // GIVEN
      const role = iam.Role.fromRoleName(stack, 'ImportedRoleId', 'importedRoleName');

      // WHEN
      const grant = eventBus.grantPutEventsTo(role);

      // THEN
      assertIdentityBasedPolicy(stack, {
        'Fn::GetAtt': ['EventBus7B8748AA', 'Arn'],
      });
      assertOnlyIdentityPolicy(stack, grant);
      expect(grant.success).toBeTruthy();
    });

    test('creates identity-based policy for imported role by ARN', () => {
      // GIVEN
      const role = iam.Role.fromRoleArn(stack, 'ImportedRoleId', 'arn:aws:iam::123456789012:role/AdminRoles/importedRoleName');

      // WHEN
      const grant = eventBus.grantPutEventsTo(role);

      // THEN
      assertIdentityBasedPolicy(stack, {
        'Fn::GetAtt': ['EventBus7B8748AA', 'Arn'],
      });
      assertOnlyIdentityPolicy(stack, grant);
      expect(grant.success).toBeTruthy();
    });

    test('creates resource-based policy for imported role by ARN when flagged not mutable', () => {
      // GIVEN
      const role = iam.Role.fromRoleArn(stack, 'ImportedRoleId', 'arn:aws:iam::123456789012:role/AdminRoles/importedRoleName', {
        mutable: false,
        addGrantsToResources: true,
      });

      // WHEN
      const grant = eventBus.grantPutEventsTo(role, 'ImportedRoleGrant');

      // THEN
      assertOnlyResourcePolicy(stack, grant);
      Template.fromStack(stack).hasResourceProperties('AWS::Events::EventBusPolicy', {
        StatementId: 'cdk-ImportedRoleGrant',
        Statement: Match.objectLike({
          Effect: 'Allow',
          Action: 'events:PutEvents',
          Principal: {
            AWS: 'arn:aws:iam::123456789012:role/AdminRoles/importedRoleName',
          },
          Resource: Match.anyValue(),
        }),
      });
      expect(grant.success).toBeTruthy();
    });

    describe('multiple grants with the same event bus', () => {
      test('creates appropriate policies for different principal types', () => {
        // GIVEN
        const servicePrincipal = new iam.ServicePrincipal('states.amazonaws.com');
        const role = new iam.Role(stack, 'Role', {
          assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
        });
        const accountPrincipal = new iam.AccountPrincipal('123456789012');

        // WHEN
        const grant1 = eventBus.grantPutEventsTo(servicePrincipal, `${defaultSid}-1`);
        const grant2 = eventBus.grantPutEventsTo(role, `${defaultSid}-2`);
        const grant3 = eventBus.grantPutEventsTo(accountPrincipal, `${defaultSid}-3`);

        // THEN
        // Count total policies
        Template.fromStack(stack).resourceCountIs('AWS::Events::EventBusPolicy', 2); // Service principal and account principal
        Template.fromStack(stack).resourceCountIs('AWS::IAM::Policy', 1); // Role

        // Verify service principal policy
        assertServicePrincipalResourcePolicy(stack, 'states.amazonaws.com');
        expect(grant1.success).toBeTruthy();

        // Verify role policy
        assertIdentityBasedPolicy(stack, {
          'Fn::GetAtt': ['EventBus7B8748AA', 'Arn'],
        });
        expect(grant2.success).toBeTruthy();

        // Verify cross-account policy
        assertCrossAccountResourcePolicy(stack, '123456789012');
        expect(grant3.success).toBeTruthy();
      });

      test('creates separate identity-based policies for different IAM principals', () => {
        // GIVEN
        const role1 = new iam.Role(stack, 'Role1', {
          assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
        });
        const role2 = new iam.Role(stack, 'Role2', {
          assumedBy: new iam.ServicePrincipal('lambda.amazonaws.com'),
        });

        // WHEN
        const grant1 = eventBus.grantPutEventsTo(role1, `${defaultSid}-1`);
        const grant2 = eventBus.grantPutEventsTo(role2, `${defaultSid}-2`);

        // THEN
        Template.fromStack(stack).resourceCountIs('AWS::IAM::Policy', 2);
        expect(grant1.success).toBeTruthy();
        expect(grant2.success).toBeTruthy();
      });
    });
  });

  describe('with feature flag EVENTBUS_POLICY_SID_REQUIRED disabled', () => {
    let app: App;
    let stack: Stack;
    let eventBus: EventBus;

    beforeEach(() => {
      app = new App({
        context: {
          [cxapi.EVENTBUS_POLICY_SID_REQUIRED]: false,
        },
      });
      stack = new Stack(app, 'Stack1');
      eventBus = new EventBus(stack, 'EventBus');
    });

    describe('feature flag behavior', () => {
      test('fails to grant to service principal without sid', () => {
        // GIVEN
        const servicePrincipal = new iam.ServicePrincipal('states.amazonaws.com');

        // WHEN
        const grant = eventBus.grantPutEventsTo(servicePrincipal);

        // THEN
        assertNoResourcePolicy(stack);
        assertNoIdentityPolicy(stack);
        assertHasWarning(stack, '/Stack1/EventBus',
          'Unable to grant PutEvents to service principal: Statement ID is required for EventBus resource policies. Either provide a \'sid\' parameter or enable the \'@aws-cdk/aws-events:requireEventBusPolicySid\' feature flag. [ack: @aws-cdk/aws-events:eventBusServicePrincipalGrant]');
        expect(grant.success).toBeFalsy();
      });

      test('fails to grant to service principal even with sid provided', () => {
        // GIVEN
        const servicePrincipal = new iam.ServicePrincipal('states.amazonaws.com');
        const explicitSid = 'MyCustomSid';

        // WHEN
        const grant = eventBus.grantPutEventsTo(servicePrincipal, explicitSid);

        // THEN
        assertNoResourcePolicy(stack);
        assertNoIdentityPolicy(stack);
        assertHasWarning(stack, '/Stack1/EventBus',
          'Unable to grant PutEvents to service principal: Statement ID is required for EventBus resource policies. Either provide a \'sid\' parameter or enable the \'@aws-cdk/aws-events:requireEventBusPolicySid\' feature flag. [ack: @aws-cdk/aws-events:eventBusServicePrincipalGrant]');
        expect(grant.success).toBeFalsy();
      });
    });
  });
});

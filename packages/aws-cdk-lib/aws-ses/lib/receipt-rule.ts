import { Construct } from 'constructs';
import type { IReceiptRuleAction } from './receipt-rule-action';
import { CfnReceiptRule } from './ses.generated';
import * as iam from '../../aws-iam';
import type { IResource } from '../../core';
import { Aws, Resource, ValidationError } from '../../core';
import type { IArrayBox } from '../../core/lib/helpers-internal';
import { Box } from '../../core/lib/helpers-internal';
import { addConstructMetadata, MethodMetadata } from '../../core/lib/metadata-resource';
import { noBoxStackTraces } from '../../core/lib/no-box-stack-traces';
import { lit } from '../../core/lib/private/literal-string';
import { propertyInjectable } from '../../core/lib/prop-injectable';
import { DropSpamSingletonFunction } from '../../custom-resource-handlers/dist/aws-ses/drop-spam-provider.generated';
import type { IReceiptRuleSetRef, IReceiptRuleRef, ReceiptRuleReference } from '../../interfaces/generated/aws-ses-interfaces.generated';

/**
 * A receipt rule.
 */
export interface IReceiptRule extends IResource, IReceiptRuleRef {
  /**
   * The name of the receipt rule.
   * @attribute
   */
  readonly receiptRuleName: string;
}

/**
 * The type of TLS policy for a receipt rule.
 */
export enum TlsPolicy {
  /**
   * Do not check for TLS.
   */
  OPTIONAL = 'Optional',

  /**
   * Bounce emails that are not received over TLS.
   */
  REQUIRE = 'Require',
}

/**
 * Options to add a receipt rule to a receipt rule set.
 */
export interface ReceiptRuleOptions {
  /**
   * An ordered list of actions to perform on messages that match at least
   * one of the recipient email addresses or domains specified in the
   * receipt rule.
   *
   * @default - No actions.
   */
  readonly actions?: IReceiptRuleAction[];

  /**
   * An existing rule after which the new rule will be placed.
   *
   * @default - The new rule is inserted at the beginning of the rule list.
   */
  readonly after?: IReceiptRuleRef;

  /**
   * Whether the rule is active.
   *
   * @default true
   */
  readonly enabled?: boolean;

  /**
   * The name for the rule
   *
   * @default - A CloudFormation generated name.
   */
  readonly receiptRuleName?: string;

  /**
   * The recipient domains and email addresses that the receipt rule applies to.
   *
   * @default - Match all recipients under all verified domains.
   */
  readonly recipients?: string[];

  /**
   * Whether to scan for spam and viruses.
   *
   * @default false
   */
  readonly scanEnabled?: boolean;

  /**
   * Whether Amazon SES should require that incoming email is delivered over a
   * connection encrypted with Transport Layer Security (TLS).
   *
   * @default - Optional which will not check for TLS.
   */
  readonly tlsPolicy?: TlsPolicy;
}

/**
 * Construction properties for a ReceiptRule.
 */
export interface ReceiptRuleProps extends ReceiptRuleOptions {
  /**
   * The name of the rule set that the receipt rule will be added to.
   */
  readonly ruleSet: IReceiptRuleSetRef;
}

/**
 * Properties of a reference to an existing receipt rule.
 */
export interface ReceiptRuleAttributes {
  /**
   * The rule set that the receipt rule belongs to.
   */
  readonly ruleSet: IReceiptRuleSetRef;

  /**
   * The name of the receipt rule.
   */
  readonly receiptRuleName: string;
}

/**
 * A new receipt rule.
 */
@propertyInjectable
@noBoxStackTraces
export class ReceiptRule extends Resource implements IReceiptRule {
  /**
   * Uniquely identifies this class.
   */
  public static readonly PROPERTY_INJECTION_ID: string = 'aws-cdk-lib.aws-ses.ReceiptRule';

  /**
   * Import an existing receipt rule by its name.
   *
   * A name alone does not identify the rule set the rule belongs to, so `ruleSetName` is not available
   * on the returned reference. Use `fromReceiptRuleAttributes()` if you need it.
   */
  public static fromReceiptRuleName(scope: Construct, id: string, receiptRuleName: string): IReceiptRule {
    class Import extends Resource implements IReceiptRule {
      public readonly receiptRuleName = receiptRuleName;

      public get receiptRuleRef(): ReceiptRuleReference {
        const self = this;
        return {
          ruleName: receiptRuleName,
          get ruleSetName(): string {
            throw new ValidationError(lit`CannotAccessRuleSetNameOfImportedReceiptRule`,
              'ruleSetName is not available on a ReceiptRule imported by rule name; use ReceiptRule.fromReceiptRuleAttributes() to get a complete reference', self);
          },
        };
      }
    }
    return new Import(scope, id);
  }

  /**
   * Import an existing receipt rule from its attributes.
   */
  public static fromReceiptRuleAttributes(scope: Construct, id: string, attrs: ReceiptRuleAttributes): IReceiptRule {
    class Import extends Resource implements IReceiptRule {
      public readonly receiptRuleName = attrs.receiptRuleName;

      public get receiptRuleRef(): ReceiptRuleReference {
        return {
          ruleName: attrs.receiptRuleName,
          ruleSetName: attrs.ruleSet.receiptRuleSetRef.ruleSetName,
        };
      }
    }
    return new Import(scope, id);
  }

  public readonly receiptRuleName: string;
  private readonly ruleSetName: string;
  private readonly actions: IArrayBox<CfnReceiptRule.ActionProperty> = Box.fromArray();

  public get receiptRuleRef(): ReceiptRuleReference {
    return {
      ruleName: this.receiptRuleName,
      ruleSetName: this.ruleSetName,
    };
  }

  constructor(scope: Construct, id: string, props: ReceiptRuleProps) {
    super(scope, id, {
      physicalName: props.receiptRuleName,
    });
    // Enhanced CDK Analytics Telemetry
    addConstructMetadata(this, props);

    this.ruleSetName = props.ruleSet.receiptRuleSetRef.ruleSetName;

    const resource = new CfnReceiptRule(this, 'Resource', {
      after: props.after?.receiptRuleRef.ruleName,
      rule: {
        actions: this.actions,
        enabled: props.enabled ?? true,
        name: this.physicalName,
        recipients: props.recipients,
        scanEnabled: props.scanEnabled,
        tlsPolicy: props.tlsPolicy,
      },
      ruleSetName: this.ruleSetName,
    });

    this.receiptRuleName = resource.ref;

    for (const action of props.actions || []) {
      this.addAction(action);
    }
  }

  /**
   * Adds an action to this receipt rule.
   */
  @MethodMetadata()
  public addAction(action: IReceiptRuleAction) {
    this.actions.push(action.bind(this));
  }
}

export interface DropSpamReceiptRuleProps extends ReceiptRuleProps {

}

/**
 * A rule added at the top of the rule set to drop spam/virus.
 *
 * @see https://docs.aws.amazon.com/ses/latest/DeveloperGuide/receiving-email-action-lambda-example-functions.html
 */
@propertyInjectable
export class DropSpamReceiptRule extends Construct {
  /**
   * Uniquely identifies this class.
   */
  public static readonly PROPERTY_INJECTION_ID: string = 'aws-cdk-lib.aws-ses.DropSpamReceiptRule';

  public readonly rule: ReceiptRule;

  constructor(scope: Construct, id: string, props: DropSpamReceiptRuleProps) {
    super(scope, id);

    const fn = new DropSpamSingletonFunction(this, 'Function', {
      uuid: '224e77f9-a32e-4b4d-ac32-983477abba16',
    });

    fn.addPermission('AllowSes', {
      action: 'lambda:InvokeFunction',
      principal: new iam.ServicePrincipal('ses.amazonaws.com'),
      sourceAccount: Aws.ACCOUNT_ID,
    });

    this.rule = new ReceiptRule(this, 'Rule', {
      actions: [
        {
          bind: () => ({
            lambdaAction: {
              functionArn: fn.functionArn,
              invocationType: 'RequestResponse',
            },
          }),
        },
      ],
      scanEnabled: true,
      ruleSet: props.ruleSet,
    });
  }
}

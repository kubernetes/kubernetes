import type { IResource } from 'aws-cdk-lib';
import { Annotations, Resource, Token } from 'aws-cdk-lib';
import type { IInputSecurityGroupRef, InputSecurityGroupReference } from 'aws-cdk-lib/aws-medialive';
import { CfnInputSecurityGroup } from 'aws-cdk-lib/aws-medialive';
import { addConstructMetadata } from 'aws-cdk-lib/core/lib/metadata-resource';
import { propertyInjectable } from 'aws-cdk-lib/core/lib/prop-injectable';
import type { Construct } from 'constructs';
import { extractResourceId } from './shared';

/**
 * Represents a MediaLive Input Security Group.
 */
export interface IInputSecurityGroup extends IResource, IInputSecurityGroupRef {
  /**
   * The ARN of the input security group.
   * @attribute
   */
  readonly inputSecurityGroupArn: string;
  /**
   * The ID of the input security group.
   * @attribute
   */
  readonly inputSecurityGroupId: string;
}

/**
 * Properties for creating a MediaLive Input Security Group.
 */
export interface InputSecurityGroupProps {
  /**
   * The list of IPv4 CIDR addresses to allow.
   */
  readonly allowlistRules: string[];
  /**
   * Tags to add to the input security group.
   * @default - no tags
   */
  readonly tags?: { [key: string]: string };
}

/**
 * Defines an AWS Elemental MediaLive Input Security Group.
 *
 * An input security group controls which IPv4 CIDR blocks can push content to a push-type input.
 */
@propertyInjectable
export class InputSecurityGroup extends Resource implements IInputSecurityGroup {
  /** Uniquely identifies this class. */
  public static readonly PROPERTY_INJECTION_ID: string = '@aws-cdk.aws-medialive-alpha.InputSecurityGroup';

  /** Import an existing input security group by its ARN. The id is parsed out of the ARN. */
  public static fromInputSecurityGroupArn(scope: Construct, id: string, inputSecurityGroupArn: string): IInputSecurityGroup {
    const inputSecurityGroupId = extractResourceId(inputSecurityGroupArn, 'InputSecurityGroup');

    class Import extends Resource implements IInputSecurityGroup {
      public readonly inputSecurityGroupArn = inputSecurityGroupArn;
      public readonly inputSecurityGroupId = inputSecurityGroupId;
      public get inputSecurityGroupRef(): InputSecurityGroupReference {
        return {
          inputSecurityGroupId: this.inputSecurityGroupId,
          inputSecurityGroupArn: this.inputSecurityGroupArn,
        };
      }
    }
    return new Import(scope, id);
  }

  public readonly inputSecurityGroupArn: string;
  public readonly inputSecurityGroupId: string;

  /** A reference to this Input Security Group resource. */
  public get inputSecurityGroupRef(): InputSecurityGroupReference {
    return {
      inputSecurityGroupId: this.inputSecurityGroupId,
      inputSecurityGroupArn: this.inputSecurityGroupArn,
    };
  }

  constructor(scope: Construct, id: string, props: InputSecurityGroupProps) {
    super(scope, id);

    addConstructMetadata(this, props);

    // Warn on fully-open CIDR blocks (0.0.0.0/0 or any /0 prefix).
    for (const cidr of props.allowlistRules) {
      if (!Token.isUnresolved(cidr) && /\/0$/.test(cidr.trim())) {
        Annotations.of(this).addWarningV2(
          '@aws-cdk/aws-medialive-alpha:openAllowlistCidr',
          `Allowlist CIDR '${cidr}' allows push requests from any IP. Restrict to the narrowest range your sources need.`,
        );
      }
    }

    const resource = new CfnInputSecurityGroup(this, 'Resource', {
      whitelistRules: props.allowlistRules.map(cidr => ({ cidr })),
      tags: props.tags ? Object.entries(props.tags).map(([key, value]) => ({ key, value })) : undefined,
    });

    this.inputSecurityGroupArn = resource.attrArn;
    this.inputSecurityGroupId = resource.ref;
  }
}

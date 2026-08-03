import type { Construct } from 'constructs';
import { CfnDBSubnetGroup } from './rds.generated';
import * as ec2 from '../../aws-ec2';
import type { IResource, RemovalPolicy } from '../../core';
import { ArnFormat, Resource, Stack, Token } from '../../core';
import { addConstructMetadata } from '../../core/lib/metadata-resource';
import { propertyInjectable } from '../../core/lib/prop-injectable';
import type { aws_rds } from '../../interfaces';

/**
 * Interface for a subnet group.
 */
export interface ISubnetGroup extends IResource, aws_rds.IDBSubnetGroupRef {
  /**
   * The name of the subnet group.
   * @attribute
   */
  readonly subnetGroupName: string;
}

/**
 * Properties for creating a SubnetGroup.
 */
export interface SubnetGroupProps {
  /**
   * Description of the subnet group.
   */
  readonly description: string;

  /**
   * The VPC to place the subnet group in.
   */
  readonly vpc: ec2.IVpc;

  /**
   * The name of the subnet group.
   *
   * @default - a name is generated
   */
  readonly subnetGroupName?: string;

  /**
   * Which subnets within the VPC to associate with this group.
   *
   * @default - private subnets
   */
  readonly vpcSubnets?: ec2.SubnetSelection;

  /**
   * The removal policy to apply when the subnet group are removed
   * from the stack or replaced during an update.
   *
   * @default RemovalPolicy.DESTROY
   */
  readonly removalPolicy?: RemovalPolicy;
}

/**
 * Class for creating a RDS DB subnet group
 *
 * @resource AWS::RDS::DBSubnetGroup
 */
@propertyInjectable
export class SubnetGroup extends Resource implements ISubnetGroup {
  /** Uniquely identifies this class. */
  public static readonly PROPERTY_INJECTION_ID: string = 'aws-cdk-lib.aws-rds.SubnetGroup';

  /**
   * Imports an existing subnet group by name.
   */
  public static fromSubnetGroupName(scope: Construct, id: string, subnetGroupName: string): ISubnetGroup {
    return new class extends Resource implements ISubnetGroup {
      public readonly subnetGroupName = subnetGroupName;
      public get dbSubnetGroupRef(): aws_rds.DBSubnetGroupReference {
        return {
          dbSubnetGroupArn: SubnetGroup.groupArn(this, this.subnetGroupName),
          dbSubnetGroupName: this.subnetGroupName,
        };
      }
    }(scope, id);
  }

  private static groupArn(scope: Construct, subnetGroupName: string): string {
    return Stack.of(scope).formatArn({
      service: 'rds',
      resource: 'subgrp',
      resourceName: subnetGroupName,
      arnFormat: ArnFormat.COLON_RESOURCE_NAME,
    });
  }

  public readonly subnetGroupName: string;

  /**
   * A reference to this subnet group
   */
  public get dbSubnetGroupRef(): aws_rds.DBSubnetGroupReference {
    return {
      dbSubnetGroupArn: SubnetGroup.groupArn(this, this.subnetGroupName),
      dbSubnetGroupName: this.subnetGroupName,
    };
  }

  constructor(scope: Construct, id: string, props: SubnetGroupProps) {
    super(scope, id);
    // Enhanced CDK Analytics Telemetry
    addConstructMetadata(this, props);

    const { subnetIds } = props.vpc.selectSubnets(props.vpcSubnets ?? { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS });

    // Using 'Default' as the resource id for historical reasons (usage from `Instance` and `Cluster`).
    const subnetGroup = new CfnDBSubnetGroup(this, 'Default', {
      dbSubnetGroupDescription: props.description,
      // names are actually stored by RDS changed to lowercase on the server side,
      // and not lowercasing them in CloudFormation means things like { Ref }
      // do not work correctly
      dbSubnetGroupName: Token.isUnresolved(props.subnetGroupName)
        ? props.subnetGroupName
        : props.subnetGroupName?.toLowerCase(),
      subnetIds,
    });

    if (props.removalPolicy) {
      subnetGroup.applyRemovalPolicy(props.removalPolicy);
    }

    this.subnetGroupName = subnetGroup.ref;
  }
}

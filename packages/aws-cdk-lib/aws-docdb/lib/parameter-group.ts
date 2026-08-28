import type { Construct } from 'constructs';
import { CfnDBClusterParameterGroup } from './docdb.generated';
import type { IResource } from '../../core';
import { Resource } from '../../core';
import { addConstructMetadata } from '../../core/lib/metadata-resource';
import { propertyInjectable } from '../../core/lib/prop-injectable';
import type { IDBClusterParameterGroupRef, DBClusterParameterGroupReference } from '../../interfaces/generated/aws-docdb-interfaces.generated';

/**
 * A parameter group
 */
export interface IClusterParameterGroup extends IResource, IDBClusterParameterGroupRef {
  /**
   * The name of this parameter group
   */
  readonly parameterGroupName: string;
}

/**
 * A new cluster or instance parameter group
 */
abstract class ClusterParameterGroupBase extends Resource implements IClusterParameterGroup {
  /**
   * Imports a parameter group
   */
  public static fromParameterGroupName(scope: Construct, id: string, parameterGroupName: string): IClusterParameterGroup {
    class Import extends Resource implements IClusterParameterGroup {
      public readonly parameterGroupName = parameterGroupName;

      public get dbClusterParameterGroupRef(): DBClusterParameterGroupReference {
        return {
          dbClusterParameterGroupName: this.parameterGroupName,
        };
      }
    }
    return new Import(scope, id);
  }

  /**
   * The name of the parameter group
   */
  public abstract readonly parameterGroupName: string;

  /**
   * A reference to this parameter group.
   */
  public get dbClusterParameterGroupRef(): DBClusterParameterGroupReference {
    return {
      dbClusterParameterGroupName: this.parameterGroupName,
    };
  }
}

/**
 * Properties for a cluster parameter group
 */
export interface ClusterParameterGroupProps {
  /**
   * Description for this parameter group
   *
   * @default a CDK generated description
   */
  readonly description?: string;

  /**
   * Database family of this parameter group
   */
  readonly family: string;

  /**
   * The name of the cluster parameter group
   *
   * @default A CDK generated name for the cluster parameter group
   */
  readonly dbClusterParameterGroupName?: string;

  /**
   * The parameters in this parameter group
   */
  readonly parameters: { [key: string]: string };
}

/**
 * A cluster parameter group
 *
 * @resource AWS::DocDB::DBClusterParameterGroup
 */
@propertyInjectable
export class ClusterParameterGroup extends ClusterParameterGroupBase implements IClusterParameterGroup {
  /** Uniquely identifies this class. */
  public static readonly PROPERTY_INJECTION_ID: string = 'aws-cdk-lib.aws-docdb.ClusterParameterGroup';
  /**
   * The name of the parameter group
   */
  public readonly parameterGroupName: string;

  constructor(scope: Construct, id: string, props: ClusterParameterGroupProps) {
    super(scope, id);
    // Enhanced CDK Analytics Telemetry
    addConstructMetadata(this, props);

    const resource = new CfnDBClusterParameterGroup(this, 'Resource', {
      name: props.dbClusterParameterGroupName,
      description: props.description || `Cluster parameter group for ${props.family}`,
      family: props.family,
      parameters: props.parameters,
    });

    this.parameterGroupName = resource.ref;
  }
}

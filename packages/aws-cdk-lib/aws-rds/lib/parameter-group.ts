import type { Construct } from 'constructs';
import type { IEngine } from './engine';
import { CfnDBClusterParameterGroup, CfnDBParameterGroup } from './rds.generated';
import type { IResource } from '../../core';
import { ArnFormat, RemovalPolicy, Resource, Stack } from '../../core';
import { ValidationError } from '../../core/lib/errors';
import type { IMapBox, IReadableBox } from '../../core/lib/helpers-internal';
import { Box } from '../../core/lib/helpers-internal';
import { addConstructMetadata, MethodMetadata } from '../../core/lib/metadata-resource';
import { lit } from '../../core/lib/private/literal-string';
import { propertyInjectable } from '../../core/lib/prop-injectable';
import type { aws_rds } from '../../interfaces';

/**
 * Options for `IParameterGroup.bindToCluster`.
 * Empty for now, but can be extended later.
 */
export interface ParameterGroupClusterBindOptions {
}

/**
 * The type returned from `IParameterGroup.bindToCluster`.
 */
export interface ParameterGroupClusterConfig {
  /** The name of this parameter group. */
  readonly parameterGroupName: string;
}

/**
 * Options for `IParameterGroup.bindToInstance`.
 * Empty for now, but can be extended later.
 */
export interface ParameterGroupInstanceBindOptions {
}

/**
 * The type returned from `IParameterGroup.bindToInstance`.
 */
export interface ParameterGroupInstanceConfig {
  /** The name of this parameter group. */
  readonly parameterGroupName: string;
}

/**
 * A parameter group.
 * Represents both a cluster parameter group,
 * and an instance parameter group.
 */
export interface IParameterGroup extends IResource, aws_rds.IDBParameterGroupRef, aws_rds.IDBClusterParameterGroupRef {
  /**
   * Method called when this Parameter Group is used when defining a database cluster.
   */
  bindToCluster(options: ParameterGroupClusterBindOptions): ParameterGroupClusterConfig;

  /**
   * Method called when this Parameter Group is used when defining a database instance.
   */
  bindToInstance(options: ParameterGroupInstanceBindOptions): ParameterGroupInstanceConfig;

  /**
   * Adds a parameter to this group.
   * If this is an imported parameter group,
   * this method does nothing.
   *
   * @returns true if the parameter was actually added
   *   (i.e., this ParameterGroup is not imported),
   *   false otherwise
   */
  addParameter(key: string, value: string): boolean;
}

/**
 * Properties for a parameter group
 */
export interface ParameterGroupProps {
  /**
   * The database engine for this parameter group.
   */
  readonly engine: IEngine;

  /**
   * The name of this parameter group.
   *
   * @default - CloudFormation-generated name
   */
  readonly name?: string;

  /**
   * Description for this parameter group
   *
   * @default a CDK generated description
   */
  readonly description?: string;

  /**
   * The parameters in this parameter group
   *
   * @default - None
   */
  readonly parameters?: { [key: string]: string };

  /**
   * The CloudFormation policy to apply when the instance is removed from the
   * stack or replaced during an update.
   *
   * @default - RemovalPolicy.DESTROY
   */
  readonly removalPolicy?: RemovalPolicy;
}

/**
 * A parameter group.
 * Represents both a cluster parameter group (AWS::RDS::DBClusterParameterGroup),
 * and an instance parameter group (AWS::RDS::DBParameterGroup).
 *
 * @resource AWS::RDS::DBParameterGroup
 */
@propertyInjectable
export class ParameterGroup extends Resource implements IParameterGroup {
  /**
   * Uniquely identifies this class.
   */
  public static readonly PROPERTY_INJECTION_ID: string = 'aws-cdk-lib.aws-rds.ParameterGroup';

  /**
   * Imports a parameter group
   *
   * `dbParameterGroupRef.dbParameterGroupArn` is an instance DB parameter group ARN, and is not valid
   * for a cluster parameter group.
   */
  public static fromParameterGroupName(scope: Construct, id: string, parameterGroupName: string): IParameterGroup {
    class Import extends Resource implements IParameterGroup {
      public bindToCluster(_options: ParameterGroupClusterBindOptions): ParameterGroupClusterConfig {
        return { parameterGroupName };
      }

      public bindToInstance(_options: ParameterGroupInstanceBindOptions): ParameterGroupInstanceConfig {
        return { parameterGroupName };
      }

      public addParameter(_key: string, _value: string): boolean {
        return false;
      }

      public get dbParameterGroupRef(): aws_rds.DBParameterGroupReference {
        return {
          dbParameterGroupName: parameterGroupName,
          dbParameterGroupArn: Stack.of(scope).formatArn({
            service: 'rds',
            resource: 'pg',
            resourceName: parameterGroupName,
            arnFormat: ArnFormat.COLON_RESOURCE_NAME,
          }),
        };
      }

      public get dbClusterParameterGroupRef(): aws_rds.DBClusterParameterGroupReference {
        return {
          dbClusterParameterGroupName: parameterGroupName,
        };
      }
    }

    return new Import(scope, id);
  }

  /**
   * Creates a standalone instance parameter group.
   *
   * This method allows you to explicitly create a parameter group
   * without binding it to a database instance.
   *
   * @returns instance parameter group (AWS::RDS::DBParameterGroup)
   */
  public static forInstance(scope: Construct, id: string, props: ParameterGroupProps): IParameterGroup {
    const parameterGroup = new ParameterGroup(scope, id, props);
    parameterGroup.createInstanceParameterGroup();
    return parameterGroup;
  }

  /**
   * Creates a standalone cluster parameter group.
   *
   * This method allows you to explicitly create a parameter group
   * without binding it to a database cluster.
   *
   * @returns cluster parameter group (AWS::RDS::DBClusterParameterGroup)
   */
  public static forCluster(scope: Construct, id: string, props: ParameterGroupProps): IParameterGroup {
    const parameterGroup = new ParameterGroup(scope, id, props);
    parameterGroup.createClusterParameterGroup();
    return parameterGroup;
  }

  private readonly parameters: IMapBox<string, string>;
  private readonly parametersObject: IReadableBox<{ [key: string]: string }>;
  private readonly family: string;
  private readonly removalPolicy?: RemovalPolicy;
  private readonly description?: string;
  private readonly name?: string;

  private clusterCfnGroup?: CfnDBClusterParameterGroup;
  private instanceCfnGroup?: CfnDBParameterGroup;

  constructor(scope: Construct, id: string, props: ParameterGroupProps) {
    super(scope, id);
    // Enhanced CDK Analytics Telemetry
    addConstructMetadata(this, props);

    const family = props.engine.parameterGroupFamily;
    if (!family) {
      throw new ValidationError(lit`ParametergroupCannotUsedEngine`, "ParameterGroup cannot be used with an engine that doesn't specify a version", this);
    }
    this.family = family;
    this.description = props.description;
    this.name = props.name;
    this.parameters = Box.fromMap(new Map(Object.entries(props.parameters ?? {})));
    this.parametersObject = this.parameters.derive(m => Object.fromEntries(m));
    this.removalPolicy = props.removalPolicy;
  }

  @MethodMetadata()
  public bindToCluster(_options: ParameterGroupClusterBindOptions): ParameterGroupClusterConfig {
    return {
      parameterGroupName: this.createClusterParameterGroup(),
    };
  }

  @MethodMetadata()
  public bindToInstance(_options: ParameterGroupInstanceBindOptions): ParameterGroupInstanceConfig {
    return {
      parameterGroupName: this.createInstanceParameterGroup(),
    };
  }

  /**
   * Add a parameter to this parameter group
   *
   * @param key The key of the parameter to be added
   * @param value The value of the parameter to be added
   */
  @MethodMetadata()
  public addParameter(key: string, value: string): boolean {
    this.parameters.put(key, value);
    return true;
  }

  /**
   * Creates the instance parameter group CloudFormation resource if it doesn't exist.
   * @returns parameter group name
   */
  private createInstanceParameterGroup(): string {
    if (!this.instanceCfnGroup) {
      const id = this.clusterCfnGroup ? 'InstanceParameterGroup' : 'Resource';
      this.instanceCfnGroup = new CfnDBParameterGroup(this, id, {
        description: this.description || `Parameter group for ${this.family}`,
        family: this.family,
        dbParameterGroupName: this.name,
        parameters: this.parametersObject,
      });
    }

    if (this.removalPolicy) {
      this.instanceCfnGroup.applyRemovalPolicy(this.removalPolicy ?? RemovalPolicy.DESTROY);
    }

    return this.instanceCfnGroup.ref;
  }

  /**
   * Creates the cluster parameter group CloudFormation resource if it doesn't exist.
   * @returns parameter group name
   */
  private createClusterParameterGroup(): string {
    if (!this.clusterCfnGroup) {
      const id = this.instanceCfnGroup ? 'ClusterParameterGroup' : 'Resource';
      this.clusterCfnGroup = new CfnDBClusterParameterGroup(this, id, {
        description: this.description || `Cluster parameter group for ${this.family}`,
        family: this.family,
        dbClusterParameterGroupName: this.name,
        parameters: this.parametersObject,
      });
    }

    if (this.removalPolicy) {
      this.clusterCfnGroup.applyRemovalPolicy(this.removalPolicy ?? RemovalPolicy.DESTROY);
    }

    return this.clusterCfnGroup.ref;
  }

  /**
   * A reference to this parameter group as a DB parameter group
   */
  public get dbParameterGroupRef(): aws_rds.DBParameterGroupReference {
    const self = this;
    return {
      get dbParameterGroupName(): string {
        return self.instanceCfnGroup?.ref ?? self.name ?? '';
      },
      get dbParameterGroupArn(): string {
        if (!self.instanceCfnGroup) {
          throw new ValidationError(lit`CannotAccessDbParameterGroupArnOfUnboundParameterGroup`,
            'this ParameterGroup is not bound to a DB instance, so it has no DB parameter group ARN - bind it with bindToInstance() or create it with ParameterGroup.forInstance()', self);
        }
        return self.instanceCfnGroup.attrDbParameterGroupArn;
      },
    };
  }

  /**
   * A reference to this parameter group as a DB cluster parameter group
   */
  public get dbClusterParameterGroupRef(): aws_rds.DBClusterParameterGroupReference {
    return {
      dbClusterParameterGroupName: this.clusterCfnGroup?.ref ?? this.name ?? '',
    };
  }
}

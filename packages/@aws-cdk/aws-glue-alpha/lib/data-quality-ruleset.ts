import * as cdk from 'aws-cdk-lib';
import { CfnDataQualityRuleset } from 'aws-cdk-lib/aws-glue';
import type { IResource, RemovalPolicy } from 'aws-cdk-lib/core';
import { Resource } from 'aws-cdk-lib/core';
import { memoizedGetter } from 'aws-cdk-lib/core/lib/helpers-internal';
import { addConstructMetadata } from 'aws-cdk-lib/core/lib/metadata-resource';
import { propertyInjectable } from 'aws-cdk-lib/core/lib/prop-injectable';
import type * as constructs from 'constructs';

/**
 * Properties of a DataQualityTargetTable.
 */
export class DataQualityTargetTable {
  /**
   * The database name of the target table.
   */
  readonly databaseName: string;

  /**
   * The table name of the target table.
   */
  readonly tableName: string;

  constructor(databaseName: string, tableName: string) {
    this.databaseName = databaseName;
    this.tableName = tableName;
  }
}

export interface IDataQualityRuleset extends IResource {
  /**
   * The ARN of the ruleset
   * @attribute
   */
  readonly rulesetArn: string;

  /**
   * The name of the ruleset
   * @attribute
   */
  readonly rulesetName: string;
}

/**
 * Construction properties for `DataQualityRuleset`
 */
export interface DataQualityRulesetProps {
  /**
   * The name of the ruleset
   * @default cloudformation generated name
   */
  readonly rulesetName?: string;

  /**
   * The client token of the ruleset
   * @attribute
   */
  readonly clientToken?: string;

  /**
   * The description of the ruleset
   * @attribute
   */
  readonly description?: string;

  /**
   * The dqdl of the ruleset
   * @attribute
   */
  readonly rulesetDqdl: string;

  /**
   *  Key-Value pairs that define tags for the ruleset.
   *  @default empty tags
   */
  readonly tags?: { [key: string]: string };

  /**
   * The target table of the ruleset
   * @attribute
   */
  readonly targetTable: DataQualityTargetTable;

  /**
   * Policy to apply when the ruleset is removed from the stack.
   *
   * @default - resource will be destroyed
   */
  readonly removalPolicy?: RemovalPolicy;
}

/**
 * A Glue Data Quality ruleset.
 */
@propertyInjectable
export class DataQualityRuleset extends Resource implements IDataQualityRuleset {
  /** Uniquely identifies this class. */
  public static readonly PROPERTY_INJECTION_ID: string = '@aws-cdk.aws-glue-alpha.DataQualityRuleset';

  public static fromRulesetArn(scope: constructs.Construct, id: string, rulesetArn: string): IDataQualityRuleset {
    class Import extends Resource implements IDataQualityRuleset {
      public rulesetArn = rulesetArn;
      public rulesetName = cdk.Arn.extractResourceName(rulesetArn, 'dataqualityruleset');
    }

    return new Import(scope, id);
  }

  public static fromRulesetName(scope: constructs.Construct, id: string, rulesetName: string): IDataQualityRuleset {
    class Import extends Resource implements IDataQualityRuleset {
      public rulesetArn = DataQualityRuleset.buildRulesetArn(scope, rulesetName);
      public rulesetName = rulesetName;
    }

    return new Import(scope, id);
  }

  private static buildRulesetArn(scope: constructs.Construct, rulesetName: string) : string {
    return cdk.Stack.of(scope).formatArn({
      service: 'glue',
      resource: 'dataqualityruleset',
      resourceName: rulesetName,
    });
  }

  private resource: CfnDataQualityRuleset;

  constructor(scope: constructs.Construct, id: string, props: DataQualityRulesetProps) {
    super(scope, id, {
      physicalName: props.rulesetName,
    });
    // Enhanced CDK Analytics Telemetry
    addConstructMetadata(this, props);

    this.resource = new CfnDataQualityRuleset(this, 'Resource', {
      clientToken: props.clientToken,
      description: props.description,
      name: props.rulesetName,
      ruleset: props.rulesetDqdl,
      tags: props.tags,
      targetTable: props.targetTable,
    });

    this.resource.applyRemovalPolicy(props.removalPolicy);
  }

  /**
   * Name of this ruleset.
   */
  @memoizedGetter
  public get rulesetName(): string {
    return this.getResourceNameAttribute(this.resource.ref);
  }

  /**
   * ARN of this ruleset.
   */
  @memoizedGetter
  public get rulesetArn(): string {
    return DataQualityRuleset.buildRulesetArn(this, this.rulesetName);
  }
}

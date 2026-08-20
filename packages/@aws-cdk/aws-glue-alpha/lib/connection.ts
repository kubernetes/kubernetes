import type * as ec2 from 'aws-cdk-lib/aws-ec2';
import { CfnConnection } from 'aws-cdk-lib/aws-glue';
import * as cdk from 'aws-cdk-lib/core';
import { lit, memoizedGetter } from 'aws-cdk-lib/core/lib/helpers-internal';
import { addConstructMetadata, MethodMetadata } from 'aws-cdk-lib/core/lib/metadata-resource';
import { propertyInjectable } from 'aws-cdk-lib/core/lib/prop-injectable';
import type * as constructs from 'constructs';
import { warnOnPlaintextSecrets } from './private/secret-detection';

/**
 * The type of the glue connection
 *
 * If you need to use a connection type that doesn't exist as a static member, you
 * can instantiate a `ConnectionType` object, e.g: `new ConnectionType('NEW_TYPE')`.
 *
 * @see https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-glue-connection-connectioninput.html#cfn-glue-connection-connectioninput-connectiontype
 */
export class ConnectionType {
  /**
   * Designates a connection to a database through Java Database Connectivity (JDBC).
   */
  public static readonly JDBC = new ConnectionType('JDBC');

  /**
   * Designates a connection to an Apache Kafka streaming platform.
   */
  public static readonly KAFKA = new ConnectionType('KAFKA');

  /**
   * Designates a connection to a MongoDB document database.
   */
  public static readonly MONGODB = new ConnectionType('MONGODB');

  /**
   * Designates a connection used for view validation by Amazon Redshift.
   */
  public static readonly VIEW_VALIDATION_REDSHIFT = new ConnectionType('VIEW_VALIDATION_REDSHIFT');

  /**
   * Designates a connection used for view validation by Amazon Athena.
   */
  public static readonly VIEW_VALIDATION_ATHENA = new ConnectionType('VIEW_VALIDATION_ATHENA');

  /**
   * Designates a network connection to a data source within an Amazon Virtual Private Cloud environment (Amazon VPC).
   */
  public static readonly NETWORK = new ConnectionType('NETWORK');

  /**
   * Uses configuration settings contained in a connector purchased from AWS Marketplace
   * to read from and write to data stores that are not natively supported by AWS Glue.
   */
  public static readonly MARKETPLACE = new ConnectionType('MARKETPLACE');

  /**
   * Uses configuration settings contained in a custom connector to read from and write to data stores
   * that are not natively supported by AWS Glue.
   */
  public static readonly CUSTOM = new ConnectionType('CUSTOM');

  /**
   * Designates a connection to Facebook Ads.
   */
  public static readonly FACEBOOKADS = new ConnectionType('FACEBOOKADS');

  /**
   * Designates a connection to Google Ads.
   */
  public static readonly GOOGLEADS = new ConnectionType('GOOGLEADS');

  /**
   * Designates a connection to Google Sheets.
   */
  public static readonly GOOGLESHEETS = new ConnectionType('GOOGLESHEETS');

  /**
   * Designates a connection to Google Analytics 4.
   */
  public static readonly GOOGLEANALYTICS4 = new ConnectionType('GOOGLEANALYTICS4');

  /**
   * Designates a connection to HubSpot.
   */
  public static readonly HUBSPOT = new ConnectionType('HUBSPOT');

  /**
   * Designates a connection to Instagram Ads.
   */
  public static readonly INSTAGRAMADS = new ConnectionType('INSTAGRAMADS');

  /**
   * Designates a connection to Intercom.
   */
  public static readonly INTERCOM = new ConnectionType('INTERCOM');

  /**
   * Designates a connection to Jira Cloud.
   */
  public static readonly JIRACLOUD = new ConnectionType('JIRACLOUD');

  /**
   * Designates a connection to Adobe Marketo Engage.
   */
  public static readonly MARKETO = new ConnectionType('MARKETO');

  /**
   * Designates a connection to Oracle NetSuite.
   */
  public static readonly NETSUITEERP = new ConnectionType('NETSUITEERP');

  /**
   * Designates a connection to Salesforce using OAuth authentication.
   */
  public static readonly SALESFORCE = new ConnectionType('SALESFORCE');

  /**
   * Designates a connection to Salesforce Marketing Cloud.
   */
  public static readonly SALESFORCEMARKETINGCLOUD = new ConnectionType('SALESFORCEMARKETINGCLOUD');

  /**
   * Designates a connection to Salesforce Marketing Cloud Account Engagement (MCAE).
   */
  public static readonly SALESFORCEPARDOT = new ConnectionType('SALESFORCEPARDOT');

  /**
   * Designates a connection to SAP OData.
   */
  public static readonly SAPODATA = new ConnectionType('SAPODATA');

  /**
   * Designates a connection to ServiceNow.
   */
  public static readonly SERVICENOW = new ConnectionType('SERVICENOW');

  /**
   * Designates a connection to Slack.
   */
  public static readonly SLACK = new ConnectionType('SLACK');

  /**
   * Designates a connection to Snapchat Ads.
   */
  public static readonly SNAPCHATADS = new ConnectionType('SNAPCHATADS');

  /**
   * Designates a connection to Stripe.
   */
  public static readonly STRIPE = new ConnectionType('STRIPE');

  /**
   * Designates a connection to Zendesk.
   */
  public static readonly ZENDESK = new ConnectionType('ZENDESK');

  /**
   * Designates a connection to Zoho CRM.
   */
  public static readonly ZOHOCRM = new ConnectionType('ZOHOCRM');

  /**
   * Designates a connection to Google BigQuery.
   */
  public static readonly BIGQUERY = new ConnectionType('BIGQUERY');

  /**
   * Designates a connection to Azure SQL Database.
   */
  public static readonly AZURESQL = new ConnectionType('AZURESQL');

  /**
   * Designates a connection to Azure Cosmos DB.
   */
  public static readonly AZURECOSMOS = new ConnectionType('AZURECOSMOS');

  /**
   * Designates a connection to Amazon OpenSearch Service.
   */
  public static readonly OPENSEARCH = new ConnectionType('OPENSEARCH');

  /**
   * Designates a connection to MySQL.
   */
  public static readonly MYSQL = new ConnectionType('MYSQL');

  /**
   * Designates a connection to PostgreSQL.
   */
  public static readonly POSTGRESQL = new ConnectionType('POSTGRESQL');

  /**
   * Designates a connection to Oracle Database.
   */
  public static readonly ORACLE = new ConnectionType('ORACLE');

  /**
   * Designates a connection to Microsoft SQL Server.
   */
  public static readonly SQLSERVER = new ConnectionType('SQLSERVER');

  /**
   * Designates a connection to SAP HANA.
   */
  public static readonly SAPHANA = new ConnectionType('SAPHANA');

  /**
   * Designates a connection to Teradata.
   */
  public static readonly TERADATA = new ConnectionType('TERADATA');

  /**
   * Designates a connection to Vertica.
   */
  public static readonly VERTICA = new ConnectionType('VERTICA');

  /**
   * Designates a connection to Amazon DynamoDB.
   */
  public static readonly DYNAMODB = new ConnectionType('DYNAMODB');

  /**
   * The name of this ConnectionType, as expected by Connection resource.
   */
  public readonly name: string;

  constructor(name: string) {
    this.name = name;
  }

  /**
   * The connection type name as expected by Connection resource.
   */
  public toString(): string {
    return this.name;
  }
}

/**
 * Interface representing a created or an imported `Connection`
 */
export interface IConnection extends cdk.IResource {
  /**
   * The name of the connection
   * @attribute
   */
  readonly connectionName: string;

  /**
   * The ARN of the connection
   * @attribute
   */
  readonly connectionArn: string;
}

/**
 * Base Connection Options
 */
export interface ConnectionOptions {
  /**
   * The name of the connection
   * @default cloudformation generated name
   */
  readonly connectionName?: string;

  /**
   * The description of the connection.
   * @default no description
   */
  readonly description?: string;

  /**
   *  Key-Value pairs that define parameters for the connection.
   *  @default empty properties
   *  @see https://docs.aws.amazon.com/glue/latest/dg/aws-glue-programming-etl-connect.html
   */
  readonly properties?: { [key: string]: string };

  /**
   * A list of criteria that can be used in selecting this connection.
   * This is useful for filtering the results of https://awscli.amazonaws.com/v2/documentation/api/latest/reference/glue/get-connections.html
   * @default no match criteria
   */
  readonly matchCriteria?: string[];

  /**
   * The list of security groups needed to successfully make this connection e.g. to successfully connect to VPC.
   * @default no security group
   */
  readonly securityGroups?: ec2.ISecurityGroup[];

  /**
   * The VPC subnet to connect to resources within a VPC. See more at https://docs.aws.amazon.com/glue/latest/dg/start-connecting.html.
   *
   * Mutually exclusive with `vpc`: provide `subnet` to pin the connection to a
   * specific subnet, or provide `vpc` (optionally with `vpcSubnets`) to let the
   * CDK select one for you.
   *
   * @default - no subnet, unless `vpc` is provided
   */
  readonly subnet?: ec2.ISubnet;

  /**
   * The VPC to connect to resources within. When provided, the CDK selects a
   * subnet from this VPC using `vpcSubnets`. A Glue connection targets a single
   * subnet, so the first subnet of the selection is used.
   *
   * Mutually exclusive with `subnet`.
   *
   * @default - no VPC, the subnet is taken from `subnet` if provided
   */
  readonly vpc?: ec2.IVpc;

  /**
   * Which subnets of `vpc` to select the connection subnet from. Only used when
   * `vpc` is provided. Since a Glue connection targets a single subnet, the
   * first subnet of the selection is used.
   *
   * @default - private subnets
   */
  readonly vpcSubnets?: ec2.SubnetSelection;
}

/**
 * Construction properties for `Connection`
 */
export interface ConnectionProps extends ConnectionOptions {
  /**
   * The type of the connection
   */
  readonly type: ConnectionType;
}

/**
 * An AWS Glue connection to a data source.
 */
@propertyInjectable
export class Connection extends cdk.Resource implements IConnection {
  /** Uniquely identifies this class. */
  public static readonly PROPERTY_INJECTION_ID: string = '@aws-cdk.aws-glue-alpha.Connection';

  /**
   * Creates a Connection construct that represents an external connection.
   *
   * @param scope The scope creating construct (usually `this`).
   * @param id The construct's id.
   * @param connectionArn arn of external connection.
   */
  public static fromConnectionArn(scope: constructs.Construct, id: string, connectionArn: string): IConnection {
    class Import extends cdk.Resource implements IConnection {
      public readonly connectionName = cdk.Arn.extractResourceName(connectionArn, 'connection');
      public readonly connectionArn = connectionArn;
    }

    return new Import(scope, id);
  }

  /**
   * Creates a Connection construct that represents an external connection.
   *
   * @param scope The scope creating construct (usually `this`).
   * @param id The construct's id.
   * @param connectionName name of external connection.
   */
  public static fromConnectionName(scope: constructs.Construct, id: string, connectionName: string): IConnection {
    class Import extends cdk.Resource implements IConnection {
      public readonly connectionName = connectionName;
      public readonly connectionArn = Connection.buildConnectionArn(scope, connectionName);
    }

    return new Import(scope, id);
  }

  private static buildConnectionArn(scope: constructs.Construct, connectionName: string): string {
    return cdk.Stack.of(scope).formatArn({
      service: 'glue',
      resource: 'connection',
      resourceName: connectionName,
    });
  }

  private readonly properties: { [key: string]: string };
  private readonly resource: CfnConnection;

  constructor(scope: constructs.Construct, id: string, props: ConnectionProps) {
    super(scope, id, {
      physicalName: props.connectionName,
    });
    // Enhanced CDK Analytics Telemetry
    addConstructMetadata(this, props);

    this.properties = props.properties || {};

    const subnet = this.resolveSubnet(props);

    const physicalConnectionRequirements = subnet || props.securityGroups ? {
      availabilityZone: subnet ? subnet.availabilityZone : undefined,
      subnetId: subnet ? subnet.subnetId : undefined,
      securityGroupIdList: props.securityGroups ? props.securityGroups.map(sg => sg.securityGroupId) : undefined,
    } : undefined;

    this.resource = new CfnConnection(this, 'Resource', {
      catalogId: cdk.Stack.of(this).account,
      connectionInput: {
        connectionProperties: cdk.Lazy.any({
          produce: () => {
            // Inspect the final property set at synthesis time so properties
            // added via `addProperty` are covered as well.
            warnOnPlaintextSecrets(
              this,
              this.properties,
              '@aws-cdk/aws-glue-alpha:plaintextConnectionSecret',
              'Reference a Secrets Manager secret through the connection\'s `SECRET_ID` property instead.',
            );
            return Object.keys(this.properties).length > 0 ? this.properties : undefined;
          },
        }),
        connectionType: props.type.name,
        description: props.description,
        matchCriteria: props.matchCriteria,
        name: props.connectionName,
        physicalConnectionRequirements,
      },
    });
  }

  /**
   * Determines the single subnet the connection should target, either from an
   * explicit `subnet` or by selecting one from `vpc`.
   */
  private resolveSubnet(props: ConnectionProps): ec2.ISubnet | undefined {
    if (props.subnet && props.vpc) {
      throw new cdk.ValidationError(lit`ConnectionSubnetConflict`, 'cannot specify both `subnet` and `vpc`; provide `subnet` for a specific subnet, or `vpc` (optionally with `vpcSubnets`) to let the CDK select one', this);
    }

    if (props.vpcSubnets && !props.vpc) {
      throw new cdk.ValidationError(lit`ConnectionVpcSubnetsWithoutVpc`, '`vpcSubnets` can only be specified together with `vpc`', this);
    }

    if (props.subnet) {
      return props.subnet;
    }

    if (props.vpc) {
      // A Glue connection targets a single subnet, so use the first subnet of the selection.
      const selected = props.vpc.selectSubnets(props.vpcSubnets);
      // `isPendingLookup` is true for imported VPCs whose subnets are not yet
      // resolved (e.g. `fromLookup`), where an empty selection is expected and
      // will materialize on a later synth. Only fail on a genuinely empty selection.
      if (!selected.isPendingLookup && selected.subnets.length === 0) {
        throw new cdk.ValidationError(lit`ConnectionNoSubnetsSelected`, '`vpcSubnets` selected no subnets from the provided `vpc`; adjust the selection or use `subnet` to target a specific subnet', this);
      }
      return selected.subnets[0];
    }

    return undefined;
  }

  /**
   * The name of the connection
   */
  @memoizedGetter
  public get connectionName(): string {
    return this.getResourceNameAttribute(this.resource.ref);
  }

  /**
   * The ARN of the connection
   */
  @memoizedGetter
  public get connectionArn(): string {
    return Connection.buildConnectionArn(this, this.connectionName);
  }

  /**
   * Add additional connection parameters
   * @param key parameter key
   * @param value parameter value
   */
  @MethodMetadata()
  public addProperty(key: string, value: string): void {
    this.properties[key] = value;
  }
}

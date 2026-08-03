import type { Construct } from 'constructs';
import type { IApi } from './api-base';
import { ApiBase } from './api-base';
import type {
  AppSyncLogConfig,
  AppSyncDomainOptions,
} from './appsync-common';
import {
  AppSyncFieldLogLevel,
  AppSyncEventResource,
} from './appsync-common';
import type { CfnApiKey } from './appsync.generated';
import { CfnApi, CfnDomainName, CfnDomainNameApiAssociation } from './appsync.generated';
import type {
  IAppSyncAuthConfig,
  AppSyncCognitoConfig,
  AppSyncLambdaAuthorizerConfig,
  AppSyncOpenIdConnectConfig,
  AppSyncAuthProvider,
} from './auth-config';
import {
  createAPIKey,
  AppSyncAuthorizationType,
} from './auth-config';
import type { ChannelNamespaceOptions } from './channel-namespace';
import { ChannelNamespace } from './channel-namespace';
import type {
  AppSyncDataSourceOptions,
  AppSyncHttpDataSourceOptions,
} from './data-source-common';
import {
  AppSyncDynamoDbDataSource,
  AppSyncHttpDataSource,
  AppSyncLambdaDataSource,
  AppSyncRdsDataSource,
  AppSyncOpenSearchDataSource,
  AppSyncEventBridgeDataSource,
} from './data-source-common';
import type { ITable } from '../../aws-dynamodb';
import type { IEventBus } from '../../aws-events';
import type { IGrantable } from '../../aws-iam';
import { Grant, ManagedPolicy, ServicePrincipal, Role } from '../../aws-iam';
import type { IFunction } from '../../aws-lambda';
import type { ILogGroup } from '../../aws-logs';
import { LogGroup, LogRetention, RetentionDays } from '../../aws-logs';
import type { IDomain } from '../../aws-opensearchservice';
import type { IDatabaseCluster, IServerlessCluster } from '../../aws-rds';
import type { ISecret } from '../../aws-secretsmanager';
import { Lazy, Names, Stack, Token, ValidationError } from '../../core';
import { addConstructMetadata } from '../../core/lib/metadata-resource';
import { lit } from '../../core/lib/private/literal-string';
import { propertyInjectable } from '../../core/lib/prop-injectable';

/**
 * Authorization configuration for the Event API
 */
export interface EventApiAuthConfig {
  /**
   * Auth providers for use in connection,
   * publish, and subscribe operations.
   * @default - API Key authorization
   */
  readonly authProviders?: AppSyncAuthProvider[];
  /**
   * Connection auth modes
   * @default - API Key authorization
   */
  readonly connectionAuthModeTypes?: AppSyncAuthorizationType[];

  /**
   * Default publish auth modes
   * @default - API Key authorization
   */
  readonly defaultPublishAuthModeTypes?: AppSyncAuthorizationType[];

  /**
   * Default subscribe auth modes
   * @default - API Key authorization
   */
  readonly defaultSubscribeAuthModeTypes?: AppSyncAuthorizationType[];
}

/**
 * Authorization configuration helper methods
 */
class AppSyncEventApiAuthConfig implements IAppSyncAuthConfig {
  /**
   * Set up OIDC Authorization configuration for GraphQL APIs and Event APIs
   * @param config - the configuration input for OpenID Connect auth mode
   * @returns CfnApi.OpenIDConnectConfigProperty | undefined
   */
  setupOpenIdConnectConfig(config?: AppSyncOpenIdConnectConfig): CfnApi.OpenIDConnectConfigProperty | undefined {
    if (!config) return undefined;
    return {
      authTtl: config.tokenExpiryFromAuth,
      clientId: config.clientId,
      iatTtl: config.tokenExpiryFromIssue,
      issuer: config.oidcProvider,
    };
  }

  /**
   * Set up Cognito Authorization configuration for Event APIs
   * @param config - the configuration input for Cognito auth mode
   * @returns CfnApi.CognitoConfigProperty | undefined
   */
  setupCognitoConfig(config?: AppSyncCognitoConfig): CfnApi.CognitoConfigProperty | undefined {
    if (!config) return undefined;
    return {
      userPoolId: config.userPool.userPoolId,
      awsRegion: config.userPool.env.region,
      appIdClientRegex: config.appIdClientRegex,
    };
  }

  /**
   * Set up Lambda Authorization configuration for GraphQL APIs and Event APIs
   * @param config - the configuration input for Lambda auth mode
   * @returns CfnApi.LambdaAuthorizerConfigProperty | undefined
   */
  setupLambdaAuthorizerConfig(config?: AppSyncLambdaAuthorizerConfig): CfnApi.LambdaAuthorizerConfigProperty | undefined {
    if (!config) return undefined;
    return {
      authorizerResultTtlInSeconds: config.resultsCacheTtl?.toSeconds(),
      authorizerUri: config.handler.functionArn,
      identityValidationExpression: config.validationRegex,
    };
  }
}

/**
 * Interface for Event API
 */
export interface IEventApi extends IApi {
  /**
   * The Authorization Types for this Event Api
   */
  readonly authProviderTypes: AppSyncAuthorizationType[];

  /**
   * The domain name of the Api's HTTP endpoint.
   *
   * @attribute
   */
  readonly httpDns: string;

  /**
   * The domain name of the Api's real-time endpoint.
   *
   * @attribute
   */
  readonly realtimeDns: string;

  /**
   * add a new channel namespace.
   * @param id the id of the channel namespace
   * @param options the options for the channel namespace
   * @returns the channel namespace
   */
  addChannelNamespace(id: string, options?: ChannelNamespaceOptions): ChannelNamespace;

  /**
   * Add a new DynamoDB data source to this API
   *
   * @param id The data source's id
   * @param table The DynamoDB table backing this data source
   * @param options The optional configuration for this data source
   */
  addDynamoDbDataSource(id: string, table: ITable, options?: AppSyncDataSourceOptions): AppSyncDynamoDbDataSource;

  /**
   * add a new http data source to this API
   *
   * @param id The data source's id
   * @param endpoint The http endpoint
   * @param options The optional configuration for this data source
   */
  addHttpDataSource(id: string, endpoint: string, options?: AppSyncHttpDataSourceOptions): AppSyncHttpDataSource;

  /**
   * Add an EventBridge data source to this api
   * @param id The data source's id
   * @param eventBus The EventBridge EventBus on which to put events
   * @param options The optional configuration for this data source
   */
  addEventBridgeDataSource(id: string, eventBus: IEventBus, options?: AppSyncDataSourceOptions): AppSyncEventBridgeDataSource;

  /**
   * add a new Lambda data source to this API
   *
   * @param id The data source's id
   * @param lambdaFunction The Lambda function to call to interact with this data source
   * @param options The optional configuration for this data source
   */
  addLambdaDataSource(id: string, lambdaFunction: IFunction, options?: AppSyncDataSourceOptions): AppSyncLambdaDataSource;

  /**
   * add a new Rds data source to this API
   *
   * @param id The data source's id
   * @param serverlessCluster The database cluster to interact with this data source
   * @param secretStore The secret store that contains the username and password for the database cluster
   * @param databaseName The optional name of the database to use within the cluster
   * @param options The optional configuration for this data source
   */
  addRdsDataSource(
    id: string,
    serverlessCluster: IServerlessCluster | IDatabaseCluster,
    secretStore: ISecret,
    databaseName?: string,
    options?: AppSyncDataSourceOptions
  ): AppSyncRdsDataSource;

  /**
   * Add a new OpenSearch data source to this API
   *
   * @param id The data source's id
   * @param domain The OpenSearch domain for this data source
   * @param options The optional configuration for this data source
   */
  addOpenSearchDataSource(id: string, domain: IDomain, options?: AppSyncDataSourceOptions): AppSyncOpenSearchDataSource;

  /**
   * Adds an IAM policy statement associated with this Event API to an IAM
   * principal's policy.
   *
   * @param grantee The principal
   * @param resources The set of resources to allow (i.e. ...:[region]:[accountId]:apis/EventApiId/...)
   * @param actions The actions that should be granted to the principal (i.e. appsync:EventPublish )
   */
  grant(grantee: IGrantable, resources: AppSyncEventResource, ...actions: string[]): Grant;

  /**
   * Adds an IAM policy statement for EventPublish access to this EventApi to an IAM
   * principal's policy.
   *
   * @param grantee The principal
   */
  grantPublish(grantee: IGrantable): Grant;

  /**
   * Adds an IAM policy statement for EventSubscribe access to this EventApi to an IAM
   * principal's policy.
   *
   * @param grantee The principal
   */
  grantSubscribe(grantee: IGrantable): Grant;

  /**
   * Adds an IAM policy statement to publish and subscribe to this API for an IAM principal's policy.
   *
   * @param grantee The principal
   */
  grantPublishAndSubscribe(grantee: IGrantable): Grant;

  /**
   * Adds an IAM policy statement for EventConnect access to this EventApi to an IAM principal's policy.
   *
   * @param grantee The principal
   */
  grantConnect(grantee: IGrantable): Grant;
}

/**
 * Base Class for Event API
 */
export abstract class EventApiBase extends ApiBase implements IEventApi {
  /**
   * The domain name of the Api's HTTP endpoint.
   */
  public abstract readonly httpDns: string;

  /**
   * The domain name of the Api's real-time endpoint.
   */
  public abstract readonly realtimeDns: string;

  /**
   * The Authorization Types for this Event Api
   */
  public abstract readonly authProviderTypes: AppSyncAuthorizationType[];

  /**
   * add a new Channel Namespace to this API
   */
  public addChannelNamespace(id: string, options?: ChannelNamespaceOptions): ChannelNamespace {
    return new ChannelNamespace(this, id, {
      api: this,
      channelNamespaceName: options?.channelNamespaceName ?? id,
      ...options,
    });
  }

  /**
   * add a new DynamoDB data source to this API
   *
   * @param id The data source's id
   * @param table The DynamoDB table backing this data source
   * @param options The optional configuration for this data source
   */
  public addDynamoDbDataSource(id: string, table: ITable, options?: AppSyncDataSourceOptions): AppSyncDynamoDbDataSource {
    return new AppSyncDynamoDbDataSource(this, id, {
      api: this,
      table,
      name: options?.name,
      description: options?.description,
    });
  }

  /**
   * add a new http data source to this API
   *
   * @param id The data source's id
   * @param endpoint The http endpoint
   * @param options The optional configuration for this data source
   */
  public addHttpDataSource(id: string, endpoint: string, options?: AppSyncHttpDataSourceOptions): AppSyncHttpDataSource {
    return new AppSyncHttpDataSource(this, id, {
      api: this,
      endpoint,
      name: options?.name,
      description: options?.description,
      authorizationConfig: options?.authorizationConfig,
    });
  }

  /**
   * add a new Lambda data source to this API
   *
   * @param id The data source's id
   * @param lambdaFunction The Lambda function to call to interact with this data source
   * @param options The optional configuration for this data source
   */
  public addLambdaDataSource(id: string, lambdaFunction: IFunction, options?: AppSyncDataSourceOptions): AppSyncLambdaDataSource {
    return new AppSyncLambdaDataSource(this, id, {
      api: this,
      lambdaFunction,
      name: options?.name,
      description: options?.description,
    });
  }

  /**
   * add a new Rds data source to this API
   * @param id The data source's id
   * @param serverlessCluster The database cluster to interact with this data source
   * @param secretStore The secret store that contains the username and password for the database cluster
   * @param databaseName The optional name of the database to use within the cluster
   * @param options The optional configuration for this data source
   */
  public addRdsDataSource(
    id: string,
    serverlessCluster: IServerlessCluster | IDatabaseCluster,
    secretStore: ISecret,
    databaseName?: string,
    options?: AppSyncDataSourceOptions,
  ): AppSyncRdsDataSource {
    return new AppSyncRdsDataSource(this, id, {
      api: this,
      name: options?.name,
      description: options?.description,
      serverlessCluster,
      secretStore,
      databaseName,
    });
  }

  /**
   * Add an EventBridge data source to this api
   * @param id The data source's id
   * @param eventBus The EventBridge EventBus on which to put events
   * @param options The optional configuration for this data source
   */
  addEventBridgeDataSource(id: string, eventBus: IEventBus, options?: AppSyncDataSourceOptions): AppSyncEventBridgeDataSource {
    return new AppSyncEventBridgeDataSource(this, id, {
      api: this,
      eventBus,
      name: options?.name,
      description: options?.description,
    });
  }

  /**
   * add a new OpenSearch data source to this API
   *
   * @param id The data source's id
   * @param domain The OpenSearch domain for this data source
   * @param options The optional configuration for this data source
   */
  public addOpenSearchDataSource(id: string, domain: IDomain, options?: AppSyncDataSourceOptions): AppSyncOpenSearchDataSource {
    return new AppSyncOpenSearchDataSource(this, id, {
      api: this,
      name: options?.name,
      description: options?.description,
      domain,
    });
  }

  /**
   * Adds an IAM policy statement associated with this Event API to an IAM
   * principal's policy.
   * [disable-awslint:no-grants]
   *
   * @param grantee The principal
   * @param resources The set of resources to allow (i.e. ...:[region]:[accountId]:apis/EventApiId/...)
   * @param actions The actions that should be granted to the principal (i.e. appsync:EventPublish )
   */
  public grant(grantee: IGrantable, resources: AppSyncEventResource, ...actions: string[]): Grant {
    if (!this.authProviderTypes.includes(AppSyncAuthorizationType.IAM)) {
      throw new ValidationError(lit`CannotGrantWithoutIam`, 'Cannot use grant method because IAM Authorization mode is missing in the auth providers on this API.', this);
    }
    return Grant.addToPrincipal({
      grantee,
      actions,
      resourceArns: resources.resourceArns(this),
    });
  }

  /**
   * Adds an IAM policy statement for EventPublish access to this EventApi to an IAM
   * principal's policy. This grants publish permission for all channels within the API.
   * [disable-awslint:no-grants]
   *
   * @param grantee The principal
   */
  public grantPublish(grantee: IGrantable): Grant {
    return this.grant(grantee, AppSyncEventResource.allChannelNamespaces(), 'appsync:EventPublish');
  }

  /**
   * Adds an IAM policy statement for EventSubscribe access to this EventApi to an IAM
   * principal's policy. This grants subscribe permission for all channels within the API.
   * [disable-awslint:no-grants]
   *
   * @param grantee The principal
   */
  public grantSubscribe(grantee: IGrantable): Grant {
    return this.grant(grantee, AppSyncEventResource.allChannelNamespaces(), 'appsync:EventSubscribe');
  }

  /**
   * Adds an IAM policy statement to publish and subscribe to this API for an IAM principal's policy.
   * This grants publish & subscribe permission for all channels within the API.
   * [disable-awslint:no-grants]
   *
   * @param grantee The principal
   */
  public grantPublishAndSubscribe(grantee: IGrantable): Grant {
    return this.grant(grantee, AppSyncEventResource.allChannelNamespaces(), 'appsync:EventPublish', 'appsync:EventSubscribe');
  }

  /**
   * Adds an IAM policy statement for EventConnect access to this EventApi to an IAM principal's policy.
   * [disable-awslint:no-grants]
   *
   * @param grantee The principal
   */
  public grantConnect(grantee: IGrantable): Grant {
    return this.grant(grantee, AppSyncEventResource.forAPI(), 'appsync:EventConnect');
  }
}

/**
 * Properties for an AppSync Event API
 */
export interface EventApiProps {
  /**
   * the name of the Event API
   */
  readonly apiName: string;

  /**
   * Optional authorization configuration
   *
   * @default - API Key authorization
   */
  readonly authorizationConfig?: EventApiAuthConfig;

  /**
   * Logging configuration for this api
   *
   * @default - None
   */
  readonly logConfig?: AppSyncLogConfig;

  /**
   * The owner contact information for an API resource.
   *
   * This field accepts any string input with a length of 0 - 256 characters.
   *
   * @default - No owner contact.
   */
  readonly ownerContact?: string;

  /**
   * The domain name configuration for the Event API
   *
   * The Route 53 hosted zone and CName DNS record must be configured in addition to this setting to
   * enable custom domain URL
   *
   * @default - no domain name
   */
  readonly domainName?: AppSyncDomainOptions;
}

/**
 * Attributes for Event API imports
 */
export interface EventApiAttributes {
  /**
   * the name of the Event API
   * @default - not needed to import API
   */
  readonly apiName?: string;

  /**
   * a unique AWS AppSync Event API identifier
   * i.e. 'lxz775lwdrgcndgz3nurvac7oa'
   */
  readonly apiId: string;

  /**
   * the ARN of the Event API
   * @default - constructed arn
   */
  readonly apiArn?: string;

  /**
   * the domain name of the Api's HTTP endpoint.
   */
  readonly httpDns: string;

  /**
   * the domain name of the Api's real-time endpoint.
   */
  readonly realtimeDns: string;

  /**
   * The Authorization Types for this Event Api
   * @default - none, required to construct event rules from imported APIs
   */
  readonly authProviderTypes?: AppSyncAuthorizationType[];
}

/**
 * An AppSync Event API
 *
 * @resource AWS::AppSync::Api
 */
@propertyInjectable
export class EventApi extends EventApiBase {
  /**
   * Uniquely identifies this class.
   */
  public static readonly PROPERTY_INJECTION_ID: string = 'aws-cdk-lib.aws-appsync.EventApi';

  /**
   * Import a Event API through this function
   *
   * @param scope scope
   * @param id id
   * @param attrs Event API Attributes of an API
   */
  public static fromEventApiAttributes(scope: Construct, id: string, attrs: EventApiAttributes): IEventApi {
    const arn =
      attrs.apiArn ??
      Stack.of(scope).formatArn({
        service: 'appsync',
        resource: 'apis',
        resourceName: attrs.apiId,
      });
    class Import extends EventApiBase {
      public readonly apiId = attrs.apiId;
      public readonly apiArn = arn;
      public readonly httpDns = attrs.httpDns;
      public readonly realtimeDns = attrs.realtimeDns;
      public readonly authProviderTypes = attrs.authProviderTypes ?? [];
    }
    return new Import(scope, id);
  }

  /**
   * a unique AWS AppSync Event API identifier
   * i.e. 'lxz775lwdrgcndgz3nurvac7oa'
   */
  public readonly apiId: string;

  /**
   * the ARN of the API
   */
  public readonly apiArn: string;

  /**
   * the domain name of the Api's HTTP endpoint.
   */
  public readonly httpDns: string;

  /**
   * the domain name of the Api's real-time endpoint.
   */
  public readonly realtimeDns: string;

  /**
   * The Authorization Types for this Event Api
   */
  public readonly authProviderTypes: AppSyncAuthorizationType[];

  /**
   * The connection auth modes for this Event Api
   */
  public readonly connectionModeTypes: AppSyncAuthorizationType[];

  /**
   * The default publish auth modes for this Event Api
   */
  public readonly defaultPublishModeTypes: AppSyncAuthorizationType[];

  /**
   * The default subscribe auth modes for this Event Api
   */
  public readonly defaultSubscribeModeTypes: AppSyncAuthorizationType[];

  /**
   * The configured API keys, if present.
   * The key of this object is an apiKey name (apiKeyConfig.name) if specified, `Default` otherwise.
   *
   * @default - no api key
   * @attribute ApiKeys
   */
  public readonly apiKeys: { [key: string]: CfnApiKey } = {};

  /**
   * the CloudWatch Log Group for this API
   */
  public readonly logGroup: ILogGroup;

  private api: CfnApi;
  private eventConfig: CfnApi.EventConfigProperty;
  private domainNameResource?: CfnDomainName;

  constructor(scope: Construct, id: string, props: EventApiProps) {
    if (props.apiName !== undefined && !Token.isUnresolved(props.apiName)) {
      if (props.apiName.length < 1 || props.apiName.length > 50) {
        throw new ValidationError(lit`ApiNameLengthInvalid`, `\`apiName\` must be between 1 and 50 characters, got: ${props.apiName.length} characters.`, scope);
      }
    }

    super(scope, id, {
      physicalName: props.apiName ?? Lazy.string({
        produce: () =>
          Names.uniqueResourceName(this, {
            maxLength: 50,
            separator: '-',
          }),
      }),
    });
    // Enhanced CDK Analytics Telemetry
    addConstructMetadata(this, props);

    const defaultAuthType: AppSyncAuthorizationType = AppSyncAuthorizationType.API_KEY;
    const defaultAuthProviders: AppSyncAuthProvider[] = [{ authorizationType: defaultAuthType }];
    const authProviders = props.authorizationConfig?.authProviders ?? defaultAuthProviders;

    this.authProviderTypes = this.setupAuthProviderTypes(authProviders);

    const connectionAuthModeTypes: AppSyncAuthorizationType[] =
      props.authorizationConfig?.connectionAuthModeTypes ?? this.authProviderTypes;
    const defaultPublishAuthModeTypes: AppSyncAuthorizationType[] =
      props.authorizationConfig?.defaultPublishAuthModeTypes ?? this.authProviderTypes;
    const defaultSubscribeAuthModeTypes: AppSyncAuthorizationType[] =
      props.authorizationConfig?.defaultSubscribeAuthModeTypes ?? this.authProviderTypes;

    this.connectionModeTypes = connectionAuthModeTypes;
    this.defaultPublishModeTypes = defaultPublishAuthModeTypes;
    this.defaultSubscribeModeTypes = defaultSubscribeAuthModeTypes;

    this.validateEventApiConfiguration(props, authProviders);

    this.eventConfig = {
      authProviders: this.mapAuthorizationProviders(authProviders),
      connectionAuthModes: this.mapAuthorizationConfig(connectionAuthModeTypes),
      defaultPublishAuthModes: this.mapAuthorizationConfig(defaultPublishAuthModeTypes),
      defaultSubscribeAuthModes: this.mapAuthorizationConfig(defaultSubscribeAuthModeTypes),
      logConfig: this.setupLogConfig(props.logConfig),
    };

    this.api = new CfnApi(this, 'Resource', {
      name: this.physicalName,
      ownerContact: props.ownerContact,
      eventConfig: this.eventConfig,
    });

    this.apiId = this.api.attrApiId;
    this.apiArn = this.api.attrApiArn;
    this.httpDns = this.api.attrDnsHttp;
    this.realtimeDns = this.api.attrDnsRealtime;

    const apiKeyConfigs = authProviders.filter((mode) => mode.authorizationType === AppSyncAuthorizationType.API_KEY);
    for (const mode of apiKeyConfigs) {
      this.apiKeys[mode.apiKeyConfig?.name ?? 'Default'] = createAPIKey(this, this.apiId, mode.apiKeyConfig);
    }

    if (authProviders.some((mode) => mode.authorizationType === AppSyncAuthorizationType.LAMBDA)) {
      const config = authProviders.find((mode: AppSyncAuthProvider) => {
        return mode.authorizationType === AppSyncAuthorizationType.LAMBDA && mode.lambdaAuthorizerConfig;
      })?.lambdaAuthorizerConfig;

      config?.handler.addPermission(`${id}-appsync`, {
        principal: new ServicePrincipal('appsync.amazonaws.com'),
        action: 'lambda:InvokeFunction',
        sourceArn: this.apiArn,
      });
    }

    if (props.domainName) {
      this.domainNameResource = new CfnDomainName(this, 'DomainName', {
        domainName: props.domainName.domainName,
        certificateArn: props.domainName.certificate.certificateRef.certificateArn,
        description: `domain for ${props.apiName} Event API`,
      });
      const domainNameAssociation = new CfnDomainNameApiAssociation(this, 'DomainAssociation', {
        domainName: props.domainName.domainName,
        apiId: this.apiId,
      });

      domainNameAssociation.addResourceDependency(this.domainNameResource);
    }

    const logGroupName = `/aws/appsync/apis/${this.apiId}`;

    if (props.logConfig) {
      const logRetention = new LogRetention(this, 'LogRetention', {
        logGroupName: logGroupName,
        retention: props.logConfig?.retention ?? RetentionDays.INFINITE,
      });
      this.logGroup = LogGroup.fromLogGroupArn(this, 'LogGroup', logRetention.logGroupArn);
    } else {
      this.logGroup = LogGroup.fromLogGroupName(this, 'LogGroup', logGroupName);
    }
  }

  /**
   * Validate Event API configuration
   */
  private validateEventApiConfiguration(props: EventApiProps, authProviders: AppSyncAuthProvider[]) {
    this.validateOwnerContact(props.ownerContact);
    this.validateAuthorizationProps(authProviders);
    this.validateAuthorizationConfig(authProviders, this.connectionModeTypes);
    this.validateAuthorizationConfig(authProviders, this.defaultPublishModeTypes);
    this.validateAuthorizationConfig(authProviders, this.defaultSubscribeModeTypes);
  }

  /**
   * Validate ownerContact property
   */
  private validateOwnerContact(ownerContact?: string) {
    if (ownerContact === undefined || Token.isUnresolved(ownerContact)) return;

    if (ownerContact.length < 1 || ownerContact.length > 256) {
      throw new ValidationError(lit`OwnerContactLengthInvalid`, `\`ownerContact\` must be between 1 and 256 characters, got: ${ownerContact.length} characters.`, this);
    }

    const ownerContactPattern = /^[A-Za-z0-9_\-\ \.]+$/;

    if (!ownerContactPattern.test(ownerContact)) {
      throw new ValidationError(lit`OwnerContactInvalidCharacters`, `\`ownerContact\` must contain only alphanumeric characters, underscores, hyphens, spaces, and periods, got: ${ownerContact}`, this);
    }
  }

  private setupLogConfig(config?: AppSyncLogConfig) {
    if (!config) return;
    const logsRoleArn: string =
      config.role?.roleRef.roleArn ??
      new Role(this, 'ApiLogsRole', {
        assumedBy: new ServicePrincipal('appsync.amazonaws.com'),
        managedPolicies: [ManagedPolicy.fromAwsManagedPolicyName('service-role/AWSAppSyncPushToCloudWatchLogs')],
      }).roleRef.roleArn;
    const fieldLogLevel: AppSyncFieldLogLevel = config.fieldLogLevel ?? AppSyncFieldLogLevel.NONE;
    return {
      cloudWatchLogsRoleArn: logsRoleArn,
      logLevel: fieldLogLevel,
    };
  }

  private setupAuthProviderTypes(authProviders?: AppSyncAuthProvider[]) {
    if (!authProviders || authProviders.length === 0) return [AppSyncAuthorizationType.API_KEY];
    const modes = authProviders.map((mode) => mode.authorizationType);
    return modes;
  }

  private mapAuthorizationProviders(authProviders: AppSyncAuthProvider[]) {
    const authConfig: IAppSyncAuthConfig = new AppSyncEventApiAuthConfig();

    return authProviders.reduce<CfnApi.AuthProviderProperty[]>((acc, mode) => {
      acc.push({
        authType: mode.authorizationType,
        cognitoConfig: authConfig.setupCognitoConfig(mode.cognitoConfig),
        openIdConnectConfig: authConfig.setupOpenIdConnectConfig(mode.openIdConnectConfig),
        lambdaAuthorizerConfig: authConfig.setupLambdaAuthorizerConfig(mode.lambdaAuthorizerConfig),
      });
      return acc;
    }, []);
  }

  private mapAuthorizationConfig(authModes: AppSyncAuthorizationType[]) {
    return authModes.map((mode) => ({ authType: mode }));
  }

  private validateAuthorizationProps(authProviders: AppSyncAuthProvider[]) {
    const keyConfigs = authProviders.filter((mode) => mode.authorizationType === AppSyncAuthorizationType.API_KEY);
    const someWithNoNames = keyConfigs.some((config) => !config.apiKeyConfig?.name);
    if (keyConfigs.length > 1 && someWithNoNames) {
      throw new ValidationError(lit`ApiKeyNamesRequired`, 'You must specify key names when configuring more than 1 API key.', this);
    }

    if (authProviders.filter((authProvider) => authProvider.authorizationType === AppSyncAuthorizationType.LAMBDA).length > 1) {
      throw new ValidationError(lit`MultipleLambdaAuthorizersNotAllowed`,
        'You can only have a single AWS Lambda function configured to authorize your API. See https://docs.aws.amazon.com/appsync/latest/devguide/security.html',
        this,
      );
    }

    if (authProviders.filter((authProvider) => authProvider.authorizationType === AppSyncAuthorizationType.IAM).length > 1) {
      throw new ValidationError(lit`DuplicateIamConfiguration`, "You can't duplicate IAM configuration. See https://docs.aws.amazon.com/appsync/latest/devguide/security.html", this);
    }

    authProviders.map((authProvider) => {
      if (authProvider.authorizationType === AppSyncAuthorizationType.OIDC && !authProvider.openIdConnectConfig) {
        throw new ValidationError(lit`OidcConfigurationMissing`, 'OPENID_CONNECT authorization type is specified but OIDC Authorizer Configuration is missing in the AuthProvider', this);
      }
      if (authProvider.authorizationType === AppSyncAuthorizationType.USER_POOL && !authProvider.cognitoConfig) {
        throw new ValidationError(lit`CognitoConfigurationMissing`, 'AMAZON_COGNITO_USER_POOLS authorization type is specified but Cognito Authorizer Configuration is missing in the AuthProvider', this);
      }
      if (authProvider.authorizationType === AppSyncAuthorizationType.LAMBDA && !authProvider.lambdaAuthorizerConfig) {
        throw new ValidationError(lit`LambdaConfigurationMissing`, 'AWS_LAMBDA authorization type is specified but Lambda Authorizer Configuration is missing in the AuthProvider', this);
      }
    });
  }

  private validateAuthorizationConfig(authProviders: AppSyncAuthProvider[], authTypes: AppSyncAuthorizationType[]) {
    for (const authType of authTypes) {
      if (!authProviders.find((authProvider) => authProvider.authorizationType === authType)) {
        throw new ValidationError(lit`AuthorizationConfigurationMissing`, `Missing authorization configuration for ${authType}`, this);
      }
    }
    if (authTypes.length === 0) {
      throw new ValidationError(lit`EmptyAuthModeTypesNotAllowed`, 'Empty AuthModeTypes array is not allowed, if specifying, you must specify a valid mode', this);
    }
  }

  /**
   * The AppSyncDomainName of the associated custom domain
   */
  public get appSyncDomainName(): string {
    if (!this.domainNameResource) {
      throw new ValidationError(lit`DomainNameConfigurationRequired`, 'Cannot retrieve the appSyncDomainName without a domainName configuration', this);
    }
    return this.domainNameResource.attrAppSyncDomainName;
  }

  /**
   * The HTTP Endpoint of the associated custom domain
   */
  public get customHttpEndpoint(): string {
    if (!this.domainNameResource) {
      throw new ValidationError(lit`DomainNameConfigurationRequired`, 'Cannot retrieve the appSyncDomainName without a domainName configuration', this);
    }
    return `https://${this.domainNameResource.attrDomainName}/event`;
  }

  /**
   * The Realtime Endpoint of the associated custom domain
   */
  public get customRealtimeEndpoint(): string {
    if (!this.domainNameResource) {
      throw new ValidationError(lit`DomainNameConfigurationRequired`, 'Cannot retrieve the appSyncDomainName without a domainName configuration', this);
    }
    return `wss://${this.domainNameResource.attrDomainName}/event/realtime`;
  }
}

import type { Construct } from 'constructs';
import { CfnApiKey, CfnGraphQLApi, CfnGraphQLSchema, CfnDomainName, CfnDomainNameApiAssociation, CfnSourceApiAssociation } from './appsync.generated';
import type { IGraphqlApi } from './graphqlapi-base';
import { GraphqlApiBase, Visibility, AuthorizationType } from './graphqlapi-base';
import type { ISchema } from './schema';
import { SchemaFile } from './schema';
import { MergeType, addSourceApiAutoMergePermission, addSourceGraphQLPermission } from './source-api-association';
import type { IUserPool } from '../../aws-cognito';
import type { IRole, IRoleRef } from '../../aws-iam';
import { ManagedPolicy, Role, ServicePrincipal } from '../../aws-iam';
import type { IFunction } from '../../aws-lambda';
import type { ILogGroup } from '../../aws-logs';
import { LogGroup, LogRetention, RetentionDays } from '../../aws-logs';
import type { CfnResource, Expiration, IResolvable } from '../../core';
import { Annotations, Duration, FeatureFlags, Lazy, Stack, Token, ValidationError } from '../../core';
import { addConstructMetadata, MethodMetadata } from '../../core/lib/metadata-resource';
import { lit } from '../../core/lib/private/literal-string';
import { propertyInjectable } from '../../core/lib/prop-injectable';
import * as cxapi from '../../cx-api';
import type { ICertificateRef } from '../../interfaces/generated/aws-certificatemanager-interfaces.generated';

/**
 * Interface to specify default or additional authorization(s)
 */
export interface AuthorizationMode {
  /**
   * One of possible four values AppSync supports
   *
   * @see https://docs.aws.amazon.com/appsync/latest/devguide/security.html
   *
   * @default - `AuthorizationType.API_KEY`
   */
  readonly authorizationType: AuthorizationType;
  /**
   * If authorizationType is `AuthorizationType.USER_POOL`, this option is required.
   * @default - none
   */
  readonly userPoolConfig?: UserPoolConfig;
  /**
   * If authorizationType is `AuthorizationType.API_KEY`, this option can be configured.
   * @default - name: 'DefaultAPIKey' | description: 'Default API Key created by CDK'
   */
  readonly apiKeyConfig?: ApiKeyConfig;
  /**
   * If authorizationType is `AuthorizationType.OIDC`, this option is required.
   * @default - none
   */
  readonly openIdConnectConfig?: OpenIdConnectConfig;
  /**
   * If authorizationType is `AuthorizationType.LAMBDA`, this option is required.
   * @default - none
   */
  readonly lambdaAuthorizerConfig?: LambdaAuthorizerConfig;
}

/**
 * enum with all possible values for Cognito user-pool default actions
 */
export enum UserPoolDefaultAction {
  /**
   * ALLOW access to API
   */
  ALLOW = 'ALLOW',
  /**
   * DENY access to API
   */
  DENY = 'DENY',
}

/**
 * Configuration for Cognito user-pools in AppSync
 */
export interface UserPoolConfig {
  /**
   * The Cognito user pool to use as identity source
   */
  readonly userPool: IUserPool;
  /**
   * the optional app id regex
   *
   * @default -  None
   */
  readonly appIdClientRegex?: string;
  /**
   * Default auth action
   *
   * @default ALLOW
   */
  readonly defaultAction?: UserPoolDefaultAction;
}

/**
 * Configuration for API Key authorization in AppSync
 */
export interface ApiKeyConfig {
  /**
   * Unique name of the API Key
   * @default - 'DefaultAPIKey'
   */
  readonly name?: string;
  /**
   * Description of API key
   * @default - 'Default API Key created by CDK'
   */
  readonly description?: string;

  /**
   * The time from creation time after which the API key expires.
   * It must be a minimum of 1 day and a maximum of 365 days from date of creation.
   * Rounded down to the nearest hour.
   *
   * @default - 7 days rounded down to nearest hour
   */
  readonly expires?: Expiration;
}

/**
 * Configuration for OpenID Connect authorization in AppSync
 */
export interface OpenIdConnectConfig {
  /**
   * The number of milliseconds an OIDC token is valid after being authenticated by OIDC provider.
   * `auth_time` claim in OIDC token is required for this validation to work.
   * @default - no validation
   */
  readonly tokenExpiryFromAuth?: number;
  /**
   * The number of milliseconds an OIDC token is valid after being issued to a user.
   * This validation uses `iat` claim of OIDC token.
   * @default - no validation
   */
  readonly tokenExpiryFromIssue?: number;
  /**
   * The client identifier of the Relying party at the OpenID identity provider.
   * A regular expression can be specified so AppSync can validate against multiple client identifiers at a time.
   * @example - 'ABCD|CDEF' // where ABCD and CDEF are two different clientId
   * @default - * (All)
   */
  readonly clientId?: string;
  /**
   * The issuer for the OIDC configuration. The issuer returned by discovery must exactly match the value of `iss` in the OIDC token.
   */
  readonly oidcProvider: string;
}

/**
 * Configuration for Lambda authorization in AppSync. Note that you can only have a single AWS Lambda function configured to authorize your API.
 */
export interface LambdaAuthorizerConfig {
  /**
   * The authorizer lambda function.
   *
   * @see https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-appsync-graphqlapi-lambdaauthorizerconfig.html
   */
  readonly handler: IFunction;

  /**
   * How long the results are cached.
   * Disable caching by setting this to 0.
   *
   * @default Duration.minutes(5)
   */
  readonly resultsCacheTtl?: Duration;

  /**
   * A regular expression for validation of tokens before the Lambda function is called.
   *
   * @default - no regex filter will be applied.
   */
  readonly validationRegex?: string;
}

/**
 * Configuration of the API authorization modes.
 */
export interface AuthorizationConfig {
  /**
   * Optional authorization configuration
   *
   * @default - API Key authorization
   */
  readonly defaultAuthorization?: AuthorizationMode;

  /**
   * Additional authorization modes
   *
   * @default - No other modes
   */
  readonly additionalAuthorizationModes?: AuthorizationMode[];
}

/**
 * log-level for fields in AppSync
 */
export enum FieldLogLevel {
  /**
   * Resolver logging is disabled
   */
  NONE = 'NONE',
  /**
   * Only Error messages appear in logs
   */
  ERROR = 'ERROR',
  /**
   * Info and Error messages appear in logs
   */
  INFO = 'INFO',
  /**
   * Debug, Info, and Error messages, appear in logs
   */
  DEBUG = 'DEBUG',
  /**
   * All messages (Debug, Error, Info, and Trace) appear in logs
   */
  ALL = 'ALL',
}

/**
 * Logging configuration for AppSync
 */
export interface LogConfig {
  /**
   * exclude verbose content
   *
   * @default false
   */
  readonly excludeVerboseContent?: boolean | IResolvable;
  /**
   * log level for fields
   *
   * @default - Use AppSync default
   */
  readonly fieldLogLevel?: FieldLogLevel;

  /**
   * The role for CloudWatch Logs
   *
   * @default - None
   */
  readonly role?: IRoleRef;

  /**
   * The number of days log events are kept in CloudWatch Logs.
   * By default AppSync keeps the logs infinitely. When updating this property,
   * unsetting it doesn't remove the log retention policy.
   * To remove the retention policy, set the value to `INFINITE`
   *
   * @default RetentionDays.INFINITE
   */
  readonly retention?: RetentionDays;
}

/**
 * Controls how data source metrics will be emitted to CloudWatch.
 */
export enum DataSourceLevelMetricsBehavior {
  /**
   * Records and emits metric data for all data sources in the request.
   */
  FULL_REQUEST_DATA_SOURCE_METRICS = 'FULL_REQUEST_DATA_SOURCE_METRICS',
  /**
   * Records and emits metric data for data sources that have the MetricsConfig value set to ENABLED.
   */
  PER_DATA_SOURCE_METRICS = 'PER_DATA_SOURCE_METRICS',
}

/**
 * Controls how operation metrics will be emitted to CloudWatch.
 */
export enum OperationLevelMetricsConfig {
  /**
   * Sends operation metrics to CloudWatch.
   */
  ENABLED = 'ENABLED',
  /**
   * Does not send operation metrics to CloudWatch.
   */
  DISABLED = 'DISABLED',
}

/**
 * Controls how resolver metrics will be emitted to CloudWatch.
 */
export enum ResolverLevelMetricsBehavior {
  /**
   * Records and emits metric data for all resolvers in the request.
   */
  FULL_REQUEST_RESOLVER_METRICS = 'FULL_REQUEST_RESOLVER_METRICS',
  /**
   * Records and emits metric data for resolvers that have the MetricsConfig value set to ENABLED.
   */
  PER_RESOLVER_METRICS = 'PER_RESOLVER_METRICS',
}

/**
 * Enhanced metrics configuration for AppSync
 */
export interface EnhancedMetricsConfig {
  /**
   * Controls how data source metrics will be emitted to CloudWatch.
   */
  readonly dataSourceLevelMetricsBehavior: DataSourceLevelMetricsBehavior;
  /**
   * Controls how operation metrics will be emitted to CloudWatch.
   */
  readonly operationLevelMetricsConfig: OperationLevelMetricsConfig;
  /**
   * Controls how resolver metrics will be emitted to CloudWatch.
   */
  readonly resolverLevelMetricsBehavior: ResolverLevelMetricsBehavior;
}

/**
 * Domain name configuration for AppSync
 */
export interface DomainOptions {
  /**
   * The certificate to use with the domain name.
   */
  readonly certificate: ICertificateRef;

  /**
   * The actual domain name. For example, `api.example.com`.
   */
  readonly domainName: string;
}

/**
 * Additional API configuration for creating a AppSync Merged API
 */
export interface SourceApiOptions {
  /**
   * Definition of source APIs associated with this Merged API
   */
  readonly sourceApis: SourceApi[];

  /**
   * IAM Role used to validate access to source APIs at runtime and to update the merged API endpoint with the source API changes
   *
   * @default - An IAM Role with acccess to source schemas will be created
   */
  readonly mergedApiExecutionRole?: Role;
}

/**
 * Configuration of source API
 */
export interface SourceApi {
  /**
   * Source API that is associated with the merged API
   *
   * @jsii suppress JSII5019 For historic reasons
   */
  readonly sourceApi: IGraphqlApi;

  /**
   * Merging option used to associate the source API to the Merged API
   *
   * @default - Auto merge. The merge is triggered automatically when the source API has changed
   */
  readonly mergeType?: MergeType;

  /**
   * Description of the Source API asssociation.
   */
  readonly description?: string;
}

/**
 * AppSync definition. Specify how you want to define your AppSync API.
 */
export abstract class Definition {
  /**
   * Schema from schema object.
   * @param schema SchemaFile.fromAsset(filePath: string) allows schema definition through schema.graphql file
   * @returns Definition with schema from file
   */
  public static fromSchema(schema: ISchema): Definition {
    return {
      schema,
    };
  }

  /**
   * Schema from file, allows schema definition through schema.graphql file
   * @param filePath the file path of the schema file
   * @returns Definition with schema from file
   */
  public static fromFile(filePath: string): Definition {
    return this.fromSchema(SchemaFile.fromAsset(filePath));
  }

  /**
   * Schema from existing AppSync APIs - used for creating a AppSync Merged API
   * @param sourceApiOptions Configuration for AppSync Merged API
   * @returns Definition with for AppSync Merged API
   */
  public static fromSourceApis(sourceApiOptions: SourceApiOptions): Definition {
    return {
      sourceApiOptions,
    };
  }

  /**
   * Schema, when AppSync API is created from schema file
   */
  readonly schema?: ISchema;

  /**
   * Source APIs for Merged API
   */
  readonly sourceApiOptions?: SourceApiOptions;
}

/**
 * Properties for an AppSync GraphQL API
 */
export interface GraphqlApiProps {
  /**
   * the name of the GraphQL API
   */
  readonly name: string;

  /**
   * Optional authorization configuration
   *
   * @default - API Key authorization
   */
  readonly authorizationConfig?: AuthorizationConfig;

  /**
   * Logging configuration for this api
   *
   * @default - None
   */
  readonly logConfig?: LogConfig;

  /**
   * Definition (schema file or source APIs) for this GraphQL Api
   */
  readonly definition?: Definition;

  /**
   * GraphQL schema definition. Specify how you want to define your schema.
   *
   * SchemaFile.fromAsset(filePath: string) allows schema definition through schema.graphql file
   *
   * @default - schema will be generated code-first (i.e. addType, addObjectType, etc.)
   * @deprecated use Definition.schema instead
   */
  readonly schema?: ISchema;
  /**
   * A flag indicating whether or not X-Ray tracing is enabled for the GraphQL API.
   *
   * @default - false
   */
  readonly xrayEnabled?: boolean;

  /**
   * A value indicating whether the API is accessible from anywhere (GLOBAL) or can only be access from a VPC (PRIVATE).
   *
   * @default - GLOBAL
   */
  readonly visibility?: Visibility;

  /**
   * The domain name configuration for the GraphQL API
   *
   * The Route 53 hosted zone and CName DNS record must be configured in addition to this setting to
   * enable custom domain URL
   *
   * @default - no domain name
   */
  readonly domainName?: DomainOptions;

  /**
   * A value indicating whether the API to enable (ENABLED) or disable (DISABLED) introspection.
   *
   * @default IntrospectionConfig.ENABLED
   */
  readonly introspectionConfig?: IntrospectionConfig;

  /**
   * A number indicating the maximum depth resolvers should be accepted when handling queries.
   * Value must be withing range of 0 to 75
   *
   * @default - The default value is 0 (or unspecified) which indicates no maximum depth.
   */
  readonly queryDepthLimit?: number;

  /**
   * A number indicating the maximum number of resolvers that should be accepted when handling queries.
   * Value must be withing range of 0 to 10000
   *
   * @default - The default value is 0 (or unspecified), which will set the limit to 10000
   */
  readonly resolverCountLimit?: number;

  /**
   * A map containing the list of resources with their properties and environment variables.
   *
   * There are a few rules you must follow when creating keys and values:
   *   - Keys must begin with a letter.
   *   - Keys must be between 2 and 64 characters long.
   *   - Keys can only contain letters, numbers, and the underscore character (_).
   *   - Values can be up to 512 characters long.
   *   - You can configure up to 50 key-value pairs in a GraphQL API.
   *
   * @default - No environment variables.
   */
  readonly environmentVariables?: { [key: string]: string };

  /**
   * The owner contact information for an API resource.
   *
   * This field accepts any string input with a length of 0 - 256 characters.
   *
   * @default - No owner contact.
   */
  readonly ownerContact?: string;

  /**
   * Enables and controls the enhanced metrics feature.
   *
   * @default - Enhanced metrics disabled.
   */
  readonly enhancedMetricsConfig?: EnhancedMetricsConfig;
}

/**
 * Attributes for GraphQL imports
 */
export interface GraphqlApiAttributes {
  /**
   * a unique AWS AppSync GraphQL API identifier
   * i.e. 'lxz775lwdrgcndgz3nurvac7oa'
   */
  readonly graphqlApiId: string;

  /**
   * the arn for the GraphQL Api
   * @default - autogenerated arn
   */
  readonly graphqlApiArn?: string;

  /**
   * The GraphQl endpoint arn for the GraphQL API
   *
   * @default - none, required to construct event rules from imported APIs
   */
  readonly graphQLEndpointArn?: string;

  /**
   * The GraphQl API visibility
   *
   * @default - GLOBAL
   */
  readonly visibility?: Visibility;

  /**
   * The Authorization Types for this GraphQL Api
   *
   * @default - none, required to construct event rules from imported APIs
   */
  readonly modes?: AuthorizationType[];
}

/**
 * Introspection configuration  for a GraphQL API
 */
export enum IntrospectionConfig {
  /**
   * Enable introspection
   */
  ENABLED = 'ENABLED',

  /**
   * Disable introspection
   */
  DISABLED = 'DISABLED',
}

/**
 * An AppSync GraphQL API
 *
 * @resource AWS::AppSync::GraphQLApi
 */
@propertyInjectable
export class GraphqlApi extends GraphqlApiBase {
  /**
   * Uniquely identifies this class.
   */
  public static readonly PROPERTY_INJECTION_ID: string = 'aws-cdk-lib.aws-appsync.GraphqlApi';

  /**
   * Import a GraphQL API through this function
   *
   * @param scope scope
   * @param id id
   * @param attrs GraphQL API Attributes of an API
   */
  public static fromGraphqlApiAttributes(scope: Construct, id: string, attrs: GraphqlApiAttributes): IGraphqlApi {
    const arn = attrs.graphqlApiArn ?? Stack.of(scope).formatArn({
      service: 'appsync',
      resource: `apis/${attrs.graphqlApiId}`,
    });
    class Import extends GraphqlApiBase {
      public readonly apiId = attrs.graphqlApiId;
      public readonly arn = arn;

      // the GraphQL endpoint ARN is not required to identify an AppSync GraphQL API
      // this value is only needed to construct event rules.
      public readonly graphQLEndpointArn = attrs.graphQLEndpointArn ?? '';
      public readonly visibility = attrs.visibility ?? Visibility.GLOBAL;
      public readonly modes = attrs.modes ?? [];

      constructor(s: Construct, i: string) {
        super(s, i);
      }
    }
    return new Import(scope, id);
  }

  /**
   * a unique AWS AppSync GraphQL API identifier
   * i.e. 'lxz775lwdrgcndgz3nurvac7oa'
   */
  public readonly apiId: string;

  /**
   * the ARN of the API
   */
  public readonly arn: string;

  /**
   * The GraphQL endpoint ARN
   */
  public readonly graphQLEndpointArn: string;

  /**
   * the URL of the endpoint created by AppSync
   *
   * @attribute GraphQlUrl
   */
  public readonly graphqlUrl: string;

  /**
   * the name of the API
   */
  public readonly name: string;

  /**
   * the visibility of the API
   */
  public readonly visibility: Visibility;

  /**
   * the schema attached to this api (only available for GraphQL APIs, not available for merged APIs)
   */
  public get schema(): ISchema {
    if (this.definition.schema) {
      return this.definition.schema;
    }
    throw new ValidationError(lit`SchemaExistAppSyncMerged`, 'Schema does not exist for AppSync merged APIs.', this);
  }

  /**
   * The Authorization Types for this GraphQL Api
   */
  public readonly modes: AuthorizationType[];

  /**
   * the configured API key, if present
   *
   * @default - no api key
   */
  public readonly apiKey?: string;

  /**
   * the CloudWatch Log Group for this API
   */
  public readonly logGroup: ILogGroup;

  private definition: Definition;
  private schemaResource?: CfnGraphQLSchema;
  private api: CfnGraphQLApi;
  private apiKeyResource?: CfnApiKey;
  private domainNameResource?: CfnDomainName;
  private mergedApiExecutionRole?: IRole;
  private environmentVariables: { [key: string]: string } = {};

  constructor(scope: Construct, id: string, props: GraphqlApiProps) {
    super(scope, id);
    // Enhanced CDK Analytics Telemetry
    addConstructMetadata(this, props);

    const defaultMode = props.authorizationConfig?.defaultAuthorization ??
      { authorizationType: AuthorizationType.API_KEY };
    const additionalModes = props.authorizationConfig?.additionalAuthorizationModes ?? [];
    const modes = [defaultMode, ...additionalModes];

    this.modes = modes.map((mode) => mode.authorizationType);

    this.validateAuthorizationProps(modes);

    if (!props.schema && !props.definition) {
      throw new ValidationError(lit`SpecifyGraphSchemaSourceProperty`, 'You must specify a GraphQL schema or source APIs in property definition.', this);
    }
    if ((props.schema !== undefined) === (props.definition !== undefined)) {
      throw new ValidationError(lit`CannotSpecifyPropertiesSchemaDefinition`, 'You cannot specify both properties schema and definition.', this);
    }
    if (props.queryDepthLimit !== undefined && (props.queryDepthLimit < 0 || props.queryDepthLimit > 75)) {
      throw new ValidationError(lit`SpecifyQueryDepthLimit`, 'You must specify a query depth limit between 0 and 75.', this);
    }
    if (props.resolverCountLimit !== undefined && (props.resolverCountLimit < 0 || props.resolverCountLimit > 10000)) {
      throw new ValidationError(lit`SpecifyResolverCountLimit`, 'You must specify a resolver count limit between 0 and 10000.', this);
    }
    if (!Token.isUnresolved(props.ownerContact) && props.ownerContact !== undefined && (props.ownerContact.length > 256)) {
      throw new ValidationError(lit`SpecifyStringCharactersLess`, 'You must specify `ownerContact` as a string of 256 characters or less.', this);
    }

    this.definition = props.schema ? Definition.fromSchema(props.schema) : props.definition!;

    if (this.definition.sourceApiOptions) {
      this.setupMergedApiExecutionRole(this.definition.sourceApiOptions);
    }

    if (props.environmentVariables !== undefined) {
      Object.entries(props.environmentVariables).forEach(([key, value]) => {
        this.addEnvironmentVariable(key, value);
      });
    }
    this.node.addValidation({ validate: () => this.validateEnvironmentVariables() });

    this.visibility = props.visibility ?? Visibility.GLOBAL;

    this.api = new CfnGraphQLApi(this, 'Resource', {
      name: props.name,
      authenticationType: defaultMode.authorizationType,
      logConfig: this.setupLogConfig(props.logConfig),
      openIdConnectConfig: this.setupOpenIdConnectConfig(defaultMode.openIdConnectConfig),
      userPoolConfig: this.setupUserPoolConfig(defaultMode.userPoolConfig),
      lambdaAuthorizerConfig: this.setupLambdaAuthorizerConfig(defaultMode.lambdaAuthorizerConfig),
      additionalAuthenticationProviders: this.setupAdditionalAuthorizationModes(additionalModes),
      xrayEnabled: props.xrayEnabled,
      visibility: props.visibility,
      mergedApiExecutionRoleArn: this.mergedApiExecutionRole?.roleArn,
      apiType: this.definition.sourceApiOptions ? 'MERGED' : undefined,
      introspectionConfig: props.introspectionConfig,
      queryDepthLimit: props.queryDepthLimit,
      resolverCountLimit: props.resolverCountLimit,
      environmentVariables: Lazy.any({ produce: () => this.renderEnvironmentVariables() }),
      ownerContact: props.ownerContact,
      enhancedMetricsConfig: this.setupEnhancedMetricsConfig(props.enhancedMetricsConfig),
    });

    this.apiId = this.api.attrApiId;
    this.arn = this.api.attrArn;
    this.graphqlUrl = this.api.attrGraphQlUrl;
    this.name = this.api.name;
    this.graphQLEndpointArn = this.api.attrGraphQlEndpointArn;

    if (this.definition.schema) {
      this.schemaResource = new CfnGraphQLSchema(this, 'Schema', this.definition.schema.bind(this));
    } else {
      this.setupSourceApiAssociations();
    }

    if (props.domainName) {
      this.domainNameResource = new CfnDomainName(this, 'DomainName', {
        domainName: props.domainName.domainName,
        certificateArn: props.domainName.certificate.certificateRef.certificateArn,
        description: `domain for ${this.name} at ${this.graphqlUrl}`,
      });
      const domainNameAssociation = new CfnDomainNameApiAssociation(this, 'DomainAssociation', {
        domainName: props.domainName.domainName,
        apiId: this.apiId,
      });

      domainNameAssociation.addResourceDependency(this.domainNameResource);
    }

    if (modes.some((mode) => mode.authorizationType === AuthorizationType.API_KEY)) {
      const config = modes.find((mode: AuthorizationMode) => {
        return mode.authorizationType === AuthorizationType.API_KEY && mode.apiKeyConfig;
      })?.apiKeyConfig;
      this.apiKeyResource = this.createAPIKey(config);
      if (this.schemaResource) {
        this.apiKeyResource.addResourceDependency(this.schemaResource);
      }
      this.apiKey = this.apiKeyResource.attrApiKey;
    }

    if (modes.some((mode) => mode.authorizationType === AuthorizationType.LAMBDA)) {
      const config = modes.find((mode: AuthorizationMode) => {
        return mode.authorizationType === AuthorizationType.LAMBDA && mode.lambdaAuthorizerConfig;
      })?.lambdaAuthorizerConfig;

      if (FeatureFlags.of(this).isEnabled(cxapi.APPSYNC_GRAPHQLAPI_SCOPE_LAMBDA_FUNCTION_PERMISSION)) {
        config?.handler.addPermission(`${id}-appsync`, {
          principal: new ServicePrincipal('appsync.amazonaws.com'),
          action: 'lambda:InvokeFunction',
          sourceArn: this.arn,
        });
      } else {
        config?.handler.addPermission(`${id}-appsync`, {
          principal: new ServicePrincipal('appsync.amazonaws.com'),
          action: 'lambda:InvokeFunction',
        });
      }
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

  private setupSourceApiAssociations() {
    this.definition.sourceApiOptions?.sourceApis.forEach(sourceApiConfig => {
      const mergeType = sourceApiConfig.mergeType ?? MergeType.AUTO_MERGE;
      let sourceApiIdentifier = sourceApiConfig.sourceApi.apiId;
      let mergedApiIdentifier = this.apiId;

      // This is protected by a feature flag because if there is an existing source api association that used the api id,
      // updating it to use ARN as identifier leads to a resource replacement. ARN is recommended going forward because it allows support
      // for both same account and cross account use cases.
      if (FeatureFlags.of(this).isEnabled(cxapi.APPSYNC_ENABLE_USE_ARN_IDENTIFIER_SOURCE_API_ASSOCIATION)) {
        sourceApiIdentifier = sourceApiConfig.sourceApi.arn;
        mergedApiIdentifier = this.arn;
      }

      const association = new CfnSourceApiAssociation(this, `${sourceApiConfig.sourceApi.node.id}Association`, {
        sourceApiIdentifier: sourceApiIdentifier,
        mergedApiIdentifier: mergedApiIdentifier,
        sourceApiAssociationConfig: {
          mergeType: mergeType,
        },
        description: sourceApiConfig.description,
      });

      // Add dependency because the schema must be created first to create the source api association.
      sourceApiConfig.sourceApi.addSchemaDependency(association);

      // Add permissions to merged api execution role
      const executionRole = this.mergedApiExecutionRole as IRole;
      addSourceGraphQLPermission(association, executionRole);

      if (mergeType === MergeType.AUTO_MERGE) {
        addSourceApiAutoMergePermission(association, executionRole);
      }
    });
  }

  private setupMergedApiExecutionRole(sourceApiOptions: SourceApiOptions) {
    if (sourceApiOptions.mergedApiExecutionRole) {
      this.mergedApiExecutionRole = sourceApiOptions.mergedApiExecutionRole;
    } else {
      this.mergedApiExecutionRole = new Role(this, 'MergedApiExecutionRole', {
        assumedBy: new ServicePrincipal('appsync.amazonaws.com'),
      });
    }
  }

  private validateAuthorizationProps(modes: AuthorizationMode[]) {
    if (modes.filter((mode) => mode.authorizationType === AuthorizationType.LAMBDA).length > 1) {
      throw new ValidationError(lit`SingleLambdaFunctionConfiguredAuthorize`, 'You can only have a single AWS Lambda function configured to authorize your API.', this);
    }
    modes.map((mode) => {
      if (mode.authorizationType === AuthorizationType.OIDC && !mode.openIdConnectConfig) {
        throw new ValidationError(lit`MissingConfiguration`, 'Missing OIDC Configuration', this);
      }
      if (mode.authorizationType === AuthorizationType.USER_POOL && !mode.userPoolConfig) {
        throw new ValidationError(lit`MissingUserPoolConfiguration`, 'Missing User Pool Configuration', this);
      }
      if (mode.authorizationType === AuthorizationType.LAMBDA && !mode.lambdaAuthorizerConfig) {
        throw new ValidationError(lit`MissingLambdaConfiguration`, 'Missing Lambda Configuration', this);
      }
    });
    if (modes.filter((mode) => mode.authorizationType === AuthorizationType.API_KEY).length > 1) {
      throw new ValidationError(lit`CanTDuplicateApiKeyConfiguration`, 'You can\'t duplicate API_KEY configuration. See https://docs.aws.amazon.com/appsync/latest/devguide/security.html', this);
    }
    if (modes.filter((mode) => mode.authorizationType === AuthorizationType.IAM).length > 1) {
      throw new ValidationError(lit`CanTDuplicateConfiguration`, 'You can\'t duplicate IAM configuration. See https://docs.aws.amazon.com/appsync/latest/devguide/security.html', this);
    }
  }

  /**
   * Add schema dependency to a given construct
   *
   * @param construct the dependee
   */
  @MethodMetadata()
  public addSchemaDependency(construct: CfnResource): boolean {
    if (this.schemaResource) {
      construct.addResourceDependency(this.schemaResource);
    }
    return true;
  }

  /**
   * Add an environment variable to the construct.
   */
  @MethodMetadata()
  public addEnvironmentVariable(key: string, value: string) {
    if (this.definition.sourceApiOptions) {
      throw new ValidationError(lit`EnvironmentVariablesSupportedMerged`, 'Environment variables are not supported for merged APIs', this);
    }
    if (!Token.isUnresolved(key) && !/^[A-Za-z]+\w*$/.test(key)) {
      throw new ValidationError(lit`MustBeBeginLetterOnly`, `Key '${key}' must begin with a letter and can only contain letters, numbers, and underscores`, this);
    }
    if (!Token.isUnresolved(key) && (key.length < 2 || key.length > 64)) {
      throw new ValidationError(lit`MustBeBetweenCharactersLong`, `Key '${key}' must be between 2 and 64 characters long, got ${key.length}`, this);
    }
    if (!Token.isUnresolved(value) && value.length > 512) {
      throw new ValidationError(lit`ValueLong`, `Value for '${key}' is too long. Values can be up to 512 characters long, got ${value.length}`, this);
    }

    this.environmentVariables[key] = value;
  }

  private validateEnvironmentVariables() {
    const errors: string[] = [];
    const entries = Object.entries(this.environmentVariables);
    if (entries.length > 50) {
      errors.push(`Only 50 environment variables can be set, got ${entries.length}`);
    }
    return errors;
  }

  private renderEnvironmentVariables() {
    return Object.entries(this.environmentVariables).length > 0 ? this.environmentVariables : undefined;
  }

  private setupLogConfig(config?: LogConfig) {
    if (!config) return undefined;
    const logsRoleArn: string = config.role?.roleRef.roleArn ?? new Role(this, 'ApiLogsRole', {
      assumedBy: new ServicePrincipal('appsync.amazonaws.com'),
      managedPolicies: [
        ManagedPolicy.fromAwsManagedPolicyName('service-role/AWSAppSyncPushToCloudWatchLogs'),
      ],
    }).roleArn;
    const fieldLogLevel: FieldLogLevel = config.fieldLogLevel ?? FieldLogLevel.NONE;
    return {
      cloudWatchLogsRoleArn: logsRoleArn,
      excludeVerboseContent: config.excludeVerboseContent,
      fieldLogLevel: fieldLogLevel,
    };
  }

  private setupOpenIdConnectConfig(config?: OpenIdConnectConfig) {
    if (!config) return undefined;
    return {
      authTtl: config.tokenExpiryFromAuth,
      clientId: config.clientId,
      iatTtl: config.tokenExpiryFromIssue,
      issuer: config.oidcProvider,
    };
  }

  private setupUserPoolConfig(config?: UserPoolConfig) {
    if (!config) return undefined;
    return {
      userPoolId: config.userPool.userPoolId,
      awsRegion: config.userPool.env.region,
      appIdClientRegex: config.appIdClientRegex,
      defaultAction: config.defaultAction || UserPoolDefaultAction.ALLOW,
    };
  }

  private setupLambdaAuthorizerConfig(config?: LambdaAuthorizerConfig) {
    if (!config) return undefined;
    return {
      authorizerResultTtlInSeconds: config.resultsCacheTtl?.toSeconds(),
      authorizerUri: config.handler.functionArn,
      identityValidationExpression: config.validationRegex,
    };
  }

  private setupAdditionalAuthorizationModes(modes?: AuthorizationMode[]) {
    if (!modes || modes.length === 0) return undefined;
    return modes.reduce<CfnGraphQLApi.AdditionalAuthenticationProviderProperty[]>((acc, mode) => [
      ...acc, {
        authenticationType: mode.authorizationType,
        userPoolConfig: this.setupUserPoolConfig(mode.userPoolConfig),
        openIdConnectConfig: this.setupOpenIdConnectConfig(mode.openIdConnectConfig),
        lambdaAuthorizerConfig: this.setupLambdaAuthorizerConfig(mode.lambdaAuthorizerConfig),
      },
    ], []);
  }

  private setupEnhancedMetricsConfig(config?: EnhancedMetricsConfig) {
    if (!config) return undefined;
    const dataSourceLevelMetricsBehavior = config.dataSourceLevelMetricsBehavior;
    if (dataSourceLevelMetricsBehavior === DataSourceLevelMetricsBehavior.FULL_REQUEST_DATA_SOURCE_METRICS ) {
      Annotations.of(this).addWarningV2('@aws-cdk/aws-appsync:fullRequestDataSourceMetrics', 'When DataSourceLevelMetricsBehavior is set to FULL_REQUEST_DATA_SOURCE_METRICS, metrics are sent to CloudWatch for all data sources used in the request, regardless of whether a data source’s MetricsConfig is set to ENABLED or DISABLED.');
    }
    const operationLevelMetricsEnabled = config.operationLevelMetricsConfig;
    const resolverLevelMetricsBehavior = config.resolverLevelMetricsBehavior;
    if (resolverLevelMetricsBehavior === ResolverLevelMetricsBehavior.FULL_REQUEST_RESOLVER_METRICS ) {
      Annotations.of(this).addWarningV2('@aws-cdk/aws-appsync:fullRequestResolverMetrics', 'When ResolverLevelMetricsBehavior is set to FULL_REQUEST_RESOLVER_METRICS, metrics are sent to CloudWatch for all resolvers used in the request, regardless of whether a resolver’s MetricsConfig is set to ENABLED or DISABLED.');
    }
    return {
      dataSourceLevelMetricsBehavior: dataSourceLevelMetricsBehavior,
      operationLevelMetricsConfig: operationLevelMetricsEnabled,
      resolverLevelMetricsBehavior: resolverLevelMetricsBehavior,
    };
  }

  private createAPIKey(config?: ApiKeyConfig) {
    if (config?.expires?.isBefore(Duration.days(1)) || config?.expires?.isAfter(Duration.days(365))) {
      throw Error('API key expiration must be between 1 and 365 days.');
    }
    const expires = config?.expires ? config?.expires.toEpoch() : undefined;
    return new CfnApiKey(this, `${config?.name || 'Default'}ApiKey`, {
      expires,
      description: config?.description,
      apiId: this.apiId,
    });
  }

  /**
   * The AppSyncDomainName of the associated custom domain
   */
  public get appSyncDomainName(): string {
    if (!this.domainNameResource) {
      throw new ValidationError(lit`CannotRetrieveAppSyncDomain`, 'Cannot retrieve the appSyncDomainName without a domainName configuration', this);
    }
    return this.domainNameResource.attrAppSyncDomainName;
  }
}

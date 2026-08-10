import type { Construct } from 'constructs';
import type { DomainNameReference, IDomainNameRef, IRestApiRef, IStageRef } from './apigateway.generated';
import { CfnDomainName } from './apigateway.generated';
import type { BasePathMappingOptions } from './base-path-mapping';
import { BasePathMapping } from './base-path-mapping';
import type { IRestApi } from './restapi';
import { EndpointType } from './restapi';
import * as apigwv2 from '../../aws-apigatewayv2';
import type { IBucket } from '../../aws-s3';
import type { IResource } from '../../core';
import { Arn, Names, Resource, Stack, Token } from '../../core';
import { ValidationError } from '../../core/lib/errors';
import { addConstructMetadata, MethodMetadata } from '../../core/lib/metadata-resource';
import { lit } from '../../core/lib/private/literal-string';
import { propertyInjectable } from '../../core/lib/prop-injectable';
import type { ICertificateRef } from '../../interfaces/generated/aws-certificatemanager-interfaces.generated';

/**
 * Options for creating an api mapping
 */
export interface ApiMappingOptions {
  /**
   * The api path name that callers of the API must provide in the URL after
   * the domain name (e.g. `example.com/base-path`). If you specify this
   * property, it can't be an empty string.
   *
   * If this is undefined, a mapping will be added for the empty path. Any request
   * that does not match a mapping will get sent to the API that has been mapped
   * to the empty path.
   *
   * @default - map requests from the domain root (e.g. `example.com`).
   */
  readonly basePath?: string;
}

/**
 * The minimum version of the SSL protocol that you want API Gateway to use for HTTPS connections.
 */
export enum SecurityPolicy {
  /** Cipher suite TLS 1.0 */
  TLS_1_0 = 'TLS_1_0',

  /** Cipher suite TLS 1.2 */
  TLS_1_2 = 'TLS_1_2',

  /**
   * Cipher suite TLS 1.3 for regional/private endpoints
   * @see https://docs.aws.amazon.com/apigateway/latest/developerguide/apigateway-custom-domain-tls-version.html
   */
  TLS13_1_3_2025_09 = 'SecurityPolicy_TLS13_1_3_2025_09',

  /**
   * Cipher suite TLS 1.3 (FIPS compliant) for regional/private endpoints
   * @see https://docs.aws.amazon.com/apigateway/latest/developerguide/apigateway-custom-domain-tls-version.html
   */
  TLS13_1_3_FIPS_2025_09 = 'SecurityPolicy_TLS13_1_3_FIPS_2025_09',

  /**
   * Cipher suite TLS 1.3 and TLS 1.2 with post-quantum cryptography for regional/private endpoints
   * @see https://docs.aws.amazon.com/apigateway/latest/developerguide/apigateway-custom-domain-tls-version.html
   */
  TLS13_1_2_PQ_2025_09 = 'SecurityPolicy_TLS13_1_2_PQ_2025_09',

  /**
   * Cipher suite TLS 1.3 and TLS 1.2 with Perfect Forward Secrecy and post-quantum cryptography for regional/private endpoints
   * @see https://docs.aws.amazon.com/apigateway/latest/developerguide/apigateway-custom-domain-tls-version.html
   */
  TLS13_1_2_PFS_PQ_2025_09 = 'SecurityPolicy_TLS13_1_2_PFS_PQ_2025_09',

  /**
   * Cipher suite TLS 1.3 for edge-optimized endpoints
   * @see https://docs.aws.amazon.com/apigateway/latest/developerguide/apigateway-custom-domain-tls-version.html
   */
  TLS13_2025_EDGE = 'SecurityPolicy_TLS13_2025_EDGE',

  /**
   * Cipher suite TLS 1.2 with Perfect Forward Secrecy for edge-optimized endpoints
   * @see https://docs.aws.amazon.com/apigateway/latest/developerguide/apigateway-custom-domain-tls-version.html
   */
  TLS12_PFS_2025_EDGE = 'SecurityPolicy_TLS12_PFS_2025_EDGE',

  /**
   * Cipher suite TLS 1.2 for edge-optimized endpoints (legacy)
   * @see https://docs.aws.amazon.com/apigateway/latest/developerguide/apigateway-custom-domain-tls-version.html
   */
  TLS12_2018_EDGE = 'SecurityPolicy_TLS12_2018_EDGE',
}

/**
 * The endpoint access mode for the domain name.
 *
 * When using enhanced security policies (those starting with `SecurityPolicy_`),
 * you must set the endpoint access mode to either `STRICT` or `BASIC`.
 * Use `STRICT` for production workloads requiring the highest security.
 * Use `BASIC` for migration scenarios or certain application architectures.
 *
 * @see https://docs.aws.amazon.com/apigateway/latest/developerguide/apigateway-security-policies.html#apigateway-security-policies-endpoint-access-mode
 */
export enum EndpointAccessMode {
  /**
   * Strict mode - only accepts connections from clients using the specified security policy.
   * Recommended for production workloads.
   */
  STRICT = 'STRICT',

  /**
   * Basic mode - one of the two valid endpoint access modes for enhanced security policies.
   * Suitable for migration scenarios or certain application architectures.
   * Note: legacy security policies (TLS_1_0, TLS_1_2) do not use this attribute.
   */
  BASIC = 'BASIC',
}

export interface DomainNameOptions {
  /**
   * The custom domain name for your API. Uppercase letters are not supported.
   */
  readonly domainName: string;

  /**
   * The reference to an AWS-managed certificate for use by the edge-optimized
   * endpoint for the domain name. For "EDGE" domain names, the certificate
   * needs to be in the US East (N. Virginia) region.
   */
  readonly certificate: ICertificateRef;

  /**
   * The type of endpoint for this DomainName.
   * @default REGIONAL
   */
  readonly endpointType?: EndpointType;

  /**
   * The Transport Layer Security (TLS) version + cipher suite for this domain name.
   * @see https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-resource-apigateway-domainname.html
   * @default SecurityPolicy.TLS_1_2
   */
  readonly securityPolicy?: SecurityPolicy;

  /**
   * The endpoint access mode for this domain name.
   *
   * When using enhanced security policies (those starting with `SecurityPolicy_`),
   * you must specify this property. STRICT is recommended for production workloads,
   * but BASIC may be needed during migration or for certain application architectures.
   *
   * @see https://docs.aws.amazon.com/apigateway/latest/developerguide/apigateway-security-policies.html#apigateway-security-policies-endpoint-access-mode
   * @default - No endpoint access mode is configured. Required for enhanced security policies.
   */
  readonly endpointAccessMode?: EndpointAccessMode;

  /**
   * The mutual TLS authentication configuration for a custom domain name.
   * @default - mTLS is not configured.
   */
  readonly mtls?: MTLSConfig;

  /**
   * The base path name that callers of the API must provide in the URL after
   * the domain name (e.g. `example.com/base-path`). If you specify this
   * property, it can't be an empty string.
   *
   * @default - map requests from the domain root (e.g. `example.com`).
   */
  readonly basePath?: string;
}

export interface DomainNameProps extends DomainNameOptions {
  /**
   * If specified, all requests to this domain will be mapped to the production
   * deployment of this API. If you wish to map this domain to multiple APIs
   * with different base paths, use `addBasePathMapping` or `addApiMapping`.
   *
   * @default - you will have to call `addBasePathMapping` to map this domain to
   * API endpoints.
   */
  readonly mapping?: IRestApi;
}

export interface IDomainName extends IResource, IDomainNameRef {
  /**
   * The domain name (e.g. `example.com`)
   *
   * @attribute DomainName
   */
  readonly domainName: string;

  /**
   * The Route53 alias target to use in order to connect a record set to this domain through an alias.
   *
   * @attribute DistributionDomainName,RegionalDomainName
   */
  readonly domainNameAliasDomainName: string;

  /**
   * The Route53 hosted zone ID to use in order to connect a record set to this domain through an alias.
   *
   * @attribute DistributionHostedZoneId,RegionalHostedZoneId
   */
  readonly domainNameAliasHostedZoneId: string;
}

@propertyInjectable
export class DomainName extends Resource implements IDomainName {
  /**
   * Uniquely identifies this class.
   */
  public static readonly PROPERTY_INJECTION_ID: string = 'aws-cdk-lib.aws-apigateway.DomainName';

  /**
   * Imports an existing domain name.
   */
  public static fromDomainNameAttributes(scope: Construct, id: string, attrs: DomainNameAttributes): IDomainName {
    class Import extends Resource implements IDomainName {
      public readonly domainName = attrs.domainName;
      public readonly domainNameAliasDomainName = attrs.domainNameAliasTarget;
      public readonly domainNameAliasHostedZoneId = attrs.domainNameAliasHostedZoneId;

      public readonly domainNameRef = {
        domainName: this.domainName,
        domainNameArn: Arn.format({
          service: 'apigateway',
          resource: 'domainnames',
          resourceName: attrs.domainName,
        }, Stack.of(scope)),
      };
    }

    return new Import(scope, id);
  }

  /** Policies that only support non-edge (regional/private) endpoints */
  private static readonly NON_EDGE_ONLY_POLICIES = [
    SecurityPolicy.TLS13_1_3_2025_09,
    SecurityPolicy.TLS13_1_3_FIPS_2025_09,
    SecurityPolicy.TLS13_1_2_PQ_2025_09,
    SecurityPolicy.TLS13_1_2_PFS_PQ_2025_09,
  ];

  /** Policies that only support edge endpoints */
  private static readonly EDGE_ONLY_POLICIES = [
    SecurityPolicy.TLS13_2025_EDGE,
    SecurityPolicy.TLS12_PFS_2025_EDGE,
    SecurityPolicy.TLS12_2018_EDGE,
  ];

  /** @jsii suppress JSII5019 For historic reasons */
  public readonly domainName: string;
  public readonly domainNameRef: DomainNameReference;
  public readonly domainNameAliasDomainName: string;
  public readonly domainNameAliasHostedZoneId: string;
  private readonly basePaths = new Set<string | undefined>();
  private readonly securityPolicy?: SecurityPolicy;
  private readonly endpointType: EndpointType;

  constructor(scope: Construct, id: string, props: DomainNameProps) {
    super(scope, id);
    // Enhanced CDK Analytics Telemetry
    addConstructMetadata(this, props);

    this.endpointType = props.endpointType || EndpointType.REGIONAL;
    const edge = this.endpointType === EndpointType.EDGE;
    this.securityPolicy = props.securityPolicy;

    if (!Token.isUnresolved(props.domainName) && /[A-Z]/.test(props.domainName)) {
      throw new ValidationError(lit`DomainNameDoesNotSupportUppercase`, `Domain name does not support uppercase letters. Got: ${props.domainName}`, scope);
    }

    // Skip all security-policy-related validations when any relevant field is a CDK token.
    // Token values are unresolved at synthesis time; CloudFormation will validate them at deploy time.
    const skipSecurityPolicyValidation =
      Token.isUnresolved(this.securityPolicy) ||
      Token.isUnresolved(this.endpointType) ||
      Token.isUnresolved(props.endpointAccessMode);

    if (!skipSecurityPolicyValidation) {
      // Validate mTLS is not used with enhanced security policies
      // See: https://docs.aws.amazon.com/apigateway/latest/developerguide/apigateway-custom-domain-tls-version.html
      if (props.mtls && this.isEnhancedSecurityPolicy(this.securityPolicy)) {
        throw new ValidationError(
          lit`MtlsNotSupportedWithEnhancedSecurityPolicy`,
          'Mutual TLS (mTLS) cannot be enabled on a domain name that uses an enhanced security policy. ' +
          'See: https://docs.aws.amazon.com/apigateway/latest/developerguide/apigateway-custom-domain-tls-version.html',
          this,
        );
      }

      // Validate endpointAccessMode is specified for enhanced security policies
      // See: https://docs.aws.amazon.com/apigateway/latest/developerguide/apigateway-security-policies.html#apigateway-security-policies-endpoint-access-mode
      if (this.isEnhancedSecurityPolicy(this.securityPolicy) &&
          props.endpointAccessMode === undefined) {
        throw new ValidationError(
          lit`EndpointAccessModeRequiredForEnhancedSecurityPolicy`,
          'Enhanced security policies require endpointAccessMode to be specified (BASIC or STRICT). ' +
          'STRICT is recommended for production workloads. ' +
          'See: https://docs.aws.amazon.com/apigateway/latest/developerguide/apigateway-security-policies.html#apigateway-security-policies-endpoint-access-mode',
          this,
        );
      }

      // Validate endpointAccessMode is not set for legacy security policies
      // CloudFormation rejects this combination: "Endpoint access mode is not supported for this security policy"
      // See: https://docs.aws.amazon.com/apigateway/latest/developerguide/apigateway-security-policies.html#apigateway-security-policies-endpoint-access-mode
      if (!this.isEnhancedSecurityPolicy(this.securityPolicy) &&
          props.endpointAccessMode !== undefined) {
        throw new ValidationError(
          lit`EndpointAccessModeNotSupportedForLegacySecurityPolicy`,
          'endpointAccessMode is not supported for legacy security policies (TLS_1_0, TLS_1_2). ' +
          'It can only be specified when using enhanced security policies (those starting with SecurityPolicy_). ' +
          'See: https://docs.aws.amazon.com/apigateway/latest/developerguide/apigateway-security-policies.html#apigateway-security-policies-endpoint-access-mode',
          this,
        );
      }

      // Validate security policy matches endpoint type
      // Regional-only policies cannot be used with EDGE endpoints and vice versa
      // See: https://docs.aws.amazon.com/apigateway/latest/developerguide/apigateway-custom-domain-tls-version.html
      this.validateSecurityPolicyEndpointType(this.securityPolicy, this.endpointType);
    }

    const mtlsConfig = this.configureMTLS(props.mtls);
    const resource = new CfnDomainName(this, 'Resource', {
      domainName: props.domainName,
      certificateArn: edge ? props.certificate.certificateRef.certificateArn : undefined,
      regionalCertificateArn: edge ? undefined : props.certificate.certificateRef.certificateArn,
      endpointConfiguration: { types: [this.endpointType] },
      mutualTlsAuthentication: mtlsConfig,
      securityPolicy: props.securityPolicy,
      endpointAccessMode: props.endpointAccessMode,
    });

    this.domainName = resource.ref;
    this.domainNameRef = resource.domainNameRef;

    this.domainNameAliasDomainName = edge
      ? resource.attrDistributionDomainName
      : resource.attrRegionalDomainName;

    this.domainNameAliasHostedZoneId = edge
      ? resource.attrDistributionHostedZoneId
      : resource.attrRegionalHostedZoneId;

    const multiLevel = this.validateBasePath(props.basePath);
    if (props.mapping && !multiLevel) {
      this.addBasePathMapping(props.mapping, {
        basePath: props.basePath,
      });
    } else if (props.mapping && multiLevel) {
      this.addApiMapping(props.mapping.deploymentStage, {
        basePath: props.basePath,
      });
    }
  }

  private validateBasePath(path?: string): boolean {
    if (this.isMultiLevel(path)) {
      if (this.endpointType === EndpointType.EDGE) {
        throw new ValidationError(lit`MultiLevelBasePathOnlySupported`, 'multi-level basePath is only supported when endpointType is EndpointType.REGIONAL', this);
      }
      // Multi-level API mappings (using ApiGatewayV2::ApiMapping) require TLS 1.2 or higher security policy.
      // TLS 1.0 does not support multi-level base path mappings.
      // See: https://docs.aws.amazon.com/apigateway/latest/developerguide/rest-api-mappings.html
      if (this.securityPolicy && !Token.isUnresolved(this.securityPolicy) && this.securityPolicy === SecurityPolicy.TLS_1_0) {
        throw new ValidationError(
          lit`DomainNameRequiresTLS12`,
          'securityPolicy must be TLS 1.2 or higher for multi-level basePath. ' +
          'See: https://docs.aws.amazon.com/apigateway/latest/developerguide/rest-api-mappings.html',
          this,
        );
      }
      return true;
    }
    return false;
  }

  private isMultiLevel(path?: string): boolean {
    return (path?.split('/').filter(x => !!x) ?? []).length >= 2;
  }

  /**
   * Maps this domain to an API endpoint.
   *
   * This uses the BasePathMapping from ApiGateway v1 which does not support multi-level paths.
   *
   * If you need to create a mapping for a multi-level path use `addApiMapping` instead.
   *
   * @param targetApi That target API endpoint, requests will be mapped to the deployment stage.
   * @param options Options for mapping to base path with or without a stage
   */
  @MethodMetadata()
  public addBasePathMapping(targetApi: IRestApiRef, options: BasePathMappingOptions = {}): BasePathMapping {
    if (this.basePaths.has(options.basePath)) {
      throw new ValidationError(lit`DomainNameAlreadyMappingPath`, `DomainName ${this.node.id} already has a mapping for path ${options.basePath}`, this);
    }
    if (this.isMultiLevel(options.basePath)) {
      throw new ValidationError(lit`BasePathMappingDoesNotSupportMultiLevel`, 'BasePathMapping does not support multi-level paths. Use "addApiMapping instead.', this);
    }

    this.basePaths.add(options.basePath);
    const basePath = options.basePath || '/';
    const id = `Map:${basePath}=>${Names.nodeUniqueId(targetApi.node)}`;
    return new BasePathMapping(this, id, {
      domainName: this,
      restApi: targetApi,
      ...options,
    });
  }

  /**
   * Maps this domain to an API endpoint.
   *
   * This uses the ApiMapping from ApiGatewayV2 which supports multi-level paths, but
   * also only supports:
   * - SecurityPolicy TLS 1.2 or higher for multi-level base paths (TLS 1.0 is not supported for multi-level paths)
   * - EndpointType.REGIONAL
   *
   * @see https://docs.aws.amazon.com/apigateway/latest/developerguide/rest-api-mappings.html
   * @param targetStage the target API stage.
   * @param options Options for mapping to a stage
   */
  @MethodMetadata()
  public addApiMapping(targetStage: IStageRef, options: ApiMappingOptions = {}): void {
    if (this.basePaths.has(options.basePath)) {
      throw new ValidationError(lit`DomainNameAlreadyMappingPath`, `DomainName ${this.node.id} already has a mapping for path ${options.basePath}`, this);
    }
    this.validateBasePath(options.basePath);

    this.basePaths.add(options.basePath);
    const id = `Map:${options.basePath ?? 'none'}=>${Names.nodeUniqueId(targetStage.node)}`;
    new apigwv2.CfnApiMapping(this, id, {
      apiId: targetStage.stageRef.restApiId,
      stage: targetStage.stageRef.stageName,
      domainName: this.domainName,
      apiMappingKey: options.basePath,
    });
  }

  private configureMTLS(mtlsConfig?: MTLSConfig): CfnDomainName.MutualTlsAuthenticationProperty | undefined {
    if (!mtlsConfig) return undefined;
    return {
      truststoreUri: mtlsConfig.bucket.s3UrlForObject(mtlsConfig.key),
      truststoreVersion: mtlsConfig.version,
    };
  }

  /**
   * Checks if the given security policy is an enhanced security policy.
   * Enhanced security policies start with 'SecurityPolicy_' prefix.
   *
   * Note: When the security policy is a CDK token (e.g., CfnParameter, cross-stack reference),
   * this method returns false to defer validation to CloudFormation deployment time.
   */
  private isEnhancedSecurityPolicy(policy?: SecurityPolicy): boolean {
    if (!policy || Token.isUnresolved(policy)) {
      return false;
    }
    return policy.startsWith('SecurityPolicy_');
  }

  /**
   * Validates that the security policy is compatible with the endpoint type.
   * Some policies are only supported for specific endpoint types.
   */
  private validateSecurityPolicyEndpointType(policy?: SecurityPolicy, endpointType?: EndpointType): void {
    if (!policy || !endpointType) {
      return;
    }

    if (endpointType === EndpointType.EDGE && DomainName.NON_EDGE_ONLY_POLICIES.includes(policy)) {
      throw new ValidationError(
        lit`SecurityPolicyNotSupportedForEdgeEndpoint`,
        `Security policy ${policy} is not supported for edge-optimized endpoints. ` +
        'Use a security policy that supports edge-optimized endpoints. ' +
        'See: https://docs.aws.amazon.com/apigateway/latest/developerguide/apigateway-custom-domain-tls-version.html',
        this,
      );
    }

    if (endpointType !== EndpointType.EDGE && DomainName.EDGE_ONLY_POLICIES.includes(policy)) {
      throw new ValidationError(
        lit`SecurityPolicyOnlySupportedForEdgeEndpoint`,
        `Security policy ${policy} is only supported for edge-optimized endpoints. ` +
        'Use a policy that supports non-edge endpoints. ' +
        'See: https://docs.aws.amazon.com/apigateway/latest/developerguide/apigateway-custom-domain-tls-version.html',
        this,
      );
    }
  }
}

export interface DomainNameAttributes {
  /**
   * The domain name (e.g. `example.com`)
   */
  readonly domainName: string;

  /**
   * The Route53 alias target to use in order to connect a record set to this domain through an alias.
   */
  readonly domainNameAliasTarget: string;

  /**
   * The Route53 hosted zone ID to use in order to connect a record set to this domain through an alias.
   */
  readonly domainNameAliasHostedZoneId: string;
}

/**
 * The mTLS authentication configuration for a custom domain name.
 */
export interface MTLSConfig {
  /**
   * The bucket that the trust store is hosted in.
   */
  readonly bucket: IBucket;

  /**
   * The key in S3 to look at for the trust store.
   */
  readonly key: string;

  /**
   *  The version of the S3 object that contains your truststore.
   *  To specify a version, you must have versioning enabled for the S3 bucket.
   *  @default - latest version
   */
  readonly version?: string;
}

import type { Construct } from 'constructs';
import type { IpAddressType } from './api';
import type { CfnDomainNameProps } from '.././index';
import { CfnDomainName } from '.././index';
import type { IBucket } from '../../../aws-s3';
import type { IResource } from '../../../core';
import { ArnFormat, Resource, Stack, Token } from '../../../core';
import { ValidationError } from '../../../core/lib/errors';
import type { IArrayBox } from '../../../core/lib/helpers-internal';
import { Box, memoizedGetter } from '../../../core/lib/helpers-internal';
import { addConstructMetadata, MethodMetadata } from '../../../core/lib/metadata-resource';
import { noBoxStackTraces } from '../../../core/lib/no-box-stack-traces';
import { lit } from '../../../core/lib/private/literal-string';
import { propertyInjectable } from '../../../core/lib/prop-injectable';
import type { ICertificateRef } from '../../../interfaces/generated/aws-certificatemanager-interfaces.generated';
import type { DomainNameReference, IDomainNameRef } from '../apigatewayv2.generated';

/**
 * The minimum version of the SSL protocol that you want API Gateway to use for HTTPS connections.
 */
export enum SecurityPolicy {
  /** Cipher suite TLS 1.0 */
  TLS_1_0 = 'TLS_1_0',

  /** Cipher suite TLS 1.2 */
  TLS_1_2 = 'TLS_1_2',
}

/**
 * Endpoint type for a domain name.
 */
export enum EndpointType {
  /**
   * For an edge-optimized custom domain name.
   */
  EDGE = 'EDGE',
  /**
   * For a regional custom domain name.
   */
  REGIONAL = 'REGIONAL',
}

/**
 * Represents an APIGatewayV2 DomainName
 * @see https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-resource-apigatewayv2-domainname.html
 */
export interface IDomainName extends IResource, IDomainNameRef {
  /**
   * The custom domain name
   * @attribute
   */
  readonly name: string;

  /**
   * The domain name associated with the regional endpoint for this custom domain name.
   * @attribute
   */
  readonly regionalDomainName: string;

  /**
   * The region-specific Amazon Route 53 Hosted Zone ID of the regional endpoint.
   * @attribute
   */
  readonly regionalHostedZoneId: string;
}

/**
 * custom domain name attributes
 */
export interface DomainNameAttributes {
  /**
   * domain name string
   */
  readonly name: string;

  /**
   * The domain name associated with the regional endpoint for this custom domain name.
   */
  readonly regionalDomainName: string;

  /**
   * The region-specific Amazon Route 53 Hosted Zone ID of the regional endpoint.
   */
  readonly regionalHostedZoneId: string;
}

/**
 * properties used for creating the DomainName
 */
export interface DomainNameProps extends EndpointOptions {
  /**
   * The custom domain name
   */
  readonly domainName: string;

  /**
   * The mutual TLS authentication configuration for a custom domain name.
   * @default - mTLS is not configured.
   */
  readonly mtls?: MTLSConfig;
}

/**
 * properties for creating a domain name endpoint
 */
export interface EndpointOptions {
  /**
   * The ACM certificate for this domain name.
   * Certificate can be both ACM issued or imported.
   */
  readonly certificate: ICertificateRef;

  /**
   * The user-friendly name of the certificate that will be used by the endpoint for this domain name.
   * @default - No friendly certificate name
   */
  readonly certificateName?: string;

  /**
   * The type of endpoint for this DomainName.
   * @default EndpointType.REGIONAL
   */
  readonly endpointType?: EndpointType;

  /**
   * The Transport Layer Security (TLS) version + cipher suite for this domain name.
   * @default SecurityPolicy.TLS_1_2
   */
  readonly securityPolicy?: SecurityPolicy;

  /**
   * A public certificate issued by ACM to validate that you own a custom domain. This parameter is required
   * only when you configure mutual TLS authentication and you specify an ACM imported or private CA certificate
   * for `certificate`. The ownership certificate validates that you have permissions to use the domain name.
   * @default - only required when configuring mTLS
   */
  readonly ownershipCertificate?: ICertificateRef;

  /**
   * The IP address types that can invoke the API.
   *
   * @see https://docs.aws.amazon.com/apigateway/latest/developerguide/http-api-ip-address-type.html
   *
   * @default undefined - AWS default is IPV4
   */
  readonly ipAddressType?: IpAddressType;
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
   * The key in S3 to look at for the trust store
   */
  readonly key: string;

  /**
   *  The version of the S3 object that contains your truststore.
   *  To specify a version, you must have versioning enabled for the S3 bucket.
   *  @default - latest version
   */
  readonly version?: string;
}

/**
 * Custom domain resource for the API
 */
@propertyInjectable
@noBoxStackTraces
export class DomainName extends Resource implements IDomainName {
  /**
   * Uniquely identifies this class.
   */
  public static readonly PROPERTY_INJECTION_ID: string = 'aws-cdk-lib.aws-apigatewayv2.DomainName';

  /**
   * Import from attributes
   */
  public static fromDomainNameAttributes(scope: Construct, id: string, attrs: DomainNameAttributes): IDomainName {
    class Import extends Resource implements IDomainName {
      public readonly regionalDomainName = attrs.regionalDomainName;
      public readonly regionalHostedZoneId = attrs.regionalHostedZoneId;
      public readonly name = attrs.name;
      public readonly domainNameRef: DomainNameReference = {
        domainName: attrs.name,
        domainNameArn: Stack.of(this).formatArn({
          service: 'apigateway',
          arnFormat: ArnFormat.SLASH_RESOURCE_SLASH_RESOURCE_NAME,
          resource: 'domainnames',
          resourceName: attrs.name,
        }),
      };
    }
    return new Import(scope, id);
  }

  public readonly name: string;
  private readonly domainNameConfigurations: IArrayBox<CfnDomainName.DomainNameConfigurationProperty>;
  private readonly resource: CfnDomainName;

  constructor(scope: Construct, id: string, props: DomainNameProps) {
    super(scope, id);
    // Enhanced CDK Analytics Telemetry
    addConstructMetadata(this, props);

    if (props.domainName === '') {
      throw new ValidationError(lit`EmptyStringDomainNameAllowed`, 'empty string for domainName not allowed', scope);
    }

    // validation for ownership certificate
    if (props.ownershipCertificate && !props.mtls) {
      throw new ValidationError(lit`OwnershipCertificateMtlsDomains`, 'ownership certificate can only be used with mtls domains', scope);
    }

    this.domainNameConfigurations = Box.fromArray([], { omitEmpty: false });

    const mtlsConfig = this.configureMTLS(props.mtls);
    const domainNameProps: CfnDomainNameProps = {
      domainName: props.domainName,
      domainNameConfigurations: this.domainNameConfigurations,
      mutualTlsAuthentication: mtlsConfig,
    };
    this.resource = new CfnDomainName(this, 'Resource', domainNameProps);
    this.name = this.resource.ref;

    if (props.certificate) {
      this.addEndpoint(props);
    }
  }

  private configureMTLS(mtlsConfig?: MTLSConfig): CfnDomainName.MutualTlsAuthenticationProperty | undefined {
    if (!mtlsConfig) return undefined;
    return {
      truststoreUri: mtlsConfig.bucket.s3UrlForObject(mtlsConfig.key),
      truststoreVersion: mtlsConfig.version,
    };
  }

  /**
   * Adds an endpoint to a domain name.
   * @param options domain name endpoint properties to be set
   */
  @MethodMetadata()
  public addEndpoint(options: EndpointOptions): void {
    const domainNameConfig: CfnDomainName.DomainNameConfigurationProperty = {
      certificateArn: options.certificate.certificateRef.certificateArn,
      certificateName: options.certificateName,
      endpointType: options.endpointType ? options.endpointType?.toString() : 'REGIONAL',
      ownershipVerificationCertificateArn: options.ownershipCertificate?.certificateRef.certificateArn,
      securityPolicy: options.securityPolicy?.toString(),
      ipAddressType: options.ipAddressType,
    };

    this.validateEndpointType(domainNameConfig.endpointType);
    this.domainNameConfigurations.push(domainNameConfig);
  }

  // validates that the new domain name configuration has a unique endpoint
  private validateEndpointType(endpointType: string | undefined) : void {
    for (let config of this.domainNameConfigurations.get()) {
      if (endpointType && endpointType == config.endpointType) {
        throw new ValidationError(lit`EndpointType`, `an endpoint with type ${endpointType} already exists`, this);
      }
    }
  }

  @memoizedGetter
  public get regionalDomainName(): string {
    return Token.asString(this.resource.getAtt('RegionalDomainName'));
  }

  @memoizedGetter
  public get regionalHostedZoneId(): string {
    return Token.asString(this.resource.getAtt('RegionalHostedZoneId'));
  }

  @memoizedGetter
  private get domainNameArn(): string {
    return Token.asString(this.resource.getAtt('DomainNameArn'));
  }

  public get domainNameRef(): DomainNameReference {
    return {
      domainName: this.name,
      domainNameArn: this.domainNameArn,
    };
  }
}

import type { Construct } from 'constructs';
import { CfnUserPoolDomain } from './cognito.generated';
import type { UserPoolClient } from './user-pool-client';
import type { IResource } from '../../core';
import { Resource, Stack, Token } from '../../core';
import { UnscopedValidationError, ValidationError } from '../../core/lib/errors';
import { addConstructMetadata, MethodMetadata } from '../../core/lib/metadata-resource';
import { lit } from '../../core/lib/private/literal-string';
import { propertyInjectable } from '../../core/lib/prop-injectable';
import type { AwsSdkCall } from '../../custom-resources';
import { AwsCustomResource, AwsCustomResourcePolicy, PhysicalResourceId } from '../../custom-resources';
import type { ICertificateRef } from '../../interfaces/generated/aws-certificatemanager-interfaces.generated';
import type { IUserPoolDomainRef, IUserPoolRef, UserPoolDomainReference } from '../../interfaces/generated/aws-cognito-interfaces.generated';

/**
 * The branding version of managed login for the domain.
 */
export enum ManagedLoginVersion {
  /**
   * The classic hosted UI.
   */
  CLASSIC_HOSTED_UI = 1,
  /**
   * The newer managed login with the branding designer.
   */
  NEWER_MANAGED_LOGIN = 2,
}

/**
 * Represents a user pool domain.
 */
export interface IUserPoolDomain extends IResource, IUserPoolDomainRef {
  /**
   * The domain that was specified to be created.
   * If `customDomain` was selected, this holds the full domain name that was specified.
   * If the `cognitoDomain` was used, it contains the prefix to the Cognito hosted domain.
   * @attribute
   */
  readonly domainName: string;
}

/**
 * Options while specifying custom domain
 * @see https://docs.aws.amazon.com/cognito/latest/developerguide/cognito-user-pools-add-custom-domain.html
 */
export interface CustomDomainOptions {
  /**
   * The custom domain name that you would like to associate with this User Pool.
   */
  readonly domainName: string;

  /**
   * The certificate to associate with this domain.
   */
  readonly certificate: ICertificateRef;
}

/**
 * Options while specifying a cognito prefix domain.
 * @see https://docs.aws.amazon.com/cognito/latest/developerguide/cognito-user-pools-assign-domain-prefix.html
 */
export interface CognitoDomainOptions {
  /**
   * The prefix to the Cognito hosted domain name that will be associated with the user pool.
   */
  readonly domainPrefix: string;
}

/**
 * Options to create a UserPoolDomain
 */
export interface UserPoolDomainOptions {
  /**
   * Associate a custom domain with your user pool
   * Either `customDomain` or `cognitoDomain` must be specified.
   * @see https://docs.aws.amazon.com/cognito/latest/developerguide/cognito-user-pools-add-custom-domain.html
   * @default - not set if `cognitoDomain` is specified, otherwise, throws an error.
   */
  readonly customDomain?: CustomDomainOptions;

  /**
   * Associate a cognito prefix domain with your user pool
   * Either `customDomain` or `cognitoDomain` must be specified.
   * @see https://docs.aws.amazon.com/cognito/latest/developerguide/cognito-user-pools-assign-domain-prefix.html
   * @default - not set if `customDomain` is specified, otherwise, throws an error.
   */
  readonly cognitoDomain?: CognitoDomainOptions;

  /**
   * A version that indicates the state of managed login.
   * This choice applies to all app clients that host services at the domain.
   *
   * @see https://docs.aws.amazon.com/cognito/latest/developerguide/cognito-user-pools-managed-login.html
   *
   * @default undefined - Cognito default setting is ManagedLoginVersion.CLASSIC_HOSTED_UI
   */
  readonly managedLoginVersion?: ManagedLoginVersion;
}

/**
 * Props for UserPoolDomain construct
 */
export interface UserPoolDomainProps extends UserPoolDomainOptions {
  /**
   * The user pool to which this domain should be associated.
   */
  readonly userPool: IUserPoolRef;
}

/**
 * Define a user pool domain
 */
@propertyInjectable
export class UserPoolDomain extends Resource implements IUserPoolDomain {
  /** Uniquely identifies this class. */
  public static readonly PROPERTY_INJECTION_ID: string = 'aws-cdk-lib.aws-cognito.UserPoolDomain';

  /**
   * Import a UserPoolDomain given its domain name
   */
  public static fromDomainName(scope: Construct, id: string, userPoolDomainName: string): IUserPoolDomain {
    class Import extends Resource implements IUserPoolDomain {
      public readonly domainName = userPoolDomainName;

      public get userPoolDomainRef(): UserPoolDomainReference {
        return {
          domain: userPoolDomainName,
          get userPoolId(): string {
            throw new UnscopedValidationError(lit`UserPoolDomainRefAvailable`, 'userPoolDomainRef is not available on imported UserPoolDomain.');
          },
        };
      }
    }

    return new Import(scope, id);
  }

  public readonly domainName: string;
  private isCognitoDomain: boolean;
  private readonly _userPool: IUserPoolRef;

  private cloudFrontCustomResource?: AwsCustomResource;
  private readonly resource: CfnUserPoolDomain;

  public get userPoolDomainRef(): UserPoolDomainReference {
    return {
      userPoolId: this._userPool.userPoolRef.userPoolId,
      domain: this.domainName,
    };
  }

  constructor(scope: Construct, id: string, props: UserPoolDomainProps) {
    super(scope, id);
    // Enhanced CDK Analytics Telemetry
    addConstructMetadata(this, props);

    this._userPool = props.userPool;

    if (!!props.customDomain === !!props.cognitoDomain) {
      throw new ValidationError(lit`ExactlyOneDomainRequired`, 'One of, and only one of, cognitoDomain or customDomain must be specified', this);
    }

    if (props.cognitoDomain?.domainPrefix &&
      !Token.isUnresolved(props.cognitoDomain?.domainPrefix) &&
      !/^[a-z0-9-]+$/.test(props.cognitoDomain.domainPrefix)) {
      throw new ValidationError(lit`DomainPrefixCognitoDomainContain`, 'domainPrefix for cognitoDomain can contain only lowercase alphabets, numbers and hyphens', this);
    }

    this.isCognitoDomain = !!props.cognitoDomain;

    const domainName = props.cognitoDomain?.domainPrefix || props.customDomain?.domainName!;
    this.resource = new CfnUserPoolDomain(this, 'Resource', {
      userPoolId: props.userPool.userPoolRef.userPoolId,
      domain: domainName,
      customDomainConfig: props.customDomain ? { certificateArn: props.customDomain.certificate.certificateRef.certificateArn } : undefined,
      managedLoginVersion: props.managedLoginVersion,
    });

    this.domainName = this.resource.ref;
  }

  /**
   * The domain name of the CloudFront distribution associated with the user pool domain.
   */
  public get cloudFrontEndpoint(): string {
    return this.resource.getAtt('CloudFrontDistribution').toString();
  }

  /**
   * The domain name of the CloudFront distribution associated with the user pool domain.
   *
   * This method creates a custom resource internally to get the CloudFront domain name.
   *
   * @deprecated use `cloudFrontEndpoint` method instead.
   */
  public get cloudFrontDomainName(): string {
    if (!this.cloudFrontCustomResource) {
      const sdkCall: AwsSdkCall = {
        service: 'CognitoIdentityServiceProvider',
        action: 'describeUserPoolDomain',
        parameters: {
          Domain: this.domainName,
        },
        physicalResourceId: PhysicalResourceId.of(this.domainName),
      };
      this.cloudFrontCustomResource = new AwsCustomResource(this, 'CloudFrontDomainName', {
        resourceType: 'Custom::UserPoolCloudFrontDomainName',
        onCreate: sdkCall,
        onUpdate: sdkCall,
        policy: AwsCustomResourcePolicy.fromSdkCalls({
          // DescribeUserPoolDomain only supports access level '*'
          // https://docs.aws.amazon.com/IAM/latest/UserGuide/list_amazoncognitouserpools.html#amazoncognitouserpools-actions-as-permissions
          resources: ['*'],
        }),
        // APIs are available in 2.1055.0
        installLatestAwsSdk: false,
      });
    }
    return this.cloudFrontCustomResource.getResponseField('DomainDescription.CloudFrontDistribution');
  }

  /**
   * The URL to the hosted UI associated with this domain
   *
   * @param options options to customize baseUrl
   */
  @MethodMetadata()
  public baseUrl(options?: BaseUrlOptions): string {
    if (this.isCognitoDomain) {
      const authDomain = 'auth' + (options?.fips ? '-fips' : '');
      return `https://${this.domainName}.${authDomain}.${Stack.of(this).region}.amazoncognito.com`;
    }
    return `https://${this.domainName}`;
  }

  /**
   * The URL to the sign in page in this domain using a specific UserPoolClient
   * @param client [disable-awslint:ref-via-interface] the user pool client that the UI will use to interact with the UserPool
   * @param options options to customize signInUrl.
   */
  @MethodMetadata()
  public signInUrl(client: UserPoolClient, options: SignInUrlOptions): string {
    let responseType: string;
    if (client.oAuthFlows.authorizationCodeGrant) {
      responseType = 'code';
    } else if (client.oAuthFlows.implicitCodeGrant) {
      responseType = 'token';
    } else {
      throw new ValidationError(lit`SignUrlSupportedClientsWithout`, 'signInUrl is not supported for clients without authorizationCodeGrant or implicitCodeGrant flow enabled', this);
    }
    const path = options.signInPath ?? '/login';
    return `${this.baseUrl(options)}${path}?client_id=${client.userPoolClientId}&response_type=${responseType}&redirect_uri=${options.redirectUri}`;
  }
}

/**
 * Options to customize the behaviour of `baseUrl()`
 */
export interface BaseUrlOptions {
  /**
   * Whether to return the FIPS-compliant endpoint
   *
   * @default return the standard URL
   */
  readonly fips?: boolean;
}

/**
 * Options to customize the behaviour of `signInUrl()`
 */
export interface SignInUrlOptions extends BaseUrlOptions {
  /**
   * Where to redirect to after sign in
   */
  readonly redirectUri: string;

  /**
   * The path in the URI where the sign-in page is located
   * @default '/login'
   */
  readonly signInPath?: string;
}

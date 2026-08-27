import type { Construct } from 'constructs';
import { validateMinimumSeconds } from './private/utils';
import * as cloudfront from '../../aws-cloudfront';
import type { OriginIpAddressType } from '../../aws-cloudfront';
import * as lambda from '../../aws-lambda';
import * as cdk from '../../core';
import { lit } from '../../core/lib/private/literal-string';

/**
 * Properties for a Lambda Function URL Origin.
 */
export interface FunctionUrlOriginProps extends cloudfront.OriginProps {
  /**
   * Specifies how long, in seconds, CloudFront waits for a response from the origin.
   * The minimum is 1 second. The maximum is governed by the origin response timeout quota, which is
   * adjustable, so the effective maximum depends on the target account.
   *
   * The default quota allows up to 120 seconds; higher values require an approved limit increase
   * in the target account, and otherwise produce an error at deploy time.
   *
   * @default Duration.seconds(30)
   */
  readonly readTimeout?: cdk.Duration;

  /**
   * Specifies how long, in seconds, CloudFront persists its connection to the origin.
   * The minimum is 1 second. The maximum is governed by the keep-alive timeout per origin quota,
   * which is adjustable, so the effective maximum depends on the target account.
   *
   * The default quota allows up to 300 seconds; higher values require an approved limit increase
   * in the target account, and otherwise produce an error at deploy time.
   *
   * @default Duration.seconds(5)
   */
  readonly keepaliveTimeout?: cdk.Duration;

  /**
   * Specifies which IP protocol CloudFront uses when connecting to your origin.
   *
   * If your origin uses both IPv4 and IPv6 protocols, you can choose dualstack to help optimize reliability.
   *
   * @default OriginIpAddressType.IPV4
   */
  readonly ipAddressType?: OriginIpAddressType;
}

/**
 * Properties for configuring a origin using a standard Lambda Functions URLs.
 */
export interface FunctionUrlOriginBaseProps extends cloudfront.OriginProps { }

/**
 * Properties for configuring a Lambda Functions URLs with OAC.
 */
export interface FunctionUrlOriginWithOACProps extends FunctionUrlOriginProps {
  /**
   * An optional Origin Access Control
   *
   * @default - an Origin Access Control will be created.
   */
  readonly originAccessControl?: cloudfront.IOriginAccessControlRef;

}

/**
 * An Origin for a Lambda Function URL.
 */
export class FunctionUrlOrigin extends cloudfront.OriginBase {
  /**
   * Create a Lambda Function URL Origin with Origin Access Control (OAC) configured
   */
  public static withOriginAccessControl(lambdaFunctionUrl: lambda.IFunctionUrl, props?: FunctionUrlOriginWithOACProps): cloudfront.IOrigin {
    return new FunctionUrlOriginWithOAC(lambdaFunctionUrl, props);
  }

  constructor(lambdaFunctionUrl: lambda.IFunctionUrl, private readonly props: FunctionUrlOriginProps = {}) {
    // Lambda Function URL is of the form 'https://<lambda-id>.lambda-url.<region>.on.aws/'
    // No need to split URL as we do with REST API, the entire URL is needed
    const domainName = cdk.Fn.select(2, cdk.Fn.split('/', lambdaFunctionUrl.url));
    super(domainName, props);

    validateMinimumSeconds('readTimeout', 1, props.readTimeout);
    validateMinimumSeconds('keepaliveTimeout', 1, props.keepaliveTimeout);
    this.validateResponseCompletionTimeoutWithReadTimeout(props.responseCompletionTimeout, props.readTimeout);
  }

  protected renderCustomOriginConfig(): cloudfront.CfnDistribution.CustomOriginConfigProperty | undefined {
    return {
      originSslProtocols: [cloudfront.OriginSslPolicy.TLS_V1_2],
      originProtocolPolicy: cloudfront.OriginProtocolPolicy.HTTPS_ONLY,
      originReadTimeout: this.props.readTimeout?.toSeconds(),
      originKeepaliveTimeout: this.props.keepaliveTimeout?.toSeconds(),
      ipAddressType: this.props.ipAddressType,
    };
  }
}

/**
 * An Origin for a Lambda Function URL with OAC.
 */
class FunctionUrlOriginWithOAC extends cloudfront.OriginBase {
  private originAccessControl?: cloudfront.IOriginAccessControlRef;
  private functionUrl: lambda.IFunctionUrl;
  private readonly props: FunctionUrlOriginWithOACProps;

  constructor(lambdaFunctionUrl: lambda.IFunctionUrl, props: FunctionUrlOriginWithOACProps = {}) {
    const domainName = cdk.Fn.select(2, cdk.Fn.split('/', lambdaFunctionUrl.url));
    super(domainName, props);
    this.functionUrl = lambdaFunctionUrl;
    this.originAccessControl = props?.originAccessControl;

    this.props = props;

    validateMinimumSeconds('readTimeout', 1, props.readTimeout);
    validateMinimumSeconds('keepaliveTimeout', 1, props.keepaliveTimeout);
  }

  protected renderCustomOriginConfig(): cloudfront.CfnDistribution.CustomOriginConfigProperty | undefined {
    return {
      originSslProtocols: [cloudfront.OriginSslPolicy.TLS_V1_2],
      originProtocolPolicy: cloudfront.OriginProtocolPolicy.HTTPS_ONLY,
      originReadTimeout: this.props.readTimeout?.toSeconds(),
      originKeepaliveTimeout: this.props.keepaliveTimeout?.toSeconds(),
      ipAddressType: this.props.ipAddressType,
    };
  }

  public bind(scope: Construct, options: cloudfront.OriginBindOptions): cloudfront.OriginBindConfig {
    const originBindConfig = super.bind(scope, options);

    if (!this.originAccessControl) {
      this.originAccessControl = new cloudfront.FunctionUrlOriginAccessControl(scope, 'FunctionUrlOriginAccessControl');
    }
    this.validateAuthType(scope);

    this.addInvokePermission(scope, options);

    return {
      ...originBindConfig,
      originProperty: {
        ...originBindConfig.originProperty!,
        originAccessControlId: this.originAccessControl?.originAccessControlRef.originAccessControlId,
      },
    };
  }

  private addInvokePermission(scope: Construct, options: cloudfront.OriginBindOptions) {
    const distributionId = options.distributionId;

    new lambda.CfnPermission(scope, `InvokeFromApiFor${options.originId}`, {
      principal: 'cloudfront.amazonaws.com',
      action: 'lambda:InvokeFunctionUrl',
      functionName: this.functionUrl.functionArn,
      sourceArn: `arn:${cdk.Aws.PARTITION}:cloudfront::${cdk.Aws.ACCOUNT_ID}:distribution/${distributionId}`,
    });
  }

  /**
   * Validation method: Ensures that when the OAC signing method is SIGV4_ALWAYS, the authType is set to AWS_IAM.
   */
  private validateAuthType(scope: Construct) {
    const cfnOriginAccessControl = this.originAccessControl?.node.children.find(
      (child) => child instanceof cloudfront.CfnOriginAccessControl,
    ) as cloudfront.CfnOriginAccessControl;
    const originConfig = cfnOriginAccessControl.originAccessControlConfig;
    const originAccessControlConfig = originConfig as cloudfront.CfnOriginAccessControl.OriginAccessControlConfigProperty;

    const isAlwaysSigning: boolean =
      originAccessControlConfig.signingBehavior === cloudfront.SigningBehavior.ALWAYS &&
      originAccessControlConfig.signingProtocol === cloudfront.SigningProtocol.SIGV4;

    const isAuthTypeIsNone: boolean = this.functionUrl.authType !== lambda.FunctionUrlAuthType.AWS_IAM;

    if (isAlwaysSigning && isAuthTypeIsNone) {
      throw new cdk.ValidationError(lit`FunctionUrlAuthTypeMustBeAwsIam`, 'The authType of the Function URL must be set to AWS_IAM when origin access control signing method is SIGV4_ALWAYS.', scope);
    }
  }
}

import { validateMinimumSeconds } from './private/utils';
import * as cloudfront from '../../aws-cloudfront';
import type * as cdk from '../../core';
import { Token, UnscopedValidationError } from '../../core';
import { lit } from '../../core/lib/private/literal-string';

/**
 * Properties for an Origin backed by an S3 website-configured bucket, load balancer, or custom HTTP server.
 */
export interface HttpOriginProps extends cloudfront.OriginProps {
  /**
   * Specifies the protocol (HTTP or HTTPS) that CloudFront uses to connect to the origin.
   *
   * @default OriginProtocolPolicy.HTTPS_ONLY
   */
  readonly protocolPolicy?: cloudfront.OriginProtocolPolicy;

  /**
   * The SSL versions to use when interacting with the origin.
   *
   * @default OriginSslPolicy.TLS_V1_2
   */
  readonly originSslProtocols?: cloudfront.OriginSslPolicy[];

  /**
   * The HTTP port that CloudFront uses to connect to the origin.
   *
   * @default 80
   */
  readonly httpPort?: number;

  /**
   * The HTTPS port that CloudFront uses to connect to the origin.
   *
   * @default 443
   */
  readonly httpsPort?: number;

  /**
   * Specifies how long, in seconds, CloudFront waits for a response from the origin, also known as the origin response timeout.
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
   * @default undefined - AWS Cloudfront default is IPv4
   */
  readonly ipAddressType?: cloudfront.OriginIpAddressType;
}

/**
 * An Origin for an HTTP server or S3 bucket configured for website hosting.
 */
export class HttpOrigin extends cloudfront.OriginBase {
  constructor(domainName: string, private readonly props: HttpOriginProps = {}) {
    super(domainName, props);

    validateMinimumSeconds('readTimeout', 1, props.readTimeout);
    validateMinimumSeconds('keepaliveTimeout', 1, props.keepaliveTimeout);
    this.validateResponseCompletionTimeoutWithReadTimeout(props.responseCompletionTimeout, props.readTimeout);

    this.validatePortNumber('httpPort', props.httpPort);
    this.validatePortNumber('httpsPort', props.httpsPort);
  }

  protected renderCustomOriginConfig(): cloudfront.CfnDistribution.CustomOriginConfigProperty | undefined {
    return {
      originSslProtocols: this.props.originSslProtocols ?? [cloudfront.OriginSslPolicy.TLS_V1_2],
      originProtocolPolicy: this.props.protocolPolicy ?? cloudfront.OriginProtocolPolicy.HTTPS_ONLY,
      httpPort: this.props.httpPort,
      httpsPort: this.props.httpsPort,
      originReadTimeout: this.props.readTimeout?.toSeconds(),
      originKeepaliveTimeout: this.props.keepaliveTimeout?.toSeconds(),
      ipAddressType: this.props.ipAddressType,
    };
  }

  private validatePortNumber(name: string, port: number | undefined) {
    if (port === undefined || Token.isUnresolved(port)) { return; }
    const isValid = Number.isInteger(port) && (port === 80 || port === 443 || (port >= 1024 && port <= 65535));
    if (!isValid) {
      throw new UnscopedValidationError(lit`InvalidPortValue`, `'${name}' must be 80, 443, or an integer between 1024 and 65535; received ${port}.`);
    }
  }
}

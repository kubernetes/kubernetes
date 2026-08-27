import { validateMinimumSeconds } from './private/utils';
import type * as apigateway from '../../aws-apigateway';
import * as cloudfront from '../../aws-cloudfront';
import * as cdk from '../../core';

/**
 * Properties for an Origin for an API Gateway REST API.
 */
export interface RestApiOriginProps extends cloudfront.OriginProps {
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
}

/**
 * An Origin for an API Gateway REST API.
 */
export class RestApiOrigin extends cloudfront.OriginBase {
  constructor(restApi: apigateway.RestApiBase, private readonly props: RestApiOriginProps = {}) {
    // urlForPath() is of the form 'https://<rest-api-id>.execute-api.<region>.amazonaws.com/<stage>'
    // Splitting on '/' gives: ['https', '', '<rest-api-id>.execute-api.<region>.amazonaws.com', '<stage>']
    // The element at index 2 is the domain name, the element at index 3 is the stage name
    super(cdk.Fn.select(2, cdk.Fn.split('/', restApi.url)), {
      originPath: props.originPath ?? `/${cdk.Fn.select(3, cdk.Fn.split('/', restApi.url))}`,
      ...props,
    });

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
    };
  }
}

import type { Construct } from 'constructs';
import { CallApiGatewayEndpointBase } from './base';
import type { CallApiGatewayEndpointBaseProps, CallApiGatewayEndpointJsonataBaseProps, CallApiGatewayEndpointJsonPathBaseProps } from './base-types';
import type * as apigateway from '../../../aws-apigateway';
import type * as iam from '../../../aws-iam';
import * as sfn from '../../../aws-stepfunctions';
import * as cdk from '../../../core';
import { propertyInjectable } from '../../../core/lib/prop-injectable';

/**
 * Base properties for calling an REST API Endpoint
 */
export interface CallApiGatewayRestApiEndpointOptions {
  /**
   * API to call
   */
  readonly api: apigateway.IRestApi;

  /**
   * Name of the stage where the API is deployed to in API Gateway
   */
  readonly stageName: string;

  /**
   * Specify a custom Region where the API is deployed, e.g. 'us-east-1'.
   *
   * @default - Uses the Region of the stack containing the `api`.
   */
  readonly region?: string;
}

/**
 * Properties for calling an REST API Endpoint using JSONPath
 */
export interface CallApiGatewayRestApiEndpointJsonPathProps extends CallApiGatewayEndpointJsonPathBaseProps, CallApiGatewayRestApiEndpointOptions {}

/**
 * Properties for calling an REST API Endpoint using JSONata
 */
export interface CallApiGatewayRestApiEndpointJsonataProps extends CallApiGatewayEndpointJsonataBaseProps, CallApiGatewayRestApiEndpointOptions {}

/**
 * Properties for calling an REST API Endpoint
 */
export interface CallApiGatewayRestApiEndpointProps extends CallApiGatewayEndpointBaseProps, CallApiGatewayRestApiEndpointOptions {}

/**
 * Call REST API endpoint as a Task
 *
 * Be aware that the header values must be arrays. When passing the Task Token
 * in the headers field `WAIT_FOR_TASK_TOKEN` integration, use
 * `JsonPath.array()` to wrap the token in an array:
 *
 * ```ts
 * import * as apigateway from 'aws-cdk-lib/aws-apigateway';
 * declare const api: apigateway.RestApi;
 *
 * new tasks.CallApiGatewayRestApiEndpoint(this, 'Endpoint', {
 *   api,
 *   stageName: 'Stage',
 *   method: tasks.HttpMethod.PUT,
 *   integrationPattern: sfn.IntegrationPattern.WAIT_FOR_TASK_TOKEN,
 *   headers: sfn.TaskInput.fromObject({
 *     TaskToken: sfn.JsonPath.array(sfn.JsonPath.taskToken),
 *   }),
 * });
 * ```
 *
 * @see https://docs.aws.amazon.com/step-functions/latest/dg/connect-api-gateway.html
 */
@propertyInjectable
export class CallApiGatewayRestApiEndpoint extends CallApiGatewayEndpointBase {
  /**
   * Uniquely identifies this class.
   */
  public static readonly PROPERTY_INJECTION_ID: string = 'aws-cdk-lib.aws-stepfunctions-tasks.CallApiGatewayRestApiEndpoint';

  /**
   * Call REST API endpoint as a Task  using JSONPath
   *
   * Be aware that the header values must be arrays. When passing the Task Token
   * in the headers field `WAIT_FOR_TASK_TOKEN` integration, use
   * `JsonPath.array()` to wrap the token in an array:
   *
   * ```ts
   * import * as apigateway from 'aws-cdk-lib/aws-apigateway';
   * declare const api: apigateway.RestApi;
   *
   * tasks.CallApiGatewayRestApiEndpoint.jsonPath(this, 'Endpoint', {
   *   api,
   *   stageName: 'Stage',
   *   method: tasks.HttpMethod.PUT,
   *   integrationPattern: sfn.IntegrationPattern.WAIT_FOR_TASK_TOKEN,
   *   headers: sfn.TaskInput.fromObject({
   *     TaskToken: sfn.JsonPath.array(sfn.JsonPath.taskToken),
   *   }),
   * });
   * ```
   *
   * @see https://docs.aws.amazon.com/step-functions/latest/dg/connect-api-gateway.html
   */
  public static jsonPath(scope: Construct, id: string, props: CallApiGatewayRestApiEndpointJsonPathProps) {
    return new CallApiGatewayRestApiEndpoint(scope, id, props);
  }
  /**
   * Call REST API endpoint as a Task using JSONata
   *
   * Be aware that the header values must be arrays. When passing the Task Token
   * in the headers field `WAIT_FOR_TASK_TOKEN` integration, use
   * `JsonPath.array()` to wrap the token in an array:
   *
   * ```ts
   * import * as apigateway from 'aws-cdk-lib/aws-apigateway';
   * declare const api: apigateway.RestApi;
   *
   * tasks.CallApiGatewayRestApiEndpoint.jsonata(this, 'Endpoint', {
   *   api,
   *   stageName: 'Stage',
   *   method: tasks.HttpMethod.PUT,
   *   integrationPattern: sfn.IntegrationPattern.WAIT_FOR_TASK_TOKEN,
   *   headers: sfn.TaskInput.fromObject({
   *     TaskToken: '{% States.Array($states.context.taskToken) %}',
   *   }),
   * });
   * ```
   *
   * @see https://docs.aws.amazon.com/step-functions/latest/dg/connect-api-gateway.html
   */
  public static jsonata(scope: Construct, id: string, props: CallApiGatewayRestApiEndpointJsonataProps) {
    return new CallApiGatewayRestApiEndpoint(scope, id, {
      ...props,
      queryLanguage: sfn.QueryLanguage.JSONATA,
    });
  }
  protected readonly taskMetrics?: sfn.TaskMetricsConfig | undefined;
  protected readonly taskPolicies?: iam.PolicyStatement[] | undefined;

  protected readonly apiEndpoint: string;
  protected readonly arnForExecuteApi: string;
  protected readonly stageName?: string;

  constructor(scope: Construct, id: string, private readonly props: CallApiGatewayRestApiEndpointProps) {
    super(scope, id, props);

    this.apiEndpoint = this.getApiEndpoint(props.region);
    this.arnForExecuteApi = props.api.arnForExecuteApi(
      props.method,
      (props.apiPath && /^{%(.*)%}$/s.test(props.apiPath)) ? '/*' : props.apiPath,
      props.stageName,
    );
    this.stageName = props.stageName;

    this.taskPolicies = this.createPolicyStatements();
  }

  private getApiEndpoint(region?: string): string {
    const apiStack = cdk.Stack.of(this.props.api);
    return `${this.props.api.restApiId}.execute-api.${region ?? apiStack.region}.${apiStack.urlSuffix}`;
  }
}

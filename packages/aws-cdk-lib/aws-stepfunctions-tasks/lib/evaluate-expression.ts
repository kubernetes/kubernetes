import type { Construct } from 'constructs';
import * as iam from '../../aws-iam';
import * as lambda from '../../aws-lambda';
import * as sfn from '../../aws-stepfunctions';
import { Token, UnscopedValidationError, ValidationError } from '../../core';
import { PATH_PATTERN, pathsInsideStringLiterals } from './private/evaluate-expression-paths';
import { lit } from '../../core/lib/private/literal-string';
import { EvalNodejsSingletonFunction } from '../../custom-resource-handlers/dist/aws-stepfunctions-tasks/eval-nodejs-provider.generated';

/**
 * Properties for EvaluateExpression
 *
 */
export interface EvaluateExpressionProps extends sfn.TaskStateBaseProps {
  /**
   * The expression to evaluate. The expression may contain state paths.
   *
   * A referenced path is only resolved where it is used as code (for example `$.a + $.b`, or
   * a function argument). To build a string that contains a value, use a template literal
   * interpolation (`${$.count}`); a path placed as literal text inside a plain string, or as
   * bare text inside a template literal (for example `'items: $.count'`), is not resolved and
   * is rejected at synth time.
   *
   * Example value: `'$.a + $.b'`
   */
  readonly expression: string;

  /**
   * The runtime language to use to evaluate the expression.
   *
   * @default - the latest Lambda node runtime available in your region.
   */
  readonly runtime?: lambda.Runtime;

  /**
   * The system architecture compatible with this lambda function.
   *
   * ARM64 architecture support varies by AWS region. Please verify that your deployment region supports Lambda functions with ARM64 architecture before specifying Architecture.ARM_64.
   *
   * @default Architecture.X86_64
   */
  readonly architecture?: lambda.Architecture;
}

/**
 * The event received by the Lambda function
 *
 * Shared definition with packages/@aws-cdk/custom-resource-handlers/lib/custom-resources/aws-stepfunctions-tasks/index.ts
 *
 * @internal
 */
export interface Event {
  /**
   * The expression to evaluate
   */
  readonly expression: string;

  /**
   * The expression attribute values
   */
  readonly expressionAttributeValues: { [key: string]: any };
}

/**
 * A Step Functions Task to evaluate an expression
 *
 * OUTPUT: the output of this task is the evaluated expression.
 *
 */
export class EvaluateExpression extends sfn.TaskStateBase {
  protected readonly taskMetrics?: sfn.TaskMetricsConfig;
  protected readonly taskPolicies?: iam.PolicyStatement[];

  private readonly evalFn: lambda.SingletonFunction;

  constructor(scope: Construct, id: string, private readonly props: EvaluateExpressionProps) {
    super(scope, id, props);

    // A path placed as literal text inside a plain string or template literal no longer
    // resolves, so fail fast at synth instead of failing at runtime or silently evaluating
    // to the wrong value.
    // Skip when the expression is an unresolved token that can't be inspected at synth time.
    if (!Token.isUnresolved(props.expression)) {
      const pathsInLiterals = pathsInsideStringLiterals(props.expression);
      const plainStringPaths = pathsInLiterals.filter((p) => p.context === 'plainString').map((p) => p.path);
      const templateTextPaths = pathsInLiterals.filter((p) => p.context === 'templateText').map((p) => p.path);
      if (plainStringPaths.length > 0) {
        throw new ValidationError(
          lit`ExpressionPathInStringLiteral`,
          'state path(s) ' + JSON.stringify(plainStringPaths) +
          ' are used as literal text inside a plain string in the EvaluateExpression \'expression\'; ' +
          'a path inside a plain string used to be resolved but is no longer resolved - use a template literal with ' +
          'interpolation instead, e.g. `...${' + plainStringPaths[0] + '}...`',
          this,
        );
      }
      if (templateTextPaths.length > 0) {
        throw new ValidationError(
          lit`ExpressionPathInStringLiteral`,
          'state path(s) ' + JSON.stringify(templateTextPaths) +
          ' are used as literal text inside a template literal in the EvaluateExpression \'expression\'; ' +
          'a path in template text used to be resolved but is no longer resolved - reference it via interpolation like ${' + templateTextPaths[0] + '} instead',
          this,
        );
      }
    }

    this.evalFn = createEvalFn(this.props.runtime, this.props.architecture, this);

    this.taskPolicies = [
      new iam.PolicyStatement({
        resources: this.evalFn.resourceArnsForGrantInvoke,
        actions: ['lambda:InvokeFunction'],
      }),
    ];
  }

  /**
   * @internal
   */
  protected _renderTask(): any {
    const matches = this.props.expression.match(PATH_PATTERN);

    let expressionAttributeValues = {};
    if (matches) {
      expressionAttributeValues = matches.reduce(
        (acc, m) => ({
          ...acc,
          [m]: sfn.JsonPath.stringAt(m), // It's okay to always use `stringAt` here
        }),
        {},
      );
    }

    const parameters: Event = {
      expression: this.props.expression,
      expressionAttributeValues,
    };
    return {
      Resource: this.evalFn.functionArn,
      Parameters: sfn.FieldUtils.renderObject(parameters),
    };
  }
}

function createEvalFn(runtime: lambda.Runtime | undefined, architecture: lambda.Architecture | undefined, scope: Construct) {
  const NO_RUNTIME = Symbol.for('no-runtime');
  const lambdaPurpose = 'Eval';

  const nodeJsGuids = {
    [lambda.Runtime.NODEJS_24_X.name]: 'a5c17f50-0de1-497b-8ec0-fabdc9086cc5',
    [lambda.Runtime.NODEJS_22_X.name]: 'b64e1fb8-9c89-4f7d-8a34-2e2a1c5f6d7e',
    [lambda.Runtime.NODEJS_20_X.name]: '9757c267-6d7c-45c2-af77-37a30d93d2c6',
    [lambda.Runtime.NODEJS_18_X.name]: '078d40d3-fb4e-4d53-94a7-9c46fc11fe02',
    [lambda.Runtime.NODEJS_16_X.name]: '2a430b68-eb4b-4026-9232-ee39b71c1db8',
    [lambda.Runtime.NODEJS_14_X.name]: 'da2d1181-604e-4a45-8694-1a6abd7fe42d',
    [lambda.Runtime.NODEJS_12_X.name]: '2b81e383-aad2-44db-8aaf-b4809ae0e3b4',
    [lambda.Runtime.NODEJS_10_X.name]: 'a0d2ce44-871b-4e74-87a1-f5e63d7c3bdc',

    // UUID used when falling back to the default node runtime, which is a token and might be different per region
    [NO_RUNTIME]: '41256dc5-4457-4273-8ed9-17bc818694e5',
  };

  const nodeJsArmGuids = {
    [lambda.Runtime.NODEJS_24_X.name]: '1662dbae-12a4-4a04-86f0-29531a7be268',
    [lambda.Runtime.NODEJS_22_X.name]: '672bfe19-76c9-448b-9584-ee75ca09edbf',
    [lambda.Runtime.NODEJS_20_X.name]: '105f0693-c082-4e9b-83f3-9007507676f0',
    [lambda.Runtime.NODEJS_18_X.name]: 'b1565b13-7d88-4f24-ac63-f9b23c325a55',
    [lambda.Runtime.NODEJS_16_X.name]: '8fa21b46-1d7c-445d-be08-6bbea646eb37',
    [lambda.Runtime.NODEJS_14_X.name]: 'cadc099b-465f-4267-afe0-e0037d263401',
    [lambda.Runtime.NODEJS_12_X.name]: 'abcd5ad4-b1eb-48f1-bec5-41d6f5c82caa',
    [lambda.Runtime.NODEJS_10_X.name]: '886204b9-1f2d-40bd-8a7e-628543da4f93',

    // UUID used when falling back to the default node runtime, which is a token and might be different per region
    [NO_RUNTIME]: 'be6d8ca1-9d2c-4db3-aeeb-f6f27f3f2f10',
  };

  const runtimeKey = runtime?.name ?? NO_RUNTIME;
  const guidsMap = architecture === lambda.Architecture.ARM_64 ? nodeJsArmGuids : nodeJsGuids;
  const uuid = guidsMap[runtimeKey];

  if (!uuid) {
    throw new UnscopedValidationError(lit`RuntimeCurrentlySupported`, `The runtime ${runtime?.name} is currently not supported.`);
  }

  return new EvalNodejsSingletonFunction(scope, 'EvalFunction', {
    uuid,
    lambdaPurpose,
    runtime,
    architecture,
  });
}

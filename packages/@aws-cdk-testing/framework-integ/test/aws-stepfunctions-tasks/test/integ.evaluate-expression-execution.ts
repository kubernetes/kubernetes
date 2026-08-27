import * as cdk from 'aws-cdk-lib';
import * as sfn from 'aws-cdk-lib/aws-stepfunctions';
import * as tasks from 'aws-cdk-lib/aws-stepfunctions-tasks';
import { IntegTest, ExpectedResult, Match } from '@aws-cdk/integ-tests-alpha';

/*
 * Runs a state machine of EvaluateExpression tasks, one per documented SAFE expression
 * shape (arithmetic, string concatenation, template-literal interpolation, method calls,
 * multi-statement object building, JSON.parse, Date construction, self-invoking functions,
 * ternaries, regex-literal arguments) plus overlapping state paths, then asserts each
 * evaluated result end-to-end via IntegTest assertions.
 *
 * Inputs are deterministic so the expected output is stable.
 *
 * Input: {
 *   "a": 3, "b": 4, "ab": 10, "first": "Jane", "last": "Doe",
 *   "runDate": "2023-11-14", "items": ["x", "y"], "jsonStr": "{\"k\":1}",
 *   "epoch": 1700000000000, "type": "Forward", "quoted": "it's Jane"
 * }
 */
const app = new cdk.App({
  postCliContext: {
    '@aws-cdk/aws-lambda:useCdkManagedLogGroup': false,
  },
});
const stack = new cdk.Stack(app, 'aws-cdk-sfn-evaluate-expression-execution-integ');

// Arithmetic (addition).
const sum = new tasks.EvaluateExpression(stack, 'Sum', {
  expression: '$.a + $.b',
  resultPath: '$.sum',
});

// Arithmetic (multiplication).
const product = new tasks.EvaluateExpression(stack, 'Product', {
  expression: '$.a * $.b',
  resultPath: '$.product',
});

// $.a is a prefix of $.ab; both must resolve to their own value.
const overlap = new tasks.EvaluateExpression(stack, 'Overlap', {
  expression: '$.a + $.ab',
  resultPath: '$.overlap',
});

// String concatenation with a separator literal.
const greeting = new tasks.EvaluateExpression(stack, 'Greeting', {
  expression: "$.first + ' ' + $.last",
  resultPath: '$.greeting',
});

// Template literal interpolating two paths.
const template = new tasks.EvaluateExpression(stack, 'Template', {
  expression: '`${$.first} ${$.last}`',
  resultPath: '$.template',
});

// Built-in function call on paths.
const max = new tasks.EvaluateExpression(stack, 'Max', {
  expression: 'Math.max($.a, $.b)',
  resultPath: '$.max',
});

// Method call on an arithmetic result.
const rounded = new tasks.EvaluateExpression(stack, 'Rounded', {
  expression: '($.a / $.b).toFixed(2)',
  resultPath: '$.rounded',
});

// Multi-statement expression building an object via a trailing expression.
const dayObj = new tasks.EvaluateExpression(stack, 'DayObj', {
  expression: 'const t = $.runDate;\n({ day: t })',
  resultPath: '$.dayObj',
});

// Template literal with interpolated paths.
const dash = new tasks.EvaluateExpression(stack, 'Dash', {
  expression: '`${$.first}-${$.last}`',
  resultPath: '$.dash',
});

// Method chain on a path.
const joined = new tasks.EvaluateExpression(stack, 'Joined', {
  expression: "($.items).join(',')",
  resultPath: '$.joined',
});

// JSON.parse of a path value.
const parsed = new tasks.EvaluateExpression(stack, 'Parsed', {
  expression: 'JSON.parse($.jsonStr)',
  resultPath: '$.parsed',
});

// Date manipulation from a fixed epoch (deterministic ISO string).
const iso = new tasks.EvaluateExpression(stack, 'Iso', {
  expression: 'new Date($.epoch).toISOString()',
  resultPath: '$.iso',
});

// A self-invoking function that returns a value.
const iife = new tasks.EvaluateExpression(stack, 'Iife', {
  expression: '(function(){ return $.a + $.b; })()',
  resultPath: '$.iife',
});

// Ternary / conditional expression.
const ternary = new tasks.EvaluateExpression(stack, 'Ternary', {
  expression: "$.type == 'Forward' ? 'A' : 'B'",
  resultPath: '$.ternary',
});

// String concatenation mixing string literals with bare paths.
const helloGreeting = new tasks.EvaluateExpression(stack, 'HelloGreeting', {
  expression: "'Hello ' + $.first + ' ' + $.last",
  resultPath: '$.helloGreeting',
});

// Regex with a quote inside its body
const replaced = new tasks.EvaluateExpression(stack, 'Replaced', {
  expression: "($.quoted).replace(/'/g, '_')",
  resultPath: '$.replaced',
});

const sm = new sfn.StateMachine(stack, 'StateMachine', {
  definitionBody: sfn.DefinitionBody.fromChainable(
    sum.next(product).next(overlap).next(greeting).next(template).next(max).next(rounded)
      .next(dayObj).next(dash).next(joined).next(parsed).next(iso).next(iife).next(ternary).next(helloGreeting).next(replaced),
  ),
});

const integ = new IntegTest(app, 'EvaluateExpressionExecutionTest', {
  testCases: [stack],
  diffAssets: true,
});

const execution = integ.assertions.awsApiCall('StepFunctions', 'startExecution', {
  stateMachineArn: sm.stateMachineArn,
  input: JSON.stringify({
    a: 3,
    b: 4,
    ab: 10,
    first: 'Jane',
    last: 'Doe',
    runDate: '2023-11-14',
    items: ['x', 'y'],
    jsonStr: '{"k":1}',
    epoch: 1700000000000,
    type: 'Forward',
    quoted: "it's Jane",
  }),
});

integ.assertions.awsApiCall('StepFunctions', 'describeExecution', {
  executionArn: execution.getAttString('executionArn'),
}).expect(ExpectedResult.objectLike({
  status: 'SUCCEEDED',
  output: Match.serializedJson(Match.objectLike({
    sum: 7,
    product: 12,
    overlap: 13,
    greeting: 'Jane Doe',
    template: 'Jane Doe',
    max: 4,
    rounded: '0.75',
    dayObj: { day: '2023-11-14' },
    dash: 'Jane-Doe',
    joined: 'x,y',
    parsed: { k: 1 },
    iso: '2023-11-14T22:13:20.000Z',
    iife: 7,
    ternary: 'A',
    helloGreeting: 'Hello Jane Doe',
    replaced: 'it_s Jane',
  })),
})).waitForAssertions({
  totalTimeout: cdk.Duration.minutes(2),
  interval: cdk.Duration.seconds(5),
});

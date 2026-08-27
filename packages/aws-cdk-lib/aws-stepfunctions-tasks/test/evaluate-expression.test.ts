import { Template } from '../../assertions';
import { Architecture, Runtime, RuntimeFamily } from '../../aws-lambda';
import * as sfn from '../../aws-stepfunctions';
import { Lazy, Stack } from '../../core';
import * as tasks from '../lib';

let stack: Stack;
beforeEach(() => {
  stack = new Stack();
});

test('Eval with Node.js', () => {
  // WHEN
  const task = new tasks.EvaluateExpression(stack, 'Task', {
    expression: '$.a + $.b',
  });
  new sfn.StateMachine(stack, 'SM', {
    definitionBody: sfn.DefinitionBody.fromChainable(task),
  });

  // THEN
  Template.fromStack(stack).hasResourceProperties('AWS::StepFunctions::StateMachine', {
    DefinitionString: {
      'Fn::Join': [
        '',
        [
          '{"StartAt":"Task","States":{"Task":{"End":true,"Type":"Task","Resource":"',
          {
            'Fn::GetAtt': ['Eval41256dc5445742738ed917bc818694e54EB1134F', 'Arn'],
          },
          '","Parameters":{"expression":"$.a + $.b","expressionAttributeValues":{"$.a.$":"$.a","$.b.$":"$.b"}}}}}',
        ],
      ],
    },
  });

  Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
    Runtime: 'nodejs24.x',
  });
});

test('expression does not contain paths', () => {
  // WHEN
  const task = new tasks.EvaluateExpression(stack, 'Task', {
    expression: '2 + 2',
  });
  new sfn.StateMachine(stack, 'SM', {
    definitionBody: sfn.DefinitionBody.fromChainable(task),
  });

  Template.fromStack(stack).hasResourceProperties('AWS::StepFunctions::StateMachine', {
    DefinitionString: {
      'Fn::Join': [
        '',
        [
          '{"StartAt":"Task","States":{"Task":{"End":true,"Type":"Task","Resource":"',
          {
            'Fn::GetAtt': ['Eval41256dc5445742738ed917bc818694e54EB1134F', 'Arn'],
          },
          '","Parameters":{"expression":"2 + 2","expressionAttributeValues":{}}}}}',
        ],
      ],
    },
  });
});

test('with dash and underscore in path', () => {
  // WHEN
  const task = new tasks.EvaluateExpression(stack, 'Task', {
    expression: '$.a_b + $.c-d + $[_e]',
  });
  new sfn.StateMachine(stack, 'SM', {
    definitionBody: sfn.DefinitionBody.fromChainable(task),
  });

  Template.fromStack(stack).hasResourceProperties('AWS::StepFunctions::StateMachine', {
    DefinitionString: {
      'Fn::Join': [
        '',
        [
          '{"StartAt":"Task","States":{"Task":{"End":true,"Type":"Task","Resource":"',
          {
            'Fn::GetAtt': ['Eval41256dc5445742738ed917bc818694e54EB1134F', 'Arn'],
          },
          '","Parameters":{"expression":"$.a_b + $.c-d + $[_e]","expressionAttributeValues":{"$.a_b.$":"$.a_b","$.c-d.$":"$.c-d","$[_e].$":"$[_e]"}}}}}',
        ],
      ],
    },
  });
});

test('With Node.js 20.x', () => {
  // WHEN
  const task = new tasks.EvaluateExpression(stack, 'Task', {
    expression: '$.a + $.b',
    runtime: new Runtime('nodejs20.x', RuntimeFamily.NODEJS),
  });
  new sfn.StateMachine(stack, 'SM', {
    definition: task,
  });

  Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
    Runtime: 'nodejs20.x',
  });
});

test('With Node.js 22.x', () => {
  // WHEN
  const task = new tasks.EvaluateExpression(stack, 'Task', {
    expression: '$.a + $.b',
    runtime: Runtime.NODEJS_22_X,
  });
  new sfn.StateMachine(stack, 'SM', {
    definition: task,
  });

  Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
    Runtime: 'nodejs22.x',
  });
});

test('With Node.js 24.x', () => {
  // WHEN
  const task = new tasks.EvaluateExpression(stack, 'Task', {
    expression: '$.a + $.b',
    runtime: Runtime.NODEJS_24_X,
  });
  new sfn.StateMachine(stack, 'SM', {
    definition: task,
  });

  Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
    Runtime: 'nodejs24.x',
  });
});

test('With ARM64 architecture', () => {
  // WHEN
  const task = new tasks.EvaluateExpression(stack, 'Task', {
    expression: '$.a + $.b',
    architecture: Architecture.ARM_64,
  });
  new sfn.StateMachine(stack, 'SM', {
    definition: task,
  });

  Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
    Architectures: ['arm64'],
  });
});

test('With X86_64 architecture', () => {
  // WHEN
  const task = new tasks.EvaluateExpression(stack, 'Task', {
    expression: '$.a + $.b',
    architecture: Architecture.X86_64,
  });
  new sfn.StateMachine(stack, 'SM', {
    definition: task,
  });

  Template.fromStack(stack).hasResourceProperties('AWS::Lambda::Function', {
    Architectures: ['x86_64'],
  });
});

describe('fails when a referenced path is inside a string or template literal', () => {
  test.each([
    ['single-quoted string', "'year=$.year'"],
    ['double-quoted string', '"year=$.year"'],
    ['template literal text (no ${})', '`year=$.year`'],
    ['bare-brace template', '`{$.year}`'],
    ['JSON.parse of a quoted path', "JSON.parse('$.year')"],
  ])('fails for a path inside a %s', (_name, expression) => {
    expect(() => new tasks.EvaluateExpression(stack, 'Task', { expression }))
      .toThrow(/are used as literal text inside a (plain string|template literal)/);
  });

  test('fails and names the offending path with a migration hint', () => {
    expect(() => new tasks.EvaluateExpression(stack, 'Task', { expression: "'hello $.user'" }))
      .toThrow(/\$\.user.*e\.g\. `\.\.\.\$\{\$\.user\}\.\.\.`/);
  });
});

describe('accepts paths in code position', () => {
  test.each([
    ['arithmetic', '$.a + $.b'],
    ['division (slash is just an operator now)', '$.a / $.b'],
    ['template interpolation ${$.x}', '`year=${$.year}`'],
    ['method call arguments', 'Math.max($.a, $.b)'],
    ['JSON.parse of a value (no quotes)', 'JSON.parse($.body)'],
    ['array literal element', '[$.a, $.b]'],
    ['string concat with path outside the quotes', "$.first + ' ' + $.last"],
    ['no referenced paths', '2 + 2'],
  ])('does not throw for %s', (_name, expression) => {
    expect(() => {
      const task = new tasks.EvaluateExpression(stack, 'Task', { expression });
      new sfn.StateMachine(stack, 'SM', { definitionBody: sfn.DefinitionBody.fromChainable(task) });
    }).not.toThrow();
  });
});

describe('skips validation for tokenized expressions', () => {
  test('does not validate a tokenized expression', () => {
    // A token cannot be inspected at synth time, so the check is skipped even for a
    // shape that would otherwise be rejected.
    expect(() => {
      const task = new tasks.EvaluateExpression(stack, 'Task', {
        expression: Lazy.string({ produce: () => "'year=$.year'" }),
      });
      new sfn.StateMachine(stack, 'SM', { definitionBody: sfn.DefinitionBody.fromChainable(task) });
    }).not.toThrow();
  });
});

describe('template literals: interpolation is accepted, bare paths are rejected', () => {
  // Happy paths: state paths referenced via `${...}` interpolation are code position and
  // resolve at runtime, so the construct accepts them.
  test.each([
    ['a single interpolation', '`Now waiting ${$.waitSeconds} seconds`'],
    ['multiple interpolations', '`${$.first} ${$.last}`'],
    ['interpolation inside an S3 path', '`s3://bucket/year=${$.year}/month=${$.month}/`'],
  ])('accepts %s', (_name, expression) => {
    expect(() => {
      const task = new tasks.EvaluateExpression(stack, 'Task', { expression });
      new sfn.StateMachine(stack, 'SM', { definitionBody: sfn.DefinitionBody.fromChainable(task) });
    }).not.toThrow();
  });

  // Failure paths: a bare path in template text (not inside `${...}`) is never interpolated,
  // so the construct rejects it at synth with the migration hint.
  test.each([
    ['a bare path in template text', '`total: $.n`'],
    ['a bare path in an S3-path template', '`s3://bucket/year=$.year/`'],
    ['a bare-brace path that looks like interpolation but is not', '`{$.x}`'],
  ])('fails for %s', (_name, expression) => {
    expect(() => new tasks.EvaluateExpression(stack, 'Task', { expression }))
      .toThrow(/are used as literal text inside a template literal/);
  });
});

describe('EvaluateExpression expression shapes', () => {
  // Synthesizes one EvaluateExpression task and returns its `expression` and
  // `expressionAttributeValues` so tests can assert the construct emits them correctly.
  // Those two fields live inside the state machine's DefinitionString, which renders as an
  // Fn::Join around the eval Lambda ARN. We're not testing the join itself; we just keep its
  // string parts (dropping the ARN token, which sits in `Resource`, not the fields we check)
  // and rejoin them to recover parseable JSON.
  function synth(expression: string): { expression: string; expressionAttributeValues: Record<string, string> } {
    const task = new tasks.EvaluateExpression(stack, 'Task', { expression });
    new sfn.StateMachine(stack, 'SM', { definitionBody: sfn.DefinitionBody.fromChainable(task) });

    const resources = Template.fromStack(stack).findResources('AWS::StepFunctions::StateMachine');
    const def = Object.values(resources)[0].Properties.DefinitionString;
    const combined: string = typeof def === 'string'
      ? def
      : def['Fn::Join'][1].map((part: unknown) => (typeof part === 'string' ? part : '')).join('');
    return JSON.parse(combined).States.Task.Parameters;
  }

  describe('synthesizes and binds each referenced path', () => {
    test.each<[string, string, string[]]>([
      ['arithmetic', '$.a + $.b', ['$.a', '$.b']],
      ['string-concat multi-line', 'const t = $.runDate;\n' + '({ regionId: $.regionId })', ['$.runDate', '$.regionId']],
      ['template literal interpolation', '`${$.Execution.Id}/requests/`', ['$.Execution.Id']],
      ['array bare elements', '[$.srcPath, $.destPath, 1]', ['$.srcPath', '$.destPath']],
      ['method chain', "($.list).filter(x => x.includes('a')).join(',')", ['$.list']],
      ['JSON.parse of a path value', 'JSON.parse($.body)', ['$.body']],
      ['JSON array built with interpolation', 'JSON.parse(`["submit","--id","${$.MarketplaceId}"]`)', ['$.MarketplaceId']],
      ['Date construction from a path', 'new Date($.epoch).toISOString()', ['$.epoch']],
      ['a self-invoking function returning a value', '(function(){ var d = new Date($.date); return { t: d.getTime() }; })()', ['$.date']],
      ['multiple statements returning an object', 'const t = $.runDate; ({ day: t })', ['$.runDate']],
      ['ternary', "$.type == 'Forward' ? 'A' : 'B'", ['$.type']],
      ['boolean comparison', "typeof $.Cause == 'string'", ['$.Cause']],
      ['a single referenced path', '$.IsPrimaryLocale', ['$.IsPrimaryLocale']],
      ['try/catch', "try { JSON.parse($.Cause).m } catch (e) { 'x' }", ['$.Cause']],
    ])('binds paths for %s', (_name, expression, paths) => {
      // Assert two things: (1) the expression lands in the task Parameters verbatim, and
      // (2) every referenced path is bound as an expressionAttributeValues key (bound via
      // JsonPath, so the key gains a `.$` suffix and the value is the path).
      const params = synth(expression);

      expect(params.expression).toEqual(expression);

      for (const path of paths) {
        expect(params.expressionAttributeValues[`${path}.$`]).toEqual(path);
      }
      expect(Object.keys(params.expressionAttributeValues)).toHaveLength(paths.length);
    });

    test('synth-time TS interpolation bakes the const into the resolved path', () => {
      const DATE_PATH = '$.currentDate';
      const params = synth(`(${DATE_PATH}).split('-')`);

      expect(params.expression).toEqual("($.currentDate).split('-')");
      expect(params.expressionAttributeValues['$.currentDate.$']).toEqual('$.currentDate');
    });

    test('synth-time TS interpolation in a path suffix', () => {
      const WF = 'Job';
      const params = synth(`$.${WF}-Result.items`);

      expect(params.expression).toEqual('$.Job-Result.items');
      expect(params.expressionAttributeValues['$.Job-Result.items.$']).toEqual('$.Job-Result.items');
    });

    test('a constant with no path binds nothing', () => {
      const params = synth('0');

      expect(params.expression).toEqual('0');
      expect(params.expressionAttributeValues).toEqual({});
    });
  });

  describe('rejects a path used as literal text inside a string or template', () => {
    test.each<[string, string]>([
      ['single-quoted string', "'x=$.count'"],
      ['double-quoted string', '"hello $.user"'],
      ['array with quoted path elements', "['--flag', '$.input.x']"],
      ['bare path in template text', '`Processing $.documentId`'],
      ['brace-wrapped bare path in template text', '`{$.Payload.Path}`'],
      ['JSON.parse of a quoted path', "JSON.parse('$.a')"],
    ])('fails for a %s', (_name, expression) => {
      expect(() => new tasks.EvaluateExpression(stack, 'Task', { expression }))
        .toThrow(/are used as literal text inside a (plain string|template literal)/);
    });

    test('fails with a plain-string message that points at a template literal', () => {
      const expression = "'x=$.count'";
      expect(() => new tasks.EvaluateExpression(stack, 'T1', { expression })).toThrow(/inside a plain string/);
      expect(() => new tasks.EvaluateExpression(stack, 'T2', { expression })).toThrow(/template literal with interpolation/);
    });

    test('fails with a template-text message that points at ${...} interpolation', () => {
      const expression = '`Processing $.documentId`';
      expect(() => new tasks.EvaluateExpression(stack, 'T1', { expression })).toThrow(/inside a template literal/);
      expect(() => new tasks.EvaluateExpression(stack, 'T2', { expression })).toThrow(/interpolation like \$\{/);
    });
  });

  describe('skips the literal-text check for tokenized expressions', () => {
    test('does not throw when the expression is an unresolved token', () => {
      // An ASL intrinsic (here JsonPath.array) is an unresolved token, so the synth-time
      // literal-text scan is skipped.
      const expression = sfn.JsonPath.array('spark-submit', '--id', sfn.JsonPath.stringAt('$.id'));

      expect(() => {
        const task = new tasks.EvaluateExpression(stack, 'Task', { expression });
        new sfn.StateMachine(stack, 'SM', { definitionBody: sfn.DefinitionBody.fromChainable(task) });
        Template.fromStack(stack);
      }).not.toThrow();
    });
  });
});

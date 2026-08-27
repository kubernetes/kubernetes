import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import type { PolicyValidationReportJson } from '@aws-cdk/cloud-assembly-schema';
import { Construct } from 'constructs';
import * as cxapi from '../../../cx-api';
import * as core from '../../lib';
import { readPerfCounters, resetCounters } from '../../lib/private/perf';

let consoleErrorMock: jest.SpyInstance;
beforeEach(() => {
  resetCounters();
  consoleErrorMock = jest.spyOn(console, 'error').mockImplementation(() => { return true; });
  jest.spyOn(console, 'log').mockImplementation(() => { return true; });
  process.exitCode = undefined;
});

afterEach(() => {
  jest.clearAllMocks();
});

const originalContextJson = process.env.CDK_CONTEXT_JSON;

beforeAll(() => {
  // These tests validate the plugin's own behavior — strict mode would mask the signals
  // by throwing before tests can assert on warnings/errors.
  process.env.CDK_CONTEXT_JSON = JSON.stringify({
    ...JSON.parse(originalContextJson ?? '{}'),
    '@aws-cdk/core:strictCfnValidateErrors': false,
  });
});

afterAll(() => {
  jest.resetAllMocks();
  process.env.CDK_CONTEXT_JSON = originalContextJson;
});

describe('CloudFormationValidatePlugin', () => {
  test('reports schema violations for invalid properties', () => {
    const app = new core.App({
      context: {
        [cxapi.VALIDATE_AGAINST_DEFAULT_RULES]: true,
        [cxapi.FAIL_SYNTH_ON_VALIDATION_ERRORS_CONTEXT]: true,
      },
    });
    const stack = new core.Stack(app, 'TestStack');
    new core.CfnResource(stack, 'MyBucket', {
      type: 'AWS::S3::Bucket',
      properties: {
        BogusProperty: 'invalid-value',
      },
    });

    expect(() => app.synth()).toThrow(/BogusProperty/);
  });

  test('downgrades errors to warnings when flag is not explicitly enabled', () => {
    const app = new core.App({
      context: {
        [cxapi.FAIL_SYNTH_ON_VALIDATION_ERRORS_CONTEXT]: true,
      },
    });
    const stack = new core.Stack(app, 'TestStack');
    new core.CfnResource(stack, 'MyBucket', {
      type: 'AWS::S3::Bucket',
      properties: {
        BogusProperty: 'invalid-value',
      },
    });

    app.synth();

    expect(process.exitCode).toBeUndefined();
    const output = consoleErrorMock.mock.calls.map((c: any[]) => c[0]).join('\n');
    expect(output).toContain('Template validation found issues in your templates');
    expect(output).toContain(cxapi.VALIDATE_AGAINST_DEFAULT_RULES);
  });

  test('template paths are assembly-relative, also for nested assemblies', () => {
    const app = new core.App({
      context: {
        [cxapi.FAIL_SYNTH_ON_VALIDATION_ERRORS_CONTEXT]: false,
      },
    });
    const stack = new core.Stack(app, 'TestStack');
    new core.CfnResource(stack, 'MyBucket', {
      type: 'AWS::S3::Bucket',
      properties: { BogusProperty: 'invalid-value' },
    });
    const stage = new core.Stage(app, 'MyStage');
    const stack2 = new core.Stack(stage, 'TestStack2');
    new core.CfnResource(stack2, 'MyBucket2', {
      type: 'AWS::S3::Bucket',
      properties: { BogusProperty: 'invalid-value' },
    });

    const asm = app.synth();

    // THEN
    const validationReport = loadValidationReport(asm);
    const report = validationReport.pluginReports.find((r: any) => r.pluginName === 'CloudFormation Validate');
    expect(report?.violations).toEqual([
      expect.objectContaining({
        ruleName: 'F3002',
        violatingConstructs: [
          expect.objectContaining({
            cloudFormationResource: expect.objectContaining({
              logicalId: 'MyBucket2',
              templatePath: 'assembly-MyStage/MyStageTestStack242E16257.template.json',
            }),
          }),
          expect.objectContaining({
            cloudFormationResource: expect.objectContaining({
              logicalId: 'MyBucket',
              templatePath: 'TestStack.template.json',
            }),
          }),
        ],
      }),
    ]);
  });

  test('succeeds with valid template', () => {
    const app = new core.App({
      context: {
        [cxapi.VALIDATE_AGAINST_DEFAULT_RULES]: true,
        [cxapi.FAIL_SYNTH_ON_VALIDATION_ERRORS_CONTEXT]: true,
      },
    });
    const stack = new core.Stack(app, 'TestStack');
    new core.CfnResource(stack, 'MyBucket', {
      type: 'AWS::S3::Bucket',
    });

    app.synth();

    expect(process.exitCode).toBeUndefined();
  });

  test('correctly reports errors at stack level instead of resource level', () => {
    const app = new core.App({
      context: {
        [cxapi.VALIDATE_AGAINST_DEFAULT_RULES]: true,
        [cxapi.FAIL_SYNTH_ON_VALIDATION_ERRORS_CONTEXT]: false,
      },
    });
    // REmove any acknowledgements for this test, since we want to see the errors
    (app.node as any)._metadata = [];

    // F0001 missing resources
    new core.Stack(app, 'TestStack');

    const report = loadValidationReport(app.synth());
    expect(report).toEqual(expect.objectContaining({
      pluginReports: expect.arrayContaining([
        expect.objectContaining({
          pluginName: 'CloudFormation Validate',
          violations: expect.arrayContaining([
            expect.objectContaining({
              ruleName: 'F0001',
              violatingConstructs: expect.arrayContaining([
                expect.objectContaining({
                  constructPath: 'TestStack',
                }),
              ]),
            }),
          ]),
        }),
      ]),

    }));
  });

  test('correctly reports errors for non-resources (e.g. Parameters) instead of resource level', () => {
    const app = new core.App({
      context: {
        [cxapi.VALIDATE_AGAINST_DEFAULT_RULES]: true,
        [cxapi.FAIL_SYNTH_ON_VALIDATION_ERRORS_CONTEXT]: false,
      },
    });
    // REmove any acknowledgements for this test, since we want to see the errors
    (app.node as any)._metadata = [];

    const stack = new core.Stack(app, 'TestStack');
    new core.CfnParameter(stack, 'MyParam', {
      type: 'Blimp',
    });

    const report = loadValidationReport(app.synth());
    expect(report).toEqual(expect.objectContaining({
      pluginReports: expect.arrayContaining([
        expect.objectContaining({
          pluginName: 'CloudFormation Validate',
          violations: expect.arrayContaining([
            expect.objectContaining({
              ruleName: 'F0001',
              violatingConstructs: expect.arrayContaining([
                expect.objectContaining({
                  // TODO: Currently this references the stack, in the future perhaps we have more information
                  // to reference the actual Parameter construct: <https://github.com/aws-cloudformation/cloudformation-validate/issues/201>
                  constructPath: 'TestStack',
                }),
              ]),
            }),
          ]),
        }),
      ]),

    }));
  });

  test('plugin can be instantiated directly with custom rules', () => {
    const plugin = new core.CloudFormationValidatePlugin({
      regoRules: [{ name: 'my-rule', content: 'package main' }],
    });

    expect(plugin.name).toBe('CloudFormation Validate');
    expect(plugin.version).toBeDefined();
    expect(plugin.ruleIds).toBeDefined();
  });

  test('user-registered instance replaces the auto-registered one', () => {
    const app = new core.App({
      context: {
        [cxapi.VALIDATE_AGAINST_DEFAULT_RULES]: true,
        [cxapi.FAIL_SYNTH_ON_VALIDATION_ERRORS_CONTEXT]: true,
      },
    });
    core.Validations.of(app).addPlugins(new core.CloudFormationValidatePlugin());
    const stack = new core.Stack(app, 'TestStack');
    new core.CfnResource(stack, 'MyBucket', {
      type: 'AWS::S3::Bucket',
    });

    app.synth();

    expect(process.exitCode).toBeUndefined();
  });

  test('fails when registering more than one CloudFormationValidatePlugin', () => {
    const app = new core.App({
      context: {
        [cxapi.VALIDATE_AGAINST_DEFAULT_RULES]: true,
        [cxapi.FAIL_SYNTH_ON_VALIDATION_ERRORS_CONTEXT]: true,
      },
    });
    core.Validations.of(app).addPlugins(new core.CloudFormationValidatePlugin());
    core.Validations.of(app).addPlugins(new core.CloudFormationValidatePlugin());

    const stack = new core.Stack(app, 'TestStack');
    new core.CfnResource(stack, 'MyBucket', {
      type: 'AWS::S3::Bucket',
    });

    expect(() => app.synth()).toThrow(/only one instance of CloudFormationValidatePlugin can be registered/);
  });

  test('fails when registered on a Stage instead of App', () => {
    const app = new core.App({
      context: {
        [cxapi.VALIDATE_AGAINST_DEFAULT_RULES]: true,
        [cxapi.FAIL_SYNTH_ON_VALIDATION_ERRORS_CONTEXT]: true,
      },
    });
    const stage = new core.Stage(app, 'MyStage');
    core.Validations.of(stage).addPlugins(new core.CloudFormationValidatePlugin());
    const stack = new core.Stack(stage, 'TestStack');
    new core.CfnResource(stack, 'MyBucket', {
      type: 'AWS::S3::Bucket',
    });

    expect(() => app.synth()).toThrow(/CloudFormationValidatePlugin can only be registered at the App level/);
  });

  test('plugin validates a template file directly', () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'cdk-validate-'));
    const templatePath = path.join(tmpDir, 'template.json');
    fs.writeFileSync(templatePath, JSON.stringify({
      Resources: {
        MyBucket: {
          Type: 'AWS::S3::Bucket',
          Properties: {
            InvalidProp: 'bad',
          },
        },
      },
    }));

    const plugin = new core.CloudFormationValidatePlugin();
    const report = plugin.validate({
      templatePaths: [templatePath],
      stackTemplates: [{ stackConstructPath: 'TestStack', templatePath }],
      appConstruct: new Construct(undefined as any, ''),
      accountId: undefined,
      region: undefined,
    });

    expect(report.success).toBe(false);
    expect(report.violations.length).toBeGreaterThan(0);
    const schemaViolation = report.violations.find(v => v.description.includes('InvalidProp'));
    expect(schemaViolation).toBeDefined();

    fs.rmSync(tmpDir, { recursive: true });
  });

  test('records validation calls, duration, and diagnostic counts by severity', () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'cdk-validate-metrics-'));
    const templatePaths = ['one.json', 'two.json'].map(fileName => path.join(tmpDir, fileName));
    for (const templatePath of templatePaths) {
      fs.writeFileSync(templatePath, JSON.stringify({ Resources: {} }));
    }

    try {
      const plugin = new core.CloudFormationValidatePlugin();
      const validateDetailed = jest.spyOn((plugin as any).engine, 'validateDetailed')
        .mockReturnValueOnce({
          diagnostics: [
            { ruleId: 'F0001', severity: 'FATAL', message: 'fatal diagnostic' },
            { ruleId: 'E0001', severity: 'ERROR', message: 'error diagnostic' },
            { ruleId: 'W0001', severity: 'WARN', message: 'warning diagnostic' },
          ],
        })
        .mockReturnValueOnce({
          diagnostics: [
            { ruleId: 'I0001', severity: 'INFO', message: 'informational diagnostic' },
            { ruleId: 'D0001', severity: 'DEBUG', message: 'debug diagnostic' },
          ],
        });

      plugin.validate({
        templatePaths,
        stackTemplates: templatePaths.map((templatePath, index) => ({
          stackConstructPath: `TestStack${index + 1}`,
          templatePath,
        })),
        appConstruct: new Construct(undefined as any, ''),
        accountId: undefined,
        region: undefined,
      });

      const counters = readPerfCounters({ telemetry: true });
      expect(validateDetailed).toHaveBeenCalledTimes(2);
      expect(counters).toMatchObject({
        [VALIDATE_DETAILED_METRIC]: {
          count: 2,
          total: expect.any(Number),
        },
        [DIAGNOSTICS_METRIC]: { count: 5, total: 0 },
        [`${DIAGNOSTICS_METRIC}.FATAL`]: { count: 1, total: 0 },
        [`${DIAGNOSTICS_METRIC}.ERROR`]: { count: 1, total: 0 },
        [`${DIAGNOSTICS_METRIC}.WARN`]: { count: 1, total: 0 },
        [`${DIAGNOSTICS_METRIC}.INFO`]: { count: 1, total: 0 },
        [`${DIAGNOSTICS_METRIC}.DEBUG`]: { count: 1, total: 0 },
      });
    } finally {
      fs.rmSync(tmpDir, { recursive: true });
    }
  });

  describe('_additionalSchemasDirectory', () => {
    let schemaDir: string;

    beforeEach(() => {
      schemaDir = fs.mkdtempSync(path.join(os.tmpdir(), 'cfn-overlay-schemas-'));
    });

    afterEach(() => {
      fs.rmSync(schemaDir, { recursive: true, force: true });
    });

    test('overlayed properties do not produce F3002', () => {
      // Write a schema that adds a custom property to WaitConditionHandle
      fs.writeFileSync(path.join(schemaDir, 'aws-cloudformation-waitconditionhandle.json'), JSON.stringify({
        typeName: 'AWS::CloudFormation::WaitConditionHandle',
        description: 'Test schema with additional property',
        properties: {
          PreGaProperty: { type: 'string', description: 'A pre-GA property' },
        },
        additionalProperties: false,
      }));

      const app = new core.App({
        context: {
          [cxapi.VALIDATE_AGAINST_DEFAULT_RULES]: true,
          [cxapi.FAIL_SYNTH_ON_VALIDATION_ERRORS_CONTEXT]: true,
        },
      });
      core.Validations.of(app).addPlugins(new core.CloudFormationValidatePlugin({
        _additionalSchemasDirectory: schemaDir,
      }));
      const stack = new core.Stack(app, 'TestStack');
      new core.CfnResource(stack, 'Handle', {
        type: 'AWS::CloudFormation::WaitConditionHandle',
        properties: {
          PreGaProperty: 'hello',
        },
      });

      // Should NOT throw — the overlay schema declares PreGaProperty as valid
      expect(() => app.synth()).not.toThrow();
    });

    test('properties not in overlay still produce violations', () => {
      // Write a schema that adds only PreGaProperty — TotallyBogus is not declared
      fs.writeFileSync(path.join(schemaDir, 'aws-cloudformation-waitconditionhandle.json'), JSON.stringify({
        typeName: 'AWS::CloudFormation::WaitConditionHandle',
        description: 'Test schema with additional property',
        properties: {
          PreGaProperty: { type: 'string' },
        },
        additionalProperties: false,
      }));

      const app = new core.App({
        context: {
          [cxapi.VALIDATE_AGAINST_DEFAULT_RULES]: true,
          [cxapi.FAIL_SYNTH_ON_VALIDATION_ERRORS_CONTEXT]: true,
        },
      });
      core.Validations.of(app).addPlugins(new core.CloudFormationValidatePlugin({
        _additionalSchemasDirectory: schemaDir,
      }));
      const stack = new core.Stack(app, 'TestStack');
      new core.CfnResource(stack, 'Handle', {
        type: 'AWS::CloudFormation::WaitConditionHandle',
        properties: {
          TotallyBogus: 'invalid',
        },
      });

      // Should still throw — TotallyBogus is NOT in the overlay
      expect(() => app.synth()).toThrow(/TotallyBogus/);
    });

    test('empty directory is a no-op', () => {
      const app = new core.App({
        context: {
          [cxapi.VALIDATE_AGAINST_DEFAULT_RULES]: true,
          [cxapi.FAIL_SYNTH_ON_VALIDATION_ERRORS_CONTEXT]: true,
        },
      });
      core.Validations.of(app).addPlugins(new core.CloudFormationValidatePlugin({
        _additionalSchemasDirectory: schemaDir, // empty dir, no schemas
      }));
      const stack = new core.Stack(app, 'TestStack');
      new core.CfnResource(stack, 'MyBucket', {
        type: 'AWS::S3::Bucket',
        properties: {
          BogusProperty: 'invalid-value',
        },
      });

      // Should still throw — no schemas loaded, default behavior
      expect(() => app.synth()).toThrow(/BogusProperty/);
    });

    test('non-existent directory is a no-op', () => {
      const plugin = new core.CloudFormationValidatePlugin({
        _additionalSchemasDirectory: '/non/existent/path',
      });

      expect(plugin.name).toBe('CloudFormation Validate');
      expect(plugin.version).toBeDefined();
    });

    test('schemas in nested subdirectories are discovered', () => {
      // Simulate the real layout: temporary-schemas/us-east-1/aws-lambda-function.json
      const regionDir = path.join(schemaDir, 'us-east-1');
      fs.mkdirSync(regionDir);
      fs.writeFileSync(path.join(regionDir, 'aws-cloudformation-waitconditionhandle.json'), JSON.stringify({
        typeName: 'AWS::CloudFormation::WaitConditionHandle',
        properties: {
          NestedProperty: { type: 'string' },
        },
        additionalProperties: false,
      }));

      const app = new core.App({
        context: {
          [cxapi.VALIDATE_AGAINST_DEFAULT_RULES]: true,
          [cxapi.FAIL_SYNTH_ON_VALIDATION_ERRORS_CONTEXT]: true,
        },
      });
      core.Validations.of(app).addPlugins(new core.CloudFormationValidatePlugin({
        _additionalSchemasDirectory: schemaDir,
      }));
      const stack = new core.Stack(app, 'TestStack');
      new core.CfnResource(stack, 'Handle', {
        type: 'AWS::CloudFormation::WaitConditionHandle',
        properties: {
          NestedProperty: 'found-in-subdir',
        },
      });

      expect(() => app.synth()).not.toThrow();
    });

    test('symlinks cause a hard failure', () => {
      // Create a real schema
      fs.writeFileSync(path.join(schemaDir, 'aws-cloudformation-waitconditionhandle.json'), JSON.stringify({
        typeName: 'AWS::CloudFormation::WaitConditionHandle',
        properties: {
          RealProperty: { type: 'string' },
        },
        additionalProperties: false,
      }));

      // Create a symlink cycle (dir pointing to parent)
      try {
        fs.symlinkSync(schemaDir, path.join(schemaDir, 'cycle-link'));
      } catch {
        // Skip test if symlinks not supported (Windows)
        return;
      }

      // Should throw — symlinks are not allowed in schema directories
      expect(() => new core.CloudFormationValidatePlugin({
        _additionalSchemasDirectory: schemaDir,
      })).toThrow(/Symbolic link found in schema directory/);
    });

    test('malformed JSON files cause a hard failure', () => {
      fs.writeFileSync(path.join(schemaDir, 'bad.json'), 'not valid json {{{');

      expect(() => new core.CloudFormationValidatePlugin({
        _additionalSchemasDirectory: schemaDir,
      })).toThrow(/Invalid JSON in schema file.*bad\.json/);
    });

    test('schema files missing typeName cause a hard failure', () => {
      fs.writeFileSync(path.join(schemaDir, 'no-typename.json'), JSON.stringify({ properties: {} }));

      expect(() => new core.CloudFormationValidatePlugin({
        _additionalSchemasDirectory: schemaDir,
      })).toThrow(/missing required "typeName" field.*no-typename\.json/);
    });

    test('non-JSON files are safely ignored', () => {
      fs.writeFileSync(path.join(schemaDir, '.keep'), '');
      fs.writeFileSync(path.join(schemaDir, 'README.md'), '# Schemas go here');
      fs.writeFileSync(path.join(schemaDir, 'aws-cloudformation-waitconditionhandle.json'), JSON.stringify({
        typeName: 'AWS::CloudFormation::WaitConditionHandle',
        properties: { MyProp: { type: 'string' } },
        additionalProperties: false,
      }));

      const app = new core.App({
        context: {
          [cxapi.VALIDATE_AGAINST_DEFAULT_RULES]: true,
          [cxapi.FAIL_SYNTH_ON_VALIDATION_ERRORS_CONTEXT]: true,
        },
      });
      core.Validations.of(app).addPlugins(new core.CloudFormationValidatePlugin({
        _additionalSchemasDirectory: schemaDir,
      }));
      const stack = new core.Stack(app, 'TestStack');
      new core.CfnResource(stack, 'Handle', {
        type: 'AWS::CloudFormation::WaitConditionHandle',
        properties: { MyProp: 'works' },
      });

      // Non-JSON files (.keep, .md) are ignored, only .json files are processed
      expect(() => app.synth()).not.toThrow();
    });
  });

  describe('_configureSingleton', () => {
    test('does not conflict with tests that register their own plugin', () => {
      // Simulate the jest hook: pre-configure the singleton with overlays
      const singletonSchemaDir = fs.mkdtempSync(path.join(os.tmpdir(), 'singleton-schemas-'));
      fs.writeFileSync(path.join(singletonSchemaDir, 'aws-cloudformation-waitconditionhandle.json'), JSON.stringify({
        typeName: 'AWS::CloudFormation::WaitConditionHandle',
        properties: { OverlayProp: { type: 'string' } },
        additionalProperties: false,
      }));
      core.CloudFormationValidatePlugin._configureSingleton({
        _additionalSchemasDirectory: singletonSchemaDir,
      });

      try {
        // Now a test registers its own plugin (like the validation plugin's own tests do)
        const app = new core.App({
          context: {
            [cxapi.VALIDATE_AGAINST_DEFAULT_RULES]: true,
            [cxapi.FAIL_SYNTH_ON_VALIDATION_ERRORS_CONTEXT]: true,
          },
        });
        core.Validations.of(app).addPlugins(new core.CloudFormationValidatePlugin());
        const stack = new core.Stack(app, 'TestStack');
        new core.CfnResource(stack, 'MyBucket', {
          type: 'AWS::S3::Bucket',
        });

        // Should NOT throw DuplicateCloudFormationValidatePlugin —
        // the user-registered plugin replaces the singleton, no double-registration.
        expect(() => app.synth()).not.toThrow();
      } finally {
        // Reset singleton for other tests
        core.CloudFormationValidatePlugin._configureSingleton({});
        fs.rmSync(singletonSchemaDir, { recursive: true, force: true });
      }
    });
  });
});

describe('CDK_VALIDATION environment variable', () => {
  const originalCdkValidation = process.env.CDK_VALIDATION;

  afterEach(() => {
    if (originalCdkValidation === undefined) {
      delete process.env.CDK_VALIDATION;
    } else {
      process.env.CDK_VALIDATION = originalCdkValidation;
    }
  });

  function appWithInvalidResource() {
    const app = new core.App({
      context: {
        [cxapi.VALIDATE_AGAINST_DEFAULT_RULES]: true,
        [cxapi.FAIL_SYNTH_ON_VALIDATION_ERRORS_CONTEXT]: true,
      },
    });
    const stack = new core.Stack(app, 'TestStack');
    new core.CfnResource(stack, 'MyBucket', {
      type: 'AWS::S3::Bucket',
      properties: {
        BogusProperty: 'invalid-value',
      },
    });
    return app;
  }

  test('CDK_VALIDATION=false disables the auto-registered validation engine', () => {
    process.env.CDK_VALIDATION = 'false';
    const app = appWithInvalidResource();

    // Would throw on BogusProperty if the default engine ran
    expect(() => app.synth()).not.toThrow();
    expect(process.exitCode).toBeUndefined();
  });

  test('CDK_VALIDATION=false does not disable explicitly registered plugins', () => {
    process.env.CDK_VALIDATION = 'false';
    const app = appWithInvalidResource();
    core.Validations.of(app).addPlugins(new core.CloudFormationValidatePlugin());

    expect(() => app.synth()).toThrow(/BogusProperty/);
  });

  test('CDK_VALIDATION=true keeps the default validation engine enabled', () => {
    process.env.CDK_VALIDATION = 'true';
    const app = appWithInvalidResource();

    expect(() => app.synth()).toThrow(/BogusProperty/);
  });

  test('unset CDK_VALIDATION keeps the default validation engine enabled', () => {
    delete process.env.CDK_VALIDATION;
    const app = appWithInvalidResource();

    expect(() => app.synth()).toThrow(/BogusProperty/);
  });
});

const VALIDATE_DETAILED_METRIC = 'CloudFormationValidate.validate';
const DIAGNOSTICS_METRIC = 'CloudFormationValidate.diagnostics';

function loadValidationReport(asm: cxapi.CloudAssembly) {
  const p = path.join(asm.directory, 'validation-report.json');
  return JSON.parse(fs.readFileSync(p, { encoding: 'utf-8' })) as PolicyValidationReportJson;
}

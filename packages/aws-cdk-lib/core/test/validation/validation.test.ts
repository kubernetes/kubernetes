import * as fs from 'fs';
import * as path from 'path';
import type { PolicyValidationReportJson } from '@aws-cdk/cloud-assembly-schema';
import { Construct } from 'constructs';
import * as cxapi from '../../../cx-api';
import * as core from '../../lib';
import type { App } from '../../lib';

const ANNOTATION_CAPTION = 'Annotation';

let consoleErrorMock: jest.SpyInstance;
beforeEach(() => {
  // These tests were written against the "subprocess" behavior of validation
  process.env.CDK_APP_MODE = 'process';
  process.env.NO_COLOR = '1';
  OUTPUT_REDACTIONS.clear();
  consoleErrorMock = jest.spyOn(console, 'error').mockImplementation(() => { return true; });
  jest.spyOn(console, 'log').mockImplementation(() => { return true; });
  process.exitCode = undefined;
});

afterEach(() => {
  jest.clearAllMocks();
});

describe('validations', () => {
  test('validation failure', () => {
    const app = new NonStrictApp({
      policyValidationBeta1: [
        new FakePlugin('test-plugin', [{
          description: 'test recommendation',
          ruleName: 'test-rule',
          severity: 'medium',
          ruleMetadata: {
            id: 'abcdefg',
          },
          violatingResources: [{
            locations: ['test-location'],
            resourceLogicalId: 'Fake',
            templatePath: 'Default.template.json',
          }],
        }]),
      ],
    });
    const stack = new core.Stack(app);
    new FailResource(stack, 'Fake');

    redactAsmDir(app.synth());
    expect(process.exitCode).toEqual(1);

    expect(mockErrorOutput()).toMatchSnapshot();
    // It has all the relevant details
    expect(mockErrorOutput()).toContain('Default/Fake');
    expect(mockErrorOutput()).toMatch(/(aws-cdk-lib.Stack|constructs.Construct)/);
  });

  test('validation success', () => {
    const app = new NonStrictApp({
      policyValidationBeta1: [
        new FakePlugin('test-plugin', []),
        new FakePlugin('test-plugin2', []),
        new FakePlugin('test-plugin3', []),
      ],
    });
    const stack = new core.Stack(app);
    new core.CfnResource(stack, 'DefaultResource', {
      type: 'Test::Resource::Fake',
      properties: {
        result: 'success',
      },
    });

    app.synth();
    expect(process.exitCode).toBeUndefined();
    // No errors from user-registered plugins; CloudFormation Validate may emit warnings
    expect(mockErrorOutput()).not.toContain('ERROR');
  });

  test('multiple stacks', () => {
    const app = new NonStrictApp({
      policyValidationBeta1: [
        new FakePlugin('test-plugin', [{
          description: 'test recommendation',
          ruleName: 'test-rule',
          violatingResources: [
            {
              locations: ['test-location'],
              resourceLogicalId: 'DefaultResource',
              templatePath: 'stack1.template.json',
            },
            {
              locations: ['test-location'],
              resourceLogicalId: 'DefaultResource',
              templatePath: 'stack2.template.json',
            },
          ],
        }]),
      ],
    });
    const stack1 = new core.Stack(app, 'stack1');
    new FailResource(stack1, 'DefaultResource');
    const stack2 = new core.Stack(app, 'stack2');
    new FailResource(stack2, 'DefaultResource');

    redactAsmDir(app.synth());
    expect(process.exitCode).toEqual(1);

    expect(mockErrorOutput()).toMatchSnapshot();
    expect(mockErrorOutput()).toContain('stack1/DefaultResource');
    expect(mockErrorOutput()).toContain('stack2/DefaultResource');
  });

  test('multiple stages', () => {
    const app = new NonStrictApp({
      policyValidationBeta1: [
        new FakePlugin('test-plugin1', [{
          description: 'do something',
          ruleName: 'test-rule1',
          violatingResources: [{
            locations: ['test-location'],
            resourceLogicalId: 'DefaultResource',
            templatePath: 'Stage1stack1DDED8B6C.template.json',
          }],
        }]),
      ],
    });
    const stage1 = new core.Stage(app, 'Stage1', {
      policyValidationBeta1: [
        new FakePlugin('test-plugin2', [{
          description: 'do something',
          ruleName: 'test-rule2',
          violatingResources: [{
            locations: ['test-location'],
            resourceLogicalId: 'DefaultResource',
            templatePath: 'Stage1stack1DDED8B6C.template.json',
          }],
        }], '1.2.3'),
      ],
    });
    const stage2 = new core.Stage(app, 'Stage2', {
      policyValidationBeta1: [
        new FakePlugin('test-plugin3', [{
          description: 'do something',
          ruleName: 'test-rule3',
          violatingResources: [{
            locations: ['test-location'],
            resourceLogicalId: 'DefaultResource',
            templatePath: 'Stage2stack259BA718E.template.json',
          }],
        }]),
      ],
    });
    const stage3 = new core.Stage(stage2, 'Stage3', {
      policyValidationBeta1: [
        new FakePlugin('test-plugin4', [{
          description: 'do something',
          ruleName: 'test-rule4',
          violatingResources: [{
            locations: ['test-location'],
            resourceLogicalId: 'DefaultResource',
            templatePath: 'Stage2Stage3stack3A378CA7D.template.json',
          }],
        }]),
      ],
    });
    const stack3 = new core.Stack(stage3, 'stack3');
    new FailResource(stack3, 'DefaultResource');
    const stack1 = new core.Stack(stage1, 'stack1');
    new FailResource(stack1, 'DefaultResource');
    const stack2 = new core.Stack(stage2, 'stack2');
    new FailResource(stack2, 'DefaultResource');

    redactAsmDir(app.synth());
    expect(process.exitCode).toEqual(1);

    expect(mockErrorOutput()).toMatchSnapshot();

    for (const cPath of ['Stage1/stack1/DefaultResource', 'Stage2/Stage3/stack3/DefaultResource', 'Stage2/stack2/DefaultResource', 'Stage1/stack1/DefaultResource']) {
      expect(mockErrorOutput()).toContain(cPath);
    }
  });

  test('multiple stages, multiple plugins', () => {
    const mockValidate = jest.fn().mockImplementation(() => {
      return {
        success: true,
        violations: [],
      };
    });
    const app = new NonStrictApp({
      policyValidationBeta1: [
        {
          name: 'test-plugin',
          validate: mockValidate,
        },
      ],
    });
    const stage1 = new core.Stage(app, 'Stage1', { });
    const stage2 = new core.Stage(app, 'Stage2', {
      policyValidationBeta1: [
        {
          name: 'test-plugin2',
          validate: mockValidate,
        },
      ],
    });
    const stage3 = new core.Stage(stage2, 'Stage3', { });
    const stack3 = new core.Stack(stage3, 'stack1');
    new FailResource(stack3, 'DefaultResource');
    const stack1 = new core.Stack(stage1, 'stack1');
    new FailResource(stack1, 'DefaultResource');
    const stack2 = new core.Stack(stage2, 'stack2');
    new FailResource(stack2, 'DefaultResource');

    app.synth();

    expect(mockValidate).toHaveBeenCalledTimes(2);
    expect(mockValidate).toHaveBeenNthCalledWith(2, expect.objectContaining({
      templatePaths: [
        expect.stringMatching(/assembly-Stage1\/Stage1stack1DDED8B6C.template.json/),
        expect.stringMatching(/assembly-Stage2\/Stage2stack259BA718E.template.json/),
        expect.stringMatching(/assembly-Stage2\/assembly-Stage2-Stage3\/Stage2Stage3stack10CD36915.template.json/),
      ],
    }));
    expect(mockValidate).toHaveBeenNthCalledWith(1, expect.objectContaining({
      templatePaths: [
        expect.stringMatching(/assembly-Stage2\/Stage2stack259BA718E.template.json/),
        expect.stringMatching(/assembly-Stage2\/assembly-Stage2-Stage3\/Stage2Stage3stack10CD36915.template.json/),
      ],
    }));
  });

  test('multiple stages, single plugin', () => {
    const mockValidate = jest.fn().mockImplementation(() => {
      return {
        success: true,
        violations: [],
      };
    });
    const app = new NonStrictApp({
      policyValidationBeta1: [
        {
          name: 'test-plugin',
          validate: mockValidate,
        },
      ],
    });
    const stage1 = new core.Stage(app, 'Stage1', { });
    const stage2 = new core.Stage(app, 'Stage2', { });
    const stage3 = new core.Stage(stage2, 'Stage3', { });
    const stack3 = new core.Stack(stage3, 'stack1');
    new FailResource(stack3, 'DefaultResource');
    const stack1 = new core.Stack(stage1, 'stack1');
    new FailResource(stack1, 'DefaultResource');
    const stack2 = new core.Stack(stage2, 'stack2');
    new FailResource(stack2, 'DefaultResource');
    app.synth();

    expect(mockValidate).toHaveBeenCalledTimes(1);
    expect(mockValidate).toHaveBeenCalledWith(expect.objectContaining({
      templatePaths: [
        expect.stringMatching(/assembly-Stage1\/Stage1stack1DDED8B6C.template.json/),
        expect.stringMatching(/assembly-Stage2\/Stage2stack259BA718E.template.json/),
        expect.stringMatching(/assembly-Stage2\/assembly-Stage2-Stage3\/Stage2Stage3stack10CD36915.template.json/),
      ],
    }));
  });

  test('multiple constructs', () => {
    const app = new NonStrictApp({
      policyValidationBeta1: [
        new FakePlugin('test-plugin', [{
          description: 'test recommendation',
          ruleName: 'test-rule',
          violatingResources: [{
            locations: ['test-location'],
            resourceLogicalId: 'SomeResource317FDD71',
            templatePath: 'Default.template.json',
          }],
        }]),
      ],
    });
    const stack = new core.Stack(app);
    new LevelTwoConstruct(stack, 'SomeResource');
    new LevelTwoConstruct(stack, 'AnotherResource');
    redactAsmDir(app.synth());
    expect(process.exitCode).toEqual(1);

    const report = mockErrorOutput();
    // The user-registered plugin only reports SomeResource
    expect(report).toContain('test recommendation');
    expect(report).toContain('Default/SomeResource');
    // Verify via the JSON report that the user plugin only flagged SomeResource
    const file = path.join(app.outdir, 'validation-report.json');
    const jsonReport = JSON.parse(fs.readFileSync(file, 'utf-8'));
    const testPluginReport = jsonReport.pluginReports.find((r: any) => r.pluginName === 'test-plugin');
    const paths = testPluginReport.violations.flatMap((v: any) =>
      v.violatingConstructs.map((c: any) => c.constructPath),
    );
    expect(paths).toContain('Default/SomeResource/Resource');
    expect(paths).not.toContain('Default/AnotherResource/Resource');
  });

  test('multiple plugins', () => {
    const app = new NonStrictApp({
      policyValidationBeta1: [
        new FakePlugin('plugin1', [{
          description: 'do something',
          ruleName: 'rule-1',
          violatingResources: [{
            locations: ['test-location'],
            resourceLogicalId: 'Fake',
            templatePath: 'Default.template.json',
          }],
        }]),
        new FakePlugin('plugin2', [{
          description: 'do another thing',
          ruleName: 'rule-2',
          violatingResources: [{
            locations: ['test-location'],
            resourceLogicalId: 'Fake',
            templatePath: 'Default.template.json',
          }],
        }]),
      ],
    });
    const stack = new core.Stack(app);
    new FailResource(stack, 'Fake');
    redactAsmDir(app.synth());
    expect(process.exitCode).toEqual(1);

    const report = mockErrorOutput();
    expect(report).toMatchSnapshot();
    expect(report).toContain('do something');
    expect(report).toContain('do another thing');
  });

  test('multiple plugins with mixed results', () => {
    const app = new NonStrictApp({
      policyValidationBeta1: [
        new FakePlugin('plugin1', []),
        new FakePlugin('plugin2', [{
          description: 'do another thing',
          ruleName: 'rule-2',
          violatingResources: [{
            locations: ['test-location'],
            resourceLogicalId: 'Fake',
            templatePath: 'Default.template.json',
          }],
        }]),
      ],
    });
    const stack = new core.Stack(app);
    new FailResource(stack, 'Fake');
    redactAsmDir(app.synth());

    const report = mockErrorOutput();
    expect(report).toMatchSnapshot();
    expect(report).toContain('do another thing');

    expect(report).toContain('plugin2');
    expect(report).not.toContain('plugin1');
  });

  test('plugin throws an error', () => {
    const app = new NonStrictApp({
      policyValidationBeta1: [
        // This plugin will throw an error
        new BrokenPlugin(),

        // But this one should still run
        new FakePlugin('test-plugin', [{
          description: 'test recommendation',
          ruleName: 'test-rule',
          violatingResources: [{
            locations: ['test-location'],
            resourceLogicalId: 'Fake',
            templatePath: 'Default.template.json',
          }],
        }]),
      ],
    });

    const stack = new core.Stack(app);
    new FailResource(stack, 'Fake');

    redactAsmDir(app.synth());
    expect(process.exitCode).toEqual(1);

    const report = mockErrorOutput();
    expect(report).toContain('test recommendation');
    expect(report).toContain('ERROR Validation plugin \'broken-plugin\' failed: Something went wrong');
  });

  test('plugin tries to modify a template', () => {
    const app = new NonStrictApp({
      policyValidationBeta1: [
        new RoguePlugin(),
      ],
    });
    const stack = new core.Stack(app);
    new FailResource(stack, 'DefaultResource');
    expect(() => {
      app.synth();
    }).toThrow(/rogue-plugin.*modified the cloud assembly/);
  });

  test('plugin that writes new files to assembly is allowed', () => {
    const app = new NonStrictApp({
      policyValidationBeta1: [
        {
          name: 'file-writer-plugin',
          validate(context: core.IPolicyValidationContextBeta1) {
            const assemblyDir = path.dirname(context.templatePaths[0]);
            fs.writeFileSync(path.join(assemblyDir, 'plugin-output.json'), '{"result":"ok"}');
            return { success: true, violations: [] };
          },
        },
      ],
    });
    const stack = new core.Stack(app);
    new core.CfnResource(stack, 'DefaultResource', {
      type: 'Test::Resource::Fake',
      properties: { result: 'success' },
    });
    expect(() => app.synth()).not.toThrow();
    const outputFile = path.join(app.outdir, 'plugin-output.json');
    expect(fs.existsSync(outputFile)).toBe(true);
    expect(JSON.parse(fs.readFileSync(outputFile, 'utf-8'))).toEqual({ result: 'ok' });
  });

  test('plugin that deletes pre-existing file is caught', () => {
    const app = new NonStrictApp({
      policyValidationBeta1: [
        {
          name: 'deleter-plugin',
          validate(context: core.IPolicyValidationContextBeta1) {
            fs.unlinkSync(context.templatePaths[0]);
            return { success: true, violations: [] };
          },
        },
      ],
    });
    const stack = new core.Stack(app);
    new core.CfnResource(stack, 'DefaultResource', {
      type: 'Test::Resource::Fake',
      properties: { result: 'success' },
    });
    expect(() => app.synth()).toThrow(/deleter-plugin.*modified the cloud assembly/);
  });

  test('failSynthOnValidationErrors=false writes JSON but does not print or fail', () => {
    const app = new NonStrictApp({
      policyValidationBeta1: [
        new FakePlugin('test-plugin', [{
          description: 'test recommendation',
          ruleName: 'test-rule',
          ruleMetadata: {
            id: 'abcdefg',
          },
          violatingResources: [{
            locations: ['test-location'],
            resourceLogicalId: 'Fake',
            templatePath: 'Default.template.json',
          }],
        }]),
      ],
      context: {
        '@aws-cdk/core:failSynthOnValidationErrors': false,
      },
    });
    const stack = new core.Stack(app);
    new FailResource(stack, 'Fake');
    redactAsmDir(app.synth());

    // No exit code set
    expect(process.exitCode).toBeUndefined();

    // JSON file is written in the new v2 format
    const file = path.join(app.outdir, 'validation-report.json');
    const report = JSON.parse(fs.readFileSync(file).toString('utf-8'));
    expect(report.version).toEqual(expect.any(String));
    expect(report.title).toEqual('Validation Report');
    const testPluginReport = report.pluginReports.find((r: any) => r.pluginName === 'test-plugin');
    expect(testPluginReport).toEqual({
      pluginName: 'test-plugin',
      conclusion: 'failure',
      violations: [
        {
          ruleName: 'test-rule',
          description: 'test recommendation',
          severity: 'error',
          ruleMetadata: { id: 'abcdefg' },
          violatingConstructs: [
            {
              constructPath: 'Default/Fake',
              constructFqn: expect.stringMatching(/(aws-cdk-lib.CfnResource|Construct)/),
              libraryVersion: expect.any(String),
              cloudFormationResource: {
                templatePath: 'Default.template.json',
                logicalId: 'Fake',
                propertyPaths: ['test-location'],
              },
              stackTraces: expect.any(Array),
            },
          ],
        },
      ],
    });

    // No console output about validation (only default plugin warnings may appear)
    const allOutput = mockErrorOutput();
    expect(allOutput).not.toContain('ERROR');
  });

  test('failSynthOnValidationErrors="false" (string) works the same as boolean false', () => {
    const app = new NonStrictApp({
      policyValidationBeta1: [
        new FakePlugin('test-plugin', [{
          description: 'test recommendation',
          ruleName: 'test-rule',
          ruleMetadata: {
            id: 'abcdefg',
          },
          violatingResources: [{
            locations: ['test-location'],
            resourceLogicalId: 'Fake',
            templatePath: 'Default.template.json',
          }],
        }]),
      ],
      context: {
        '@aws-cdk/core:failSynthOnValidationErrors': 'false',
      },
    });
    const stack = new core.Stack(app);
    new core.CfnResource(stack, 'Fake', {
      type: 'Test::Resource::Fake',
      properties: {
        result: 'failure',
      },
    });
    redactAsmDir(app.synth());

    expect(process.exitCode).toBeUndefined();

    const file = path.join(app.outdir, 'validation-report.json');
    expect(fs.existsSync(file)).toBe(true);

    const allOutput = mockErrorOutput();
    expect(allOutput).not.toContain('ERROR');
  });

  test('Pretty print as default', () => {
    const app = new NonStrictApp({
      policyValidationBeta1: [
        new FakePlugin('test-plugin', [{
          description: 'test recommendation',
          ruleName: 'test-rule',
          ruleMetadata: {
            id: 'abcdefg',
          },
          violatingResources: [{
            locations: ['test-location'],
            resourceLogicalId: 'Fake',
            templatePath: 'Default.template.json',
          }],
        }]),
      ],
      context: {
      },
    });
    const stack = new core.Stack(app);
    new FailResource(stack, 'Fake');
    redactAsmDir(app.synth());
    expect(process.exitCode).toEqual(1);
    const consoleOut = mockErrorOutput();
    expect(consoleOut).toContain('A copy of this report can be found');
  });

  test('both formats enabled by default', () => {
    const app = new NonStrictApp({
      policyValidationBeta1: [
        new FakePlugin('test-plugin', [{
          description: 'test recommendation',
          ruleName: 'test-rule',
          violatingResources: [{
            locations: ['test-location'],
            resourceLogicalId: 'Fake',
            templatePath: 'Default.template.json',
          }],
        }]),
      ],
    });
    const stack = new core.Stack(app);
    new FailResource(stack, 'Fake');
    redactAsmDir(app.synth());
    expect(process.exitCode).toEqual(1);

    // JSON file written
    const file = path.join(app.outdir, 'validation-report.json');
    expect(fs.existsSync(file)).toBe(true);

    // Pretty print also output
    const consoleReport = mockErrorOutput();
    expect(consoleReport).toContain('test recommendation');
  });

  test('failSynthOnValidationErrors=false succeeds even with validation failures', () => {
    const app = new NonStrictApp({
      policyValidationBeta1: [
        new FakePlugin('test-plugin', [{
          description: 'test recommendation',
          ruleName: 'test-rule',
          violatingResources: [{
            locations: ['test-location'],
            resourceLogicalId: 'Fake',
            templatePath: 'Default.template.json',
          }],
        }]),
      ],
      context: {
        '@aws-cdk/core:failSynthOnValidationErrors': false,
      },
    });
    const stack = new core.Stack(app);
    new FailResource(stack, 'Fake');
    redactAsmDir(app.synth());

    // No failure exit code
    expect(process.exitCode).toBeUndefined();

    // JSON file is still written
    const file = path.join(app.outdir, 'validation-report.json');
    expect(fs.existsSync(file)).toBe(true);
    const report = JSON.parse(fs.readFileSync(file, 'utf-8'));
    expect(report.pluginReports[0].conclusion).toEqual('failure');
  });

  test('validationReportJson=true writes legacy report alongside new report', () => {
    const app = new NonStrictApp({
      policyValidationBeta1: [
        new FakePlugin('test-plugin', [{
          description: 'test recommendation',
          ruleName: 'test-rule',
          violatingResources: [{
            locations: ['test-location'],
            resourceLogicalId: 'Fake',
            templatePath: 'Default.template.json',
          }],
        }]),
      ],
      context: {
        '@aws-cdk/core:validationReportJson': true,
      },
    });
    const stack = new core.Stack(app);
    new FailResource(stack, 'Fake');
    redactAsmDir(app.synth());

    // New format is always written
    const newFile = path.join(app.outdir, 'validation-report.json');
    expect(fs.existsSync(newFile)).toBe(true);
    const newReport = JSON.parse(fs.readFileSync(newFile, 'utf-8'));
    expect(newReport.version).toBeDefined();
    expect(newReport.pluginReports[0].conclusion).toEqual('failure');

    // Legacy format is also written when opted in
    const legacyFile = path.join(app.outdir, 'policy-validation-report.json');
    expect(fs.existsSync(legacyFile)).toBe(true);
    const legacyReport = JSON.parse(fs.readFileSync(legacyFile, 'utf-8'));
    expect(legacyReport.pluginReports[0].summary.status).toEqual('failure');
    expect(legacyReport.pluginReports[0].summary.pluginName).toEqual('test-plugin');
  });

  test('legacy report is NOT written by default', () => {
    const app = new NonStrictApp({
      policyValidationBeta1: [
        new FakePlugin('test-plugin', [{
          description: 'test recommendation',
          ruleName: 'test-rule',
          violatingResources: [{
            locations: ['test-location'],
            resourceLogicalId: 'Fake',
            templatePath: 'Default.template.json',
          }],
        }]),
      ],
    });
    const stack = new core.Stack(app);
    new FailResource(stack, 'Fake');
    redactAsmDir(app.synth());

    // New format written
    expect(fs.existsSync(path.join(app.outdir, 'validation-report.json'))).toBe(true);

    // Legacy format NOT written
    expect(fs.existsSync(path.join(app.outdir, 'policy-validation-report.json'))).toBe(false);
  });

  test('Multi format', () => {
    const app = new NonStrictApp({
      policyValidationBeta1: [
        new FakePlugin('test-plugin', [{
          description: 'test recommendation',
          ruleName: 'test-rule',
          ruleMetadata: {
            id: 'abcdefg',
          },
          violatingResources: [{
            locations: ['test-location'],
            resourceLogicalId: 'Fake',
            templatePath: 'Default.template.json',
          }],
        }]),
      ],
    });
    const stack = new core.Stack(app);
    new FailResource(stack, 'Fake');
    redactAsmDir(app.synth());
    expect(process.exitCode).toEqual(1);
    const file = path.join(app.outdir, 'validation-report.json');
    const report = JSON.parse(fs.readFileSync(file).toString('utf-8'));
    expect(report.version).toEqual(expect.any(String));
    expect(report.title).toEqual('Validation Report');
    const testPluginReport = report.pluginReports.find((r: any) => r.pluginName === 'test-plugin');
    expect(testPluginReport).toEqual({
      pluginName: 'test-plugin',
      conclusion: 'failure',
      violations: [
        {
          ruleName: 'test-rule',
          description: 'test recommendation',
          severity: 'error',
          ruleMetadata: { id: 'abcdefg' },
          violatingConstructs: [
            {
              constructPath: 'Default/Fake',
              constructFqn: expect.stringMatching(/(aws-cdk-lib.CfnResource|Construct)/),
              libraryVersion: expect.any(String),
              cloudFormationResource: {
                templatePath: 'Default.template.json',
                logicalId: 'Fake',
                propertyPaths: ['test-location'],
              },
              stackTraces: expect.any(Array),
            },
          ],
        },
      ],
    });
    const consoleOut = mockErrorOutput();
    expect(consoleOut).toContain('A copy of this report can be found');
  });

  test('a plugin implementing Beta1 is assignable to IPolicyValidationPlugin', () => {
    // GIVEN
    const beta1Plugin: core.IPolicyValidationPluginBeta1 = new FakePlugin('beta1-plugin', []);

    // WHEN
    const plugin: core.IPolicyValidationPlugin = beta1Plugin;

    // THEN
    expect(plugin.name).toEqual('beta1-plugin');
  });

  describe('annotation report integration', () => {
    const annotationReportContext = { [cxapi.ANNOTATIONS_IN_VALIDATION_REPORT]: true };

    test('annotation warnings appear in validation report', () => {
      const app = new NonStrictApp({ context: annotationReportContext });
      const stack = new core.Stack(app, 'MyStack');
      const construct = new Construct(stack, 'MyConstruct');
      new FailResource(construct, 'Resource');

      core.Annotations.of(construct).addWarningV2('my-lib:SomeWarning', 'This is a warning');

      redactAsmDir(app.synth());

      // Warnings alone should not fail
      expect(process.exitCode).toBeUndefined();

      // Should show the annotation report
      const output = mockErrorOutput();
      expect(output).toContain(ANNOTATION_CAPTION);
      expect(output).toContain('my-lib:SomeWarning');
    });

    test('annotation warnings have the right template path, even in nested assemblies', () => {
      // GIVEN
      const app = new NonStrictApp({ context: annotationReportContext });
      const stack = new core.Stack(app, 'MyStack');
      const r1 = new FailResource(stack, 'Resource');
      core.Annotations.of(r1).addWarningV2('my-lib:SomeWarning', 'This is a warning');

      const stage = new core.Stage(app, 'MyStage');
      const stack2 = new core.Stack(stage, 'Stack2');
      const r2 = new FailResource(stack2, 'Resource2');
      core.Annotations.of(r2).addWarningV2('my-lib:SomeWarning', 'This is a warning');

      // WHEN
      const asm = redactAsmDir(app.synth());

      // THEN
      const validationReport: PolicyValidationReportJson = loadJson(path.join(asm.directory, 'validation-report.json'));
      const report = validationReport.pluginReports.find(r => r.pluginName === 'Construct Annotations');
      expect(report?.violations).toEqual([
        expect.objectContaining({
          ruleName: 'Annotation::my-lib:SomeWarning',
          violatingConstructs: [
            expect.objectContaining({
              cloudFormationResource: expect.objectContaining({
                logicalId: 'Resource',
                templatePath: 'MyStack.template.json',
              }),
            }),
            expect.objectContaining({
              cloudFormationResource: expect.objectContaining({
                logicalId: 'Resource2',
                templatePath: 'assembly-MyStage/MyStageStack2DAB805FC.template.json',
              }),
            }),
          ],
        }),
      ]);
    });

    test('annotation warnings in nested stage appear in validation report', () => {
      const app = new NonStrictApp({ context: annotationReportContext });
      const stage = new core.Stage(app, 'Stage');
      const stack = new core.Stack(stage, 'MyStack');
      const construct = new Construct(stack, 'MyConstruct');
      new FailResource(construct, 'Resource');

      core.Annotations.of(construct).addWarningV2('my-lib:SomeWarning', 'This is a warning');

      redactAsmDir(app.synth());

      // Warnings alone should not fail
      expect(process.exitCode).toBeUndefined();

      // Should show the annotation report
      const output = mockErrorOutput();
      expect(output).toContain(ANNOTATION_CAPTION);
      expect(output).toContain('my-lib:SomeWarning');
    });

    test('annotation errors cause validation failure', () => {
      const app = new NonStrictApp({ context: annotationReportContext });
      const stack = new core.Stack(app, 'MyStack');
      const construct = new Construct(stack, 'MyConstruct');
      new FailResource(construct, 'Resource');

      core.Annotations.of(construct).addError('Something is broken');

      redactAsmDir(app.synth());

      expect(process.exitCode).toEqual(1);

      const output = mockErrorOutput();
      expect(output).toContain(ANNOTATION_CAPTION);
      expect(output).toContain('Something is broken');
      expect(output).toMatch(/Error/i);
    });

    test('Annotations.addWarningV2 can be acknowleged via Annotations.acknowledgeWarning', () => {
      const app = new NonStrictApp({ context: annotationReportContext });
      const stack = new core.Stack(app, 'MyStack');
      const construct = new Construct(stack, 'MyConstruct');
      new FailResource(construct, 'Resource');

      core.Annotations.of(construct).addWarningV2('my-lib:AckedWarning', 'This warning is acknowledged');
      core.Annotations.of(construct).acknowledgeWarning('my-lib:AckedWarning');

      redactAsmDir(app.synth());

      // No annotations left, so no report at all
      const output = mockErrorOutput();
      expect(output).not.toContain('AckedWarning');
    });

    // We make suppressible using both the old and new prefixes, to ensure that both are supported
    test.each([
      'Construct-Annotations::',
      'Annotation::',
      'annotation::',
    ])('Annotations.addWarningV2 can be acknowledged via Validations using: %p', (prefix) => {
      const app = new NonStrictApp({ context: annotationReportContext });
      const stack = new core.Stack(app, 'MyStack');
      const construct = new Construct(stack, 'MyConstruct');
      new FailResource(construct, 'Resource');

      core.Annotations.of(construct).addWarningV2('my-lib:AckedWarning', 'This warning is acknowledged');
      core.Validations.of(construct).acknowledge({
        id: `${prefix}my-lib:AckedWarning`,
        reason: 'Acceptable for testing',
      });

      redactAsmDir(app.synth());

      // No annotations left, so no report at all
      const output = mockErrorOutput();
      expect(output).not.toContain('AckedWarning');
    });

    test('partial acknowledgment only excludes acknowledged warnings', () => {
      const app = new NonStrictApp({ context: annotationReportContext });
      const stack = new core.Stack(app, 'MyStack');
      const construct = new Construct(stack, 'MyConstruct');
      new FailResource(construct, 'Resource');

      core.Validations.of(construct).addWarning('AckedRule', 'This one is acknowledged');
      core.Validations.of(construct).addWarning('KeptRule', 'This one is not');
      core.Validations.of(construct).acknowledge({ id: 'AckedRule', reason: 'Accepted risk' });

      redactAsmDir(app.synth());

      const output = mockErrorOutput();
      expect(output).toContain(ANNOTATION_CAPTION);
      expect(output).not.toContain('Annotation::AckedRule');
      expect(output).toContain('Annotation::KeptRule');
    });

    test('annotation report works alongside plugin reports', () => {
      const app = new NonStrictApp({
        context: annotationReportContext,
        policyValidationBeta1: [
          new FakePlugin('test-plugin', [{
            description: 'plugin violation',
            ruleName: 'plugin-rule',
            violatingResources: [{
              locations: ['test-location'],
              resourceLogicalId: 'Fake',
              templatePath: 'Default.template.json',
            }],
          }]),
        ],
      });
      const stack = new core.Stack(app);
      new FailResource(stack, 'Fake');

      core.Annotations.of(stack).addWarningV2('my-lib:StackWarning', 'Stack-level warning');

      redactAsmDir(app.synth());

      expect(process.exitCode).toEqual(1);

      const output = mockErrorOutput();
      // Both plugin and annotation reports should appear
      expect(output).toContain('test-plugin');
      expect(output).toContain(ANNOTATION_CAPTION);
      expect(output).toContain('plugin-rule');
      expect(output).toContain('my-lib:StackWarning');
    });

    test('annotation report with no plugins registered still produces output', () => {
      const app = new NonStrictApp({ context: annotationReportContext });
      const stack = new core.Stack(app, 'MyStack');
      const construct = new Construct(stack, 'MyConstruct');
      new FailResource(construct, 'Resource');

      core.Annotations.of(construct).addError('Error without plugins');

      redactAsmDir(app.synth());

      expect(process.exitCode).toEqual(1);

      const output = mockErrorOutput();
      expect(output).toContain(ANNOTATION_CAPTION);
      expect(output).toContain('Error without plugins');
    });

    test('annotations on constructs without CfnResource use construct path', () => {
      const app = new NonStrictApp({ context: annotationReportContext });
      const stack = new core.Stack(app, 'MyStack');
      const construct = new Construct(stack, 'Orphan');
      // No CfnResource child

      core.Annotations.of(construct).addWarningV2('my-lib:OrphanWarning', 'Warning on orphan');

      redactAsmDir(app.synth());

      const output = consoleErrorMock.mock.calls.map((c: any[]) => c[0]).join('\n');
      expect(output).toContain(ANNOTATION_CAPTION);
      expect(output).toContain('my-lib:OrphanWarning');
      // Construct path is provided directly
      expect(output).toContain('MyStack/Orphan');
    });

    test('Validations.of().addWarning appears in annotation report', () => {
      const app = new NonStrictApp({ context: annotationReportContext });
      const stack = new core.Stack(app, 'MyStack');
      const construct = new Construct(stack, 'MyConstruct');
      new FailResource(construct, 'Resource');

      core.Validations.of(construct).addWarning('MyRule', 'Validation warning');

      redactAsmDir(app.synth());

      const output = mockErrorOutput();
      expect(output).toContain(ANNOTATION_CAPTION);
      expect(output).toContain('Annotation::MyRule');
    });

    test('Validations.of().addError appears in annotation report and fails', () => {
      const app = new NonStrictApp({ context: annotationReportContext });
      const stack = new core.Stack(app, 'MyStack');
      const construct = new Construct(stack, 'MyConstruct');
      new FailResource(construct, 'Resource');

      core.Validations.of(construct).addError('MyError', 'Validation error');

      redactAsmDir(app.synth());

      expect(process.exitCode).toEqual(1);

      const output = mockErrorOutput();
      expect(output).toContain(ANNOTATION_CAPTION);
      expect(output).toMatch(/error/i);

      // The error violation itself should show "Rule" not "Acknowledge with"
      const errorSection = output.split(`(${ANNOTATION_CAPTION})`)[0] + `(${ANNOTATION_CAPTION})`;
      expect(errorSection).not.toMatch(/acknowledge/i);
      expect(output).toContain('Rule Annotation::MyError');
    });

    test('extractRuleName regex matches addWarningV2 ack tag format', () => {
      // This test verifies the coupling between the [ack: <id>] tag format
      // produced by Annotations.addWarningV2 and the regex in extractRuleName.
      // If the tag format in annotations.ts changes, this test should fail.
      const app = new NonStrictApp({ context: annotationReportContext });
      const stack = new core.Stack(app, 'MyStack');
      const construct = new Construct(stack, 'MyConstruct');
      new FailResource(construct, 'Resource');

      core.Annotations.of(construct).addWarningV2('my-lib:TestId', 'Test message');

      // Verify the metadata contains the expected tag format
      const warning = construct.node.metadata.find(m => m.type === 'aws:cdk:warning');
      expect(warning?.data).toContain('[ack: my-lib:TestId]');

      // Verify the report extracts the ID correctly
      redactAsmDir(app.synth());
      const output = mockErrorOutput();
      expect(output).toContain('my-lib:TestId');
    });

    test('plugin violations can be suppressed via Validations.acknowledge()', () => {
      const app = new NonStrictApp({ context: annotationReportContext });
      const stack = new core.Stack(app);
      new core.CfnResource(stack, 'MyBucket', {
        type: 'AWS::S3::Bucket',
        properties: {},
      });

      core.Validations.of(app).addPlugins(
        new FakePlugin('test-plugin', [{
          description: 'S3 Bucket should have versioning enabled',
          ruleName: 'S3_BUCKET_VERSIONING_ENABLED',
          severity: 'error',
          violatingResources: [{
            locations: ['Properties/VersioningConfiguration'],
            resourceLogicalId: 'MyBucket',
            templatePath: 'Default.template.json',
          }],
        }]),
      );

      // Suppress the error-level violation using <pluginName>::<ruleId>
      core.Validations.of(stack).acknowledge({ id: 'test-plugin::S3_BUCKET_VERSIONING_ENABLED', reason: 'Not needed for this bucket' });

      redactAsmDir(app.synth());

      const output = mockErrorOutput();
      expect(output).not.toContain('S3_BUCKET_VERSIONING_ENABLED');
    });

    test('suppressed violations appear in validation-report.json', () => {
      const app = new NonStrictApp({
        context: {
          ...annotationReportContext,
          '@aws-cdk/core:failSynthOnValidationErrors': false,
        },
      });
      const stack = new core.Stack(app);
      new core.CfnResource(stack, 'MyBucket', {
        type: 'AWS::S3::Bucket',
        properties: {},
      });

      core.Validations.of(app).addPlugins(
        new FakePlugin('test-plugin', [{
          description: 'S3 Bucket should have versioning enabled',
          ruleName: 'S3_BUCKET_VERSIONING_ENABLED',
          severity: 'error',
          violatingResources: [{
            locations: ['Properties/VersioningConfiguration'],
            resourceLogicalId: 'MyBucket',
            templatePath: 'Default.template.json',
          }],
        }]),
      );

      core.Validations.of(stack).acknowledge({ id: 'test-plugin::S3_BUCKET_VERSIONING_ENABLED', reason: 'Not needed for this bucket' });

      redactAsmDir(app.synth());

      const file = path.join(app.outdir, 'validation-report.json');
      const report = JSON.parse(fs.readFileSync(file, 'utf-8'));
      expect(report.pluginReports).toHaveLength(1);
      expect(report.pluginReports[0].violations).toHaveLength(0);
      expect(report.pluginReports[0].suppressedViolations).toHaveLength(1);
      const sv = report.pluginReports[0].suppressedViolations[0];
      expect(sv).toEqual(expect.objectContaining({
        ruleName: 'S3_BUCKET_VERSIONING_ENABLED',
        description: 'S3 Bucket should have versioning enabled',
        acknowledgedId: 'test-plugin::S3_BUCKET_VERSIONING_ENABLED',
        reason: 'Not needed for this bucket',
        acknowledgedAt: 'Default',
      }));
      expect(sv.violatingConstructs[0].stackTraces).toBeDefined();
      expect(sv.acknowledgedStackTrace).toBeDefined();
      expect(sv.acknowledgedStackTrace).toContain('validation.test.ts');
    });

    test('fatal plugin violations cannot be suppressed', () => {
      const app = new NonStrictApp({ context: annotationReportContext });
      const stack = new core.Stack(app);
      new core.CfnResource(stack, 'Fake', {
        type: 'AWS::S3::Bucket',
        properties: {},
      });

      core.Validations.of(app).addPlugins(
        new FakePlugin('test-plugin', [{
          description: 'Unknown resource type',
          ruleName: 'E9001',
          severity: 'fatal',
          violatingResources: [{
            locations: [],
            resourceLogicalId: 'BadResource',
            templatePath: 'Default.template.json',
          }],
        }]),
      );

      // Attempt to suppress the fatal violation
      core.Validations.of(stack).acknowledge({ id: 'test-plugin::E9001', reason: 'Trying to suppress fatal' });

      redactAsmDir(app.synth());

      const output = mockErrorOutput();
      // Fatal violations remain despite acknowledgment
      expect(output).toContain('Rule test-plugin::E9001');
      expect(output).toContain('Unknown resource type');
    });

    test('plugin names with spaces use dashes in suppression IDs', () => {
      const app = new NonStrictApp({ context: annotationReportContext });
      const stack = new core.Stack(app);
      new core.CfnResource(stack, 'Fake', {
        type: 'AWS::S3::Bucket',
        properties: {},
      });

      core.Validations.of(app).addPlugins(
        new FakePlugin('My Plugin', [{
          description: 'Some violation',
          ruleName: 'MY RULE',
          severity: 'error',
          violatingResources: [{
            locations: [],
            resourceLogicalId: 'Fake',
            templatePath: 'Default.template.json',
          }],
        }]),
      );

      // Suppress using dashes instead of spaces
      core.Validations.of(stack).acknowledge({ id: 'My-Plugin::MY-RULE', reason: 'OK' });

      redactAsmDir(app.synth());

      const output = mockErrorOutput();
      expect(output).not.toContain('MY RULE');
    });

    test('validation report JSON is always written to assembly directory', () => {
      const app = new NonStrictApp({
        context: annotationReportContext,
      });
      const stack = new core.Stack(app, 'MyStack');
      const construct = new Construct(stack, 'MyConstruct');
      new FailResource(construct, 'Resource');

      core.Annotations.of(construct).addWarningV2('my-lib:AlwaysWritten', 'Report always in assembly');

      const assembly = redactAsmDir(app.synth());

      const reportPath = path.join(assembly.directory, 'validation-report.json');
      expect(fs.existsSync(reportPath)).toBe(true);
      const report = JSON.parse(fs.readFileSync(reportPath, 'utf-8'));
      const annotationsReport = report.pluginReports.find((r: any) => r.pluginName === 'Construct Annotations');
      expect(annotationsReport).toBeDefined();
    });
  });

  describe('Validations.of()', () => {
    test('addPlugins adds plugin to enclosing stage', () => {
      // GIVEN
      const app = new NonStrictApp();
      const plugin = new FakePlugin('test-plugin', []);

      // WHEN
      core.Validations.of(app).addPlugins(plugin);

      // THEN
      expect(app.policyValidationBeta1.map(p => p.name)).toContain('test-plugin');
    });

    test('addPlugins from nested construct resolves to enclosing stage', () => {
      // GIVEN
      const app = new NonStrictApp();
      const stack = new core.Stack(app, 'MyStack');
      const plugin = new FakePlugin('test-plugin', []);

      // WHEN
      core.Validations.of(stack).addPlugins(plugin);

      // THEN - plugin is registered on the app (enclosing stage), not the stack
      expect(app.policyValidationBeta1.map(p => p.name)).toContain('test-plugin');
    });

    test('throws when addPlugins called without enclosing stage', () => {
      // GIVEN
      const construct = new Construct(undefined as any, '');

      // THEN
      expect(() => core.Validations.of(construct).addPlugins(new FakePlugin('test', []))).toThrow(/without an enclosing Stage/);
    });

    test('plugin added via addPlugins runs during synth', () => {
      // GIVEN
      const app = new NonStrictApp();
      const stack = new core.Stack(app);
      new FailResource(stack, 'Fake');

      // WHEN
      core.Validations.of(app).addPlugins(new FakePlugin('added-plugin', [{
        description: 'test recommendation',
        ruleName: 'test-rule',
        violatingResources: [{
          locations: ['test-location'],
          resourceLogicalId: 'Fake',
          templatePath: 'Default.template.json',
        }],
      }]));
      redactAsmDir(app.synth());

      // THEN - exitCode 1 means the plugin ran and reported violations
      expect(process.exitCode).toEqual(1);
    });

    test('addWarning adds warning metadata to construct', () => {
      // GIVEN
      const app = new NonStrictApp();
      const stack = new core.Stack(app, 'MyStack');
      const construct = new Construct(stack, 'MyConstruct');

      // WHEN
      core.Validations.of(construct).addWarning('my-lib:MyWarning', 'Something is off');

      // THEN
      const warnings = construct.node.metadata.filter(m => m.type === 'aws:cdk:warning');
      expect(warnings).toHaveLength(1);
      expect(warnings[0].data).toContain('Something is off');
      expect(warnings[0].data).toContain('[ack: Annotation::my-lib:MyWarning]');
    });

    test('addError adds error metadata with id to construct', () => {
      // GIVEN
      const app = new NonStrictApp();
      const stack = new core.Stack(app, 'MyStack');
      const construct = new Construct(stack, 'MyConstruct');

      // WHEN
      core.Validations.of(construct).addError('my-lib:MyError', 'Something is wrong');

      // THEN
      const errors = construct.node.metadata.filter(m => m.type === 'aws:cdk:error');
      expect(errors).toHaveLength(1);
      expect(errors[0].data).toBe('Something is wrong (Annotation::my-lib:MyError)');
    });

    test('acknowledge routes annotation rules to Annotations.acknowledgeWarning', () => {
      // GIVEN
      const app = new NonStrictApp();
      const stack = new core.Stack(app, 'MyStack');
      const construct = new Construct(stack, 'MyConstruct');
      core.Validations.of(construct).addWarning('SomeWarning', 'This is a warning');

      // WHEN - no prefix defaults to annotation rule
      core.Validations.of(construct).acknowledge({ id: 'SomeWarning', reason: 'Accepted risk' });

      // THEN - existing warning is removed
      const warningsAfterAck = construct.node.metadata.filter(m => m.type === 'aws:cdk:warning');
      expect(warningsAfterAck).toHaveLength(0);
    });

    test('relative templatePaths are absolutized correctly', () => {
      const app = new NonStrictApp({
        policyValidationBeta1: [
          new RelativePathPlugin('test-plugin', [{
            description: 'test recommendation',
            ruleName: 'test-rule',
            severity: 'medium',
            ruleMetadata: {
              id: 'abcdefg',
            },
            violatingResources: [{
              locations: ['test-location'],
              resourceLogicalId: 'Fake',
              templatePath: 'cdk.out/Bla.template.json',
            }],
          }]),
        ],
      });
      const stack = new core.Stack(app);
      new FailResource(stack, 'Fake');

      const assembly = app.synth();
      const report = loadJson(path.join(assembly.directory, 'validation-report.json'));
      expect(report).toEqual(expect.objectContaining({
        pluginReports: expect.arrayContaining([
          expect.objectContaining({
            pluginName: 'test-plugin',
            violations: expect.arrayContaining([
              expect.objectContaining({
                violatingConstructs: expect.arrayContaining([
                  expect.objectContaining({
                    cloudFormationResource: expect.objectContaining({
                      templatePath: 'cdk.out/Bla.template.json',
                    }),
                  }),
                ]),
              }),
            ]),
          }),
        ]),
      }));
    });

    test('acknowledge records to construct metadata', () => {
      // GIVEN
      const app = new NonStrictApp();
      const stack = new core.Stack(app, 'MyStack');
      const construct = new Construct(stack, 'MyConstruct');

      // WHEN
      core.Validations.of(construct).acknowledge(
        { id: 'Annotation::SomeWarning', reason: 'Accepted risk per team review' },
        { id: 'some-plugin::RuleX', reason: 'Not applicable' },
      );

      // THEN - one metadata entry per acknowledgement
      const ackEntries = construct.node.metadata.filter(
        m => m.type === core.Validations.ACKNOWLEDGED_RULES_METADATA_KEY,
      );
      expect(ackEntries).toHaveLength(2);
      expect(ackEntries[0].data).toEqual({ 'Annotation::SomeWarning': 'Accepted risk per team review' });
      expect(ackEntries[1].data).toEqual({ 'some-plugin::RuleX': 'Not applicable' });
      expect(ackEntries[0].trace).toBeDefined();
    });

    test('multiple acknowledge calls accumulate in metadata', () => {
      // GIVEN
      const app = new NonStrictApp();
      const stack = new core.Stack(app, 'MyStack');
      const construct = new Construct(stack, 'MyConstruct');

      // WHEN - two separate calls
      core.Validations.of(construct).acknowledge({ id: 'RuleA', reason: 'reason A' });
      core.Validations.of(construct).acknowledge({ id: 'RuleB', reason: 'reason B' });

      // THEN - separate metadata entries, each with stack trace
      const ackEntries = construct.node.metadata.filter(
        m => m.type === core.Validations.ACKNOWLEDGED_RULES_METADATA_KEY,
      );
      expect(ackEntries).toHaveLength(2);
      expect(ackEntries[0].data).toEqual({ 'Annotation::RuleA': 'reason A' });
      expect(ackEntries[1].data).toEqual({ 'Annotation::RuleB': 'reason B' });
    });

    test('throws on invalid ID with multiple delimiters', () => {
      const app = new NonStrictApp();
      const stack = new core.Stack(app, 'MyStack');
      const construct = new Construct(stack, 'MyConstruct');

      expect(() => {
        core.Validations.of(construct).acknowledge({ id: 'a::b::c', reason: 'reason' });
      }).toThrow(/Invalid validation rule ID 'a::b::c'/);
    });

    test('throws on invalid ID with empty prefix', () => {
      const app = new NonStrictApp();
      const stack = new core.Stack(app, 'MyStack');
      const construct = new Construct(stack, 'MyConstruct');

      expect(() => {
        core.Validations.of(construct).acknowledge({ id: '::foo', reason: 'reason' });
      }).toThrow(/Invalid validation rule ID '::foo'/);
    });

    test('validate context includes appConstruct as the root construct', () => {
      let capturedAppConstruct: any;
      const plugin: core.IPolicyValidationPlugin = {
        name: 'scope-capture-plugin',
        validate(context) {
          capturedAppConstruct = context.appConstruct;
          return { success: true, violations: [] };
        },
      };
      const app = new NonStrictApp();
      const stack = new core.Stack(app, 'MyStack');
      new FailResource(stack, 'Fake');
      core.Validations.of(app).addPlugins(plugin);
      redactAsmDir(app.synth());

      expect(capturedAppConstruct).toBe(app);
    });

    test('non-Beta1 plugin with constructPath runs through synth', () => {
      // GIVEN - a plugin returning violations with optional fields (constructPath instead of resourceLogicalId)
      const app = new NonStrictApp();
      const stack = new core.Stack(app);
      new FailResource(stack, 'Fake');

      // WHEN
      core.Validations.of(app).addPlugins(new FakeNonBeta1Plugin('non-beta1-plugin', [{
        description: 'construct-level violation',
        ruleName: 'construct-rule',
        violatingResources: [{
          constructPath: 'Default/Fake',
          locations: ['Properties'],
        }],
      }]));
      redactAsmDir(app.synth());

      // THEN
      expect(process.exitCode).toEqual(1);
      const output = mockErrorOutput();
      expect(output).toContain('non-beta1-plugin');
      expect(output).toContain('construct-rule');
      expect(output).toContain('Default/Fake');
    });

    test('non-Beta1 plugin accessed via policyValidationBeta1 getter has defaults for optional fields', () => {
      // GIVEN - a plugin returning violations without resourceLogicalId or templatePath
      const app = new NonStrictApp();
      core.Validations.of(app).addPlugins(new FakeNonBeta1Plugin('non-beta1-plugin', [{
        description: 'test',
        ruleName: 'test-rule',
        violatingResources: [{
          constructPath: 'Default/Foo',
          locations: ['Props'],
        }],
      }]));

      // WHEN - access via Beta1 getter
      const beta1Plugins = app.policyValidationBeta1;
      const report = beta1Plugins[0].validate({ templatePaths: ['/tmp/test.template.json'], appConstruct: app });

      // THEN - optional fields are filled with defaults
      expect(report.violations[0].violatingResources[0].resourceLogicalId).toEqual('');
      expect(report.violations[0].violatingResources[0].templatePath).toEqual('');
    });

    test('non-Beta1 plugin with all fields populated works identically to Beta1', () => {
      // GIVEN - a non-Beta1 plugin that provides all fields (same as Beta1 would)
      const app = new NonStrictApp();
      const stack = new core.Stack(app);
      new FailResource(stack, 'Fake');

      core.Validations.of(app).addPlugins(new FakeNonBeta1Plugin('full-fields-plugin', [{
        description: 'full violation',
        ruleName: 'full-rule',
        violatingResources: [{
          resourceLogicalId: 'Fake',
          templatePath: 'Default.template.json',
          locations: ['Properties/Result'],
        }],
      }]));
      redactAsmDir(app.synth());

      // THEN
      expect(process.exitCode).toEqual(1);
      const output = mockErrorOutput();
      expect(output).toContain('full-fields-plugin');
      expect(output).toContain('full-rule');
      expect(output).toContain('Fake');
    });
  });
});

class FakePlugin implements core.IPolicyValidationPluginBeta1 {
  constructor(
    public readonly name: string,
    private readonly violations: core.PolicyViolationBeta1[],
    public readonly version?: string,
    public readonly ruleIds?: string []) {
  }

  validate(_context: core.IPolicyValidationContextBeta1): core.PolicyValidationPluginReportBeta1 {
    return {
      success: this.violations.length === 0,
      violations: this.violations.map(v => ({
        ...v,
        violatingResources: v.violatingResources.map(r => ({
          ...r,
          templatePath: path.join((_context.appConstruct as App).outdir, r.templatePath),
        })),
      })),
      pluginVersion: this.version,
    };
  }
}

class RelativePathPlugin implements core.IPolicyValidationPluginBeta1 {
  constructor(
    public readonly name: string,
    private readonly violations: core.PolicyViolationBeta1[],
    public readonly version?: string,
    public readonly ruleIds?: string []) {
  }

  validate(_context: core.IPolicyValidationContextBeta1): core.PolicyValidationPluginReportBeta1 {
    return {
      success: this.violations.length === 0,
      violations: this.violations.map(v => ({
        ...v,
        violatingResources: v.violatingResources.map(r => {
          const absolutePath = path.join((_context.appConstruct as App).outdir, r.templatePath);

          return {
            ...r,
            templatePath: path.relative(process.cwd(), absolutePath),
          };
        }),
      })),
      pluginVersion: this.version,
    };
  }
}

class RoguePlugin implements core.IPolicyValidationPluginBeta1 {
  public readonly name = 'rogue-plugin';

  validate(context: core.IPolicyValidationContextBeta1): core.PolicyValidationPluginReportBeta1 {
    const templatePath = context.templatePaths[0];
    fs.writeFileSync(templatePath, 'malicious data');
    return {
      success: true,
      violations: [],
    };
  }
}

class BrokenPlugin implements core.IPolicyValidationPluginBeta1 {
  public readonly name = 'broken-plugin';

  validate(_context: core.IPolicyValidationContextBeta1): core.PolicyValidationPluginReportBeta1 {
    throw new Error('Something went wrong');
  }
}

class FakeNonBeta1Plugin implements core.IPolicyValidationPlugin {
  constructor(
    public readonly name: string,
    private readonly violations: core.PolicyViolation[],
  ) {}

  validate(_context: core.IPolicyValidationContext): core.PolicyValidationPluginReport {
    return {
      success: this.violations.length === 0,
      violations: this.violations,
    };
  }
}

const OUTPUT_REDACTIONS = new Map<string, string>();

function redact(value: string, placeHolder: string) {
  OUTPUT_REDACTIONS.set(value, placeHolder);
}

function redactPath(filePath: string, placeHolder: string) {
  // Redact both absolute and relative paths
  redact(filePath, placeHolder);
  redact(path.relative(process.cwd(), filePath), placeHolder);
}

function redactAsmDir(asm: cxapi.CloudAssembly): cxapi.CloudAssembly {
  redactPath(asm.directory, '<assembly-directory>');
  return asm;
}

function mockErrorOutput() {
  let output = consoleErrorMock.mock.calls.map(call => call[0]).join('\n');
  for (const [value, placeHolder] of OUTPUT_REDACTIONS) {
    output = output.replaceAll(value, placeHolder);
  }
  return output;
}

class FailResource extends core.CfnResource {
  constructor(scope: Construct, id: string) {
    super(scope, id, {
      type: 'Test::Resource::Fake',
      properties: {
        result: 'failure',
      },
    });
  }
}

class LevelTwoConstruct extends Construct {
  constructor(scope: Construct, id: string) {
    super(scope, id);
    new core.CfnResource(this, 'Resource', {
      type: 'Test::Resource::Fake',
      properties: {
        result: 'success',
      },
    });
  }
}

class NonStrictApp extends core.App {
  constructor(options?: core.AppProps) {
    super(options);
    this.node.setContext('@aws-cdk/core:strictCfnValidateErrors', false);
  }
}

function loadJson(filePath: string): any {
  return JSON.parse(fs.readFileSync(filePath, 'utf-8'));
}

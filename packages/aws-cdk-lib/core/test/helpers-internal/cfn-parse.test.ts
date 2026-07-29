import { App, CfnResource, Stack } from '../../../core';
import type { ICfnFinder } from '../../../core/lib/helpers-internal/cfn-parse';
import { CfnParser } from '../../../core/lib/helpers-internal/cfn-parse';

describe('cfn-parse', () => {
  let app: App;
  let stack: Stack;
  let parser: CfnParser;

  beforeEach(() => {
    app = new App();
    stack = new Stack(app, 'TestStack');

    // Create a mock finder for the CfnParser
    const mockFinder: ICfnFinder = {
      findCondition: jest.fn(),
      findMapping: jest.fn(),
      findRefTarget: jest.fn(),
      findResource: jest.fn(),
    };

    // Create a CfnParser instance
    parser = new CfnParser({
      finder: mockFinder,
      parameters: {},
    });
  });

  test('handleAttributes correctly parses all properties in AutoScalingRollingUpdate', () => {
    // Create a CfnResource to test with
    const cfnResource = new CfnResource(stack, 'TestResource', {
      type: 'AWS::AutoScaling::AutoScalingGroup',
      properties: {
        MinSize: '1',
        MaxSize: '10',
        DesiredCapacity: '2',
      },
    });

    // Create resource attributes with UpdatePolicy containing AutoScalingRollingUpdate
    const resourceAttributes = {
      UpdatePolicy: {
        AutoScalingRollingUpdate: {
          MaxBatchSize: 2,
          MinInstancesInService: 1,
          MinSuccessfulInstancesPercent: 75,
          MinActiveInstancesPercent: 50,
          PauseTime: 'PT5M',
          SuspendProcesses: ['HealthCheck', 'ReplaceUnhealthy'],
          WaitOnResourceSignals: true,
        },
      },
    };

    // Call handleAttributes to process the resource attributes
    parser.handleAttributes(cfnResource, resourceAttributes, 'TestResource');

    // Verify the update policy was correctly parsed
    expect(cfnResource.cfnOptions.updatePolicy).toBeDefined();
    expect(cfnResource.cfnOptions.updatePolicy?.autoScalingRollingUpdate).toBeDefined();

    const rollingUpdate = cfnResource.cfnOptions.updatePolicy?.autoScalingRollingUpdate;
    expect(rollingUpdate?.maxBatchSize).toBe(2);
    expect(rollingUpdate?.minInstancesInService).toBe(1);
    expect(rollingUpdate?.minSuccessfulInstancesPercent).toBe(75);
    expect(rollingUpdate?.minActiveInstancesPercent).toBe(50);
    expect(rollingUpdate?.pauseTime).toBe('PT5M');
    expect(rollingUpdate?.suspendProcesses).toEqual(['HealthCheck', 'ReplaceUnhealthy']);
    expect(rollingUpdate?.waitOnResourceSignals).toBe(true);
  });

  test('handleAttributes correctly parses AutoScalingRollingUpdate without MinActiveInstancesPercent', () => {
    // Create a CfnResource to test with
    const cfnResource = new CfnResource(stack, 'TestResource', {
      type: 'AWS::AutoScaling::AutoScalingGroup',
      properties: {
        MinSize: '1',
        MaxSize: '10',
        DesiredCapacity: '2',
      },
    });

    // Create resource attributes with UpdatePolicy containing AutoScalingRollingUpdate
    // but without MinActiveInstancesPercent
    const resourceAttributes = {
      UpdatePolicy: {
        AutoScalingRollingUpdate: {
          MaxBatchSize: 2,
          MinInstancesInService: 1,
          MinSuccessfulInstancesPercent: 75,
          PauseTime: 'PT5M',
          SuspendProcesses: ['HealthCheck', 'ReplaceUnhealthy'],
          WaitOnResourceSignals: true,
        },
      },
    };

    // Call handleAttributes to process the resource attributes
    parser.handleAttributes(cfnResource, resourceAttributes, 'TestResource');

    // Verify the update policy was correctly parsed
    expect(cfnResource.cfnOptions.updatePolicy).toBeDefined();
    expect(cfnResource.cfnOptions.updatePolicy?.autoScalingRollingUpdate).toBeDefined();

    const rollingUpdate = cfnResource.cfnOptions.updatePolicy?.autoScalingRollingUpdate;
    expect(rollingUpdate?.maxBatchSize).toBe(2);
    expect(rollingUpdate?.minInstancesInService).toBe(1);
    expect(rollingUpdate?.minSuccessfulInstancesPercent).toBe(75);
    expect(rollingUpdate?.minActiveInstancesPercent).toBeUndefined();
    expect(rollingUpdate?.pauseTime).toBe('PT5M');
    expect(rollingUpdate?.suspendProcesses).toEqual(['HealthCheck', 'ReplaceUnhealthy']);
    expect(rollingUpdate?.waitOnResourceSignals).toBe(true);
  });

  test('handleAttributes correctly parses all properties in AutoScalingInstanceRefresh', () => {
    const cfnResource = new CfnResource(stack, 'TestResource', {
      type: 'AWS::AutoScaling::AutoScalingGroup',
      properties: { MinSize: '1', MaxSize: '10', DesiredCapacity: '2' },
    });

    const resourceAttributes = {
      UpdatePolicy: {
        AutoScalingInstanceRefresh: {
          Strategy: 'Rolling',
          Preferences: {
            MinHealthyPercentage: 90,
            MaxHealthyPercentage: 110,
            InstanceWarmup: 300,
            SkipMatching: true,
            CheckpointPercentages: [20, 50, 100],
            CheckpointDelay: 3600,
            BakeTime: 600,
            AlarmSpecification: { Alarms: ['my-alarm-1', 'my-alarm-2'] },
            ScaleInProtectedInstances: 'Wait',
            StandbyInstances: 'Ignore',
          },
        },
      },
    };

    parser.handleAttributes(cfnResource, resourceAttributes, 'TestResource');

    const instanceRefresh = cfnResource.cfnOptions.updatePolicy?.autoScalingInstanceRefresh;
    expect(instanceRefresh?.strategy).toBe('Rolling');
    const prefs = instanceRefresh?.preferences;
    expect(prefs?.minHealthyPercentage).toBe(90);
    expect(prefs?.maxHealthyPercentage).toBe(110);
    expect(prefs?.instanceWarmup).toBe(300);
    expect(prefs?.skipMatching).toBe(true);
    expect(prefs?.checkpointPercentages).toEqual([20, 50, 100]);
    expect(prefs?.checkpointDelay).toBe(3600);
    expect(prefs?.bakeTime).toBe(600);
    expect(prefs?.alarmSpecification?.alarms).toEqual(['my-alarm-1', 'my-alarm-2']);
    expect(prefs?.scaleInProtectedInstances).toBe('Wait');
    expect(prefs?.standbyInstances).toBe('Ignore');
  });
});

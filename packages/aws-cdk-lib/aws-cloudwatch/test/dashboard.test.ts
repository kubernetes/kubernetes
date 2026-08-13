import { Annotations, Match, Template } from '../../assertions';
import { App, Duration, Stack } from '../../core';
import type { IWidget } from '../lib';
import {
  Dashboard, DashboardVariable, DefaultValue,
  GraphWidget,
  MathExpression,
  PeriodOverride,
  TextWidget,
  TextWidgetBackground, Values,
  VariableInputType,
  VariableType,
} from '../lib';

describe('Dashboard', () => {
  test('concrete widgets are assignable to the dashboard widget interface', () => {
    // Regression coverage for exactOptionalPropertyTypes users.
    const stack = new Stack();
    const dashboard = new Dashboard(stack, 'Dash');
    const widget = new GraphWidget({ width: 1, height: 1 });
    const widgets: IWidget[] = [widget];

    dashboard.addWidgets(...widgets);

    expect(widget.warnings).toEqual([]);
    expect(widget.warningsV2).toEqual({});
  });

  test('widgets in different adds are laid out underneath each other', () => {
    // GIVEN
    const stack = new Stack();
    const dashboard = new Dashboard(stack, 'Dash');

    // WHEN
    dashboard.addWidgets(new TextWidget({
      width: 10,
      height: 2,
      markdown: 'first',
      background: TextWidgetBackground.SOLID,
    }));
    dashboard.addWidgets(new TextWidget({
      width: 1,
      height: 4,
      markdown: 'second',
      background: TextWidgetBackground.TRANSPARENT,
    }));
    dashboard.addWidgets(new TextWidget({
      width: 4,
      height: 1,
      markdown: 'third',
    }));

    // THEN
    const resources = Template.fromStack(stack).findResources('AWS::CloudWatch::Dashboard');
    expect(Object.keys(resources).length).toEqual(1);
    const key = Object.keys(resources)[0];
    hasWidgets(resources[key].Properties, [
      { type: 'text', width: 10, height: 2, x: 0, y: 0, properties: { markdown: 'first', background: TextWidgetBackground.SOLID } },
      { type: 'text', width: 1, height: 4, x: 0, y: 2, properties: { markdown: 'second', background: TextWidgetBackground.TRANSPARENT } },
      { type: 'text', width: 4, height: 1, x: 0, y: 6, properties: { markdown: 'third' } },
    ]);
  });

  test('widgets in same add are laid out next to each other', () => {
    // GIVEN
    const stack = new Stack();
    const dashboard = new Dashboard(stack, 'Dash');

    // WHEN
    dashboard.addWidgets(
      new TextWidget({
        width: 10,
        height: 2,
        markdown: 'first',
      }),
      new TextWidget({
        width: 1,
        height: 4,
        markdown: 'second',
      }),
      new TextWidget({
        width: 4,
        height: 1,
        markdown: 'third',
      }),
    );

    // THEN
    const resources = Template.fromStack(stack).findResources('AWS::CloudWatch::Dashboard');
    expect(Object.keys(resources).length).toEqual(1);
    const key = Object.keys(resources)[0];
    hasWidgets(resources[key].Properties, [
      { type: 'text', width: 10, height: 2, x: 0, y: 0, properties: { markdown: 'first' } },
      { type: 'text', width: 1, height: 4, x: 10, y: 0, properties: { markdown: 'second' } },
      { type: 'text', width: 4, height: 1, x: 11, y: 0, properties: { markdown: 'third' } },
    ]);
  });

  test('tokens in widgets are retained', () => {
    // GIVEN
    const stack = new Stack();
    const dashboard = new Dashboard(stack, 'Dash');

    // WHEN
    dashboard.addWidgets(
      new GraphWidget({ width: 1, height: 1 }), // GraphWidget has internal reference to current region
    );

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CloudWatch::Dashboard', {
      DashboardBody: {
        'Fn::Join': ['', [
          '{"widgets":[{"type":"metric","width":1,"height":1,"x":0,"y":0,"properties":{"view":"timeSeries","region":"',
          { Ref: 'AWS::Region' },
          '","yAxis":{}}}]}',
        ]],
      },
    });
  });

  test('dashboard body includes non-widget fields', () => {
    // GIVEN
    const stack = new Stack();
    const dashboard = new Dashboard(stack, 'Dash',
      {
        start: '-9H',
        end: '2018-12-17T06:00:00.000Z',
        periodOverride: PeriodOverride.INHERIT,
      });

    // WHEN
    dashboard.addWidgets(
      new GraphWidget({ width: 1, height: 1 }), // GraphWidget has internal reference to current region
    );

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CloudWatch::Dashboard', {
      DashboardBody: {
        'Fn::Join': ['', [
          '{"start":"-9H","end":"2018-12-17T06:00:00.000Z","periodOverride":"inherit",\
"widgets":[{"type":"metric","width":1,"height":1,"x":0,"y":0,"properties":{"view":"timeSeries","region":"',
          { Ref: 'AWS::Region' },
          '","yAxis":{}}}]}',
        ]],
      },
    });
  });

  test('defaultInterval test', () => {
    // GIVEN
    const stack = new Stack();
    // WHEN
    const dashboard = new Dashboard(stack, 'Dash', {
      defaultInterval: Duration.days(7),
    });
    dashboard.addWidgets(
      new GraphWidget({ width: 1, height: 1 }), // GraphWidget has internal reference to current region
    );

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CloudWatch::Dashboard', {
      DashboardBody: {
        'Fn::Join': ['', [
          '{"start":"-P7D",\
"widgets":[{"type":"metric","width":1,"height":1,"x":0,"y":0,"properties":{"view":"timeSeries","region":"',
          { Ref: 'AWS::Region' },
          '","yAxis":{}}}]}',
        ]],
      },
    });
  });

  test('throws if both defaultInterval and start are specified', () => {
    // GIVEN
    const stack = new Stack();
    // WHEN
    const toThrow = () => {
      new Dashboard(stack, 'Dash', {
        start: '-P7D',
        defaultInterval: Duration.days(7),
      });
    };

    // THEN
    expect(() => toThrow()).toThrow(/both properties defaultInterval and start cannot be set at once/);
  });

  test('throws if end is specified but start is not', () => {
    // GIVEN
    const stack = new Stack();
    // WHEN
    const toThrow = () => {
      new Dashboard(stack, 'Dash', {
        end: '2018-12-17T06:00:00.000Z',
      });
    };

    // THEN
    expect(() => toThrow()).toThrow(/If you specify a value for end, you must also specify a value for start./);
  });

  test('DashboardName is set when provided', () => {
    // GIVEN
    const app = new App();
    const stack = new Stack(app, 'MyStack');

    // WHEN
    new Dashboard(stack, 'MyDashboard', {
      dashboardName: 'MyCustomDashboardName',
    });

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CloudWatch::Dashboard', {
      DashboardName: 'MyCustomDashboardName',
    });
  });

  test('DashboardName is not generated if not provided', () => {
    // GIVEN
    const app = new App();
    const stack = new Stack(app, 'MyStack');

    // WHEN
    new Dashboard(stack, 'MyDashboard');

    // THEN
    Template.fromStack(stack).hasResourceProperties('AWS::CloudWatch::Dashboard', {});
  });

  test('throws if DashboardName is not valid', () => {
    // GIVEN
    const app = new App();
    const stack = new Stack(app, 'MyStack');

    // WHEN
    const toThrow = () => {
      new Dashboard(stack, 'MyDashboard', {
        dashboardName: 'My Invalid Dashboard Name',
      });
    };

    // THEN
    expect(() => toThrow()).toThrow(/field dashboardName contains invalid characters/);
  });

  test('dashboardArn should not include a region', () => {
    // GIVEN
    const app = new App();
    const stack = new Stack(app, 'MyStack', {
      env: {
        region: 'invalid-region',
      },
    });

    // WHEN
    const dashboard = new Dashboard(stack, 'MyStack');

    // THEN
    expect(dashboard.dashboardArn).not.toContain('invalid-region');
  });

  test('metric warnings are added to dashboard', () => {
    const app = new App();
    const stack = new Stack(app, 'MyStack');
    const m = new MathExpression({ expression: 'oops' });

    // WHEN
    new Dashboard(stack, 'MyDashboard', {
      widgets: [
        [new GraphWidget({ left: [m] }), new TextWidget({ markdown: 'asdf' })],
      ],
    });

    // THEN
    const template = Annotations.fromStack(stack);
    template.hasWarning('/MyStack/MyDashboard', Match.stringLikeRegexp("Math expression 'oops' references unknown identifiers"));
  });

  test('dashboard has initial select/pattern variable', () => {
    // GIVEN
    const stack = new Stack();
    // WHEN
    new Dashboard(stack, 'Dashboard', {
      variables: [new DashboardVariable({
        type: VariableType.PATTERN,
        value: 'us-east-1',
        inputType: VariableInputType.SELECT,
        id: 'region3',
        label: 'RegionPatternWithValues',
        defaultValue: DefaultValue.value('us-east-1'),
        visible: true,
        values: Values.fromValues({ label: 'IAD', value: 'us-east-1' }, { label: 'DUB', value: 'us-west-2' }),
      })],
    });

    // THEN
    const resources = Template.fromStack(stack).findResources('AWS::CloudWatch::Dashboard');
    expect(Object.keys(resources).length).toEqual(1);
    const key = Object.keys(resources)[0];
    hasVariables(resources[key].Properties, [
      { defaultValue: 'us-east-1', id: 'region3', inputType: 'select', label: 'RegionPatternWithValues', pattern: 'us-east-1', type: 'pattern', values: [{ label: 'IAD', value: 'us-east-1' }, { label: 'DUB', value: 'us-west-2' }], visible: true },
    ]);
  });

  test('dashboard has initial and added variable', () => {
    // GIVEN
    const stack = new Stack();
    const dashboard = new Dashboard(stack, 'Dashboard', {
      variables: [new DashboardVariable({
        type: VariableType.PATTERN,
        value: 'us-east-1',
        inputType: VariableInputType.SELECT,
        id: 'region3',
        label: 'RegionPatternWithValues',
        defaultValue: DefaultValue.value('us-east-1'),
        visible: true,
        values: Values.fromValues({ label: 'IAD', value: 'us-east-1' }, { label: 'DUB', value: 'us-west-2' }),
      })],
    });

    dashboard.addVariable(new DashboardVariable({
      type: VariableType.PATTERN,
      value: 'us-east-1',
      inputType: VariableInputType.INPUT,
      id: 'region2',
      label: 'RegionPattern',
      defaultValue: DefaultValue.value('us-east-1'),
      visible: true,
    }));

    // THEN
    const resources = Template.fromStack(stack).findResources('AWS::CloudWatch::Dashboard');
    expect(Object.keys(resources).length).toEqual(1);
    const key = Object.keys(resources)[0];
    hasVariables(resources[key].Properties, [
      { defaultValue: 'us-east-1', id: 'region3', inputType: 'select', label: 'RegionPatternWithValues', pattern: 'us-east-1', type: 'pattern', values: [{ label: 'IAD', value: 'us-east-1' }, { label: 'DUB', value: 'us-west-2' }], visible: true },
      { defaultValue: 'us-east-1', id: 'region2', inputType: 'input', label: 'RegionPattern', pattern: 'us-east-1', type: 'pattern', visible: true },
    ]);
  });

  test('dashboard has property/select search variable', () => {
    // GIVEN
    const stack = new Stack();
    const dashboard = new Dashboard(stack, 'Dashboard');

    // WHEN
    dashboard.addVariable(new DashboardVariable({
      defaultValue: DefaultValue.FIRST,
      id: 'InstanceId',
      label: 'Instance',
      inputType: VariableInputType.SELECT,
      type: VariableType.PROPERTY,
      value: 'InstanceId',
      values: Values.fromSearchComponents({
        namespace: 'AWS/EC2',
        dimensions: ['InstanceId'],
        metricName: 'CPUUtilization',
        populateFrom: 'InstanceId',
      }),
      visible: true,
    }));

    // THEN
    const resources = Template.fromStack(stack).findResources('AWS::CloudWatch::Dashboard');
    expect(Object.keys(resources).length).toEqual(1);
    const key = Object.keys(resources)[0];
    hasVariables(resources[key].Properties, [
      { defaultValue: '__FIRST', id: 'InstanceId', inputType: 'select', label: 'Instance', populateFrom: 'InstanceId', property: 'InstanceId', search: '{AWS/EC2,InstanceId} MetricName="CPUUtilization"', type: 'property', visible: true },
    ]);
  });

  test('dashboard has input/pattern value variable', () => {
    // GIVEN
    const stack = new Stack();
    const dashboard = new Dashboard(stack, 'Dashboard');

    // WHEN
    dashboard.addVariable(new DashboardVariable({
      type: VariableType.PATTERN,
      value: 'us-east-1',
      inputType: VariableInputType.INPUT,
      id: 'region2',
      label: 'RegionPattern',
      defaultValue: DefaultValue.value('us-east-1'),
      visible: true,
    }));

    // THEN
    const resources = Template.fromStack(stack).findResources('AWS::CloudWatch::Dashboard');
    expect(Object.keys(resources).length).toEqual(1);
    const key = Object.keys(resources)[0];
    hasVariables(resources[key].Properties, [
      { defaultValue: 'us-east-1', id: 'region2', inputType: 'input', label: 'RegionPattern', pattern: 'us-east-1', type: 'pattern', visible: true },
    ]);
  });

  test('dashboard has property/radio value variable', () => {
    // GIVEN
    const stack = new Stack();
    const dashboard = new Dashboard(stack, 'Dashboard');

    // WHEN
    dashboard.addVariable(new DashboardVariable({
      type: VariableType.PROPERTY,
      value: 'region',
      inputType: VariableInputType.RADIO,
      id: 'region3',
      label: 'RegionRadio',
      defaultValue: DefaultValue.value('us-east-1'),
      visible: true,
      values: Values.fromValues({ label: 'IAD', value: 'us-east-1' }, { label: 'DUB', value: 'us-west-2' }),
    }));

    // THEN
    const resources = Template.fromStack(stack).findResources('AWS::CloudWatch::Dashboard');
    expect(Object.keys(resources).length).toEqual(1);
    const key = Object.keys(resources)[0];
    hasVariables(resources[key].Properties, [
      { defaultValue: 'us-east-1', id: 'region3', inputType: 'radio', label: 'RegionRadio', property: 'region', type: 'property', values: [{ label: 'IAD', value: 'us-east-1' }, { label: 'DUB', value: 'us-west-2' }], visible: true },
    ]);
  });

  test('dashboard variable fails if unsupported input inputType', () => {
    expect(() => new DashboardVariable({
      defaultValue: DefaultValue.FIRST,
      id: 'InstanceId',
      label: 'Instance',
      inputType: VariableInputType.INPUT,
      type: VariableType.PROPERTY,
      value: 'InstanceId',
      values: Values.fromSearchComponents({
        namespace: 'AWS/EC2',
        dimensions: ['InstanceId'],
        metricName: 'CPUUtilization',
        populateFrom: 'InstanceId',
      }),
      visible: true,
    })).toThrow(/inputType INPUT cannot be combined with values. Please choose either SELECT or RADIO or remove 'values' from options/);
  });

  test('dashboard variable fails if no values provided for select or radio inputType', () => {
    [VariableInputType.SELECT, VariableInputType.RADIO].forEach(inputType => {
      expect(() => new DashboardVariable({
        inputType,
        type: VariableType.PATTERN,
        value: 'us-east-1',
        id: 'region3',
        label: 'RegionPatternWithValues',
        defaultValue: DefaultValue.value('us-east-1'),
        visible: true,
      })).toThrow(`Variable with inputType (${inputType}) requires values to be set`);
    });
  });

  test('search values fail if empty dimensions', () => {
    expect(() => Values.fromSearchComponents({
      namespace: 'AWS/EC2',
      dimensions: [],
      metricName: 'CPUUtilization',
      populateFrom: 'InstanceId',
    })).toThrow(/Empty dimensions provided. Please specify one dimension at least/);
  });

  test('search values fail if populateFrom is not present in dimensions', () => {
    expect(() => Values.fromSearchComponents({
      namespace: 'AWS/EC2',
      dimensions: ['InstanceId'],
      metricName: 'CPUUtilization',
      populateFrom: 'DontExist',
    })).toThrow('populateFrom (DontExist) is not present in dimensions');
  });
});

/**
 * Returns a property predicate that checks that the given Dashboard has the indicated widgets
 */
function hasWidgets(props: any, widgets: any[]) {
  let actualWidgets: any[] = [];
  try {
    actualWidgets = JSON.parse(props.DashboardBody).widgets;
  } catch (e) {
    console.error('Error parsing', props);
    throw e;
  }
  expect(actualWidgets).toEqual(expect.arrayContaining(widgets));
}

/**
 * Returns a property predicate that checks that the given Dashboard has the indicated variables
 */
function hasVariables(props: any, variables: any[]) {
  let actualVariables: any[] = [];
  try {
    actualVariables = JSON.parse(props.DashboardBody).variables;
  } catch (e) {
    console.error('Error parsing', props);
    throw e;
  }
  expect(actualVariables).toEqual(expect.arrayContaining(variables));
}

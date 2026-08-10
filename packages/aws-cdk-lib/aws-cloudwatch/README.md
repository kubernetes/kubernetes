# Amazon CloudWatch Construct Library


## Metric objects

Metric objects represent a metric that is emitted by AWS services or your own
application, such as `CPUUsage`, `FailureCount` or `Bandwidth`.

Metric objects can be constructed directly or are exposed by resources as
attributes. Resources that expose metrics will have functions that look
like `metricXxx()` which will return a Metric object, initialized with defaults
that make sense.

For example, `lambda.Function` objects have the `fn.metricErrors()` method, which
represents the amount of errors reported by that Lambda function:

```ts
declare const fn: lambda.Function;

const errors = fn.metricErrors();
```

`Metric` objects can be account and region aware. You can specify `account` and `region` as properties of the metric, or use the `metric.attachTo(Construct)` method. `metric.attachTo()` will automatically copy the `region` and `account` fields of the `Construct`, which can come from anywhere in the Construct tree.

You can also instantiate `Metric` objects to reference any
[published metric](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/aws-services-cloudwatch-metrics.html)
that's not exposed using a convenience method on the CDK construct.
For example:

```ts
const hostedZone = new route53.HostedZone(this, 'MyHostedZone', { zoneName: "example.org" });
const metric = new cloudwatch.Metric({
  namespace: 'AWS/Route53',
  metricName: 'DNSQueries',
  dimensionsMap: {
    HostedZoneId: hostedZone.hostedZoneId
  }
});
```

### Instantiating a new Metric object

If you want to reference a metric that is not yet exposed by an existing construct,
you can instantiate a `Metric` object to represent it. For example:

```ts
const metric = new cloudwatch.Metric({
  namespace: 'MyNamespace',
  metricName: 'MyMetric',
  dimensionsMap: {
    ProcessingStep: 'Download'
  }
});
```

### Metric ID

Metrics can be assigned a unique identifier using the `id` property. This is
useful when referencing metrics in math expressions:

```ts
const metric = new cloudwatch.Metric({
  namespace: 'AWS/Lambda',
  metricName: 'Invocations',
  dimensionsMap: {
    FunctionName: 'MyFunction'
  },
  id: 'invocations'
});
```

The `id` must start with a lowercase letter and can only contain letters, numbers, and underscores.

### Metric Visible
Metrics can be hidden from dashboard graphs using the `visible` property:

```ts
declare const fn: lambda.Function;

const metric = fn.metricErrors({
  visible: false
});
```

By default, all metrics are visible (`visible: true`). Setting `visible: false`
hides the metric from dashboard visualizations while still allowing it to be
used in math expressions given that it has an `id` set to it.

### Metric Math

Math expressions are supported by instantiating the `MathExpression` class.
For example, a math expression that sums two other metrics looks like this:

```ts
declare const fn: lambda.Function;

const allProblems = new cloudwatch.MathExpression({
  expression: "errors + throttles",
  usingMetrics: {
    errors: fn.metricErrors(),
    throttles: fn.metricThrottles(),
  }
});
```

You can use `MathExpression` objects like any other metric, including using
them in other math expressions:

```ts
declare const fn: lambda.Function;
declare const allProblems: cloudwatch.MathExpression;

const problemPercentage = new cloudwatch.MathExpression({
  expression: "(problems / invocations) * 100",
  usingMetrics: {
    problems: allProblems,
    invocations: fn.metricInvocations()
  }
});
```

### Metric ID Usage in Math Expressions

When metrics have custom IDs, you can reference them directly in math expressions.

```ts
declare const fn: lambda.Function;

const invocations = fn.metricInvocations({
  id: 'lambda_invocations',
});

const errors = fn.metricErrors({
  id: 'lambda_errors',
});
```

When metrics have predefined IDs, they can be referenced directly in math expressions by their ID without requiring the `usingMetrics` property.

```ts
const errorRate = new cloudwatch.MathExpression({
  expression: 'lambda_errors / lambda_invocations * 100',
  label: 'Error Rate (%)',
});
```

### Search Expressions

Search expressions allow you to dynamically discover and display metrics that match specific criteria, making them ideal for monitoring dynamic infrastructure where the exact metrics aren't known in advance. A single search expression can return up to 500 time series.


#### Using SearchExpression Class

Following is an example of a search expression that returns all CPUUtilization metrics with the graph showing the Average statistic with an aggregation period of 5 minutes:

```ts
const cpuUtilization = new cloudwatch.SearchExpression({
  expression: "SEARCH('{AWS/EC2,InstanceId} MetricName=\"CPUUtilization\"', 'Average', 900)",
  label: 'EC2 CPU Utilization',
  color: '#ff7f0e',
});
```

Cross-account and cross-region search expressions are also supported. Use
the `searchAccount` and `searchRegion` properties to specify the account
and/or region to evaluate the search expression against.

```ts
const crossAccountSearch = new cloudwatch.SearchExpression({
  expression: "SEARCH('{AWS/Lambda,FunctionName} MetricName=\"Invocations\"', 'Sum', 300)",
  searchAccount: '123456789012',
  searchRegion: 'us-west-2',
  label: 'Production Lambda Invocations',
});
```

### Aggregation

To graph or alarm on metrics you must aggregate them first, using a function
like `Average` or a percentile function like `P99`. By default, most Metric objects
returned by CDK libraries will be configured as `Average` over `300 seconds` (5 minutes).
The exception is if the metric represents a count of discrete events, such as
failures. In that case, the Metric object will be configured as `Sum` over `300
seconds`, i.e. it represents the number of times that event occurred over the
time period.

If you want to change the default aggregation of the Metric object (for example,
the function or the period), you can do so by passing additional parameters
to the metric function call:

```ts
declare const fn: lambda.Function;

const minuteErrorRate = fn.metricErrors({
  statistic: cloudwatch.Stats.AVERAGE,
  period: Duration.minutes(1),
  label: 'Lambda failure rate'
});
```

The `statistic` field accepts a `string`; the `cloudwatch.Stats` object has a
number of predefined factory functions that help you constructs strings that are
appropriate for CloudWatch. The `metricErrors` function also allows changing the
metric label or color, which will be useful when embedding them in graphs (see
below).

> Rates versus Sums
>
> The reason for using `Sum` to count discrete events is that *some* events are
> emitted as either `0` or `1` (for example `Errors` for a Lambda) and some are
> only emitted as `1` (for example `NumberOfMessagesPublished` for an SNS
> topic).
>
> In case `0`-metrics are emitted, it makes sense to take the `Average` of this
> metric: the result will be the fraction of errors over all executions.
>
> If `0`-metrics are not emitted, the `Average` will always be equal to `1`,
> and not be very useful.
>
> In order to simplify the mental model of `Metric` objects, we default to
> aggregating using `Sum`, which will be the same for both metrics types. If you
> happen to know the Metric you want to alarm on makes sense as a rate
> (`Average`) you can always choose to change the statistic.

### Available Aggregation Statistics

For your metrics aggregation, you can use the following statistics:

| Statistic                |    Short format     |                 Long format                  | Factory name         |
| ------------------------ | :-----------------: | :------------------------------------------: | -------------------- |
| SampleCount (n)          |         ❌          |                      ❌                      | `Stats.SAMPLE_COUNT` |
| Average (avg)            |         ❌          |                      ❌                      | `Stats.AVERAGE`      |
| Sum                      |         ❌          |                      ❌                      | `Stats.SUM`          |
| Minimum (min)            |         ❌          |                      ❌                      | `Stats.MINIMUM`      |
| Maximum (max)            |         ❌          |                      ❌                      | `Stats.MAXIMUM`      |
| Interquartile mean (IQM) |         ❌          |                      ❌                      | `Stats.IQM`          |
| Percentile (p)           |        `p99`        |                      ❌                      | `Stats.p(99)`        |
| Winsorized mean (WM)     | `wm99` = `WM(:99%)` | `WM(x:y) \| WM(x%:y%) \| WM(x%:) \| WM(:y%)` | `Stats.wm(10, 90)`   |
| Trimmed count (TC)       | `tc99` = `TC(:99%)` | `TC(x:y) \| TC(x%:y%) \| TC(x%:) \| TC(:y%)` | `Stats.tc(10, 90)`   |
| Trimmed sum (TS)         | `ts99` = `TS(:99%)` | `TS(x:y) \| TS(x%:y%) \| TS(x%:) \| TS(:y%)` | `Stats.ts(10, 90)`   |
| Percentile rank (PR)     |         ❌          |        `PR(x:y) \| PR(x:) \| PR(:y)`         | `Stats.pr(10, 5000)` |

The most common values are provided in the `cloudwatch.Stats` class. You can provide any string if your statistic is not in the class.

Read more at [CloudWatch statistics definitions](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/Statistics-definitions.html).

```ts
declare const hostedZone: route53.HostedZone;

new cloudwatch.Metric({
  namespace: 'AWS/Route53',
  metricName: 'DNSQueries',
  dimensionsMap: {
    HostedZoneId: hostedZone.hostedZoneId
  },
  statistic: cloudwatch.Stats.SAMPLE_COUNT,
  period: Duration.minutes(5)
});

new cloudwatch.Metric({
  namespace: 'AWS/Route53',
  metricName: 'DNSQueries',
  dimensionsMap: {
    HostedZoneId: hostedZone.hostedZoneId
  },
  statistic: cloudwatch.Stats.p(99),
  period: Duration.minutes(5)
});

new cloudwatch.Metric({
  namespace: 'AWS/Route53',
  metricName: 'DNSQueries',
  dimensionsMap: {
    HostedZoneId: hostedZone.hostedZoneId
  },
  statistic: 'TS(7.5%:90%)',
  period: Duration.minutes(5)
});
```

### Labels

Metric labels are displayed in the legend of graphs that include the metrics.

You can use [dynamic labels](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/graph-dynamic-labels.html)
to show summary information about the displayed time series
in the legend. For example, if you use:

```ts
declare const fn: lambda.Function;

const minuteErrorRate = fn.metricErrors({
  statistic: cloudwatch.Stats.SUM,
  period: Duration.hours(1),

  // Show the maximum hourly error count in the legend
  label: '[max: ${MAX}] Lambda failure rate',
});
```

As the metric label, the maximum value in the visible range will
be shown next to the time series name in the graph's legend.

If the metric is a math expression producing more than one time series, the
maximum will be individually calculated and shown for each time series produce
by the math expression.

## Alarms

Alarms can be created on metrics in one of two ways. Either create an `Alarm`
object, passing the `Metric` object to set the alarm on:

```ts
declare const fn: lambda.Function;

new cloudwatch.Alarm(this, 'Alarm', {
  metric: fn.metricErrors(),
  threshold: 100,
  evaluationPeriods: 2,
});
```

Alternatively, you can call `metric.createAlarm()`:

```ts
declare const fn: lambda.Function;

fn.metricErrors().createAlarm(this, 'Alarm', {
  threshold: 100,
  evaluationPeriods: 2,
});
```

The most important properties to set while creating an Alarms are:

- `threshold`: the value to compare the metric against.
- `comparisonOperator`: the comparison operation to use, defaults to `metric >= threshold`.
- `evaluationPeriods`: how many consecutive periods the metric has to be
  breaching the threshold for the alarm to trigger.

To create a cross-account alarm, make sure you have enabled [cross-account functionality](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/Cross-Account-Cross-Region.html) in CloudWatch. Then, set the `account` property in the `Metric` object either manually or via the `metric.attachTo()` method.

Please note that it is **not possible** to:

- Create a cross-Account alarm that has `evaluateLowSampleCountPercentile: "ignore"`. The reason is that the only
  way to pass an AccountID is to use the [`Metrics` field of the Alarm resource](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-resource-cloudwatch-alarm.html#cfn-cloudwatch-alarm-metrics). If we use the `Metrics` field, the CloudWatch event that is
  used to evaluate the Alarm doesn't have a `SampleCount` field anymore ("[When CloudWatch evaluates alarms, periods are aggregated into single data points](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/Create-alarm-on-metric-math-expression.html)"). The result is that the Alarm cannot evaluate at all.
- Create a cross-Region alarm ([source](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/Cross-Account-Cross-Region.html)).

### Alarm Actions

To add actions to an alarm, use the integration classes from the
`aws-cdk-lib/aws-cloudwatch-actions` package. For example, to post a message to
an SNS topic when an alarm breaches, do the following:

```ts
import * as cw_actions from 'aws-cdk-lib/aws-cloudwatch-actions';
declare const alarm: cloudwatch.Alarm;

const topic = new sns.Topic(this, 'Topic');
alarm.addAlarmAction(new cw_actions.SnsAction(topic));
```

#### Notification formats

Alarms can be created in one of two "formats":

- With "top-level parameters" (these are the classic style of CloudWatch Alarms).
- With a list of metrics specifications (these are the modern style of CloudWatch Alarms).

For backwards compatibility, CDK will try to create classic, top-level CloudWatch alarms
as much as possible, unless you are using features that cannot be expressed in that format.
Features that require the new-style alarm format are:

- Metric math
- Cross-account metrics
- Labels

The difference between these two does not impact the functionality of the alarm
in any way, *except* that the format of the notifications the Alarm generates is
different between them. This affects both the notifications sent out over SNS,
as well as the EventBridge events generated by this Alarm. If you are writing
code to consume these notifications, be sure to handle both formats.

### Composite Alarms

[Composite Alarms](https://aws.amazon.com/about-aws/whats-new/2020/03/amazon-cloudwatch-now-allows-you-to-combine-multiple-alarms/)
can be created from existing Alarm resources.

```ts
declare const alarm1: cloudwatch.Alarm;
declare const alarm2: cloudwatch.Alarm;
declare const alarm3: cloudwatch.Alarm;
declare const alarm4: cloudwatch.Alarm;

const alarmRule = cloudwatch.AlarmRule.anyOf(
  cloudwatch.AlarmRule.allOf(
    cloudwatch.AlarmRule.anyOf(
      alarm1,
      cloudwatch.AlarmRule.fromAlarm(alarm2, cloudwatch.AlarmState.OK),
      alarm3,
    ),
    cloudwatch.AlarmRule.not(cloudwatch.AlarmRule.fromAlarm(alarm4, cloudwatch.AlarmState.INSUFFICIENT_DATA)),
  ),
  cloudwatch.AlarmRule.fromBoolean(false),
);

new cloudwatch.CompositeAlarm(this, 'MyAwesomeCompositeAlarm', {
  alarmRule,
});
```

#### Actions Suppressor

If you want to disable actions of a Composite Alarm based on a certain condition, you can use [Actions Suppression](https://www.amazonaws.cn/en/new/2022/amazon-cloudwatch-supports-composite-alarm-actions-suppression/).

```ts
declare const alarm1: cloudwatch.Alarm;
declare const alarm2: cloudwatch.Alarm;
declare const onAlarmAction: cloudwatch.IAlarmAction;
declare const onOkAction: cloudwatch.IAlarmAction;
declare const actionsSuppressor: cloudwatch.Alarm;

const alarmRule = cloudwatch.AlarmRule.anyOf(alarm1, alarm2);

const myCompositeAlarm = new cloudwatch.CompositeAlarm(this, 'MyAwesomeCompositeAlarm', {
  alarmRule,
  actionsSuppressor,
});
myCompositeAlarm.addAlarmAction(onAlarmAction);
myCompositeAlarm.addOkAction(onOkAction);
```

In the provided example, if `actionsSuppressor` is in `ALARM` state, `onAlarmAction` won't be triggered even if `myCompositeAlarm` goes into `ALARM` state.
Similar, if `actionsSuppressor` is in `ALARM` state and `myCompositeAlarm` goes from `ALARM` into `OK` state, `onOkAction` won't be triggered.

### A note on units

In CloudWatch, Metrics datums are emitted with units, such as `seconds` or
`bytes`. When `Metric` objects are given a `unit` attribute, it will be used to
*filter* the stream of metric datums for datums emitted using the same `unit`
attribute.

In particular, the `unit` field is *not* used to rescale datums or alarm threshold
values (for example, it cannot be used to specify an alarm threshold in
*Megabytes* if the metric stream is being emitted as *bytes*).

You almost certainly don't want to specify the `unit` property when creating
`Metric` objects (which will retrieve all datums regardless of their unit),
unless you have very specific requirements. Note that in any case, CloudWatch
only supports filtering by `unit` for Alarms, not in Dashboard graphs.

Please see the following GitHub issue for a discussion on real unit
calculations in CDK: https://github.com/aws/aws-cdk/issues/5595

## Anomaly Detection Alarms

CloudWatch anomaly detection applies machine learning algorithms to create a model of expected metric behavior. You can use anomaly detection to:

- Detect anomalies with minimal configuration
- Visualize expected metric behavior
- Create alarms that trigger when metrics deviate from expected patterns

### Creating an Anomaly Detection Alarm

To build an Anomaly Detection Alarm, you should create a MathExpression that
uses an `ANOMALY_DETECTION_BAND()` function, and use one of the band comparison
operators (see the next section). Anomaly Detection Alarms have a dynamic
threshold, not a fixed one, so the value for `threshold` is ignored. Specify the
value `0` or use the symbolic `Alarm.ANOMALY_DETECTION_NO_THRESHOLD` value.

You can use the `AnomalyDetectionAlarm` class for convenience, which takes care
of building the right metric math expression and passing in a magic value for
the treshold for you:

```ts
// Create a metric
const metric = new cloudwatch.Metric({
  namespace: 'AWS/EC2',
  metricName: 'CPUUtilization',
  statistic: 'Average',
  period: Duration.hours(1), // Alarm will use the metric's period
});

// Create an anomaly detection alarm
const alarm = new cloudwatch.AnomalyDetectionAlarm(this, 'AnomalyAlarm', {
  metric: metric,
  evaluationPeriods: 1,

  // Number of standard deviations for the band (default: 2)
  stdDevs: 2,
  // Alarm outside on either side of the band, or just below or above it (default: outside)
  comparisonOperator: cloudwatch.ComparisonOperator.LESS_THAN_LOWER_OR_GREATER_THAN_UPPER_THRESHOLD,
  alarmDescription: 'Alarm when metric is outside the expected band',
});
```

### Comparison Operators for Anomaly Detection

When creating an anomaly detection alarm, you must use one of the following comparison operators:

- `LESS_THAN_LOWER_OR_GREATER_THAN_UPPER_THRESHOLD`: Alarm when the metric is outside the band, on either side of it
- `GREATER_THAN_UPPER_THRESHOLD`: Alarm only when the metric is above the band
- `LESS_THAN_LOWER_THRESHOLD`: Alarm only when the metric is below the band

For more information on anomaly detection in CloudWatch, see the [AWS documentation](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/CloudWatch_Anomaly_Detection.html).

## PromQL Alarms

PromQL alarms evaluate a [PromQL](https://prometheus.io/docs/prometheus/latest/querying/basics/) instant query against metrics ingested through the CloudWatch OTLP endpoint on a recurring schedule. All matching time series returned by the query are considered breaching, and each is tracked independently as a _contributor_.

### How PromQL Alarms Differ from Standard CloudWatch Alarms

The existing `Alarm` construct is designed around the standard CloudWatch alarm model: a metric, a comparison operator, a threshold, and evaluation periods. PromQL alarms have a fundamentally different configuration model:

- Instead of using alarm condition (threshold + comparison), all matching time series returned by the PromQL alarm query are considered to be breaching.
- Instead of `evaluationPeriods` and `period`, PromQL alarms use `evaluationInterval`, `pendingPeriod`, and `recoveryPeriod`.
- PromQL alarms do not use `MetricName`, `Namespace`, `Statistic`, `Dimensions`, or `Threshold` properties.
- PromQL alarms only work with metrics ingested through the CloudWatch OTLP endpoint. Standard CloudWatch namespace metrics (e.g., `AWS/EC2`, `AWS/Lambda`) are not queryable via PromQL.

### Creating a PromQL Alarm

```ts
new cloudwatch.PromQLAlarm(this, 'HighCpuAlarm', {
  alarmName: 'HighCpuAlarm',
  alarmDescription: 'Alarm when average CPU exceeds 80%',
  query: 'avg(cpu_utilization_percent) > 80',
  evaluationInterval: Duration.seconds(60),
});
```

### Pending and Recovery Periods

Use `pendingPeriod` to require a contributor to breach continuously for a duration before entering ALARM state (equivalent to Prometheus `for` duration). Use `recoveryPeriod` to require a contributor to stop breaching for a duration before returning to OK state (equivalent to Prometheus `keep_firing_for` duration).

```ts
new cloudwatch.PromQLAlarm(this, 'HighLatencyAlarm', {
  alarmDescription: 'P99 latency exceeds 500ms for 5 minutes',
  query: 'histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m])) > 0.5',
  evaluationInterval: Duration.seconds(60),
  pendingPeriod: Duration.seconds(300),
  recoveryPeriod: Duration.seconds(600),
});
```

### Adding Actions

`PromQLAlarm` extends `AlarmBase`, so you can add alarm, OK, and insufficient-data actions the same way as standard alarms:

```ts
import * as cloudwatch_actions from 'aws-cdk-lib/aws-cloudwatch-actions';

declare const topic: sns.Topic;

const alarm = new cloudwatch.PromQLAlarm(this, 'ServiceDownAlarm', {
  query: 'up == 0',
  evaluationInterval: Duration.seconds(60),
  pendingPeriod: Duration.seconds(300),
  recoveryPeriod: Duration.seconds(300),
});

alarm.addAlarmAction(new cloudwatch_actions.SnsAction(topic));
alarm.addOkAction(new cloudwatch_actions.SnsAction(topic));
```

### Importing an Existing PromQL Alarm

```ts
const importedByArn = cloudwatch.PromQLAlarm.fromPromQLAlarmArn(
  this, 'ImportedAlarm',
  'arn:aws:cloudwatch:us-east-1:123456789012:alarm:MyPromQLAlarm',
);

const importedByName = cloudwatch.PromQLAlarm.fromPromQLAlarmName(
  this, 'ImportedByName',
  'MyPromQLAlarm',
);
```

### Using in Composite Alarms

Since `PromQLAlarm` implements `IAlarm`, it can be used in composite alarm rules alongside standard alarms:

```ts
declare const promqlAlarm: cloudwatch.PromQLAlarm;
declare const standardAlarm: cloudwatch.Alarm;

new cloudwatch.CompositeAlarm(this, 'CompositeAlarm', {
  alarmRule: cloudwatch.AlarmRule.allOf(promqlAlarm, standardAlarm),
});
```

## Alarm Mute Rules

Alarm Mute Rules is a CloudWatch feature that provides you a mechanism to
automatically mute alarm actions during predefined time windows.
When you create a mute rule, you define specific time periods and target
alarms whose actions will be muted.
CloudWatch will continue monitoring and evaluating alarm states while preventing
unwanted notifications or automated alarm actions during expected operational events.

```ts
declare const alarm1: cloudwatch.Alarm;
declare const alarm2: cloudwatch.Alarm;

const alarmMuteRule = new cloudwatch.AlarmMuteRule(this, 'AlarmMuteRule', {
  alarms: [alarm1],
  // Defines the mute period begins at 0:00 everyday in UTC
  schedule: cloudwatch.ScheduleExpression.cron({ minute: '0', hour: '0' }),
  // Specifies the mute rule lasts 1 hour.
  duration: Duration.hours(1),
})

// The mute target can be added after construction.
alarmMuteRule.addAlarm(alarm2);
```

For more information on alarm mute rules, see the [AWS documentation](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/alarm-mute-rules.html).

## Dashboards

Dashboards are set of Widgets stored server-side which can be accessed quickly
from the AWS console. Available widgets are graphs of a metric over time, the
current value of a metric, or a static piece of Markdown which explains what the
graphs mean.

The following widgets are available:

- `GraphWidget` -- shows any number of metrics on both the left and right
  vertical axes.
- `AlarmWidget` -- shows the graph and alarm line for a single alarm.
- `SingleValueWidget` -- shows the current value of a set of metrics.
- `TextWidget` -- shows some static Markdown.
- `AlarmStatusWidget` -- shows the status of your alarms in a grid view.

### Graph widget

A graph widget can display any number of metrics on either the `left` or
`right` vertical axis:

```ts
declare const dashboard: cloudwatch.Dashboard;
declare const executionCountMetric: cloudwatch.Metric;
declare const errorCountMetric: cloudwatch.Metric;

dashboard.addWidgets(new cloudwatch.GraphWidget({
  title: "Executions vs error rate",

  left: [executionCountMetric],

  right: [errorCountMetric.with({
    statistic: cloudwatch.Stats.AVERAGE,
    label: "Error rate",
    color: cloudwatch.Color.GREEN,
  })]
}));
```

Using the methods `addLeftMetric()` and `addRightMetric()` you can add metrics to a graph widget later on.

Graph widgets can also display annotations attached to the left or right y-axis or the x-axis.

```ts
declare const dashboard: cloudwatch.Dashboard;

dashboard.addWidgets(new cloudwatch.GraphWidget({
  // ...

  leftAnnotations: [
    { value: 1800, label: Duration.minutes(30).toHumanString(), color: cloudwatch.Color.RED, },
    { value: 3600, label: '1 hour', color: '#2ca02c', }
  ],
  verticalAnnotations: [
    { date: '2022-10-19T00:00:00Z', label: 'Deployment', color: cloudwatch.Color.RED, }
  ]
}));
```

The graph legend can be adjusted from the default position at bottom of the widget.

```ts
declare const dashboard: cloudwatch.Dashboard;

dashboard.addWidgets(new cloudwatch.GraphWidget({
  // ...

  legendPosition: cloudwatch.LegendPosition.RIGHT,
}));
```

The graph can publish live data within the last minute that has not been fully aggregated.

```ts
declare const dashboard: cloudwatch.Dashboard;

dashboard.addWidgets(new cloudwatch.GraphWidget({
  // ...

  liveData: true,
}));
```

The graph view can be changed from default 'timeSeries' to 'bar' or 'pie'.

```ts
declare const dashboard: cloudwatch.Dashboard;

dashboard.addWidgets(new cloudwatch.GraphWidget({
  // ...

  view: cloudwatch.GraphWidgetView.BAR,
}));
```

The `displayLabelsOnChart` property can be set to `true` to show labels on the chart. Note that this only has an effect when the `view` property is set to `cloudwatch.GraphWidgetView.PIE`.

```ts
declare const dashboard: cloudwatch.Dashboard;

dashboard.addWidgets(new cloudwatch.GraphWidget({
  // ...

  view: cloudwatch.GraphWidgetView.PIE,
  displayLabelsOnChart: true,
}));
```

The `start` and `end` properties can be used to specify the time range for each graph widget independently from those of the dashboard.
The parameters can be specified at `GraphWidget`, `GaugeWidget`, and `SingleValueWidget`.

```ts
declare const dashboard: cloudwatch.Dashboard;

dashboard.addWidgets(new cloudwatch.GraphWidget({
  // ...

  start: '-P7D',
  end: '2018-12-17T06:00:00.000Z',
}));
```

### Table Widget

A `TableWidget` can display any number of metrics in tabular form.

```ts
declare const dashboard: cloudwatch.Dashboard;
declare const executionCountMetric: cloudwatch.Metric;

dashboard.addWidgets(new cloudwatch.TableWidget({
  title: "Executions",
  metrics: [executionCountMetric],
}));
```

The `layout` property can be used to invert the rows and columns of the table.
The default `cloudwatch.TableLayout.HORIZONTAL` means that metrics are shown in rows and datapoints in columns.
`cloudwatch.TableLayout.VERTICAL` means that metrics are shown in columns and datapoints in rows.

```ts
declare const dashboard: cloudwatch.Dashboard;

dashboard.addWidgets(new cloudwatch.TableWidget({
  // ...

  layout: cloudwatch.TableLayout.VERTICAL,
}));
```

The `summary` property allows customizing the table to show summary columns (`columns` sub property),
whether to make the summary columns sticky remaining in view while scrolling (`sticky` sub property),
and to optionally only present summary columns (`hideNonSummaryColumns` sub property).

```ts
declare const dashboard: cloudwatch.Dashboard;

dashboard.addWidgets(new cloudwatch.TableWidget({
  // ...

  summary: {
    columns: [cloudwatch.TableSummaryColumn.AVERAGE],
    hideNonSummaryColumns: true,
    sticky: true,
  },
}));
```

The `thresholds` property can be used to highlight cells with a color when the datapoint value falls within the threshold.

```ts
declare const dashboard: cloudwatch.Dashboard;

dashboard.addWidgets(new cloudwatch.TableWidget({
  // ...

  thresholds: [
    cloudwatch.TableThreshold.above(1000, cloudwatch.Color.RED),
    cloudwatch.TableThreshold.between(500, 1000, cloudwatch.Color.ORANGE),
    cloudwatch.TableThreshold.below(500, cloudwatch.Color.GREEN),
  ],
}));
```

The `showUnitsInLabel` property can be used to display what unit is associated with a metric in the label column.

```ts
declare const dashboard: cloudwatch.Dashboard;

dashboard.addWidgets(new cloudwatch.TableWidget({
  // ...

  showUnitsInLabel: true,
}));
```

The `fullPrecision` property can be used to show as many digits as can fit in a cell, before rounding.

```ts
declare const dashboard: cloudwatch.Dashboard;

dashboard.addWidgets(new cloudwatch.TableWidget({
  // ...

  fullPrecision: true,
}));
```

### Gauge widget

Gauge graph requires the max and min value of the left Y axis, if no value is informed the limits will be from 0 to 100.

```ts
declare const dashboard: cloudwatch.Dashboard;
declare const errorAlarm: cloudwatch.Alarm;
declare const gaugeMetric: cloudwatch.Metric;

dashboard.addWidgets(new cloudwatch.GaugeWidget({
  metrics: [gaugeMetric],
  leftYAxis: {
    min: 0,
    max: 1000,
  }
}));
```

### Alarm widget

An alarm widget shows the graph and the alarm line of a single alarm:

```ts
declare const dashboard: cloudwatch.Dashboard;
declare const errorAlarm: cloudwatch.Alarm;

dashboard.addWidgets(new cloudwatch.AlarmWidget({
  title: "Errors",
  alarm: errorAlarm,
}));
```

### Single value widget

A single-value widget shows the latest value of a set of metrics (as opposed
to a graph of the value over time):

```ts
declare const dashboard: cloudwatch.Dashboard;
declare const visitorCount: cloudwatch.Metric;
declare const purchaseCount: cloudwatch.Metric;

dashboard.addWidgets(new cloudwatch.SingleValueWidget({
  metrics: [visitorCount, purchaseCount],
}));
```

Show as many digits as can fit, before rounding.


```ts
declare const dashboard: cloudwatch.Dashboard;

dashboard.addWidgets(new cloudwatch.SingleValueWidget({
  metrics: [ /* ... */ ],

  fullPrecision: true,
}));
```

Sparkline allows you to glance the trend of a metric by displaying a simplified linegraph below the value. You can't use `sparkline: true` together with `setPeriodToTimeRange: true`

```ts
declare const dashboard: cloudwatch.Dashboard;

dashboard.addWidgets(new cloudwatch.SingleValueWidget({
  metrics: [ /* ... */ ],

  sparkline: true,
}));
```

Period allows you to set the default period for the widget:

```ts
declare const dashboard: cloudwatch.Dashboard;

dashboard.addWidgets(new cloudwatch.SingleValueWidget({
  metrics: [ /* ... */ ],

  period: Duration.minutes(15),
}));
```

### Text widget

A text widget shows an arbitrary piece of MarkDown. Use this to add explanations
to your dashboard:

```ts
declare const dashboard: cloudwatch.Dashboard;

dashboard.addWidgets(new cloudwatch.TextWidget({
  markdown: '# Key Performance Indicators'
}));
```

Optionally set the TextWidget background to be transparent

```ts
declare const dashboard: cloudwatch.Dashboard;

dashboard.addWidgets(new cloudwatch.TextWidget({
  markdown: '# Key Performance Indicators',
  background: cloudwatch.TextWidgetBackground.TRANSPARENT
}));
```

### Alarm Status widget

An alarm status widget displays instantly the status of any type of alarms and gives the
ability to aggregate one or more alarms together in a small surface.

```ts
declare const dashboard: cloudwatch.Dashboard;
declare const errorAlarm: cloudwatch.Alarm;

dashboard.addWidgets(
  new cloudwatch.AlarmStatusWidget({
    alarms: [errorAlarm],
  })
);
```

An alarm status widget only showing firing alarms, sorted by state and timestamp:

```ts
declare const dashboard: cloudwatch.Dashboard;
declare const errorAlarm: cloudwatch.Alarm;

dashboard.addWidgets(new cloudwatch.AlarmStatusWidget({
  title: "Errors",
  alarms: [errorAlarm],
  sortBy: cloudwatch.AlarmStatusWidgetSortBy.STATE_UPDATED_TIMESTAMP,
  states: [cloudwatch.AlarmState.ALARM],
}));
```

### Query results widget

A `LogQueryWidget` shows the results of a query from Logs Insights:

```ts
declare const dashboard: cloudwatch.Dashboard;

dashboard.addWidgets(new cloudwatch.LogQueryWidget({
  logGroupNames: ['my-log-group'],
  view: cloudwatch.LogQueryVisualizationType.TABLE,
  // The lines will be automatically combined using '\n|'.
  queryLines: [
    'fields @message',
    'filter @message like /Error/',
  ]
}));
```

Log Insights QL is the default query language. You may specify an [alternate query language: OpenSearch PPL or SQL](https://aws.amazon.com/blogs/aws/new-amazon-cloudwatch-and-amazon-opensearch-service-launch-an-integrated-analytics-experience/), if desired:

```ts
declare const dashboard: cloudwatch.Dashboard;

dashboard.addWidgets(new cloudwatch.LogQueryWidget({
  logGroupNames: ['my-log-group'],
  view: cloudwatch.LogQueryVisualizationType.TABLE,
  queryString: "SELECT count(*) as count FROM 'my-log-group'",
  queryLanguage: cloudwatch.LogQueryLanguage.SQL,
}));
```

### Custom widget

A `CustomWidget` shows the result of an AWS Lambda function:

```ts
declare const dashboard: cloudwatch.Dashboard;

// Import or create a lambda function
const fn = lambda.Function.fromFunctionArn(
  dashboard,
  'Function',
  'arn:aws:lambda:us-east-1:123456789012:function:MyFn',
);

dashboard.addWidgets(new cloudwatch.CustomWidget({
  functionArn: fn.functionArn,
  title: 'My lambda baked widget',
}));
```

You can learn more about custom widgets in the [Amazon Cloudwatch User Guide](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/add_custom_widget_dashboard.html).

### Dashboard Layout

The widgets on a dashboard are visually laid out in a grid that is 24 columns
wide. Normally you specify X and Y coordinates for the widgets on a Dashboard,
but because this is inconvenient to do manually, the library contains a simple
layout system to help you lay out your dashboards the way you want them to.

Widgets have a `width` and `height` property, and they will be automatically
laid out either horizontally or vertically stacked to fill out the available
space.

Widgets are added to a Dashboard by calling `add(widget1, widget2, ...)`.
Widgets given in the same call will be laid out horizontally. Widgets given
in different calls will be laid out vertically. To make more complex layouts,
you can use the following widgets to pack widgets together in different ways:

- `Column`: stack two or more widgets vertically.
- `Row`: lay out two or more widgets horizontally.
- `Spacer`: take up empty space

### Column widget

A column widget contains other widgets and they will be laid out in a
vertical column. Widgets will be put one after another in order.

```ts
declare const widgetA: cloudwatch.IWidget;
declare const widgetB: cloudwatch.IWidget;

new cloudwatch.Column(widgetA, widgetB);
```

You can add a widget after object instantiation with the method
`addWidget()`. Each new widget will be put at the bottom of the column.

### Row widget

A row widget contains other widgets and they will be laid out in a
horizontal row. Widgets will be put one after another in order.
If the total width of the row exceeds the max width of the grid of 24
columns, the row will wrap automatically and adapt its height.

```ts
declare const widgetA: cloudwatch.IWidget;
declare const widgetB: cloudwatch.IWidget;

new cloudwatch.Row(widgetA, widgetB);
```

You can add a widget after object instantiation with the method
`addWidget()`.

### Interval duration for dashboard

Interval duration for metrics in dashboard. You can specify `defaultInterval` with
the relative time (e.g. 7 days) as `Duration.days(7)`.

```ts
import * as cw from 'aws-cdk-lib/aws-cloudwatch';

const dashboard = new cw.Dashboard(this, 'Dash', {
  defaultInterval: Duration.days(7),
});
```

Here, the dashboard would show the metrics for the last 7 days.

### Dashboard variables

Dashboard variables are a convenient way to create flexible dashboards that display different content depending
on the value of an input field within a dashboard. They create a dashboard on which it's possible to quickly switch between
different Lambda functions, Amazon EC2 instances, etc.

You can learn more about Dashboard variables in the [Amazon Cloudwatch User Guide](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/cloudwatch_dashboard_variables.html)

There are two types of dashboard variables available: a property variable and a pattern variable.
- Property variables can change any JSON property in the JSON source of a dashboard like `region`. It can also change the dimension name for a metric.
- Pattern variables use a regular expression pattern to change all or part of a JSON property.

A use case of a **property variable** is a dashboard with the ability to toggle the `region` property to see the same dashboard in different regions:

```ts
import * as cw from 'aws-cdk-lib/aws-cloudwatch';

const dashboard = new cw.Dashboard(this, 'Dash', {
  defaultInterval: Duration.days(7),
  variables: [new cw.DashboardVariable({
    id: 'region',
    type: cw.VariableType.PROPERTY,
    label: 'Region',
    inputType: cw.VariableInputType.RADIO,
    value: 'region',
    values: cw.Values.fromValues({ label: 'IAD', value: 'us-east-1' }, { label: 'DUB', value: 'us-west-2' }),
    defaultValue: cw.DefaultValue.value('us-east-1'),
    visible: true,
  })],
});
```

This example shows how to change `region` everywhere, assuming the current dashboard is showing region `us-east-1` already, by using **pattern variable**

```ts
import * as cw from 'aws-cdk-lib/aws-cloudwatch';

const dashboard = new cw.Dashboard(this, 'Dash', {
  defaultInterval: Duration.days(7),
  variables: [new cw.DashboardVariable({
    id: 'region2',
    type: cw.VariableType.PATTERN,
    label: 'RegionPattern',
    inputType: cw.VariableInputType.INPUT,
    value: 'us-east-1',
    defaultValue: cw.DefaultValue.value('us-east-1'),
    visible: true,
  })],
});
```

The following example generates a Lambda function variable, with a radio button for each function. Functions are discovered by a metric query search.
The `values` with `cw.Values.fromSearchComponents` indicates that the values will be populated from `FunctionName` values retrieved from the search expression `{AWS/Lambda,FunctionName} MetricName=\"Duration\"`.
The `defaultValue` with `cw.DefaultValue.FIRST` indicates that the default value will be the first value returned from the search.

```ts
import * as cw from 'aws-cdk-lib/aws-cloudwatch';

const dashboard = new cw.Dashboard(this, 'Dash', {
  defaultInterval: Duration.days(7),
  variables: [new cw.DashboardVariable({
    id: 'functionName',
    type: cw.VariableType.PATTERN,
    label: 'Function',
    inputType: cw.VariableInputType.RADIO,
    value: 'originalFuncNameInDashboard',
    // equivalent to cw.Values.fromSearch('{AWS/Lambda,FunctionName} MetricName=\"Duration\"', 'FunctionName')
    values: cw.Values.fromSearchComponents({
      namespace: 'AWS/Lambda',
      dimensions: ['FunctionName'],
      metricName: 'Duration',
      populateFrom: 'FunctionName',
    }),
    defaultValue: cw.DefaultValue.FIRST,
    visible: true,
  })],
});
```

You can add a variable after object instantiation with the method `dashboard.addVariable()`.

### Cross-Account Visibility

Both Log and Metric Widget objects support cross-account visibility by allowing you to specify the AWS Account ID that the data (logs or metrics) originates from.

**Prerequisites:**
1. The monitoring account must be set up as a monitoring account
2. The source account must grant permissions to the monitoring account
3. Appropriate IAM roles and policies must be configured

For detailed setup instructions, see [Cross-Account Cross-Region CloudWatch Console](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/Cross-Account-Cross-Region.html).


To use this feature, you can set the `accountId` property on `LogQueryWidget`, `GraphWidget`, `AlarmWidget`, `SingleValueWidget`, and `GaugeWidget` constructs:

```ts
declare const dashboard: cloudwatch.Dashboard;

dashboard.addWidgets(new cloudwatch.GraphWidget({
  // ...
  accountId: '123456789012',
}));

dashboard.addWidgets(new cloudwatch.LogQueryWidget({
  logGroupNames: ['my-log-group'],
  // ...
  accountId: '123456789012',
  queryLines: [
    'fields @message',
  ],
}));
```

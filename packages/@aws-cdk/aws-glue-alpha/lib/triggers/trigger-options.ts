import * as events from 'aws-cdk-lib/aws-events';
import type { CfnCrawler } from 'aws-cdk-lib/aws-glue';
import type * as cdk from 'aws-cdk-lib/core';
import type { JobState, CrawlerState, ConditionLogicalOperator, PredicateLogical } from '../constants';
import type { IJob } from '../jobs/job'; // Use IJob interface instead of concrete class
import type { ISecurityConfiguration } from '../security-configuration';

/**
 * Represents a trigger action.
 */
export interface Action {
  /**
   * The job to be executed.
   *
   * @default - no job is executed
   */
  readonly job?: IJob;

  /**
   * The job arguments used when this trigger fires.
   *
   * @default - no arguments are passed to the job
   */
  readonly arguments?: { [key: string]: string };

  /**
   * The job run timeout. This is the maximum time that a job run can consume resources before it is terminated and enters TIMEOUT status.
   *
   * @default - the default timeout value set in the job definition
   */
  readonly timeout?: cdk.Duration;

  /**
   * The `SecurityConfiguration` to be used with this action.
   *
   * @default - no security configuration is used
   */
  readonly securityConfiguration?: ISecurityConfiguration;

  /**
   * The name of the crawler to be used with this action.
   *
   * @default - no crawler is used
   */
  readonly crawler?: CfnCrawler;
}

/**
 * Represents a trigger schedule.
 */
export class TriggerSchedule {
  /**
   * Creates a new TriggerSchedule instance with a cron expression.
   *
   * @param options The cron options for the schedule.
   * @returns A new TriggerSchedule instance.
   */
  public static cron(options: events.CronOptions): TriggerSchedule {
    return new TriggerSchedule(events.Schedule.cron(options).expressionString);
  }

  /**
   * Creates a new TriggerSchedule instance with a custom expression.
   *
   * @param expression The custom expression for the schedule.
   * @returns A new TriggerSchedule instance.
   */
  public static expression(expression: string): TriggerSchedule {
    return new TriggerSchedule(expression);
  }

  /**
   * @param expressionString The expression string for the schedule.
   */
  private constructor(public readonly expressionString: string) {}
}

/**
 * Represents a trigger predicate.
 */
export interface Predicate {
  /**
   * The logical operator to be applied to the conditions.
   *
   * @default - PredicateLogical.AND if multiple conditions are provided, no logical operator if only one condition
   */
  readonly logical?: PredicateLogical;

  /**
   * A list of the conditions that determine when the trigger will fire.
   *
   * @default - no conditions are provided
   */
  readonly conditions?: Condition[];
}

/**
 * Represents a trigger condition.
 */
export interface Condition {
  /**
   * The logical operator for the condition.
   *
   * @default ConditionLogicalOperator.EQUALS
   */
  readonly logicalOperator?: ConditionLogicalOperator;

  /**
   * The job to which this condition applies.
   *
   * @default - no job is specified
   */
  readonly job?: IJob;

  /**
   * The condition job state.
   *
   * @default - no job state is specified
   */
  readonly state?: JobState;

  /**
   * The name of the crawler to which this condition applies.
   *
   * @default - no crawler is specified
   */
  readonly crawlerName?: string;

  /**
   * The condition crawler state.
   *
   * @default - no crawler state is specified
   */
  readonly crawlState?: CrawlerState;
}

/**
 * Represents event trigger batch condition.
 */
export interface EventBatchingCondition {
  /**
   * Number of events that must be received from Amazon EventBridge before EventBridge event trigger fires.
   */
  readonly batchSize: number;

  /**
   * Window of time in seconds after which EventBridge event trigger fires.
   *
   * @default - 900 seconds
   */
  readonly batchWindow?: cdk.Duration;
}

/**
 * Properties for configuring a Glue Trigger
 */
export interface TriggerOptions {
  /**
   * A name for the trigger.
   *
   * @default - no name is provided
   */
  readonly name?: string;

  /**
   * A description for the trigger.
   *
   * @default - no description
   */
  readonly description?: string;

  /**
   * The actions initiated by this trigger.
   */
  readonly actions: Action[];
}

/**
 * Properties for configuring an on-demand Glue Trigger.
 */
export interface OnDemandTriggerOptions extends TriggerOptions {}

/**
 * Properties for configuring a daily-scheduled Glue Trigger.
 */
export interface DailyScheduleTriggerOptions extends TriggerOptions {
  /**
   * Whether to start the trigger on creation or not.
   *
   * @default - false
   */
  readonly startOnCreation?: boolean;
}

/**
 * Properties for configuring a weekly-scheduled Glue Trigger.
 */
export interface WeeklyScheduleTriggerOptions extends DailyScheduleTriggerOptions {}

/**
 * Properties for configuring a custom-scheduled Glue Trigger.
 */
export interface CustomScheduledTriggerOptions extends WeeklyScheduleTriggerOptions {
  /**
   * The custom schedule for the trigger.
   */
  readonly schedule: TriggerSchedule;
}

/**
 * Properties for configuring an Event Bridge based Glue Trigger.
 */
export interface NotifyEventTriggerOptions extends TriggerOptions {
  /**
   * Batch condition for the trigger.
   *
   * @default - no batch condition
   */
  readonly eventBatchingCondition?: EventBatchingCondition;
}

/**
 * Properties for configuring a Condition (Predicate) based Glue Trigger.
 */
export interface ConditionalTriggerOptions extends DailyScheduleTriggerOptions{
  /**
   * The predicate for the trigger.
   */
  readonly predicate: Predicate;
}

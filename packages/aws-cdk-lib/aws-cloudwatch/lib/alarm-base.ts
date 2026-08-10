import type { IAlarmAction } from './alarm-action';
import type { AlarmMuteRuleOptions } from './alarm-mute-rule';
import { AlarmMuteRule } from './alarm-mute-rule';
import type { IResource } from '../../core';
import { Resource } from '../../core';
import type { IArrayBox } from '../../core/lib/helpers-internal';
import { Box } from '../../core/lib/helpers-internal';
import type { IAlarmRef, AlarmReference } from '../../interfaces/generated/aws-cloudwatch-interfaces.generated';

/**
 * Interface for Alarm Rule.
 */
export interface IAlarmRule {

  /**
   * serialized representation of Alarm Rule to be used when building the Composite Alarm resource.
   */
  renderAlarmRule(): string;

}

/**
 * Represents a CloudWatch Alarm
 */
export interface IAlarm extends IAlarmRule, IResource, IAlarmRef {
  /**
   * Alarm ARN (i.e. arn:aws:cloudwatch:<region>:<account-id>:alarm:Foo)
   *
   * @attribute
   */
  readonly alarmArn: string;

  /**
   * Name of the alarm
   *
   * @attribute
   */
  readonly alarmName: string;
}

/**
 * The base class for Alarm and CompositeAlarm resources.
 */
export abstract class AlarmBase extends Resource implements IAlarm {
  /**
   * @attribute
   */
  public abstract readonly alarmArn: string;
  public abstract readonly alarmName: string;

  /** @internal */
  protected readonly _alarmActionArns: IArrayBox<string> = Box.fromArray();
  /** @internal */
  protected readonly _insufficientDataActionArns: IArrayBox<string> = Box.fromArray();
  /** @internal */
  protected readonly _okActionArns: IArrayBox<string> = Box.fromArray();

  protected get alarmActionArns(): string[] | undefined {
    return this._alarmActionArns.length > 0 ? [...this._alarmActionArns] : undefined;
  }
  protected set alarmActionArns(value: string[] | undefined) {
    this._alarmActionArns.set(value ?? []);
  }
  protected get insufficientDataActionArns(): string[] | undefined {
    return this._insufficientDataActionArns.length > 0 ? [...this._insufficientDataActionArns] : undefined;
  }
  protected set insufficientDataActionArns(value: string[] | undefined) {
    this._insufficientDataActionArns.set(value ?? []);
  }
  protected get okActionArns(): string[] | undefined {
    return this._okActionArns.length > 0 ? [...this._okActionArns] : undefined;
  }
  protected set okActionArns(value: string[] | undefined) {
    this._okActionArns.set(value ?? []);
  }

  public get alarmRef(): AlarmReference {
    return {
      alarmName: this.alarmName,
      alarmArn: this.alarmArn,
    };
  }

  /**
   * AlarmRule indicating ALARM state for Alarm.
   */
  public renderAlarmRule(): string {
    return `ALARM("${this.alarmArn}")`;
  }

  /**
   * Trigger this action if the alarm fires
   *
   * Typically SnsAction or AutoScalingAction.
   */
  public addAlarmAction(...actions: IAlarmAction[]) {
    this._alarmActionArns.push(...actions.map(a => a.bind(this, this).alarmActionArn));
  }

  /**
   * Trigger this action if there is insufficient data to evaluate the alarm
   *
   * Typically SnsAction or AutoScalingAction.
   */
  public addInsufficientDataAction(...actions: IAlarmAction[]) {
    this._insufficientDataActionArns.push(...actions.map(a => a.bind(this, this).alarmActionArn));
  }

  /**
   * Trigger this action if the alarm returns from breaching state into ok state
   *
   * Typically SnsAction or AutoScalingAction.
   */
  public addOkAction(...actions: IAlarmAction[]) {
    this._okActionArns.push(...actions.map(a => a.bind(this, this).alarmActionArn));
  }

  /**
   * Add an alarm mute rule.
   */
  public addAlarmMuteRule(id: string, options: AlarmMuteRuleOptions) {
    return new AlarmMuteRule(this, id, { ...options, alarms: [this] });
  }
}

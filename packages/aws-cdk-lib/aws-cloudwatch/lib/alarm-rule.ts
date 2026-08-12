import type { IAlarm, IAlarmRule } from './alarm-base';
import { Token, Tokenization, UnscopedValidationError } from '../../core';
import { lit } from '../../core/lib/private/literal-string';
import type { IAlarmRef } from '../../interfaces/generated/aws-cloudwatch-interfaces.generated';

/**
 * Enumeration indicates state of Alarm used in building Alarm Rule.
 */
export enum AlarmState {
  /**
   * State indicates resource is in ALARM
   */
  ALARM = 'ALARM',

  /**
   * State indicates resource is not in ALARM
   */
  OK = 'OK',

  /**
   * State indicates there is not enough data to determine is resource is in ALARM
   */
  INSUFFICIENT_DATA = 'INSUFFICIENT_DATA',
}

/**
 * Enumeration of supported Composite Alarms operators.
 */
enum Operator {
  AND = 'AND',
  OR = 'OR',
  NOT = 'NOT',
}

/**
 * Options for the AT_LEAST AlarmRule wrapper function.
 */
export interface AtLeastOptions {
  /**
   * Alarms to evaluate in the AT_LEAST expression.
   *
   * Must contain at least one alarm.
   */
  readonly operands: IAlarm[];

  /**
   * Threshold for the AT_LEAST expression.
   *
   * Use `AtLeastThreshold.count()` for an absolute number
   * or `AtLeastThreshold.percentage()` for a percentage.
   */
  readonly threshold: AtLeastThreshold;
}

/**
 * Internal configuration returned by threshold binding.
 */
interface AtLeastThresholdConfig {
  /**
   * The threshold as a string for the AT_LEAST expression.
   *
   * Either a number (e.g. "2") or a percentage (e.g. "60%").
   */
  readonly threshold: string;
}

/**
 * Threshold configuration for the AT_LEAST composite alarm rule expression.
 *
 * Use `AtLeastThreshold.count()` for an absolute number or
 * `AtLeastThreshold.percentage()` for a percentage-based threshold.
 */
export abstract class AtLeastThreshold {
  /**
   * Creates a count-based threshold for the AT_LEAST expression.
   *
   * The count must be a positive integer between 1 and the number of operands.
   *
   * @param count the minimum number of alarms that must be in the specified state
   */
  public static count(count: number): AtLeastThreshold {
    return new AtLeastThresholdCount(count);
  }

  /**
   * Creates a percentage-based threshold for the AT_LEAST expression.
   *
   * The percentage must be an integer between 1 and 100.
   *
   * @param percentage the minimum percentage of alarms that must be in the specified state
   */
  public static percentage(percentage: number): AtLeastThreshold {
    return new AtLeastThresholdPercentage(percentage);
  }

  /**
   * Called when the alarm rule is rendered to bind the threshold and validate.
   *
   * @internal
   */
  public abstract _bind(operands: IAlarm[]): AtLeastThresholdConfig;
}

/** @internal */
class AtLeastThresholdCount extends AtLeastThreshold {
  constructor(private readonly count: number) {
    super();
  }

  /**
   * @internal
   */
  public _bind(operands: IAlarm[]): AtLeastThresholdConfig {
    if (Token.isUnresolved(this.count)) {
      return { threshold: Tokenization.stringifyNumber(this.count) };
    }

    if (this.count < 1 || operands.length < this.count || !Number.isInteger(this.count)) {
      throw new UnscopedValidationError(
        lit`InvalidAtLeastCount`,
        `count must be an integer between 1 and the number of operands (${operands.length}), got ${this.count}`,
      );
    }

    return { threshold: `${this.count}` };
  }
}

/** @internal */
class AtLeastThresholdPercentage extends AtLeastThreshold {
  constructor(private readonly percentage: number) {
    super();
  }

  /**
   * @internal
   */
  public _bind(_operands: IAlarm[]): AtLeastThresholdConfig {
    if (Token.isUnresolved(this.percentage)) {
      return { threshold: `${Tokenization.stringifyNumber(this.percentage)}%` };
    }

    if (this.percentage < 1 || 100 < this.percentage || !Number.isInteger(this.percentage)) {
      throw new UnscopedValidationError(
        lit`InvalidAtLeastPercentage`,
        `percentage must be an integer between 1 and 100, got ${this.percentage}`,
      );
    }

    return { threshold: `${this.percentage}%` };
  }
}

/**
 * Class with static functions to build AlarmRule for Composite Alarms.
 */
export class AlarmRule {
  /**
   * function to join all provided AlarmRules with AND operator.
   *
   * @param operands IAlarmRules to be joined with AND operator.
   */
  public static allOf(...operands: IAlarmRule[]): IAlarmRule {
    return this.concat(Operator.AND, ...operands);
  }

  /**
   * function to join all provided AlarmRules with OR operator.
   *
   * @param operands IAlarmRules to be joined with OR operator.
   */
  public static anyOf(...operands: IAlarmRule[]): IAlarmRule {
    return this.concat(Operator.OR, ...operands);
  }

  /**
   * function to wrap provided AlarmRule in NOT operator.
   *
   * @param operand IAlarmRule to be wrapped in NOT operator.
   */
  public static not(operand: IAlarmRule): IAlarmRule {
    return new class implements IAlarmRule {
      public renderAlarmRule(): string {
        return `(NOT (${operand.renderAlarmRule()}))`;
      }
    };
  }

  /**
   * function to create an AT_LEAST expression for the given alarm state.
   *
   * @param alarmState the alarm state to evaluate against.
   * @param options operands and threshold for the AT_LEAST expression.
   */
  public static atLeast(alarmState: AlarmState, options: AtLeastOptions): IAlarmRule {
    return this.renderAtLeast(`${alarmState}`, options);
  }

  /**
   * function to create an AT_LEAST expression for the negated alarm state.
   *
   * @param alarmState the alarm state to negate and evaluate against.
   * @param options operands and threshold for the AT_LEAST expression.
   */
  public static atLeastNot(alarmState: AlarmState, options: AtLeastOptions): IAlarmRule {
    return this.renderAtLeast(`${Operator.NOT} ${alarmState}`, options);
  }

  /**
   * function to build TRUE/FALSE intent for Rule Expression.
   *
   * @param value boolean value to be used in rule expression.
   */
  public static fromBoolean(value: boolean): IAlarmRule {
    return new class implements IAlarmRule {
      public renderAlarmRule(): string {
        return `${String(value).toUpperCase()}`;
      }
    };
  }

  /**
   * function to build Rule Expression for given IAlarm and AlarmState.
   *
   * @param alarm IAlarm to be used in Rule Expression.
   * @param alarmState AlarmState to be used in Rule Expression.
   */
  public static fromAlarm(alarm: IAlarmRef, alarmState: AlarmState): IAlarmRule {
    return new class implements IAlarmRule {
      public renderAlarmRule(): string {
        return `${alarmState}("${alarm.alarmRef.alarmArn}")`;
      }
    };
  }

  /**
   * function to build Rule Expression for given Alarm Rule string.
   *
   * @param alarmRule string to be used in Rule Expression.
   */
  public static fromString(alarmRule: string): IAlarmRule {
    return new class implements IAlarmRule {
      public renderAlarmRule(): string {
        return alarmRule;
      }
    };
  }

  private static concat(operator: Operator, ...operands: IAlarmRule[]): IAlarmRule {
    return new class implements IAlarmRule {
      public renderAlarmRule(): string {
        if (operands.length === 0) {
          throw new UnscopedValidationError(
            lit`NoOperandsDetectedForAlarmRule`,
            `Did not detect any operands for AlarmRule.${operator === Operator.AND ? 'allOf' : 'anyOf'}()`,
          );
        }

        const expression = operands
          .map(operand => `${operand.renderAlarmRule()}`)
          .join(` ${operator} `);

        return `(${expression})`;
      }
    };
  }

  private static renderAtLeast(stateExpression: string, props: AtLeastOptions): IAlarmRule {
    return new class implements IAlarmRule {
      public renderAlarmRule(): string {
        if (props.operands.length === 0) {
          throw new UnscopedValidationError(
            lit`NoOperandsDetectedForAtLeast`,
            `Did not detect any operands for AT_LEAST ${stateExpression}`,
          );
        }

        const thresholdConfig = props.threshold._bind(props.operands);
        const concatAlarms = props.operands
          .map(operand => `"${operand.alarmArn}"`)
          .join(', ');

        return `AT_LEAST(${thresholdConfig.threshold}, ${stateExpression}, (${concatAlarms}))`;
      }
    };
  }
}

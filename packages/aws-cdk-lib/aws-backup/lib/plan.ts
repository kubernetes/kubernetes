import type { Construct } from 'constructs';
import { CfnBackupPlan } from './backup.generated';
import { toIBackupVault } from './private/ref-utils';
import type { BackupPlanCopyActionProps } from './rule';
import { BackupPlanRule } from './rule';
import type { BackupSelectionOptions } from './selection';
import { BackupSelection } from './selection';
import type { IBackupVault } from './vault';
import { BackupVault } from './vault';
import type { IResource } from '../../core';
import { ArnFormat, Resource, ValidationError } from '../../core';
import type { IArrayBox } from '../../core/lib/helpers-internal';
import { Box } from '../../core/lib/helpers-internal';
import { addConstructMetadata, MethodMetadata } from '../../core/lib/metadata-resource';
import { noBoxStackTraces } from '../../core/lib/no-box-stack-traces';
import { lit } from '../../core/lib/private/literal-string';
import { propertyInjectable } from '../../core/lib/prop-injectable';
import type { BackupPlanReference, IBackupPlanRef, IBackupVaultRef } from '../../interfaces/generated/aws-backup-interfaces.generated';

/**
 * A backup plan
 */
export interface IBackupPlan extends IResource, IBackupPlanRef {
  /**
   * The identifier of the backup plan.
   *
   * @attribute
   */
  readonly backupPlanId: string;
}

/**
 * Properties for a BackupPlan
 */
export interface BackupPlanProps {
  /**
   * The display name of the backup plan.
   *
   * @default - A CDK generated name
   */
  readonly backupPlanName?: string;

  /**
   * The backup vault where backups are stored
   *
   * @default - use the vault defined at the rule level. If not defined a new
   * common vault for the plan will be created
   */
  readonly backupVault?: IBackupVaultRef;

  /**
   * Rules for the backup plan. Use `addRule()` to add rules after
   * instantiation.
   *
   * @default - use `addRule()` to add rules
   */
  readonly backupPlanRules?: BackupPlanRule[];

  /**
   * Enable Windows VSS backup.
   *
   * @see https://docs.aws.amazon.com/aws-backup/latest/devguide/windows-backups.html
   *
   * @default false
   */
  readonly windowsVss?: boolean;
}

/**
 * A backup plan
 */
@propertyInjectable
@noBoxStackTraces
export class BackupPlan extends Resource implements IBackupPlan {
  /** Uniquely identifies this class. */
  public static readonly PROPERTY_INJECTION_ID: string = 'aws-cdk-lib.aws-backup.BackupPlan';

  /**
   * Import an existing backup plan
   */
  public static fromBackupPlanId(scope: Construct, id: string, backupPlanId: string): IBackupPlan {
    class Import extends Resource implements IBackupPlan {
      public readonly backupPlanId = backupPlanId;

      public get backupPlanRef(): BackupPlanReference {
        return {
          backupPlanId: this.backupPlanId,
          backupPlanArn: this.stack.formatArn({
            service: 'backup',
            resource: 'backup-plan',
            resourceName: this.backupPlanId,
            arnFormat: ArnFormat.COLON_RESOURCE_NAME,
          }),
        };
      }
    }
    return new Import(scope, id);
  }

  /**
   * Daily with 35 day retention
   */
  public static daily35DayRetention(scope: Construct, id: string, backupVault?: IBackupVaultRef) {
    const plan = new BackupPlan(scope, id, { backupVault });
    plan.addRule(BackupPlanRule.daily());
    return plan;
  }

  /**
   * Daily and monthly with 1 year retention
   */
  public static dailyMonthly1YearRetention(scope: Construct, id: string, backupVault?: IBackupVaultRef) {
    const plan = new BackupPlan(scope, id, { backupVault });
    plan.addRule(BackupPlanRule.daily());
    plan.addRule(BackupPlanRule.monthly1Year());
    return plan;
  }

  /**
   * Daily, weekly and monthly with 5 year retention
   */
  public static dailyWeeklyMonthly5YearRetention(scope: Construct, id: string, backupVault?: IBackupVaultRef) {
    const plan = new BackupPlan(scope, id, { backupVault });
    plan.addRule(BackupPlanRule.daily());
    plan.addRule(BackupPlanRule.weekly());
    plan.addRule(BackupPlanRule.monthly5Year());
    return plan;
  }

  /**
   * Daily, weekly and monthly with 7 year retention
   */
  public static dailyWeeklyMonthly7YearRetention(scope: Construct, id: string, backupVault?: IBackupVaultRef) {
    const plan = new BackupPlan(scope, id, { backupVault });
    plan.addRule(BackupPlanRule.daily());
    plan.addRule(BackupPlanRule.weekly());
    plan.addRule(BackupPlanRule.monthly7Year());
    return plan;
  }

  public readonly backupPlanId: string;

  /**
   * The ARN of the backup plan
   *
   * @attribute
   */
  public readonly backupPlanArn: string;

  /**
   * Version Id
   *
   * @attribute
   */
  public readonly versionId: string;

  public get backupPlanRef(): BackupPlanReference {
    return {
      backupPlanId: this.backupPlanId,
      backupPlanArn: this.backupPlanArn,
    };
  }

  private readonly rules: IArrayBox<CfnBackupPlan.BackupRuleResourceTypeProperty>;
  private _backupVault?: IBackupVaultRef;

  constructor(scope: Construct, id: string, props: BackupPlanProps = {}) {
    super(scope, id);
    // Enhanced CDK Analytics Telemetry
    addConstructMetadata(this, props);

    this.rules = Box.fromArray();

    const plan = new CfnBackupPlan(this, 'Resource', {
      backupPlan: {
        advancedBackupSettings: this.advancedBackupSettings(props),
        backupPlanName: props.backupPlanName || id,
        backupPlanRule: this.rules,
      },
    });

    this.backupPlanId = plan.attrBackupPlanId;
    this.backupPlanArn = plan.attrBackupPlanArn;
    this.versionId = plan.attrVersionId;

    this._backupVault = props.backupVault;

    for (const rule of props.backupPlanRules || []) {
      this.addRule(rule);
    }

    this.node.addValidation({ validate: () => this.validatePlan() });
  }

  private advancedBackupSettings(props: BackupPlanProps) {
    if (!props.windowsVss) {
      return undefined;
    }
    return [{
      backupOptions: {
        WindowsVSS: 'enabled',
      },
      resourceType: 'EC2',
    }];
  }

  /**
   * Adds a rule to a plan
   *
   * @param rule the rule to add
   */
  @MethodMetadata()
  public addRule(rule: BackupPlanRule) {
    let vault: IBackupVaultRef;
    if (rule.props.backupVault) {
      vault = rule.props.backupVault;
    } else if (this._backupVault) {
      vault = this._backupVault;
    } else {
      this._backupVault = new BackupVault(this, 'Vault');
      vault = this._backupVault;
    }

    this.rules.push({
      completionWindowMinutes: rule.props.completionWindow?.toMinutes(),
      lifecycle: (rule.props.deleteAfter || rule.props.moveToColdStorageAfter) && {
        deleteAfterDays: rule.props.deleteAfter?.toDays(),
        moveToColdStorageAfterDays: rule.props.moveToColdStorageAfter?.toDays(),
      },
      ruleName: rule.props.ruleName ?? `${this.node.id}Rule${this.rules.length}`,
      scheduleExpression: rule.props.scheduleExpression?.expressionString,
      scheduleExpressionTimezone: rule.props.scheduleExpressionTimezone?.timezoneName,
      startWindowMinutes: rule.props.startWindow?.toMinutes(),
      enableContinuousBackup: rule.props.enableContinuousBackup,
      targetBackupVault: vault.backupVaultRef.backupVaultName,
      copyActions: rule.props.copyActions?.map(this.planCopyActions),
      recoveryPointTags: rule.props.recoveryPointTags,
      indexActions: rule.props.indexActions?.map(indexAction => ({
        resourceTypes: indexAction.resourceTypes.map(resourceType => resourceType.value),
      })),
    });
  }

  private planCopyActions(this: void, props: BackupPlanCopyActionProps): CfnBackupPlan.CopyActionResourceTypeProperty {
    return {
      destinationBackupVaultArn: props.destinationBackupVault.backupVaultRef.backupVaultArn,
      lifecycle: (props.deleteAfter || props.moveToColdStorageAfter) && {
        deleteAfterDays: props.deleteAfter?.toDays(),
        moveToColdStorageAfterDays: props.moveToColdStorageAfter?.toDays(),
      },
    };
  }

  /**
   * The backup vault where backups are stored if not defined at
   * the rule level
   */
  public get backupVault(): IBackupVault {
    if (!this._backupVault) {
      // This cannot happen but is here to make TypeScript happy
      throw new ValidationError(lit`BackupVault`, 'No backup vault!', this);
    }

    return toIBackupVault(this._backupVault);
  }

  /**
   * Adds a selection to this plan
   */
  @MethodMetadata()
  public addSelection(id: string, options: BackupSelectionOptions): BackupSelection {
    return new BackupSelection(this, id, {
      backupPlan: this,
      ...options,
    });
  }

  private validatePlan() {
    if (this.rules.length === 0) {
      return ['A backup plan must have at least 1 rule.'];
    }

    return [];
  }
}

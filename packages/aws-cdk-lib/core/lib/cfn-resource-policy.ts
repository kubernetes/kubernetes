/**
 * Associate the CreationPolicy attribute with a resource to prevent its status from reaching create complete until
 * AWS CloudFormation receives a specified number of success signals or the timeout period is exceeded. To signal a
 * resource, you can use the cfn-signal helper script or SignalResource API. AWS CloudFormation publishes valid signals
 * to the stack events so that you track the number of signals sent.
 *
 * The creation policy is invoked only when AWS CloudFormation creates the associated resource. Currently, the only
 * AWS CloudFormation resources that support creation policies are AWS::AutoScaling::AutoScalingGroup, AWS::EC2::Instance,
 * AWS::CloudFormation::WaitCondition and AWS::AppStream::Fleet.
 *
 * Use the CreationPolicy attribute when you want to wait on resource configuration actions before stack creation proceeds.
 * For example, if you install and configure software applications on an EC2 instance, you might want those applications to
 * be running before proceeding. In such cases, you can add a CreationPolicy attribute to the instance, and then send a success
 * signal to the instance after the applications are installed and configured. For a detailed example, see Deploying Applications
 * on Amazon EC2 with AWS CloudFormation.
 */
export interface CfnCreationPolicy {
  /**
   * For an Auto Scaling group replacement update, specifies how many instances must signal success for the
   * update to succeed.
   */
  readonly autoScalingCreationPolicy?: CfnResourceAutoScalingCreationPolicy;

  /**
   * When AWS CloudFormation creates the associated resource, configures the number of required success signals and
   * the length of time that AWS CloudFormation waits for those signals.
   */
  readonly resourceSignal?: CfnResourceSignal;

  /**
   * For an AppStream Fleet creation, specifies that the fleet is started after creation.
   */
  readonly startFleet?: boolean;
}

/**
 * For an Auto Scaling group replacement update, specifies how many instances must signal success for the
 * update to succeed.
 */
export interface CfnResourceAutoScalingCreationPolicy {
  /**
   * Specifies the percentage of instances in an Auto Scaling replacement update that must signal success for the
   * update to succeed. You can specify a value from 0 to 100. AWS CloudFormation rounds to the nearest tenth of a percent.
   * For example, if you update five instances with a minimum successful percentage of 50, three instances must signal success.
   * If an instance doesn't send a signal within the time specified by the Timeout property, AWS CloudFormation assumes that the
   * instance wasn't created.
   */
  readonly minSuccessfulInstancesPercent?: number;
}

/**
 * When AWS CloudFormation creates the associated resource, configures the number of required success signals and
 * the length of time that AWS CloudFormation waits for those signals.
 */
export interface CfnResourceSignal {

  /**
   * The number of success signals AWS CloudFormation must receive before it sets the resource status as CREATE_COMPLETE.
   * If the resource receives a failure signal or doesn't receive the specified number of signals before the timeout period
   * expires, the resource creation fails and AWS CloudFormation rolls the stack back.
   */
  readonly count?: number;

  /**
   * The length of time that AWS CloudFormation waits for the number of signals that was specified in the Count property.
   * The timeout period starts after AWS CloudFormation starts creating the resource, and the timeout expires no sooner
   * than the time you specify but can occur shortly thereafter. The maximum time that you can specify is 12 hours.
   */
  readonly timeout?: string;
}

/**
 * With the DeletionPolicy attribute you can preserve or (in some cases) backup a resource when its stack is deleted.
 * You specify a DeletionPolicy attribute for each resource that you want to control. If a resource has no DeletionPolicy
 * attribute, AWS CloudFormation deletes the resource by default. Note that this capability also applies to update operations
 * that lead to resources being removed.
 */
export enum CfnDeletionPolicy {
  /**
   * AWS CloudFormation deletes the resource and all its content if applicable during stack deletion. You can add this
   * deletion policy to any resource type. By default, if you don't specify a DeletionPolicy, AWS CloudFormation deletes
   * your resources. However, be aware of the following considerations:
   */
  DELETE = 'Delete',

  /**
   * AWS CloudFormation keeps the resource without deleting the resource or its contents when its stack is deleted.
   * You can add this deletion policy to any resource type. Note that when AWS CloudFormation completes the stack deletion,
   * the stack will be in Delete_Complete state; however, resources that are retained continue to exist and continue to incur
   * applicable charges until you delete those resources.
   */
  RETAIN = 'Retain',

  /**
   * RetainExceptOnCreate behaves like Retain for stack operations, except for the stack operation that initially created the resource.
   * If the stack operation that created the resource is rolled back, CloudFormation deletes the resource. For all other stack operations,
   * such as stack deletion, CloudFormation retains the resource and its contents. The result is that new, empty, and unused resources are deleted,
   * while in-use resources and their data are retained.
   */
  RETAIN_EXCEPT_ON_CREATE = 'RetainExceptOnCreate',

  /**
   * For resources that support snapshots (AWS::EC2::Volume, AWS::ElastiCache::CacheCluster, AWS::ElastiCache::ReplicationGroup,
   * AWS::RDS::DBInstance, AWS::RDS::DBCluster, and AWS::Redshift::Cluster), AWS CloudFormation creates a snapshot for the
   * resource before deleting it. Note that when AWS CloudFormation completes the stack deletion, the stack will be in the
   * Delete_Complete state; however, the snapshots that are created with this policy continue to exist and continue to
   * incur applicable charges until you delete those snapshots.
   */
  SNAPSHOT = 'Snapshot',
}

/**
 * Use the UpdatePolicy attribute to specify how AWS CloudFormation handles updates to the AWS::AutoScaling::AutoScalingGroup
 * resource. AWS CloudFormation invokes one of three update policies depending on the type of change you make or whether a
 * scheduled action is associated with the Auto Scaling group.
 */
export interface CfnUpdatePolicy {

  /**
   * Specifies whether an Auto Scaling group and the instances it contains are replaced during an update. During replacement,
   * AWS CloudFormation retains the old group until it finishes creating the new one. If the update fails, AWS CloudFormation
   * can roll back to the old Auto Scaling group and delete the new Auto Scaling group.
   */
  readonly autoScalingReplacingUpdate?: CfnAutoScalingReplacingUpdate;

  /**
   * To specify how AWS CloudFormation handles rolling updates for an Auto Scaling group, use the AutoScalingRollingUpdate
   * policy. Rolling updates enable you to specify whether AWS CloudFormation updates instances that are in an Auto Scaling
   * group in batches or all at once.
   */
  readonly autoScalingRollingUpdate?: CfnAutoScalingRollingUpdate;

  /**
   * To specify how AWS CloudFormation handles instance refresh for an Auto Scaling group, use the
   * AutoScalingInstanceRefresh policy. This policy triggers an instance refresh when certain properties
   * of the Auto Scaling group change (such as launch template or mixed instances policy), replacing
   * instances gradually while maintaining availability.
   */
  readonly autoScalingInstanceRefresh?: CfnAutoScalingInstanceRefresh;

  /**
   * To specify how AWS CloudFormation handles updates for the MinSize, MaxSize, and DesiredCapacity properties when
   * the AWS::AutoScaling::AutoScalingGroup resource has an associated scheduled action, use the AutoScalingScheduledAction
   * policy.
   */
  readonly autoScalingScheduledAction?: CfnAutoScalingScheduledAction;

  /**
   * To perform an AWS CodeDeploy deployment when the version changes on an AWS::Lambda::Alias resource,
   * use the CodeDeployLambdaAliasUpdate update policy.
   */
  readonly codeDeployLambdaAliasUpdate?: CfnCodeDeployLambdaAliasUpdate;

  /**
   * To modify a replication group's shards by adding or removing shards, rather than replacing the entire
   * AWS::ElastiCache::ReplicationGroup resource, use the UseOnlineResharding update policy.
   */
  readonly useOnlineResharding?: boolean;

  /**
   * To upgrade an Amazon ES domain to a new version of Elasticsearch rather than replacing the entire
   * AWS::Elasticsearch::Domain resource, use the EnableVersionUpgrade update policy.
   */
  readonly enableVersionUpgrade?: boolean;

}

/**
 * To specify how AWS CloudFormation handles rolling updates for an Auto Scaling group, use the AutoScalingRollingUpdate
 * policy. Rolling updates enable you to specify whether AWS CloudFormation updates instances that are in an Auto Scaling
 * group in batches or all at once.
 */
export interface CfnAutoScalingRollingUpdate {

  /**
   * Specifies the maximum number of instances that AWS CloudFormation updates.
   */
  readonly maxBatchSize?: number;

  /**
   * Specifies the minimum number of instances that must be in service within the Auto Scaling group while AWS
   * CloudFormation updates old instances.
   */
  readonly minInstancesInService?: number;

  /**
   * Specifies the percentage of instances in an Auto Scaling rolling update that must signal success for an update to succeed.
   * You can specify a value from 0 to 100. AWS CloudFormation rounds to the nearest tenth of a percent. For example, if you
   * update five instances with a minimum successful percentage of 50, three instances must signal success.
   *
   * If an instance doesn't send a signal within the time specified in the PauseTime property, AWS CloudFormation assumes
   * that the instance wasn't updated.
   *
   * If you specify this property, you must also enable the WaitOnResourceSignals and PauseTime properties.
   */
  readonly minSuccessfulInstancesPercent?: number;

  /**
   * Specifies the percentage of instances in an Auto Scaling group that must remain in service while AWS CloudFormation
   * updates old instances. You can specify a value from 0 to 100. AWS CloudFormation rounds to the nearest tenth of a percent.
   * For example, if you update five instances with a minimum active percentage of 50, three instances must remain in service.
   */
  readonly minActiveInstancesPercent?: number;

  /**
   * The amount of time that AWS CloudFormation pauses after making a change to a batch of instances to give those instances
   * time to start software applications. For example, you might need to specify PauseTime when scaling up the number of
   * instances in an Auto Scaling group.
   *
   * If you enable the WaitOnResourceSignals property, PauseTime is the amount of time that AWS CloudFormation should wait
   * for the Auto Scaling group to receive the required number of valid signals from added or replaced instances. If the
   * PauseTime is exceeded before the Auto Scaling group receives the required number of signals, the update fails. For best
   * results, specify a time period that gives your applications sufficient time to get started. If the update needs to be
   * rolled back, a short PauseTime can cause the rollback to fail.
   *
   * Specify PauseTime in the ISO8601 duration format (in the format PT#H#M#S, where each # is the number of hours, minutes,
   * and seconds, respectively). The maximum PauseTime is one hour (PT1H).
   */
  readonly pauseTime?: string;

  /**
   * Specifies the Auto Scaling processes to suspend during a stack update. Suspending processes prevents Auto Scaling from
   * interfering with a stack update. For example, you can suspend alarming so that Auto Scaling doesn't execute scaling
   * policies associated with an alarm. For valid values, see the ScalingProcesses.member.N parameter for the SuspendProcesses
   * action in the Auto Scaling API Reference.
   */
  readonly suspendProcesses?: string[];

  /**
   * Specifies whether the Auto Scaling group waits on signals from new instances during an update. Use this property to
   * ensure that instances have completed installing and configuring applications before the Auto Scaling group update proceeds.
   * AWS CloudFormation suspends the update of an Auto Scaling group after new EC2 instances are launched into the group.
   * AWS CloudFormation must receive a signal from each new instance within the specified PauseTime before continuing the update.
   * To signal the Auto Scaling group, use the cfn-signal helper script or SignalResource API.
   *
   * To have instances wait for an Elastic Load Balancing health check before they signal success, add a health-check
   * verification by using the cfn-init helper script. For an example, see the verify_instance_health command in the Auto Scaling
   * rolling updates sample template.
   */
  readonly waitOnResourceSignals?: boolean;
}

/**
 * Specifies whether an Auto Scaling group and the instances it contains are replaced during an update. During replacement,
 * AWS CloudFormation retains the old group until it finishes creating the new one. If the update fails, AWS CloudFormation
 * can roll back to the old Auto Scaling group and delete the new Auto Scaling group.
 *
 * While AWS CloudFormation creates the new group, it doesn't detach or attach any instances. After successfully creating
 * the new Auto Scaling group, AWS CloudFormation deletes the old Auto Scaling group during the cleanup process.
 *
 * When you set the WillReplace parameter, remember to specify a matching CreationPolicy. If the minimum number of
 * instances (specified by the MinSuccessfulInstancesPercent property) don't signal success within the Timeout period
 * (specified in the CreationPolicy policy), the replacement update fails and AWS CloudFormation rolls back to the old
 * Auto Scaling group.
 */
export interface CfnAutoScalingReplacingUpdate {
  readonly willReplace?: boolean;
}

/**
 * With scheduled actions, the group size properties of an Auto Scaling group can change at any time. When you update a
 * stack with an Auto Scaling group and scheduled action, AWS CloudFormation always sets the group size property values of
 * your Auto Scaling group to the values that are defined in the AWS::AutoScaling::AutoScalingGroup resource of your template,
 * even if a scheduled action is in effect.
 *
 * If you do not want AWS CloudFormation to change any of the group size property values when you have a scheduled action in
 * effect, use the AutoScalingScheduledAction update policy to prevent AWS CloudFormation from changing the MinSize, MaxSize,
 * or DesiredCapacity properties unless you have modified these values in your template.\
 */
export interface CfnAutoScalingScheduledAction {
  /*
  * Specifies whether AWS CloudFormation ignores differences in group size properties between your current Auto Scaling
  * group and the Auto Scaling group described in the AWS::AutoScaling::AutoScalingGroup resource of your template during
  * a stack update. If you modify any of the group size property values in your template, AWS CloudFormation uses the modified
  * values and updates your Auto Scaling group.
  */
  readonly ignoreUnmodifiedGroupSizeProperties?: boolean;
}

/**
 * To perform an AWS CodeDeploy deployment when the version changes on an AWS::Lambda::Alias resource,
 * use the CodeDeployLambdaAliasUpdate update policy.
 */
export interface CfnCodeDeployLambdaAliasUpdate {
  /**
   * The name of the AWS CodeDeploy application.
   */
  readonly applicationName: string;

  /**
   * The name of the AWS CodeDeploy deployment group. This is where the traffic-shifting policy is set.
   */
  readonly deploymentGroupName: string;

  /**
   * The name of the Lambda function to run before traffic routing starts.
   */
  readonly beforeAllowTrafficHook?: string;

  /**
   * The name of the Lambda function to run after traffic routing completes.
   */
  readonly afterAllowTrafficHook?: string;
}

/**
 * To specify how AWS CloudFormation handles instance refresh for an Auto Scaling group, use the
 * AutoScalingInstanceRefresh update policy. When properties that trigger an instance refresh change
 * (such as LaunchTemplate or MixedInstancesPolicy), CloudFormation starts an instance refresh to
 * replace instances gradually.
 *
 * This policy is mutually exclusive with AutoScalingRollingUpdate.
 */
export interface CfnAutoScalingInstanceRefresh {
  /**
   * The strategy to use for the instance refresh.
   */
  readonly strategy: string;

  /**
   * The preferences for the instance refresh.
   */
  readonly preferences?: CfnAutoScalingInstanceRefreshPreferences;
}

/**
 * Preferences for the AutoScalingInstanceRefresh update policy.
 */
export interface CfnAutoScalingInstanceRefreshPreferences {
  /**
   * The minimum percentage of the group to keep in service, healthy, and ready to use
   * to support your workload during the instance refresh.
   *
   * @default - the value set in the Auto Scaling group's instance maintenance policy, if defined; otherwise 100 when the strategy is Rolling, or 90 when the strategy is ReplaceRootVolume
   */
  readonly minHealthyPercentage?: number;

  /**
   * The maximum percentage of the group that can be in service and healthy, or pending,
   * to support your workload during the instance refresh. Used to control batch size.
   *
   * @default - the value set in the Auto Scaling group's instance maintenance policy, if defined; otherwise 110 when the strategy is Rolling, or 100 when the strategy is ReplaceRootVolume
   */
  readonly maxHealthyPercentage?: number;

  /**
   * The number of seconds to wait after a new instance enters the InService state before
   * moving on to replacing the next instance.
   *
   * @default - the group's DefaultInstanceWarmup if defined; otherwise the group's HealthCheckGracePeriod
   */
  readonly instanceWarmup?: number;

  /**
   * Indicates whether skip matching is enabled. If true, instances that already match the
   * desired configuration are not replaced.
   *
   * @default true
   */
  readonly skipMatching?: boolean;

  /**
   * Threshold values for each checkpoint in ascending order. Each number is a percentage of
   * the total number of instances in the group. When the percentage of instances that have
   * been replaced reaches a checkpoint, the refresh waits for the configured checkpoint delay
   * before continuing.
   *
   * @default - no checkpoints
   */
  readonly checkpointPercentages?: number[];

  /**
   * The number of seconds to wait after a checkpoint is reached before continuing.
   *
   * @default - 3600 seconds (1 hour), applied only when checkpointPercentages is set
   */
  readonly checkpointDelay?: number;

  /**
   * The number of seconds after an instance refresh completes successfully before
   * CloudFormation considers the update successful.
   *
   * @default 0
   */
  readonly bakeTime?: number;

  /**
   * The CloudWatch alarms to monitor during the instance refresh. If any alarm goes into
   * ALARM state, the instance refresh fails.
   */
  readonly alarmSpecification?: CfnAutoScalingInstanceRefreshAlarmSpecification;

  /**
   * Specifies the behavior of instances that are protected from scale in during an instance refresh.
   *
   * @default 'Wait'
   */
  readonly scaleInProtectedInstances?: string;

  /**
   * Specifies the behavior of instances in standby during an instance refresh.
   *
   * @default 'Wait'
   */
  readonly standbyInstances?: string;
}

/**
 * Alarm specification for the AutoScalingInstanceRefresh update policy.
 */
export interface CfnAutoScalingInstanceRefreshAlarmSpecification {
  /**
   * The names of the CloudWatch alarms to monitor.
   */
  readonly alarms?: string[];
}

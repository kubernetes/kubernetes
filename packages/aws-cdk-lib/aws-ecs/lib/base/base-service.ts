import type { Construct, IConstruct } from 'constructs';
import { ScalableTaskCount } from './scalable-task-count';
import type { ServiceManagedVolume } from './service-managed-volume';
import * as appscaling from '../../../aws-applicationautoscaling';
import * as cloudwatch from '../../../aws-cloudwatch';
import * as ec2 from '../../../aws-ec2';
import type * as elb from '../../../aws-elasticloadbalancing';
import * as elbv2 from '../../../aws-elasticloadbalancingv2';
import * as iam from '../../../aws-iam';
import type * as kms from '../../../aws-kms';
import * as cloudmap from '../../../aws-servicediscovery';
import type {
  IResolvable,
  IResource,
} from '../../../core';
import {
  Annotations,
  Arn,
  ArnFormat,
  Duration,
  FeatureFlags,
  Fn,
  Lazy,
  Resource,
  Stack,
  Token,
  ValidationError,
} from '../../../core';
import type { IArrayBox, IBox } from '../../../core/lib/helpers-internal';
import { Box, memoizedGetter } from '../../../core/lib/helpers-internal';
import { noBoxStackTraces } from '../../../core/lib/no-box-stack-traces';
import { lit } from '../../../core/lib/private/literal-string';
import * as cxapi from '../../../cx-api';
import type { IServiceRef, ServiceReference } from '../../../interfaces/generated/aws-ecs-interfaces.generated';
import { RegionInfo } from '../../../region-info';
import type { IAlternateTarget } from '../alternate-target-configuration';
import type {
  LoadBalancerTargetOptions,
  TaskDefinition,
} from '../base/task-definition';
import {
  NetworkMode,
  TaskDefinitionRevision,
} from '../base/task-definition';
import type { CapacityProviderStrategy, ICluster } from '../cluster';
import { Cluster, ExecuteCommandLogging } from '../cluster';
import type { ContainerDefinition, Protocol } from '../container-definition';
import type { IDeploymentLifecycleHookTarget } from '../deployment-lifecycle-hook-target';
import { CfnService } from '../ecs.generated';
import type { LogDriver, LogDriverConfig } from '../log-drivers/log-driver';

/**
 * The interface for a service.
 */
export interface IService extends IResource, IServiceRef {
  /**
   * The Amazon Resource Name (ARN) of the service.
   *
   * @attribute
   */
  readonly serviceArn: string;

  /**
   * The name of the service.
   *
   * @attribute
   */
  readonly serviceName: string;
}

/**
 * The deployment controller to use for the service.
 */
export interface DeploymentController {
  /**
   * The deployment controller type to use.
   *
   * @default DeploymentControllerType.ECS
   */
  readonly type?: DeploymentControllerType;
}

/**
 * The deployment circuit breaker to use for the service
 */
export interface DeploymentCircuitBreaker {
  /**
   * Whether to enable the deployment circuit breaker logic
   * @default true
   */
  readonly enable?: boolean;

  /**
   * Whether to enable rollback on deployment failure
   *
   * @default false
   */
  readonly rollback?: boolean;
}

/**
 * Configuration for forcing a new deployment of the service.
 */
export interface ForceNewDeployment {
  /**
   * Whether to enable the force-new-deployment mechanism for the service.
   *
   * Setting this to `true` enables the mechanism, but on its own it does not
   * force a new deployment on every `cdk deploy`: CloudFormation only starts a
   * new deployment when it detects a change in the template, and the signal for
   * that is the `nonce` value changing between deployments. If `nonce` is not
   * provided or its value stays the same across deployments, no new deployment
   * is forced. When set to `false`, the `ForceNewDeployment` property is rendered
   * with `EnableForceNewDeployment: false`.
   *
   * To force a new deployment on every `cdk deploy`, provide a `nonce` with a
   * unique, time-varying value such as a timestamp, random string, or sequence
   * number (e.g. `Date.now().toString()`).
   *
   * @see https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-properties-ecs-service-forcenewdeployment.html
   */
  readonly enabled: boolean;

  /**
   * A unique nonce value that signals Amazon ECS to start a new deployment.
   *
   * When you change this value, it triggers a new deployment even though no
   * other service parameters have changed. Use a stable, time-varying value
   * like a commit hash, image digest, or version string.
   *
   * If not provided and `enabled` is `true`, only `EnableForceNewDeployment`
   * is set without a nonce.
   *
   * Must be between 1 and 255 characters.
   *
   * @default - no nonce
   */
  readonly nonce?: string;
}

/**
 * Configuration for traffic shift during progressive deployments
 */
export interface TrafficShiftConfig {
  /**
   * The percentage of production traffic to shift in each step.
   * - For linear deployment: multiples of 0.1 from 3.0 to 100.0
   * - For canary deployment: multiples of 0.1 from 0.1 to 100.0
   *
   * @default - 10.0 for linear, 5.0 for canary
   */
  readonly stepPercent?: number;

  /**
   * The duration to wait between traffic shifting steps.
   * Valid values are 0 to 1440 minutes (24 hours).
   *
   * @default - Duration.minutes(6) for linear, Duration.minutes(10) for canary
   */
  readonly stepBakeTime?: Duration;
}

/**
 * Deployment behavior when an ECS Service Deployment Alarm is triggered
 */
export enum AlarmBehavior {
  /**
   * ROLLBACK_ON_ALARM causes the service to roll back to the previous deployment
   * when any deployment alarm enters the 'Alarm' state. The Cloudformation stack
   * will be rolled back and enter state "UPDATE_ROLLBACK_COMPLETE".
   */
  ROLLBACK_ON_ALARM = 'ROLLBACK_ON_ALARM',
  /**
   * FAIL_ON_ALARM causes the deployment to fail immediately when any deployment
   * alarm enters the 'Alarm' state. In order to restore functionality, you must
   * roll the stack forward by pushing a new version of the ECS service.
   */
  FAIL_ON_ALARM = 'FAIL_ON_ALARM',
}

/**
 * Options for deployment alarms
 */
export interface DeploymentAlarmOptions {
  /**
   * Default rollback on alarm
   * @default AlarmBehavior.ROLLBACK_ON_ALARM
   */
  readonly behavior?: AlarmBehavior;
}

/**
 * Configuration for deployment alarms
 */
export interface DeploymentAlarmConfig extends DeploymentAlarmOptions {
  /**
   * List of alarm names to monitor during deployments
   */
  readonly alarmNames: string[];
}

export interface EcsTarget {
  /**
   * The name of the container.
   */
  readonly containerName: string;

  /**
   * The port number of the container. Only applicable when using application/network load balancers.
   *
   * @default - Container port of the first added port mapping.
   */
  readonly containerPort?: number;

  /**
   * The protocol used for the port mapping. Only applicable when using application load balancers.
   *
   * @default Protocol.TCP
   */
  readonly protocol?: Protocol;

  /**
   * ID for a target group to be created.
   */
  readonly newTargetGroupId: string;

  /**
   * Listener and properties for adding target group to the listener.
   */
  readonly listener: ListenerConfig;
}

/**
 * Interface for ECS load balancer target.
 */
export interface IEcsLoadBalancerTarget extends elbv2.IApplicationLoadBalancerTarget, elbv2.INetworkLoadBalancerTarget, elb.ILoadBalancerTarget {
}

/**
 * The format of Service Connect access logs.
 *
 * @see https://docs.aws.amazon.com/AmazonECS/latest/developerguide/service-connect-envoy-access-logs.html
 */
export enum ServiceConnectAccessLogFormat {
  /**
   * Human-readable text format for access logs.
   */
  TEXT = 'TEXT',

  /**
   * Structured JSON format for access logs.
   * This format is well-suited for integration with log analysis tools.
   */
  JSON = 'JSON',
}

/**
 * Configuration for Service Connect access logs.
 *
 * Service Connect access logs provide detailed telemetry about individual requests processed by the Service Connect proxy,
 * including HTTP methods, paths, response codes, and timing information.
 *
 * @see https://docs.aws.amazon.com/AmazonECS/latest/developerguide/service-connect-envoy-access-logs.html
 */
export interface ServiceConnectAccessLogConfiguration {
  /**
   * The format for Service Connect access log output.
   *
   * - TEXT: Human-readable text format
   * - JSON: Structured JSON format for log analysis tools
   */
  readonly format: ServiceConnectAccessLogFormat;

  /**
   * Whether to include query parameters in Service Connect access logs.
   *
   * When enabled, query parameters from HTTP requests are included in the access logs.
   * Consider security and privacy implications as query parameters may contain sensitive information such as request IDs and tokens.
   *
   * @default undefined - AWS ECS default is false, which means that query parameters are not included in access logs
   */
  readonly includeQueryParameters?: boolean;
}

/**
 * Interface for Service Connect configuration.
 */
export interface ServiceConnectProps {
  /**
   * The cloudmap namespace to register this service into.
   *
   * @default the cloudmap namespace specified on the cluster.
   */
  readonly namespace?: string;

  /**
   * The list of Services, including a port mapping, terse client alias, and optional intermediate DNS name.
   *
   * This property may be left blank if the current ECS service does not need to advertise any ports via Service Connect.
   *
   * @default none
   */
  readonly services?: ServiceConnectService[];

  /**
   * The log driver configuration to use for the Service Connect agent logs.
   *
   * @default - none
   */
  readonly logDriver?: LogDriver;

  /**
   * The configuration for Service Connect access logs.
   *
   * Access logs provide detailed telemetry about individual requests processed by the　Service Connect proxy.
   *
   * @default undefined - AWS ECS default is disabled, which means that access logs are not recorded
   */
  readonly accessLogConfiguration?: ServiceConnectAccessLogConfiguration;
}

/**
 * Interface for service connect Service props.
 */
export interface ServiceConnectService {
  /**
   * portMappingName specifies which port and protocol combination should be used for this
   * service connect service.
   */
  readonly portMappingName: string;

  /**
   * Optionally specifies an intermediate dns name to register in the CloudMap namespace.
   * This is required if you wish to use the same port mapping name in more than one service.
   *
   * @default - port mapping name
   */
  readonly discoveryName?: string;

  /**
   * The terse DNS alias to use for this port mapping in the service connect mesh.
   * Service Connect-enabled clients will be able to reach this service at
   * http://dnsName:port.
   *
   * @default - No alias is created. The service is reachable at `portMappingName.namespace:port`.
   */
  readonly dnsName?: string;

  /**
   * The port for clients to use to communicate with this service via Service Connect.
   *
   * @default the container port specified by the port mapping in portMappingName.
   */
  readonly port?: number;

  /**
   * Optional. The port on the Service Connect agent container to use for traffic ingress to this service.
   *
   * @default - none
   */
  readonly ingressPortOverride?: number;

  /**
   * The amount of time in seconds a connection for Service Connect will stay active while idle.
   *
   * A value of 0 can be set to disable `idleTimeout`.
   *
   * If `idleTimeout` is set to a time that is less than `perRequestTimeout`, the connection will close
   * when the `idleTimeout` is reached and not the `perRequestTimeout`.
   *
   * @default - Duration.minutes(5) for HTTP/HTTP2/GRPC, Duration.hours(1) for TCP.
   */
  readonly idleTimeout?: Duration;

  /**
   * The amount of time waiting for the upstream to respond with a complete response per request for
   * Service Connect.
   *
   * A value of 0 can be set to disable `perRequestTimeout`.
   * Can only be set when the `appProtocol` for the application container is HTTP/HTTP2/GRPC.
   *
   * If `idleTimeout` is set to a time that is less than `perRequestTimeout`, the connection will close
   * when the `idleTimeout` is reached and not the `perRequestTimeout`.
   *
   * @default - Duration.seconds(15)
   */
  readonly perRequestTimeout?: Duration;

  /**
   * A reference to an object that represents a Transport Layer Security (TLS) configuration.
   *
   * @default - none
   */
  readonly tls?: ServiceConnectTlsConfiguration;
}

/**
 * TLS configuration for Service Connect service
 */
export interface ServiceConnectTlsConfiguration {
  /**
   * The ARN of the certificate root authority that secures your service.
   *
   * @default - none
   */
  readonly awsPcaAuthorityArn?: string;

  /**
   * The KMS key used for encryption and decryption.
   *
   * @default - none
   */
  readonly kmsKey?: kms.IKeyRef;

  /**
   * The IAM role that's associated with the Service Connect TLS.
   *
   * @default - none
   */
  readonly role?: iam.IRole;
}

/**
 * The properties for the base Ec2Service or FargateService service.
 */
export interface BaseServiceOptions {
  /**
   * The name of the cluster that hosts the service.
   */
  readonly cluster: ICluster;

  /**
   * The desired number of instantiations of the task definition to keep running on the service.
   *
   * @default - When creating the service, default is 1; when updating the service, default uses
   * the current task number.
   */
  readonly desiredCount?: number;

  /**
   * The name of the service.
   *
   * @default - CloudFormation-generated name.
   */
  readonly serviceName?: string;

  /**
   * The maximum number of tasks, specified as a percentage of the Amazon ECS
   * service's DesiredCount value, that can run in a service during a
   * deployment.
   *
   * @default - 100 if daemon, otherwise 200
   */
  readonly maxHealthyPercent?: number;

  /**
   * The minimum number of tasks, specified as a percentage of
   * the Amazon ECS service's DesiredCount value, that must
   * continue to run and remain healthy during a deployment.
   *
   * @default - 0 if daemon, otherwise 50
   */
  readonly minHealthyPercent?: number;

  /**
   * The period of time, in seconds, that the Amazon ECS service scheduler ignores unhealthy
   * Elastic Load Balancing target health checks after a task has first started.
   *
   * @default - defaults to 60 seconds if at least one load balancer is in-use and it is not already set
   */
  readonly healthCheckGracePeriod?: Duration;

  /**
   * The options for configuring an Amazon ECS service to use service discovery.
   *
   * @default - AWS Cloud Map service discovery is not enabled.
   */
  readonly cloudMapOptions?: CloudMapOptions;

  /**
   * Specifies whether to propagate the tags from the task definition or the service to the tasks in the service
   *
   * Valid values are: PropagatedTagSource.SERVICE, PropagatedTagSource.TASK_DEFINITION or PropagatedTagSource.NONE
   *
   * @default PropagatedTagSource.NONE
   */
  readonly propagateTags?: PropagatedTagSource;

  /**
   * Specifies whether to propagate the tags from the task definition or the service to the tasks in the service.
   * Tags can only be propagated to the tasks within the service during service creation.
   *
   * @deprecated Use `propagateTags` instead.
   * @default PropagatedTagSource.NONE
   */
  readonly propagateTaskTagsFrom?: PropagatedTagSource;

  /**
   * Specifies whether to enable Amazon ECS managed tags for the tasks within the service. For more information, see
   * [Tagging Your Amazon ECS Resources](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ecs-using-tags.html)
   *
   * @default false
   */
  readonly enableECSManagedTags?: boolean;

  /**
   * Specifies which deployment controller to use for the service. For more information, see
   * [Amazon ECS Deployment Types](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/deployment-types.html)
   *
   * @default - Rolling update (ECS)
   */
  readonly deploymentController?: DeploymentController;

  /**
   * Whether to enable the deployment circuit breaker. If this property is defined, circuit breaker will be implicitly
   * enabled.
   * @default - disabled
   */
  readonly circuitBreaker?: DeploymentCircuitBreaker;

  /**
   * The alarm(s) to monitor during deployment, and behavior to apply if at least one enters a state of alarm
   * during the deployment or bake time.
   *
   *
   * @default - No alarms will be monitored during deployment.
   */
  readonly deploymentAlarms?: DeploymentAlarmConfig;

  /**
   * A list of Capacity Provider strategies used to place a service.
   *
   * @default - undefined
   *
   */
  readonly capacityProviderStrategies?: CapacityProviderStrategy[];

  /**
   * Whether to enable the ability to execute into a container
   *
   *  @default - undefined
   */
  readonly enableExecuteCommand?: boolean;

  /**
   * Configuration for Service Connect.
   *
   * @default No ports are advertised via Service Connect on this service, and the service
   * cannot make requests to other services via Service Connect.
   */
  readonly serviceConnectConfiguration?: ServiceConnectProps;

  /**
   * Revision number for the task definition or `latest` to use the latest active task revision.
   *
   * @default - Uses the revision of the passed task definition deployed by CloudFormation
   */
  readonly taskDefinitionRevision?: TaskDefinitionRevision;

  /**
   * Configuration details for a volume used by the service. This allows you to specify
   * details about the EBS volume that can be attached to ECS tasks.
   *
   * @default - undefined
   */
  readonly volumeConfigurations?: ServiceManagedVolume[];

  /**
   * The deployment strategy to use for the service.
   * @default ROLLING
   */
  readonly deploymentStrategy?: DeploymentStrategy;

  /**
   * bake time minutes for service.
   * @default - none
   */
  readonly bakeTime?: Duration;

  /**
   * The lifecycle hooks to execute during deployment stages
   * @default - none;
   */
  readonly lifecycleHooks?: IDeploymentLifecycleHookTarget[];

  /**
   * Configuration for linear deployment strategy.
   * Only valid when deploymentStrategy is set to LINEAR.
   *
   * @default - no linear configuration
   */
  readonly linearConfiguration?: TrafficShiftConfig;

  /**
   * Configuration for canary deployment strategy.
   * Only valid when deploymentStrategy is set to CANARY.
   *
   * @default - no canary configuration
   */
  readonly canaryConfiguration?: TrafficShiftConfig;

  /**
   * Configuration for forcing a new deployment of the service.
   *
   * By default, deployments aren't forced. You can use this option to start
   * a new deployment with no service definition changes. For example, you can
   * update a service's tasks to use a newer Docker image with the same
   * image/tag combination (`my_image:latest`) or to roll Fargate tasks onto
   * a newer platform version.
   *
   * This is equivalent to calling the `forceNewDeployment()` method, but allows
   * you to configure it declaratively at construction time, including the ability
   * to explicitly disable it with `enabled: false`.
   *
   * @see https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-ecs-service-forcenewdeployment.html
   * @default - no forced deployment
   */
  readonly forceNewDeployment?: ForceNewDeployment;

}

/**
 * Complete base service properties that are required to be supplied by the implementation
 * of the BaseService class.
 */
export interface BaseServiceProps extends BaseServiceOptions {
  /**
   * The launch type on which to run your service.
   *
   * LaunchType will be omitted if capacity provider strategies are specified on the service.
   *
   * @see https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-resource-ecs-service.html#cfn-ecs-service-capacityproviderstrategy
   *
   * Valid values are: LaunchType.ECS or LaunchType.FARGATE or LaunchType.EXTERNAL
   */
  readonly launchType: LaunchType;
}

/**
 * Base class for configuring listener when registering targets.
 */
export abstract class ListenerConfig {
  /**
   * Create a config for adding target group to ALB listener.
   */
  public static applicationListener(listener: elbv2.ApplicationListener, props?: elbv2.AddApplicationTargetsProps): ListenerConfig {
    return new ApplicationListenerConfig(listener, props);
  }

  /**
   * Create a config for adding target group to NLB listener.
   */
  public static networkListener(listener: elbv2.NetworkListener, props?: elbv2.AddNetworkTargetsProps): ListenerConfig {
    return new NetworkListenerConfig(listener, props);
  }

  /**
   * Create and attach a target group to listener.
   */
  public abstract addTargets(id: string, target: LoadBalancerTargetOptions, service: BaseService): void;
}

/**
 * Class for configuring application load balancer listener when registering targets.
 */
class ApplicationListenerConfig extends ListenerConfig {
  constructor(private readonly listener: elbv2.ApplicationListener, private readonly props?: elbv2.AddApplicationTargetsProps) {
    super();
  }

  /**
   * Create and attach a target group to listener.
   */
  public addTargets(id: string, target: LoadBalancerTargetOptions, service: BaseService) {
    const props = this.props || {};
    const protocol = props.protocol;
    const port = props.port ?? (protocol === elbv2.ApplicationProtocol.HTTPS ? 443 : 80);
    this.listener.addTargets(id, {
      ...props,
      targets: [
        service.loadBalancerTarget({
          ...target,
        }),
      ],
      port,
    });
  }
}

/**
 * Class for configuring network load balancer listener when registering targets.
 */
class NetworkListenerConfig extends ListenerConfig {
  constructor(private readonly listener: elbv2.NetworkListener, private readonly props?: elbv2.AddNetworkTargetsProps) {
    super();
  }

  /**
   * Create and attach a target group to listener.
   */
  public addTargets(id: string, target: LoadBalancerTargetOptions, service: BaseService) {
    const port = this.props?.port ?? 80;
    this.listener.addTargets(id, {
      ... this.props,
      targets: [
        service.loadBalancerTarget({
          ...target,
        }),
      ],
      port,
    });
  }
}

/**
 * The interface for BaseService.
 */
export interface IBaseService extends IService {
  /**
   * The cluster that hosts the service.
   */
  readonly cluster: ICluster;
}

/**
 * The base class for Ec2Service and FargateService services.
 */
@noBoxStackTraces
export abstract class BaseService extends Resource
  implements IBaseService, elbv2.IApplicationLoadBalancerTarget, elbv2.INetworkLoadBalancerTarget, elb.ILoadBalancerTarget {
  /**
   * Import an existing ECS/Fargate Service using the service cluster format.
   * The format is the "new" format "arn:aws:ecs:region:aws_account_id:service/cluster-name/service-name".
   * @see https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ecs-account-settings.html#ecs-resource-ids
   */
  public static fromServiceArnWithCluster(scope: Construct, id: string, serviceArn: string): IBaseService {
    const stack = Stack.of(scope);
    const arn = stack.splitArn(serviceArn, ArnFormat.SLASH_RESOURCE_NAME);
    const resourceName = Arn.extractResourceName(serviceArn, 'service');
    let clusterName: string;
    let serviceName: string;
    if (Token.isUnresolved(resourceName)) {
      clusterName = Fn.select(0, Fn.split('/', resourceName));
      serviceName = Fn.select(1, Fn.split('/', resourceName));
    } else {
      const resourceNameParts = resourceName.split('/');
      if (resourceNameParts.length !== 2) {
        throw new ValidationError(lit`ResourceNameServiceArn`, `resource name ${resourceName} from service ARN: ${serviceArn} is not using the ARN cluster format`, scope);
      }
      clusterName = resourceNameParts[0];
      serviceName = resourceNameParts[1];
    }

    const clusterArn = Stack.of(scope).formatArn({
      partition: arn.partition,
      region: arn.region,
      account: arn.account,
      service: 'ecs',
      resource: 'cluster',
      resourceName: clusterName,
    });

    const cluster = Cluster.fromClusterArn(scope, `${id}Cluster`, clusterArn);

    class Import extends Resource implements IBaseService {
      public readonly serviceArn = serviceArn;
      public readonly serviceName = serviceName;
      public readonly cluster = cluster;

      public get serviceRef(): ServiceReference {
        return {
          serviceArn: this.serviceArn,
        };
      }
    }

    return new Import(scope, id, {
      environmentFromArn: serviceArn,
    });
  }

  private static MIN_PORT = 1;
  private static MAX_PORT = 65535;

  /**
   * The security groups which manage the allowed network traffic for the service.
   */
  public readonly connections: ec2.Connections = new ec2.Connections();

  /**
   * A reference to this service.
   */
  public get serviceRef(): ServiceReference {
    return {
      serviceArn: this.serviceArn,
    };
  }

  /**
   * The task definition to use for tasks in the service.
   */
  public readonly taskDefinition: TaskDefinition;

  /**
   * The cluster that hosts the service.
   */
  public readonly cluster: ICluster;

  /**
   * The details of the AWS Cloud Map service.
   */
  protected cloudmapService?: cloudmap.Service;

  /**
   * A list of Elastic Load Balancing load balancer objects, containing the load balancer name, the container
   * name (as it appears in a container definition), and the container port to access from the load balancer.
   */
  private _loadBalancers: IArrayBox<CfnService.LoadBalancerProperty> = Box.fromArray([], { omitEmpty: false });

  /**
   * A list of Elastic Load Balancing load balancer objects, containing the load balancer name, the container
   * name (as it appears in a container definition), and the container port to access from the load balancer.
   */
  private readonly _networkConfigurationBox: IBox<CfnService.NetworkConfigurationProperty | undefined> = Box.fromValue(undefined);

  protected get networkConfiguration(): CfnService.NetworkConfigurationProperty | undefined {
    return this._networkConfigurationBox.getMutable();
  }
  protected set networkConfiguration(value: CfnService.NetworkConfigurationProperty | undefined) {
    this._networkConfigurationBox.set(value);
  }

  /**
   * The deployment alarms property - this will be rendered directly and lazily as the CfnService.alarms
   * property.
   */
  private readonly _deploymentAlarms: IBox<CfnService.DeploymentAlarmsProperty | undefined> = Box.fromValue(undefined);

  protected get deploymentAlarms(): CfnService.DeploymentAlarmsProperty | undefined {
    return this._deploymentAlarms.getMutable();
  }
  protected set deploymentAlarms(value: CfnService.DeploymentAlarmsProperty | undefined) {
    this._deploymentAlarms.set(value);
  }

  /**
   * The details of the service discovery registries to assign to this service.
   * For more information, see Service Discovery.
   */
  private _serviceRegistries: IArrayBox<CfnService.ServiceRegistryProperty> = Box.fromArray();

  /**
   * The service connect configuration for this service.
   */
  private readonly _serviceConnectConfig: IBox<CfnService.ServiceConnectConfigurationProperty | undefined> = Box.fromValue(undefined);

  /**
   * Whether this service is using the ECS deployment controller.
   * @internal
   */
  private readonly isEcsDeploymentController: boolean;

  private readonly resource: CfnService;
  private scalableTaskCount?: ScalableTaskCount;

  /**
   * All volumes
   */
  private readonly _volumes: IArrayBox<ServiceManagedVolume>;

  /**
   * A deployment lifecycle hook runs custom logic at specific stages of the deployment process.
   * @default - none
   */
  private readonly _lifecycleHooks: IArrayBox<IDeploymentLifecycleHookTarget>;

  @memoizedGetter
  public get serviceArn(): string {
    return this.getResourceArnAttribute(this.resource.ref, {
      service: 'ecs',
      resource: 'service',
      resourceName: `${this.cluster.clusterName}/${this.physicalName}`,
    });
  }

  /**
   * The details of the service discovery registries to assign to this service.
   * For more information, see Service Discovery.
   */
  protected set serviceRegistries(sr: CfnService.ServiceRegistryProperty[]) {
    this._serviceRegistries.set(sr);
  }

  /**
   * The details of the service discovery registries to assign to this service.
   * For more information, see Service Discovery.
   */
  protected get serviceRegistries(): CfnService.ServiceRegistryProperty[] {
    return this._serviceRegistries.getMutable();
  }

  /**
   * A list of Elastic Load Balancing load balancer objects, containing the load balancer name, the container
   * name (as it appears in a container definition), and the container port to access from the load balancer.
   */
  public get loadBalancers(): Array<CfnService.LoadBalancerProperty> {
    return this._loadBalancers.getMutable();
  }

  /**
   * A list of Elastic Load Balancing load balancer objects, containing the load balancer name, the container
   * name (as it appears in a container definition), and the container port to access from the load balancer.
   */
  public set loadBalancers(value: Array<CfnService.LoadBalancerProperty>) {
    this._loadBalancers.set(value);
  }

  @memoizedGetter
  public get serviceName(): string {
    return this.getResourceNameAttribute(this.resource.attrName);
  }

  /**
   * The deployment strategy for the service
   */
  private readonly deploymentStrategy?: DeploymentStrategy;

  /**
   * Constructs a new instance of the BaseService class.
   */
  constructor(
    scope: Construct,
    id: string,
    props: BaseServiceProps,
    additionalProps: any,
    taskDefinition: TaskDefinition) {
    super(scope, id, {
      physicalName: props.serviceName,
    });
    if (props.propagateTags && props.propagateTaskTagsFrom) {
      throw new ValidationError(lit`OnlySpecifyEitherPropagateTags`, 'You can only specify either propagateTags or propagateTaskTagsFrom. Alternatively, you can leave both blank', this);
    }

    this.taskDefinition = taskDefinition;
    this._volumes = Box.fromArray();
    this._lifecycleHooks = Box.fromArray();

    // launchType will set to undefined if using external DeploymentController or capacityProviderStrategies
    const launchType = props.deploymentController?.type === DeploymentControllerType.EXTERNAL ||
      props.capacityProviderStrategies !== undefined ?
      undefined : props.launchType;

    const propagateTagsFromSource = props.propagateTaskTagsFrom ?? props.propagateTags ?? PropagatedTagSource.NONE;
    const deploymentController = this.getDeploymentController(props);

    // Determine if this service is using the ECS deployment controller
    this.isEcsDeploymentController = !deploymentController || deploymentController.type === DeploymentControllerType.ECS;
    this.deploymentStrategy = props.deploymentStrategy;

    if (props.linearConfiguration) {
      this.validateLinearConfiguration(props.linearConfiguration);
    }
    if (props.canaryConfiguration) {
      this.validateCanaryConfiguration(props.canaryConfiguration);
    }

    this.resource = new CfnService(this, 'Service', {
      desiredCount: props.desiredCount,
      serviceName: this.physicalName,
      loadBalancers: this._loadBalancers.derive(lbs => lbs.length > 0 ? lbs : undefined),
      deploymentConfiguration: {
        maximumPercent: props.maxHealthyPercent || 200,
        minimumHealthyPercent: props.minHealthyPercent === undefined ? 50 : props.minHealthyPercent,
        deploymentCircuitBreaker: props.circuitBreaker ? {
          enable: props.circuitBreaker.enable ?? true,
          rollback: props.circuitBreaker.rollback ?? false,
        } : undefined,
        alarms: this._deploymentAlarms,
        strategy: props.deploymentStrategy,
        bakeTimeInMinutes: props.bakeTime?.toMinutes(),
        linearConfiguration: props.linearConfiguration ? {
          stepPercent: props.linearConfiguration.stepPercent,
          stepBakeTimeInMinutes: props.linearConfiguration.stepBakeTime?.toMinutes(),
        } : undefined,
        canaryConfiguration: props.canaryConfiguration ? {
          canaryPercent: props.canaryConfiguration.stepPercent,
          canaryBakeTimeInMinutes: props.canaryConfiguration.stepBakeTime?.toMinutes(),
        } : undefined,
        lifecycleHooks: Lazy.any({ produce: () => this.renderLifecycleHooks() }, { omitEmptyArray: true }),
      },
      propagateTags: propagateTagsFromSource === PropagatedTagSource.NONE ? undefined : props.propagateTags,
      enableEcsManagedTags: props.enableECSManagedTags ?? false,
      deploymentController: deploymentController,
      launchType: launchType,
      enableExecuteCommand: props.enableExecuteCommand,
      capacityProviderStrategy: props.capacityProviderStrategies,
      healthCheckGracePeriodSeconds: this.evaluateHealthGracePeriod(props.healthCheckGracePeriod),
      /* role: never specified, supplanted by Service Linked Role */
      networkConfiguration: this._networkConfigurationBox,
      serviceRegistries: this._serviceRegistries,
      serviceConnectConfiguration: this._serviceConnectConfig,
      volumeConfigurations: this._volumes.derive(_ => this.renderVolumes()),
      ...additionalProps,
    });

    this.node.addDependency(this.taskDefinition.taskRole);

    if (props.deploymentController?.type === DeploymentControllerType.EXTERNAL) {
      Annotations.of(this).addWarningV2('@aws-cdk/aws-ecs:externalDeploymentController', 'taskDefinition and launchType are blanked out when using external deployment controller.');
    }

    if (props.circuitBreaker && !this.isEcsDeploymentController) {
      Annotations.of(this)._addTrackableError(lit`CircuitBreakerRequiresEcsController`, 'Deployment circuit breaker requires the ECS deployment controller.');
    }

    if (!props.circuitBreaker && this.isEcsDeploymentController) {
      // If we *could* use a circuit breaker, then let's recommend users to do so. It makes detecting errors sooo much faster.
      Annotations.of(this).addWarningV2('@aws-cdk/aws-ecs:shouldUseCircuitBreaker', 'Enable the \'circuitBreaker\' property to trigger a quicker deployment failure if tasks are failing to start up (without this setting deployments may take up to 3 hours to fail).');
    }

    if (props.deploymentAlarms && !this.isEcsDeploymentController) {
      throw new ValidationError(lit`RequiresDeploymentAlarmsRequires`, 'Deployment alarms requires the ECS deployment controller.', this);
    }

    if (props.forceNewDeployment !== undefined) {
      if (!this.isEcsDeploymentController) {
        throw new ValidationError(lit`ForceNewDeploymentRequiresEcsController`, 'forceNewDeployment requires the ECS deployment controller.', this);
      }

      const enabled = props.forceNewDeployment.enabled;
      const nonce = props.forceNewDeployment.nonce;

      if (nonce !== undefined && !Token.isUnresolved(nonce) && (nonce.length < 1 || nonce.length > 255)) {
        throw new ValidationError(lit`ForceNewDeploymentNonceInvalidLength`, `forceNewDeployment nonce must be between 1 and 255 characters, got ${nonce.length}`, this);
      }

      this.resource.forceNewDeployment = {
        enableForceNewDeployment: enabled,
        forceNewDeploymentNonce: nonce,
      };
    }

    if (
      props.deploymentController?.type === DeploymentControllerType.CODE_DEPLOY
      && props.taskDefinitionRevision
      && props.taskDefinitionRevision !== TaskDefinitionRevision.LATEST
    ) {
      throw new ValidationError(lit`CodeDeployDeploymentControllerOnlyUsed`, 'CODE_DEPLOY deploymentController can only be used with the `latest` task definition revision', this);
    }

    if (props.minHealthyPercent === undefined) {
      Annotations.of(this).addWarningV2('@aws-cdk/aws-ecs:minHealthyPercent', 'minHealthyPercent has not been configured so the default value of 50% is used. The number of running tasks will decrease below the desired count during deployments etc. See https://github.com/aws/aws-cdk/issues/31705');
    }

    if (props.deploymentController?.type === DeploymentControllerType.CODE_DEPLOY) {
      // Strip the revision ID from the service's task definition property to
      // prevent new task def revisions in the stack from triggering updates
      // to the stack's ECS service resource
      this.resource.taskDefinition = taskDefinition.family;
      this.node.addDependency(taskDefinition);
    } else if (props.taskDefinitionRevision) {
      this.resource.taskDefinition = taskDefinition.family;
      if (props.taskDefinitionRevision !== TaskDefinitionRevision.LATEST) {
        this.resource.taskDefinition += `:${props.taskDefinitionRevision.revision}`;
      }
      this.node.addDependency(taskDefinition);
    }

    this.cluster = props.cluster;

    if (props.cloudMapOptions) {
      this.enableCloudMap(props.cloudMapOptions);
    }

    if (props.serviceConnectConfiguration) {
      this.enableServiceConnect(props.serviceConnectConfiguration);
    }

    if (props.volumeConfigurations) {
      props.volumeConfigurations.forEach(v => this.addVolume(v));
    }

    if (props.enableExecuteCommand) {
      this.enableExecuteCommand();

      const logging = this.cluster.executeCommandConfiguration?.logging ?? ExecuteCommandLogging.DEFAULT;

      if (this.cluster.executeCommandConfiguration?.kmsKey) {
        this.enableExecuteCommandEncryption(logging);
      }
      if (logging !== ExecuteCommandLogging.NONE) {
        this.executeCommandLogConfiguration();
      }
    }

    if (props.deploymentAlarms) {
      if (props.deploymentAlarms.alarmNames.length === 0) {
        throw new ValidationError(lit`IsRequiredLeastAlarmName`, 'at least one alarm name is required when specifying deploymentAlarms, received empty array', this);
      }
      this.deploymentAlarms = {
        alarmNames: props.deploymentAlarms.alarmNames,
        enable: true,
        rollback: props.deploymentAlarms.behavior !== AlarmBehavior.FAIL_ON_ALARM,
      };
      // CloudWatch alarms is only supported for Amazon ECS services that use the rolling update (ECS) deployment controller.
    } else if (this.isEcsDeploymentController && this.deploymentAlarmsAvailableInRegion()) {
      // Only set default deployment alarms settings when feature flag is not enabled.
      if (!FeatureFlags.of(this).isEnabled(cxapi.ECS_REMOVE_DEFAULT_DEPLOYMENT_ALARM)) {
        this.deploymentAlarms = {
          alarmNames: [],
          enable: false,
          rollback: false,
        };
      }
    }

    if (props.lifecycleHooks) {
      if (this.isEcsDeploymentController) {
        props.lifecycleHooks.forEach(target => this.addLifecycleHook(target));
      } else {
        throw new ValidationError(lit`RequiresDeploymentLifecycleHooks`, 'Deployment lifecycle hooks requires the ECS deployment controller.', this);
      }
    }

    this.node.defaultChild = this.resource;
  }

  /**
   * Forces a new deployment of the service.
   *
   * This can be used to trigger a deployment without changing the task definition or desired count.
   * ECS will start a new deployment even if there are no changes to the service configuration.
   *
   * **Important:** When called without a nonce, a timestamp is generated automatically, which means
   * every `cdk synth` produces a different template and every `cdk deploy` triggers a new deployment
   * regardless of whether any code has changed. To avoid this, provide a stable nonce value that only
   * changes when you intentionally want to force a redeployment (e.g., an image digest or a version string).
   *
   * @param nonce - A unique string (1-255 characters) that signals ECS to start a new deployment.
   * If not provided, a timestamp-based nonce is generated.
   */
  public forceNewDeployment(nonce?: string) {
    // ForceNewDeployment is only supported with the ECS deployment controller.
    // CODE_DEPLOY and EXTERNAL controllers manage deployments externally and do not
    // support this CloudFormation property. The AWS documentation does not explicitly
    // state this restriction; a documentation update has been requested.
    if (!this.isEcsDeploymentController) {
      throw new ValidationError(lit`EcsControllerRequired`, 'forceNewDeployment requires the ECS deployment controller.', this);
    }

    const resolvedNonce = nonce ?? new Date().toISOString();

    if (!Token.isUnresolved(resolvedNonce) && (resolvedNonce.length < 1 || resolvedNonce.length > 255)) {
      throw new ValidationError(lit`ForceDeploymentErrorInvalidLength`, `forceNewDeployment nonce must be between 1 and 255 characters, got ${resolvedNonce.length}`, this);
    }

    this.resource.forceNewDeployment = {
      enableForceNewDeployment: true,
      forceNewDeploymentNonce: resolvedNonce,
    };
  }

  /**
   * Add a deployment lifecycle hook target
   * @param target The lifecycle hook target to add
   */
  public addLifecycleHook(target: IDeploymentLifecycleHookTarget) {
    if (!this.isEcsDeploymentController) {
      throw new ValidationError(lit`RequiresDeploymentLifecycleHooks`, 'Deployment lifecycle hooks requires the ECS deployment controller.', this);
    }
    this._lifecycleHooks.push(target);
  }

  private renderLifecycleHooks(): CfnService.DeploymentLifecycleHookProperty[] {
    return this._lifecycleHooks.get().map((target) => {
      const config = target.bind(this);
      return {
        hookTargetArn: config.targetArn,
        roleArn: config.role!.roleArn,
        lifecycleStages: config.lifecycleStages.map(stage => stage.toString()),
      };
    });
  }

  /**
   * Adds a volume to the Service.
   */
  public addVolume(volume: ServiceManagedVolume) {
    this._volumes.push(volume);
  }

  private renderVolumes(): CfnService.ServiceVolumeConfigurationProperty[] {
    if (this._volumes.length > 1) {
      throw new ValidationError(lit`OnlyVolumeSpecifiedVolumeConfigurations`, `Only one EBS volume can be specified for 'volumeConfigurations', got: ${this._volumes.length}`, this);
    }
    return this._volumes.get().map(renderVolume);
    function renderVolume(spec: ServiceManagedVolume): CfnService.ServiceVolumeConfigurationProperty {
      const tagSpecifications = spec.config?.tagSpecifications?.map(ebsTagSpec => {
        return {
          resourceType: 'volume',
          propagateTags: ebsTagSpec.propagateTags,
          tags: ebsTagSpec.tags ? Object.entries(ebsTagSpec.tags).map(([key, value]) => ({
            key: key,
            value: value,
          })) : undefined,
        } as CfnService.EBSTagSpecificationProperty;
      });

      return {
        name: spec.name,
        managedEbsVolume: spec.config && {
          roleArn: spec.role.roleArn,
          encrypted: spec.config.encrypted,
          filesystemType: spec.config.fileSystemType,
          iops: spec.config.iops,
          kmsKeyId: spec.config.kmsKeyId?.keyId,
          throughput: spec.config.throughput,
          volumeType: spec.config.volumeType,
          snapshotId: spec.config.snapShotId,
          sizeInGiB: spec.config.size?.toGibibytes(),
          volumeInitializationRate: spec.config.volumeInitializationRate?.toMebibytes(),
          tagSpecifications: tagSpecifications,
        },
      };
    }
  }

  /**
   * Enable Deployment Alarms which take advantage of arbitrary alarms and configure them after service initialization.
   * If you have already enabled deployment alarms, this function can be used to tell ECS about additional alarms that
   * should interrupt a deployment.
   *
   * New alarms specified in subsequent calls of this function will be appended to the existing list of alarms.
   *
   * The same Alarm Behavior must be used on all deployment alarms. If you specify different AlarmBehavior values in
   * multiple calls to this function, or the Alarm Behavior used here doesn't match the one used in the service
   * constructor, an error will be thrown.
   *
   * If the alarm's metric references the service, you cannot pass `Alarm.alarmName` here. That will cause a circular
   * dependency between the service and its deployment alarm. See this package's README for options to alarm on service
   * metrics, and avoid this circular dependency.
   *
   */
  public enableDeploymentAlarms(alarmNames: string[], options?: DeploymentAlarmOptions) {
    if (alarmNames.length === 0) {
      throw new ValidationError(lit`IsRequiredLeastAlarmName`, 'at least one alarm name is required when calling enableDeploymentAlarms(), received empty array', this);
    }

    alarmNames.forEach(alarmName => {
      if (Token.isUnresolved(alarmName)) {
        Annotations.of(this).addInfo(
          `Deployment alarm (${JSON.stringify(this.stack.resolve(alarmName))}) enabled on ${this.node.id} may cause a circular dependency error when this stack deploys. The alarm name references the alarm's logical id, or another resource. See the 'Deployment alarms' section in the module README for more details.`,
        );
      }
    });

    if (this.deploymentAlarms?.enable && options?.behavior) {
      if (
        (AlarmBehavior.ROLLBACK_ON_ALARM === options.behavior && !this.deploymentAlarms.rollback) ||
        (AlarmBehavior.FAIL_ON_ALARM === options.behavior && this.deploymentAlarms.rollback)
      ) {
        throw new ValidationError(lit`DeploymentAlarmsServiceSame`, `all deployment alarms on an ECS service must have the same AlarmBehavior. Attempted to enable deployment alarms with ${options.behavior}, but alarms were previously enabled with ${this.deploymentAlarms.rollback ? AlarmBehavior.ROLLBACK_ON_ALARM : AlarmBehavior.FAIL_ON_ALARM}`, this);
      }
    }

    if (!this.deploymentAlarms?.enable) {
      this.deploymentAlarms = {
        enable: true,
        alarmNames: alarmNames,
        rollback: options?.behavior !== AlarmBehavior.FAIL_ON_ALARM,
      };
    } else {
      // If deployment alarms have previously been enabled, we only need to add
      // the new alarm names, since rollback behaviors can't be updated/mixed.
      this.deploymentAlarms.alarmNames.concat(alarmNames);
    }
  }

  /**
   * Enable Service Connect on this service.
   */
  public enableServiceConnect(config?: ServiceConnectProps) {
    if (this._serviceConnectConfig.get() !== undefined) {
      throw new ValidationError(lit`ServiceConnectConfigurationCannot`, 'Service connect configuration cannot be specified more than once.', this);
    }

    this.validateServiceConnectConfiguration(config);

    let cfg = config || {};

    /**
     * Namespace already exists as validated in validateServiceConnectConfiguration.
     * Resolve which namespace to use by picking:
     * 1. The namespace defined in service connect config.
     * 2. The namespace defined in the cluster's defaultCloudMapNamespace property.
     */
    let namespace;
    if (this.cluster.defaultCloudMapNamespace) {
      namespace = this.cluster.defaultCloudMapNamespace.namespaceName;
    }

    if (cfg.namespace) {
      namespace = cfg.namespace;
    }

    /**
     * Map services to CFN property types. This block manages:
     * 1. Finding the correct port.
     * 2. Client alias enumeration
     */
    const services = cfg.services?.map(svc => {
      const containerPort = this.taskDefinition.findPortMappingByName(svc.portMappingName)?.containerPort;
      if (!containerPort) {
        throw new ValidationError(lit`PortMappingNameDoes`, `Port mapping with name ${svc.portMappingName} does not exist.`, this);
      }
      const alias = {
        port: svc.port || containerPort,
        dnsName: svc.dnsName,
      };

      const tls: CfnService.ServiceConnectTlsConfigurationProperty | undefined = svc.tls ? {
        issuerCertificateAuthority: {
          awsPcaAuthorityArn: svc.tls.awsPcaAuthorityArn,
        },
        kmsKey: svc.tls.kmsKey?.keyRef.keyArn,
        roleArn: svc.tls.role?.roleArn,
      } : undefined;

      return {
        portName: svc.portMappingName,
        discoveryName: svc.discoveryName,
        ingressPortOverride: svc.ingressPortOverride,
        clientAliases: [alias],
        timeout: this.renderTimeout(svc.idleTimeout, svc.perRequestTimeout),
        tls,
      } as CfnService.ServiceConnectServiceProperty;
    });

    let logConfig: LogDriverConfig | undefined;
    if (cfg.logDriver && this.taskDefinition.defaultContainer) {
      // Default container existence is validated in validateServiceConnectConfiguration.
      // We only need the default container so that bind() can get the task definition from the container definition.
      logConfig = cfg.logDriver.bind(this, this.taskDefinition.defaultContainer);
    }

    this._serviceConnectConfig.set({
      enabled: true,
      logConfiguration: logConfig,
      namespace: namespace,
      services: services,
      accessLogConfiguration: cfg.accessLogConfiguration ? {
        format: cfg.accessLogConfiguration.format,
        includeQueryParameters: cfg.accessLogConfiguration.includeQueryParameters ? 'ENABLED' : 'DISABLED',
      } : undefined,
    });
  }

  /**
   * Validate Service Connect Configuration
   */
  private validateServiceConnectConfiguration(config?: ServiceConnectProps) {
    if (!this.taskDefinition.defaultContainer) {
      throw new ValidationError(lit`TaskDefinitionLeastContainer`, 'Task definition must have at least one container to enable service connect.', this);
    }

    // Check the implicit enable case; when config isn't specified or namespace isn't specified, we need to check that there is a namespace on the cluster.
    if ((!config || !config.namespace) && !this.cluster.defaultCloudMapNamespace) {
      throw new ValidationError(lit`MustBeNamespaceDefinedEither`, 'Namespace must be defined either in serviceConnectConfig or cluster.defaultCloudMapNamespace', this);
    }

    // When config isn't specified, return.
    if (!config) {
      return;
    }

    // accessLogConfiguration controls the format of Envoy proxy access logs, but the actual
    // log delivery is handled by logDriver (logConfiguration). Without logDriver, the Envoy
    // sidecar has no log driver and its stdout is dropped, access logs never reach any destination.
    if (config.accessLogConfiguration && !config.logDriver) {
      throw new ValidationError(lit`AccessLogConfigurationRequiresLogDriver`,
        'accessLogConfiguration requires logDriver to be set. Without logDriver, access logs are not delivered to any destination.',
        this,
      );
    }

    if (!config.services) {
      return;
    }
    let portNames = new Map<string, string[]>();
    config.services.forEach(serviceConnectService => {
      // port must exist on the task definition
      if (!this.taskDefinition.findPortMappingByName(serviceConnectService.portMappingName)) {
        throw new ValidationError(lit`PortMappingDoesExist`, `Port Mapping '${serviceConnectService.portMappingName}' does not exist on the task definition.`, this);
      }

      // Check that no two service connect services use the same discovery name.
      const discoveryName = serviceConnectService.discoveryName || serviceConnectService.portMappingName;
      if (portNames.get(serviceConnectService.portMappingName)?.includes(discoveryName)) {
        throw new ValidationError(lit`CannotCreateMultiple`, `Cannot create multiple services with the discoveryName '${discoveryName}'.`, this);
      }

      let currentDiscoveries = portNames.get(serviceConnectService.portMappingName);
      if (!currentDiscoveries) {
        portNames.set(serviceConnectService.portMappingName, [discoveryName]);
      } else {
        currentDiscoveries.push(discoveryName);
        portNames.set(serviceConnectService.portMappingName, currentDiscoveries);
      }

      // IngressPortOverride should be within the valid port range if it exists.
      if (serviceConnectService.ingressPortOverride && !this.isValidPort(serviceConnectService.ingressPortOverride)) {
        throw new ValidationError(lit`IngressPortOverrideValid`, `ingressPortOverride ${serviceConnectService.ingressPortOverride} is not valid.`, this);
      }

      // clientAlias.port should be within the valid port range
      if (serviceConnectService.port &&
        !this.isValidPort(serviceConnectService.port)) {
        throw new ValidationError(lit`ClientAliasPortValid`, `Client Alias port ${serviceConnectService.port} is not valid.`, this);
      }

      // tls.awsPcaAuthorityArn should be an ARN
      const awsPcaAuthorityArn = serviceConnectService.tls?.awsPcaAuthorityArn;
      if (awsPcaAuthorityArn && !Token.isUnresolved(awsPcaAuthorityArn) && !awsPcaAuthorityArn.startsWith('arn:')) {
        throw new ValidationError(lit`AwsPcaAuthorityArnStartArnLeast`, `awsPcaAuthorityArn must start with "arn:" and have at least 6 components; received ${awsPcaAuthorityArn}`, this);
      }
    });
  }

  /**
   * Validate Canary Configuration
   */
  private validateCanaryConfiguration(config: TrafficShiftConfig) {
    if (this.deploymentStrategy !== DeploymentStrategy.CANARY) {
      throw new ValidationError(lit`RequiresCanaryConfigurationRequires`, 'Canary configuration requires deploymentStrategy to be set to CANARY', this);
    }

    if (config.stepPercent !== undefined && !Token.isUnresolved(config.stepPercent)) {
      if (!Number.isFinite(config.stepPercent) || config.stepPercent < 0.1 || config.stepPercent > 100.0) {
        throw new ValidationError(lit`MustBeCanaryDeploymentStepPercent`, `Canary deployment stepPercent must be between 0.1 and 100.0, received ${config.stepPercent}`, this);
      }
      if (!Number.isInteger(config.stepPercent * 10)) {
        throw new ValidationError(lit`MustBeCanaryDeploymentStepPercent`, `Canary deployment stepPercent must be a multiple of 0.1, received ${config.stepPercent}`, this);
      }
    }

    if (config.stepBakeTime !== undefined && !config.stepBakeTime.isUnresolved()) {
      const minutes = config.stepBakeTime.toMinutes({ integral: false });
      if (!Number.isInteger(minutes)) {
        throw new ValidationError(lit`MustBeCanaryDeploymentStepBakeTime`, `Canary deployment stepBakeTime must be a whole number of minutes, received ${minutes} minutes`, this);
      }
      if (minutes < 0 || minutes > 1440) {
        throw new ValidationError(lit`MustBeCanaryDeploymentStepBakeTime`, `Canary deployment stepBakeTime must be between 0 and 1440 minutes, received ${minutes} minutes`, this);
      }
    }
  }

  /**
   * Validate Linear Configuration
   */
  private validateLinearConfiguration(config: TrafficShiftConfig) {
    if (this.deploymentStrategy !== DeploymentStrategy.LINEAR) {
      throw new ValidationError(lit`RequiresLinearConfigurationRequires`, 'Linear configuration requires deploymentStrategy to be set to LINEAR', this);
    }

    if (config.stepPercent !== undefined && !Token.isUnresolved(config.stepPercent)) {
      if (!Number.isFinite(config.stepPercent) || config.stepPercent < 3.0 || config.stepPercent > 100.0) {
        throw new ValidationError(lit`MustBeLinearDeploymentStepPercent`, `Linear deployment stepPercent must be between 3.0 and 100.0, received ${config.stepPercent}`, this);
      }
      if (!Number.isInteger(config.stepPercent * 10)) {
        throw new ValidationError(lit`MustBeLinearDeploymentStepPercent`, `Linear deployment stepPercent must be a multiple of 0.1, received ${config.stepPercent}`, this);
      }
    }

    if (config.stepBakeTime !== undefined && !config.stepBakeTime.isUnresolved()) {
      const minutes = config.stepBakeTime.toMinutes({ integral: false });
      if (!Number.isInteger(minutes)) {
        throw new ValidationError(lit`MustBeLinearDeploymentStepBakeTime`, `Linear deployment stepBakeTime must be a whole number of minutes, received ${minutes} minutes`, this);
      }
      if (minutes < 0 || minutes > 1440) {
        throw new ValidationError(lit`MustBeLinearDeploymentStepBakeTime`, `Linear deployment stepBakeTime must be between 0 and 1440 minutes, received ${minutes} minutes`, this);
      }
    }
  }

  /**
   * Determines if a port is valid
   *
   * @param port: The port number
   * @returns boolean whether the port is valid
   */
  private isValidPort(port?: number): boolean {
    return !!(port && Number.isInteger(port) && port >= BaseService.MIN_PORT && port <= BaseService.MAX_PORT);
  }

  /**
   * The CloudMap service created for this service, if any.
   */
  public get cloudMapService(): cloudmap.IService | undefined {
    return this.cloudmapService;
  }

  private getDeploymentController(props: BaseServiceProps): DeploymentController | undefined {
    if (props.deploymentController) {
      // The customer is always right
      return props.deploymentController;
    }
    const disableCircuitBreakerEcsDeploymentControllerFeatureFlag =
      FeatureFlags.of(this).isEnabled(cxapi.ECS_DISABLE_EXPLICIT_DEPLOYMENT_CONTROLLER_FOR_CIRCUIT_BREAKER);

    if (!disableCircuitBreakerEcsDeploymentControllerFeatureFlag && props.circuitBreaker) {
      // This is undesirable behavior (the controller is implicitly ECS anyway when left
      // undefined, so specifying it is not necessary but DOES trigger a CFN replacement)
      // but we leave it in for backwards compat.
      return {
        type: DeploymentControllerType.ECS,
      };
    }

    return undefined;
  }

  private executeCommandLogConfiguration() {
    const reducePermissions = FeatureFlags.of(this).isEnabled(cxapi.REDUCE_EC2_FARGATE_CLOUDWATCH_PERMISSIONS);
    const logConfiguration = this.cluster.executeCommandConfiguration?.logConfiguration;

    // When Feature Flag is false, keep the previous behaviour for non-breaking changes.
    // When Feature Flag is true and when cloudwatch log group is specified in logConfiguration, then
    // append the necessary permissions to the task definition.
    if (!reducePermissions || logConfiguration?.cloudWatchLogGroup) {
      this.taskDefinition.addToTaskRolePolicy(new iam.PolicyStatement({
        actions: [
          'logs:DescribeLogGroups',
        ],
        resources: ['*'],
      }));

      const logGroupArn = logConfiguration?.cloudWatchLogGroup ? `arn:${this.stack.partition}:logs:${this.env.region}:${this.env.account}:log-group:${logConfiguration.cloudWatchLogGroup.logGroupRef.logGroupName}:*` : '*';
      this.taskDefinition.addToTaskRolePolicy(new iam.PolicyStatement({
        actions: [
          'logs:CreateLogStream',
          'logs:DescribeLogStreams',
          'logs:PutLogEvents',
        ],
        resources: [logGroupArn],
      }));
    }

    if (logConfiguration?.s3Bucket?.bucketRef.bucketName) {
      this.taskDefinition.addToTaskRolePolicy(new iam.PolicyStatement({
        actions: [
          's3:GetBucketLocation',
        ],
        resources: ['*'],
      }));
      this.taskDefinition.addToTaskRolePolicy(new iam.PolicyStatement({
        actions: [
          's3:PutObject',
        ],
        resources: [`arn:${this.stack.partition}:s3:::${logConfiguration.s3Bucket.bucketRef.bucketName}/*`],
      }));
      if (logConfiguration.s3EncryptionEnabled) {
        this.taskDefinition.addToTaskRolePolicy(new iam.PolicyStatement({
          actions: [
            's3:GetEncryptionConfiguration',
          ],
          resources: [`arn:${this.stack.partition}:s3:::${logConfiguration.s3Bucket.bucketRef.bucketName}`],
        }));
      }
    }
  }

  private enableExecuteCommandEncryption(logging: ExecuteCommandLogging) {
    this.taskDefinition.addToTaskRolePolicy(new iam.PolicyStatement({
      actions: [
        'kms:Decrypt',
        'kms:GenerateDataKey',
      ],
      resources: [`${this.cluster.executeCommandConfiguration?.kmsKey?.keyArn}`],
    }));

    this.cluster.executeCommandConfiguration?.kmsKey?.addToResourcePolicy(new iam.PolicyStatement({
      actions: [
        'kms:*',
      ],
      resources: ['*'],
      principals: [new iam.ArnPrincipal(`arn:${this.stack.partition}:iam::${this.env.account}:root`)],
    }));

    if (logging === ExecuteCommandLogging.DEFAULT || this.cluster.executeCommandConfiguration?.logConfiguration?.cloudWatchEncryptionEnabled) {
      this.cluster.executeCommandConfiguration?.kmsKey?.addToResourcePolicy(new iam.PolicyStatement({
        actions: [
          'kms:Encrypt*',
          'kms:Decrypt*',
          'kms:ReEncrypt*',
          'kms:GenerateDataKey*',
          'kms:Describe*',
        ],
        resources: ['*'],
        principals: [new iam.ServicePrincipal(`logs.${this.env.region}.amazonaws.com`)],
        conditions: {
          ArnLike: { 'kms:EncryptionContext:aws:logs:arn': `arn:${this.stack.partition}:logs:${this.env.region}:${this.env.account}:*` },
        },
      }));
    }
  }

  /**
   * This method is called to attach this service to an Application Load Balancer.
   *
   * Don't call this function directly. Instead, call `listener.addTargets()`
   * to add this service to a load balancer.
   */
  public attachToApplicationTargetGroup(targetGroup: elbv2.IApplicationTargetGroup): elbv2.LoadBalancerTargetProps {
    return this.defaultLoadBalancerTarget.attachToApplicationTargetGroup(targetGroup);
  }

  /**
   * Registers the service as a target of a Classic Load Balancer (CLB).
   *
   * Don't call this. Call `loadBalancer.addTarget()` instead.
   *
   * @param loadBalancer [disable-awslint:ref-via-interface]
   */
  public attachToClassicLB(loadBalancer: elb.LoadBalancer): void {
    return this.defaultLoadBalancerTarget.attachToClassicLB(loadBalancer);
  }

  /**
   * Return a load balancing target for a specific container and port.
   *
   * Use this function to create a load balancer target if you want to load balance to
   * another container than the first essential container or the first mapped port on
   * the container.
   *
   * Use the return value of this function where you would normally use a load balancer
   * target, instead of the `Service` object itself.
   *
   * @example
   *
   * declare const listener: elbv2.ApplicationListener;
   * declare const service: ecs.BaseService;
   * listener.addTargets('ECS', {
   *   port: 80,
   *   targets: [service.loadBalancerTarget({
   *     containerName: 'MyContainer',
   *     containerPort: 1234,
   *   })],
   * });
   */
  public loadBalancerTarget(options: LoadBalancerTargetOptions): IEcsLoadBalancerTarget {
    if (options.alternateTarget && !this.isEcsDeploymentController) {
      throw new ValidationError(lit`RequiresDeploymentLifecycleHooks`, 'Deployment lifecycle hooks requires the ECS deployment controller.', this);
    }

    const self = this;
    const target = this.taskDefinition._validateTarget(options);
    const connections = self.connections;
    return {
      attachToApplicationTargetGroup(targetGroup: elbv2.ApplicationTargetGroup): elbv2.LoadBalancerTargetProps {
        targetGroup.registerConnectable(self, self.taskDefinition._portRangeFromPortMapping(target.portMapping));
        return self.attachToELBv2(targetGroup, target.containerName, target.portMapping.containerPort!, options.alternateTarget);
      },
      attachToNetworkTargetGroup(targetGroup: elbv2.NetworkTargetGroup): elbv2.LoadBalancerTargetProps {
        return self.attachToELBv2(targetGroup, target.containerName, target.portMapping.containerPort!, options.alternateTarget);
      },
      connections,
      attachToClassicLB(loadBalancer: elb.LoadBalancer): void {
        return self.attachToELB(loadBalancer, target.containerName, target.portMapping.containerPort!);
      },
    };
  }

  /**
   * Use this function to create all load balancer targets to be registered in this service, add them to
   * target groups, and attach target groups to listeners accordingly.
   *
   * Alternatively, you can use `listener.addTargets()` to create targets and add them to target groups.
   *
   * @example
   *
   * declare const listener: elbv2.ApplicationListener;
   * declare const service: ecs.BaseService;
   * service.registerLoadBalancerTargets(
   *   {
   *     containerName: 'web',
   *     containerPort: 80,
   *     newTargetGroupId: 'ECS',
   *     listener: ecs.ListenerConfig.applicationListener(listener, {
   *       protocol: elbv2.ApplicationProtocol.HTTPS
   *     }),
   *   },
   * )
   */
  public registerLoadBalancerTargets(...targets: EcsTarget[]) {
    for (const target of targets) {
      target.listener.addTargets(target.newTargetGroupId, {
        containerName: target.containerName,
        containerPort: target.containerPort,
        protocol: target.protocol,
      }, this);
    }
  }

  /**
   * This method is called to attach this service to a Network Load Balancer.
   *
   * Don't call this function directly. Instead, call `listener.addTargets()`
   * to add this service to a load balancer.
   */
  public attachToNetworkTargetGroup(targetGroup: elbv2.INetworkTargetGroup): elbv2.LoadBalancerTargetProps {
    return this.defaultLoadBalancerTarget.attachToNetworkTargetGroup(targetGroup);
  }

  /**
   * An attribute representing the minimum and maximum task count for an AutoScalingGroup.
   */
  public autoScaleTaskCount(props: appscaling.EnableScalingProps) {
    if (this.scalableTaskCount) {
      throw new ValidationError(lit`AutoScalingTaskCountAlready`, 'AutoScaling of task count already enabled for this service', this);
    }

    return this.scalableTaskCount = new ScalableTaskCount(this, 'TaskCount', {
      serviceNamespace: appscaling.ServiceNamespace.ECS,
      resourceId: `service/${this.cluster.clusterName}/${this.serviceName}`,
      dimension: 'ecs:service:DesiredCount',
      role: this.makeAutoScalingRole(),
      ...props,
    });
  }

  /**
   * Enable CloudMap service discovery for the service
   *
   * @returns The created CloudMap service
   */
  public enableCloudMap(options: CloudMapOptions): cloudmap.Service {
    const sdNamespace = options.cloudMapNamespace ?? this.cluster.defaultCloudMapNamespace;
    if (sdNamespace === undefined) {
      throw new ValidationError(lit`CannotEnableService`, 'Cannot enable service discovery if a Cloudmap Namespace has not been created in the cluster.', this);
    }

    if (sdNamespace.type === cloudmap.NamespaceType.HTTP) {
      throw new ValidationError(lit`CannotEnableService`, 'Cannot enable DNS service discovery for HTTP Cloudmap Namespace.', this);
    }

    // Determine DNS type based on network mode
    const networkMode = this.taskDefinition.networkMode;
    if (networkMode === NetworkMode.NONE) {
      throw new ValidationError(lit`CannotServiceDiscovery`, 'Cannot use a service discovery if NetworkMode is None. Use Bridge, Host or AwsVpc instead.', this);
    }

    // Bridge or host network mode requires SRV records
    let dnsRecordType = options.dnsRecordType;

    if (networkMode === NetworkMode.BRIDGE || networkMode === NetworkMode.HOST) {
      if (dnsRecordType === undefined) {
        dnsRecordType = cloudmap.DnsRecordType.SRV;
      }
      if (dnsRecordType !== cloudmap.DnsRecordType.SRV) {
        throw new ValidationError(lit`MustBeRecordsUsedNetwork`, 'SRV records must be used when network mode is Bridge or Host.', this);
      }
    }

    // Default DNS record type for AwsVpc network mode is A Records
    if (networkMode === NetworkMode.AWS_VPC) {
      if (dnsRecordType === undefined) {
        dnsRecordType = cloudmap.DnsRecordType.A;
      }
    }

    const { containerName, containerPort } = determineContainerNameAndPort(this, {
      taskDefinition: this.taskDefinition,
      dnsRecordType: dnsRecordType!,
      container: options.container,
      containerPort: options.containerPort,
    });

    const cloudmapService = new cloudmap.Service(this, 'CloudmapService', {
      namespace: sdNamespace,
      name: options.name,
      dnsRecordType: dnsRecordType!,
      customHealthCheck: { failureThreshold: options.failureThreshold || 1 },
      dnsTtl: options.dnsTtl,
    });

    const serviceArn = cloudmapService.serviceArn;

    // add Cloudmap service to the ECS Service's serviceRegistry
    this.addServiceRegistry({
      arn: serviceArn,
      containerName,
      containerPort,
    });

    this.cloudmapService = cloudmapService;

    return cloudmapService;
  }

  /**
   * Associates this service with a CloudMap service
   */
  public associateCloudMapService(options: AssociateCloudMapServiceOptions): void {
    const service = options.service;

    const { containerName, containerPort } = determineContainerNameAndPort(this, {
      taskDefinition: this.taskDefinition,
      dnsRecordType: service.dnsRecordType,
      container: options.container,
      containerPort: options.containerPort,
    });

    // add Cloudmap service to the ECS Service's serviceRegistry
    this.addServiceRegistry({
      arn: service.serviceArn,
      containerName,
      containerPort,
    });
  }

  /**
   * This method returns the specified CloudWatch metric name for this service.
   */
  public metric(metricName: string, props?: cloudwatch.MetricOptions): cloudwatch.Metric {
    return new cloudwatch.Metric({
      namespace: 'AWS/ECS',
      metricName,
      dimensionsMap: { ClusterName: this.cluster.clusterName, ServiceName: this.serviceName },
      ...props,
    }).attachTo(this);
  }

  /**
   * This method returns the CloudWatch metric for this service's memory utilization.
   *
   * @default average over 5 minutes
   */
  public metricMemoryUtilization(props?: cloudwatch.MetricOptions): cloudwatch.Metric {
    return this.metric('MemoryUtilization', props);
  }

  /**
   * This method returns the CloudWatch metric for this service's CPU utilization.
   *
   * @default average over 5 minutes
   */
  public metricCpuUtilization(props?: cloudwatch.MetricOptions): cloudwatch.Metric {
    return this.metric('CPUUtilization', props);
  }

  /**
   * This method is called to create a networkConfiguration.
   * @deprecated use configureAwsVpcNetworkingWithSecurityGroups instead.
   */

  protected configureAwsVpcNetworking(vpc: ec2.IVpc, assignPublicIp?: boolean, vpcSubnets?: ec2.SubnetSelection, securityGroup?: ec2.ISecurityGroup) {
    if (vpcSubnets === undefined) {
      vpcSubnets = assignPublicIp ? { subnetType: ec2.SubnetType.PUBLIC } : {};
    }
    if (securityGroup === undefined) {
      securityGroup = new ec2.SecurityGroup(this, 'SecurityGroup', { vpc });
    }
    this.connections.addSecurityGroup(securityGroup);

    this.networkConfiguration = {
      awsvpcConfiguration: {
        assignPublicIp: assignPublicIp ? 'ENABLED' : 'DISABLED',
        subnets: vpc.selectSubnets(vpcSubnets).subnetIds,
        securityGroups: [securityGroup!.securityGroupId],
      },
    };
  }

  /**
   * This method is called to create a networkConfiguration.
   */
  // eslint-disable-next-line max-len
  protected configureAwsVpcNetworkingWithSecurityGroups(vpc: ec2.IVpc, assignPublicIp?: boolean, vpcSubnets?: ec2.SubnetSelection, securityGroups?: ec2.ISecurityGroup[]) {
    if (vpcSubnets === undefined) {
      vpcSubnets = assignPublicIp ? { subnetType: ec2.SubnetType.PUBLIC } : {};
    }
    if (securityGroups === undefined || securityGroups.length === 0) {
      securityGroups = [new ec2.SecurityGroup(this, 'SecurityGroup', { vpc })];
    }

    securityGroups.forEach((sg) => { this.connections.addSecurityGroup(sg); }, this);

    this.networkConfiguration = {
      awsvpcConfiguration: {
        assignPublicIp: assignPublicIp ? 'ENABLED' : 'DISABLED',
        subnets: vpc.selectSubnets(vpcSubnets).subnetIds,
        securityGroups: securityGroups.map((sg) => sg.securityGroupId),
      },
    };
  }

  private renderServiceRegistry(registry: ServiceRegistry): CfnService.ServiceRegistryProperty {
    return {
      registryArn: registry.arn,
      containerName: registry.containerName,
      containerPort: registry.containerPort,
    };
  }

  /**
   * Shared logic for attaching to an ELB
   */
  private attachToELB(loadBalancer: elb.LoadBalancer, containerName: string, containerPort: number): void {
    if (this.taskDefinition.networkMode === NetworkMode.AWS_VPC) {
      throw new ValidationError(lit`CannotClassicLoad`, 'Cannot use a Classic Load Balancer if NetworkMode is AwsVpc. Use Host or Bridge instead.', this);
    }
    if (this.taskDefinition.networkMode === NetworkMode.NONE) {
      throw new ValidationError(lit`CannotClassicLoad`, 'Cannot use a Classic Load Balancer if NetworkMode is None. Use Host or Bridge instead.', this);
    }

    this._loadBalancers.push({
      loadBalancerName: loadBalancer.loadBalancerName,
      containerName,
      containerPort,
    });
  }

  /**
   * Shared logic for attaching to an ELBv2
   */
  private attachToELBv2(
    targetGroup: elbv2.ITargetGroup,
    containerName: string,
    containerPort: number,
    alternateTarget?: IAlternateTarget): elbv2.LoadBalancerTargetProps {
    if (this.taskDefinition.networkMode === NetworkMode.NONE) {
      throw new ValidationError(lit`CannotLoadBalancer`, 'Cannot use a load balancer if NetworkMode is None. Use Bridge, Host or AwsVpc instead.', this);
    }

    const advancedConfiguration = alternateTarget?.bind(this);
    this._loadBalancers.push({
      targetGroupArn: targetGroup.targetGroupArn,
      containerName,
      containerPort,
      advancedConfiguration,
    });

    // Service creation can only happen after the load balancer has
    // been associated with our target group(s), so add ordering dependency.
    this.resource.node.addDependency(targetGroup.loadBalancerAttached);

    const targetType = this.taskDefinition.networkMode === NetworkMode.AWS_VPC ? elbv2.TargetType.IP : elbv2.TargetType.INSTANCE;
    return { targetType };
  }

  private get defaultLoadBalancerTarget() {
    return this.loadBalancerTarget({
      containerName: this.taskDefinition.defaultContainer!.containerName,
    });
  }

  /**
   * Generate the role that will be used for autoscaling this service
   */
  private makeAutoScalingRole(): iam.IRole {
    // Use a Service Linked Role.
    return iam.Role.fromRoleArn(this, 'ScalingRole', Stack.of(this).formatArn({
      region: '',
      service: 'iam',
      resource: 'role/aws-service-role/ecs.application-autoscaling.amazonaws.com',
      resourceName: 'AWSServiceRoleForApplicationAutoScaling_ECSService',
    }));
  }

  /**
   * Associate Service Discovery (Cloud Map) service
   */
  private addServiceRegistry(registry: ServiceRegistry) {
    if (this._serviceRegistries.length >= 1) {
      throw new ValidationError(lit`CannotAssociateGiven`, 'Cannot associate with the given service discovery registry. ECS supports at most one service registry per service.', this);
    }

    const sr = this.renderServiceRegistry(registry);
    this._serviceRegistries.push(sr);
  }

  /**
   *  Return the default grace period when load balancers are configured and
   *  healthCheckGracePeriod is not already set
   */
  private evaluateHealthGracePeriod(providedHealthCheckGracePeriod?: Duration): IResolvable {
    return this._loadBalancers.derive(
      lbs => providedHealthCheckGracePeriod?.toSeconds() ?? (lbs.length > 0 ? 60 : undefined),
    );
  }

  private enableExecuteCommand() {
    this.taskDefinition.addToTaskRolePolicy(new iam.PolicyStatement({
      actions: [
        'ssmmessages:CreateControlChannel',
        'ssmmessages:CreateDataChannel',
        'ssmmessages:OpenControlChannel',
        'ssmmessages:OpenDataChannel',
      ],
      resources: ['*'],
    }));
  }

  private deploymentAlarmsAvailableInRegion(): boolean {
    const unsupportedPartitions = ['aws-cn', 'aws-us-gov', 'aws-iso', 'aws-iso-b'];
    const currentRegion = RegionInfo.get(this.stack.resolve(this.stack.region));
    if (currentRegion.partition) {
      return !unsupportedPartitions.includes(currentRegion.partition);
    }
    return true;
  }

  private renderTimeout(idleTimeout?: Duration, perRequestTimeout?: Duration): CfnService.TimeoutConfigurationProperty | undefined {
    if (!idleTimeout && !perRequestTimeout) return undefined;
    if (idleTimeout && idleTimeout.toMilliseconds() > 0 && idleTimeout.toMilliseconds() < Duration.seconds(1).toMilliseconds()) {
      throw new ValidationError(lit`MustBeIdleTimeoutLeastSecond`, `idleTimeout must be at least 1 second or 0 to disable it, got ${idleTimeout.toMilliseconds()}ms.`, this);
    }
    if (perRequestTimeout && perRequestTimeout.toMilliseconds() > 0 && perRequestTimeout.toMilliseconds() < Duration.seconds(1).toMilliseconds()) {
      throw new ValidationError(lit`MustBePerRequestTimeoutLeastSecond`, `perRequestTimeout must be at least 1 second or 0 to disable it, got ${perRequestTimeout.toMilliseconds()}ms.`, this);
    }
    return {
      idleTimeoutSeconds: idleTimeout?.toSeconds(),
      perRequestTimeoutSeconds: perRequestTimeout?.toSeconds(),
    };
  }

  /**
   * Checks if the service is using the ECS deployment controller.
   * @returns true if the service is using the ECS deployment controller or if no deployment controller is specified (defaults to ECS)
   */
  public isUsingECSDeploymentController(): boolean {
    return this.isEcsDeploymentController;
  }
}

/**
 * The options to enabling AWS Cloud Map for an Amazon ECS service.
 */
export interface CloudMapOptions {
  /**
   * The name of the Cloud Map service to attach to the ECS service.
   *
   * @default CloudFormation-generated name
   */
  readonly name?: string;

  /**
   * The service discovery namespace for the Cloud Map service to attach to the ECS service.
   *
   * @default - the defaultCloudMapNamespace associated to the cluster
   */
  readonly cloudMapNamespace?: cloudmap.INamespace;

  /**
   * The DNS record type that you want AWS Cloud Map to create. The supported record types are A or SRV.
   *
   * @default - DnsRecordType.A if TaskDefinition.networkMode = AWS_VPC, otherwise DnsRecordType.SRV
   */
  readonly dnsRecordType?: cloudmap.DnsRecordType.A | cloudmap.DnsRecordType.SRV;

  /**
   * The amount of time that you want DNS resolvers to cache the settings for this record.
   *
   * @default Duration.minutes(1)
   */
  readonly dnsTtl?: Duration;

  /**
   * The number of 30-second intervals that you want Cloud Map to wait after receiving an UpdateInstanceCustomHealthStatus
   * request before it changes the health status of a service instance.
   *
   * NOTE: This is used for HealthCheckCustomConfig
   */
  readonly failureThreshold?: number;

  /**
   * The container to point to for a SRV record.
   * @default - the task definition's default container
   */
  readonly container?: ContainerDefinition;

  /**
   * The port to point to for a SRV record.
   * @default - the default port of the task definition's default container
   */
  readonly containerPort?: number;
}

/**
 * The options for using a cloudmap service.
 */
export interface AssociateCloudMapServiceOptions {
  /**
   * The cloudmap service to register with.
   */
  readonly service: cloudmap.IService;

  /**
   * The container to point to for a SRV record.
   * @default - the task definition's default container
   */
  readonly container?: ContainerDefinition;

  /**
   * The port to point to for a SRV record.
   * @default - the default port of the task definition's default container
   */
  readonly containerPort?: number;
}

/**
 * Service Registry for ECS service
 */
interface ServiceRegistry {
  /**
   * Arn of the Cloud Map Service that will register a Cloud Map Instance for your ECS Service
   */
  readonly arn: string;

  /**
   * The container name value, already specified in the task definition, to be used for your service discovery service.
   * If the task definition that your service task specifies uses the bridge or host network mode,
   * you must specify a containerName and containerPort combination from the task definition.
   * If the task definition that your service task specifies uses the awsvpc network mode and a type SRV DNS record is
   * used, you must specify either a containerName and containerPort combination or a port value, but not both.
   */
  readonly containerName?: string;

  /**
   * The container port value, already specified in the task definition, to be used for your service discovery service.
   * If the task definition that your service task specifies uses the bridge or host network mode,
   * you must specify a containerName and containerPort combination from the task definition.
   * If the task definition that your service task specifies uses the awsvpc network mode and a type SRV DNS record is
   * used, you must specify either a containerName and containerPort combination or a port value, but not both.
   */
  readonly containerPort?: number;
}

/**
 * The launch type of an ECS service
 */
export enum LaunchType {
  /**
   * The service will be launched using the EC2 launch type
   */
  EC2 = 'EC2',

  /**
   * The service will be launched using the FARGATE launch type
   */
  FARGATE = 'FARGATE',

  /**
   * The service will be launched using the EXTERNAL launch type
   */
  EXTERNAL = 'EXTERNAL',
}

/**
 * The deployment controller type to use for the service.
 */
export enum DeploymentControllerType {
  /**
   * The rolling update (ECS) deployment type involves replacing the current
   * running version of the container with the latest version.
   */
  ECS = 'ECS',

  /**
   * The blue/green (CODE_DEPLOY) deployment type uses the blue/green deployment model powered by AWS CodeDeploy
   */
  CODE_DEPLOY = 'CODE_DEPLOY',

  /**
   * The external (EXTERNAL) deployment type enables you to use any third-party deployment controller
   */
  EXTERNAL = 'EXTERNAL',
}

/**
 * The deployment stratergy to use for ECS controller
 */
export enum DeploymentStrategy {
  /**
   * Rolling update deployment
   */
  ROLLING = 'ROLLING',
  /**
   * Blue/green deployment
   */
  BLUE_GREEN = 'BLUE_GREEN',
  /**
   * Linear deployment with progressive traffic shifting
   */
  LINEAR = 'LINEAR',
  /**
   * Canary deployment with fixed traffic percentage testing
   */
  CANARY = 'CANARY',
}

/**
 * Propagate tags from either service or task definition
 */
export enum PropagatedTagSource {
  /**
   * Propagate tags from service
   */
  SERVICE = 'SERVICE',

  /**
   * Propagate tags from task definition
   */
  TASK_DEFINITION = 'TASK_DEFINITION',

  /**
   * Do not propagate
   */
  NONE = 'NONE',
}

/**
 * Options for `determineContainerNameAndPort`
 */
interface DetermineContainerNameAndPortOptions {
  dnsRecordType: cloudmap.DnsRecordType;
  taskDefinition: TaskDefinition;
  container?: ContainerDefinition;
  containerPort?: number;
}

/**
 * Determine the name of the container and port to target for the service registry.
 */
function determineContainerNameAndPort(scope: IConstruct, options: DetermineContainerNameAndPortOptions) {
  // If the record type is SRV, then provide the containerName and containerPort to target.
  // We use the name of the default container and the default port of the default container
  // unless the user specifies otherwise.
  if (options.dnsRecordType === cloudmap.DnsRecordType.SRV) {
    // Ensure the user-provided container is from the right task definition.
    if (options.container && options.container.taskDefinition != options.taskDefinition) {
      throw new ValidationError(lit`CannotDiscoveryContainer`, 'Cannot add discovery for a container from another task definition', scope);
    }

    const container = options.container ?? options.taskDefinition.defaultContainer!;

    // Ensure that any port given by the user is mapped.
    if (options.containerPort && !container.portMappings.some(mapping => mapping.containerPort === options.containerPort)) {
      throw new ValidationError(lit`CannotDiscoveryContainer`, 'Cannot add discovery for a container port that has not been mapped', scope);
    }

    return {
      containerName: container.containerName,
      containerPort: options.containerPort ?? options.taskDefinition.defaultContainer!.containerPort,
    };
  }

  return {};
}

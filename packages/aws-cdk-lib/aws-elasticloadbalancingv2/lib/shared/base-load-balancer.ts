import type { Construct, IDependable } from 'constructs';
import { IpAddressType } from './enums';
import { ifUndefined, mapTagMapToCxschema, renderAttributes } from './util';
import * as ec2 from '../../../aws-ec2';
import * as iam from '../../../aws-iam';
import { PolicyStatement, ServicePrincipal } from '../../../aws-iam';
import type * as s3 from '../../../aws-s3';
import * as cxschema from '../../../cloud-assembly-schema';
import type { IResource } from '../../../core';
import { CfnResource, ContextProvider, Resource, Stack, Token } from '../../../core';
import { UnscopedValidationError, ValidationError } from '../../../core/lib/errors';
import type { IMapBox } from '../../../core/lib/helpers-internal';
import { Box } from '../../../core/lib/helpers-internal';
import { lit } from '../../../core/lib/private/literal-string';
import * as cxapi from '../../../cx-api';
import type { aws_elasticloadbalancingv2 } from '../../../interfaces';
import { RegionInfo } from '../../../region-info';
import { CfnLoadBalancer } from '../elasticloadbalancingv2.generated';

/**
 * The prefix to use for source NAT for a dual-stack network load balancer with UDP listeners.
 */
export class SourceNatIpv6Prefix {
  /**
   * Use an automatically assigned IPv6 prefix
   */
  public static autoAssigned(): SourceNatIpv6Prefix {
    return new SourceNatIpv6Prefix('auto_assigned');
  }

  /**
   * Use a custom IPv6 prefix with /80 netmask
   * @param prefix The IPv6 prefix
   */
  public static fromIpv6Prefix(prefix: string): SourceNatIpv6Prefix {
    if (!prefix.includes('/')) {
      throw new UnscopedValidationError(lit`Ipv6PrefixIncludeNetmask`, `IPv6 prefix must include netmask (e.g. 2001:db8::/80), got ${prefix}`);
    }

    const [_ipv6, netmask] = prefix.split('/');
    if (netmask !== '80') {
      throw new UnscopedValidationError(lit`Ipv6PrefixNetmask`, `IPv6 prefix must have a /80 netmask, got ${netmask}`);
    }

    return new SourceNatIpv6Prefix(prefix);
  }

  /**
   * @param prefix The IPv6 prefix
   */
  constructor(public readonly prefix: string) {}
}

/**
 * Specifies a subnet for a load balancer
 */
export interface SubnetMapping {
  /**
   * The subnet.
   */
  readonly subnet: ec2.ISubnet;

  /**
   * The allocation ID of the Elastic IP address for an internet-facing load balancer.
   *
   * @default undefined - AWS default is to allocate a new IP address for internet-facing load balancers
   */
  readonly allocationId?: string;

  /**
   * The IPv6 address.
   *
   * @default undefined - AWS default is to allocate an IPv6 address from the subnet's pool
   */
  readonly ipv6Address?: string;

  /**
   * The private IPv4 address for an internal load balancer.
   *
   * @default undefined - AWS default is to allocate a private IPv4 address from the subnet's pool
   */
  readonly privateIpv4Address?: string;

  /**
   * The IPv6 prefix to use for source NAT for a dual-stack network load balancer with UDP listeners.
   *
   * Specify an IPv6 prefix (/80 netmask) from the subnet CIDR block
   * or `SourceNatIpv6Prefix.autoAssigned()` to use an IPv6 prefix selected at random from the subnet CIDR block.
   *
   * @default undefined - AWS default is `SourceNatIpv6Prefix.autoAssigned()` for IPv6 load balancers
   */
  readonly sourceNatIpv6Prefix?: SourceNatIpv6Prefix;
}

/**
 * Shared properties of both Application and Network Load Balancers
 */
export interface BaseLoadBalancerProps {
  /**
   * Name of the load balancer
   *
   * @default - Automatically generated name.
   */
  readonly loadBalancerName?: string;

  /**
   * The VPC network to place the load balancer in
   */
  readonly vpc: ec2.IVpc;

  /**
   * Whether the load balancer has an internet-routable address
   *
   * @default false
   */
  readonly internetFacing?: boolean;

  /**
   * Which subnets place the load balancer in
   *
   * @default - the Vpc default strategy.
   *
   */
  readonly vpcSubnets?: ec2.SubnetSelection;

  /**
   * Indicates whether deletion protection is enabled.
   *
   * @default false
   */
  readonly deletionProtection?: boolean;

  /**
   * Indicates whether cross-zone load balancing is enabled.
   *
   * @default - false for Network Load Balancers and true for Application Load Balancers.
   * This can not be `false` for Application Load Balancers.
   * @see https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-elasticloadbalancingv2-loadbalancer-loadbalancerattribute.html
   */
  readonly crossZoneEnabled?: boolean;

  /**
   * Indicates whether the load balancer blocks traffic through the Internet Gateway (IGW).
   *
   * @default - false for internet-facing load balancers and true for internal load balancers
   */
  readonly denyAllIgwTraffic?: boolean;

  /**
   * The minimum capacity (LCU) for a load balancer.
   *
   * @default undefined - ELB default is 0 LCU
   *
   * @see https://docs.aws.amazon.com/elasticloadbalancing/latest/application/capacity-unit-reservation.html
   * @see https://docs.aws.amazon.com/elasticloadbalancing/latest/network/capacity-unit-reservation.html
   * @see https://exampleloadbalancer.com/ondemand_capacity_reservation_calculator.html
   */
  readonly minimumCapacityUnit?: number;
}

export interface ILoadBalancerV2 extends IResource, aws_elasticloadbalancingv2.ILoadBalancerRef {
  /**
   * The canonical hosted zone ID of this load balancer
   *
   * Example value: `Z2P70J7EXAMPLE`
   *
   * @attribute
   */
  readonly loadBalancerCanonicalHostedZoneId: string;

  /**
   * The DNS name of this load balancer
   *
   * Example value: `my-load-balancer-424835706.us-west-2.elb.amazonaws.com`
   *
   * @attribute
   */
  readonly loadBalancerDnsName: string;
}

/**
 * Options for looking up load balancers
 */
export interface BaseLoadBalancerLookupOptions {
  /**
   * Find by load balancer's ARN
   * @default - does not search by load balancer arn
   */
  readonly loadBalancerArn?: string;

  /**
   * Match load balancer tags.
   * @default - does not match load balancers by tags
   */
  readonly loadBalancerTags?: Record<string, string>;
}

/**
 * Options for query context provider
 * @internal
 */
export interface LoadBalancerQueryContextProviderOptions {
  /**
   * User's lookup options
   */
  readonly userOptions: BaseLoadBalancerLookupOptions;

  /**
   * Type of load balancer
   */
  readonly loadBalancerType: cxschema.LoadBalancerType;
}

/**
 * Base class for both Application and Network Load Balancers
 */
export abstract class BaseLoadBalancer extends Resource {
  /**
   * Queries the load balancer context provider for load balancer info.
   * @internal
   */
  protected static _queryContextProvider(scope: Construct, options: LoadBalancerQueryContextProviderOptions) {
    if (Token.isUnresolved(options.userOptions.loadBalancerArn)
      || Object.values(options.userOptions.loadBalancerTags ?? {}).some(Token.isUnresolved)) {
      throw new ValidationError(lit`ArgumentsLookUpLoadBalancer`, 'All arguments to look up a load balancer must be concrete (no Tokens)', scope);
    }

    let cxschemaTags: cxschema.Tag[] | undefined;
    if (options.userOptions.loadBalancerTags) {
      cxschemaTags = mapTagMapToCxschema(options.userOptions.loadBalancerTags);
    }

    const props: cxapi.LoadBalancerContextResponse = ContextProvider.getValue(scope, {
      provider: cxschema.ContextProvider.LOAD_BALANCER_PROVIDER,
      props: {
        loadBalancerArn: options.userOptions.loadBalancerArn,
        loadBalancerTags: cxschemaTags,
        loadBalancerType: options.loadBalancerType,
      } as cxschema.LoadBalancerContextQuery,
      dummyValue: {
        ipAddressType: cxapi.LoadBalancerIpAddressType.DUAL_STACK,
        // eslint-disable-next-line @cdklabs/no-literal-partition
        loadBalancerArn: `arn:aws:elasticloadbalancing:us-west-2:123456789012:loadbalancer/${options.loadBalancerType}/my-load-balancer/50dc6c495c0c9188`,
        loadBalancerCanonicalHostedZoneId: 'Z3DZXE0EXAMPLE',
        loadBalancerDnsName: 'my-load-balancer-123456789012.us-west-2.elb.amazonaws.com',
        securityGroupIds: ['sg-1234'],
        vpcId: 'vpc-12345',
      } as cxapi.LoadBalancerContextResponse,
    }).value;

    return props;
  }

  /**
   * The canonical hosted zone ID of this load balancer
   *
   * Example value: `Z2P70J7EXAMPLE`
   *
   * @attribute
   */
  public readonly loadBalancerCanonicalHostedZoneId: string;

  /**
   * The DNS name of this load balancer
   *
   * Example value: `my-load-balancer-424835706.us-west-2.elb.amazonaws.com`
   *
   * @attribute
   */
  public readonly loadBalancerDnsName: string;

  /**
   * The full name of this load balancer
   *
   * Example value: `app/my-load-balancer/50dc6c495c0c9188`
   *
   * @attribute
   */
  public readonly loadBalancerFullName: string;

  /**
   * The name of this load balancer
   *
   * Example value: `my-load-balancer`
   *
   * @attribute
   */
  public readonly loadBalancerName: string;

  /**
   * The ARN of this load balancer
   *
   * Example value: `arn:aws:elasticloadbalancing:us-west-2:123456789012:loadbalancer/app/my-internal-load-balancer/50dc6c495c0c9188`
   *
   * @attribute
   */
  public readonly loadBalancerArn: string;

  /**
   * A reference to this load balancer
   */
  public get loadBalancerRef(): aws_elasticloadbalancingv2.LoadBalancerReference {
    return {
      loadBalancerArn: this.loadBalancerArn,
    };
  }

  /**
   * @attribute
   */
  public readonly loadBalancerSecurityGroups: string[];

  /**
   * The VPC this load balancer has been created in.
   *
   * This property is always defined (not `null` or `undefined`) for sub-classes of `BaseLoadBalancer`.
   */
  public readonly vpc?: ec2.IVpc;

  /**
   * Attributes set on this load balancer
   */
  private readonly attributes: IMapBox<string, string | undefined> = Box.fromMap();

  constructor(scope: Construct, id: string, baseProps: BaseLoadBalancerProps, additionalProps: any) {
    super(scope, id, {
      physicalName: baseProps.loadBalancerName,
    });

    const internetFacing = ifUndefined(baseProps.internetFacing, false);

    if (baseProps.vpcSubnets && additionalProps.subnetMappings) {
      throw new ValidationError(lit`SpecifyEither`, 'You can specify either `vpcSubnets` or `subnetMappings`, not both.', this);
    }

    let subnetIds: string[] | undefined;
    let subnetMappings: SubnetMapping[] | undefined = additionalProps.subnetMappings;
    let internetConnectivityEstablishedSubnets: IDependable | undefined;

    if (additionalProps.ipAddressType === IpAddressType.DUAL_STACK_WITHOUT_PUBLIC_IPV4 &&
        additionalProps.type !== cxschema.LoadBalancerType.APPLICATION) {
      throw new ValidationError(lit`IpaddresstypeDualStackWithoutPublicIpv4OnlyUsed`, `'ipAddressType' DUAL_STACK_WITHOUT_PUBLIC_IPV4 can only be used with Application Load Balancer, got ${additionalProps.type}`, this);
    }

    this.vpc = baseProps.vpc;

    if (subnetMappings) {
      if (internetFacing) {
        const mappedSubnetIds = subnetMappings.map(mapping => mapping.subnet.subnetId);
        internetConnectivityEstablishedSubnets = baseProps.vpc.selectSubnets(
          { subnetFilters: [ec2.SubnetFilter.byIds(mappedSubnetIds)] },
        ).internetConnectivityEstablished;
      }

      const { subnetMappings: _, ...cfnProps } = additionalProps;
      additionalProps = cfnProps;
    } else {
      const vpcSubnets = ifUndefined(baseProps.vpcSubnets,
        (internetFacing ? { subnetType: ec2.SubnetType.PUBLIC } : {}));

      const result = baseProps.vpc.selectSubnets(vpcSubnets);
      subnetIds = result.subnetIds;
      internetConnectivityEstablishedSubnets = result.internetConnectivityEstablished;
    }

    const resource = new CfnLoadBalancer(this, 'Resource', {
      name: this.physicalName,
      subnets: subnetIds,
      subnetMappings: subnetMappings?.map((mapping) => {
        return ({
          subnetId: mapping.subnet.subnetId,
          allocationId: mapping.allocationId,
          privateIPv4Address: mapping.privateIpv4Address,
          iPv6Address: mapping.ipv6Address,
          sourceNatIpv6Prefix: mapping.sourceNatIpv6Prefix?.prefix,
        });
      }),
      scheme: internetFacing ? 'internet-facing' : 'internal',
      loadBalancerAttributes: this.attributes.derive(m => {
        const ret = renderAttributes(Object.fromEntries(m));
        return ret.length > 0 ? ret : undefined;
      }),
      minimumLoadBalancerCapacity: baseProps.minimumCapacityUnit ? {
        capacityUnits: baseProps.minimumCapacityUnit,
      } : undefined,
      ...additionalProps,
    });

    if (internetFacing && internetConnectivityEstablishedSubnets) {
      resource.node.addDependency(internetConnectivityEstablishedSubnets);
    }

    this.setAttribute('deletion_protection.enabled', baseProps.deletionProtection ? 'true' : 'false');

    if (baseProps.crossZoneEnabled !== undefined) {
      this.setAttribute('load_balancing.cross_zone.enabled', baseProps.crossZoneEnabled === true ? 'true' : 'false');
    }

    if (baseProps.denyAllIgwTraffic !== undefined) {
      if (additionalProps.ipAddressType === IpAddressType.DUAL_STACK) {
        this.setAttribute('ipv6.deny_all_igw_traffic', baseProps.denyAllIgwTraffic.toString());
      } else {
        throw new ValidationError(lit`DenyIgwTrafficSetLoad`, `'denyAllIgwTraffic' may only be set on load balancers with ${IpAddressType.DUAL_STACK} addressing.`, this);
      }
    }

    this.loadBalancerCanonicalHostedZoneId = resource.attrCanonicalHostedZoneId;
    this.loadBalancerDnsName = resource.attrDnsName;
    this.loadBalancerFullName = resource.attrLoadBalancerFullName;
    this.loadBalancerName = resource.attrLoadBalancerName;
    this.loadBalancerArn = resource.ref;
    this.loadBalancerSecurityGroups = resource.attrSecurityGroups;

    this.node.addValidation({ validate: this.validateLoadBalancer.bind(this) });
  }

  /**
   * Enable access logging for this load balancer.
   *
   * A region must be specified on the stack containing the load balancer; you cannot enable logging on
   * environment-agnostic stacks. See https://docs.aws.amazon.com/cdk/latest/guide/environments.html
   */
  public logAccessLogs(bucket: s3.IBucket, prefix?: string) {
    prefix = prefix || '';
    this.setAttribute('access_logs.s3.enabled', 'true');
    this.setAttribute('access_logs.s3.bucket', bucket.bucketName.toString());
    this.setAttribute('access_logs.s3.prefix', prefix);

    const logsDeliveryServicePrincipal = new ServicePrincipal('delivery.logs.amazonaws.com');
    bucket.addToResourcePolicy(new PolicyStatement({
      actions: ['s3:PutObject'],
      principals: [this.resourcePolicyPrincipal()],
      resources: [
        bucket.arnForObjects(`${prefix ? prefix + '/' : ''}AWSLogs/${Stack.of(this).account}/*`),
      ],
    }));
    bucket.addToResourcePolicy(
      new PolicyStatement({
        actions: ['s3:PutObject'],
        principals: [logsDeliveryServicePrincipal],
        resources: [
          bucket.arnForObjects(`${prefix ? prefix + '/' : ''}AWSLogs/${this.env.account}/*`),
        ],
        conditions: {
          StringEquals: { 's3:x-amz-acl': 'bucket-owner-full-control' },
        },
      }),
    );
    bucket.addToResourcePolicy(
      new PolicyStatement({
        actions: ['s3:GetBucketAcl'],
        principals: [logsDeliveryServicePrincipal],
        resources: [bucket.bucketArn],
      }),
    );

    // make sure the bucket's policy is created before the ALB (see https://github.com/aws/aws-cdk/issues/1633)
    // at the L1 level to avoid creating a circular dependency (see https://github.com/aws/aws-cdk/issues/27528
    // and https://github.com/aws/aws-cdk/issues/27928)
    const lb = this.node.defaultChild;
    const bucketPolicy = bucket.policy?.node.defaultChild;
    if (lb && bucketPolicy && CfnResource.isCfnResource(lb) && CfnResource.isCfnResource(bucketPolicy)) {
      lb.addResourceDependency(bucketPolicy);
    }
  }

  /**
   * Set a non-standard attribute on the load balancer
   *
   * @see https://docs.aws.amazon.com/elasticloadbalancing/latest/application/application-load-balancers.html#load-balancer-attributes
   */
  public setAttribute(key: string, value: string | undefined) {
    this.attributes.put(key, value);
  }

  /**
   * Remove an attribute from the load balancer
   */
  public removeAttribute(key: string) {
    this.setAttribute(key, undefined);
  }

  protected resourcePolicyPrincipal(): iam.IPrincipal {
    const region = Stack.of(this).region;
    if (Token.isUnresolved(region)) {
      throw new ValidationError(lit`RegionRequiredEnableBvAccess`, 'Region is required to enable ELBv2 access logging', this);
    }

    const account = RegionInfo.get(region).elbv2Account;
    if (!account) {
      // New Regions use a service principal
      // https://docs.aws.amazon.com/elasticloadbalancing/latest/classic/enable-access-logs.html#attach-bucket-policy
      return new iam.ServicePrincipal('logdelivery.elasticloadbalancing.amazonaws.com');
    }

    return new iam.AccountPrincipal(account);
  }

  protected validateLoadBalancer(): string[] {
    const ret = new Array<string>();

    // https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-resource-elasticloadbalancingv2-loadbalancer.html#cfn-elasticloadbalancingv2-loadbalancer-name
    const loadBalancerName = this.physicalName;
    if (!Token.isUnresolved(loadBalancerName) && loadBalancerName !== undefined) {
      if (loadBalancerName.length > 32) {
        ret.push(`Load balancer name: "${loadBalancerName}" can have a maximum of 32 characters.`);
      }
      if (loadBalancerName.startsWith('internal-')) {
        ret.push(`Load balancer name: "${loadBalancerName}" must not begin with "internal-".`);
      }
      if (loadBalancerName.startsWith('-') || loadBalancerName.endsWith('-')) {
        ret.push(`Load balancer name: "${loadBalancerName}" must not begin or end with a hyphen.`);
      }
      if (!/^[0-9a-z-]+$/i.test(loadBalancerName)) {
        ret.push(`Load balancer name: "${loadBalancerName}" must contain only alphanumeric characters or hyphens.`);
      }
    }

    return ret;
  }
}

import type { Construct } from 'constructs';
import { Connections } from './connections';
import type { ISecurityGroupRef, SecurityGroupReference } from './ec2.generated';
import { CfnSecurityGroup, CfnSecurityGroupEgress, CfnSecurityGroupIngress } from './ec2.generated';
import type { EgressRuleConfig, IngressRuleConfig, IPeer } from './peer';
import { Peer } from './peer';
import { Port } from './port';
import type { IVpc } from './vpc';
import * as cxschema from '../../cloud-assembly-schema';
import type {
  IResource,
  ResourceProps,
} from '../../core';
import {
  Annotations,
  ContextProvider,
  Names,
  Resource,
  Stack,
  Token,
  ValidationError,
} from '../../core';
import type { IArrayBox } from '../../core/lib/helpers-internal';
import { Box } from '../../core/lib/helpers-internal';
import { addConstructMetadata, MethodMetadata } from '../../core/lib/metadata-resource';
import { noBoxStackTraces } from '../../core/lib/no-box-stack-traces';
import { lit } from '../../core/lib/private/literal-string';
import { propertyInjectable } from '../../core/lib/prop-injectable';
import type * as cxapi from '../../cx-api';

const SECURITY_GROUP_SYMBOL = Symbol.for('@aws-cdk/iam.SecurityGroup');

const SECURITY_GROUP_DISABLE_INLINE_RULES_CONTEXT_KEY = '@aws-cdk/aws-ec2.securityGroupDisableInlineRules';

/**
 * Interface for security group-like objects
 */
export interface ISecurityGroup extends IResource, IPeer, ISecurityGroupRef {
  /**
   * ID for the current security group
   * @attribute
   */
  readonly securityGroupId: string;

  /**
   * Whether the SecurityGroup has been configured to allow all outbound traffic
   */
  readonly allowAllOutbound: boolean;

  /**
   * Add an ingress rule for the current security group
   *
   * `remoteRule` controls where the Rule object is created if the peer is also a
   * securityGroup and they are in different stack. If false (default) the
   * rule object is created under the current SecurityGroup object. If true and the
   * peer is also a SecurityGroup, the rule object is created under the remote
   * SecurityGroup object.
   */
  addIngressRule(peer: IPeer, connection: Port, description?: string, remoteRule?: boolean): void;

  /**
   * Add an egress rule for the current security group
   *
   * `remoteRule` controls where the Rule object is created if the peer is also a
   * securityGroup and they are in different stack. If false (default) the
   * rule object is created under the current SecurityGroup object. If true and the
   * peer is also a SecurityGroup, the rule object is created under the remote
   * SecurityGroup object.
   */
  addEgressRule(peer: IPeer, connection: Port, description?: string, remoteRule?: boolean): void;
}

/**
 * A SecurityGroup that is not created in this template
 */
abstract class SecurityGroupBase extends Resource implements ISecurityGroup {
  /**
   * Return whether the indicated object is a security group
   */
  public static isSecurityGroup(x: any): x is SecurityGroupBase {
    return SECURITY_GROUP_SYMBOL in x;
  }

  public abstract readonly securityGroupId: string;
  public abstract readonly allowAllOutbound: boolean;
  public abstract readonly allowAllIpv6Outbound: boolean;

  public readonly canInlineRule = false;
  public readonly connections: Connections = new Connections({ securityGroups: [this] });
  public readonly defaultPort?: Port;

  private peerAsTokenCount: number = 0;

  constructor(scope: Construct, id: string, props?: ResourceProps) {
    super(scope, id, props);

    Object.defineProperty(this, SECURITY_GROUP_SYMBOL, { value: true });
  }

  public get securityGroupRef(): SecurityGroupReference {
    return {
      securityGroupId: this.securityGroupId,
    };
  }

  public get uniqueId() {
    return Names.nodeUniqueId(this.node);
  }

  public addIngressRule(peer: IPeer, connection: Port, description?: string, remoteRule?: boolean) {
    if (description === undefined) {
      description = `from ${peer.uniqueId}:${connection}`;
    }

    const { scope, id } = this.determineRuleScope(peer, connection, 'from', remoteRule);

    // Skip duplicates
    if (scope.node.tryFindChild(id) === undefined) {
      new CfnSecurityGroupIngress(scope, id, {
        groupId: this.securityGroupId,
        ...peer.toIngressRuleConfig(),
        ...connection.toRuleJson(),
        description,
      });
    }
  }

  public addEgressRule(peer: IPeer, connection: Port, description?: string, remoteRule?: boolean) {
    if (description === undefined) {
      description = `to ${peer.uniqueId}:${connection}`;
    }

    const { scope, id } = this.determineRuleScope(peer, connection, 'to', remoteRule);

    // Skip duplicates
    if (scope.node.tryFindChild(id) === undefined) {
      new CfnSecurityGroupEgress(scope, id, {
        groupId: this.securityGroupId,
        ...peer.toEgressRuleConfig(),
        ...connection.toRuleJson(),
        description,
      });
    }
  }

  public toIngressRuleConfig(): IngressRuleConfig {
    return { sourceSecurityGroupId: this.securityGroupId };
  }

  public toEgressRuleConfig(): EgressRuleConfig {
    return { destinationSecurityGroupId: this.securityGroupId };
  }

  /**
   * Determine where to parent a new ingress/egress rule
   *
   * A SecurityGroup rule is parented under the group it's related to, UNLESS
   * we're in a cross-stack scenario with another Security Group. In that case,
   * we respect the 'remoteRule' flag and will parent under the other security
   * group.
   *
   * This is necessary to avoid cyclic dependencies between stacks, since both
   * ingress and egress rules will reference both security groups, and a naive
   * parenting will lead to the following situation:
   *
   *   ╔════════════════════╗         ╔════════════════════╗
   *   ║  ┌───────────┐     ║         ║    ┌───────────┐   ║
   *   ║  │  GroupA   │◀────╬─┐   ┌───╬───▶│  GroupB   │   ║
   *   ║  └───────────┘     ║ │   │   ║    └───────────┘   ║
   *   ║        ▲           ║ │   │   ║          ▲         ║
   *   ║        │           ║ │   │   ║          │         ║
   *   ║        │           ║ │   │   ║          │         ║
   *   ║  ┌───────────┐     ║ └───┼───╬────┌───────────┐   ║
   *   ║  │  EgressA  │─────╬─────┘   ║    │ IngressB  │   ║
   *   ║  └───────────┘     ║         ║    └───────────┘   ║
   *   ║                    ║         ║                    ║
   *   ╚════════════════════╝         ╚════════════════════╝
   *
   * By having the ability to switch the parent, we avoid the cyclic reference by
   * keeping all rules in a single stack.
   *
   * If this happens, we also have to change the construct ID, because
   * otherwise we might have two objects with the same ID if we have
   * multiple reversed security group relationships.
   *
   *   ╔═══════════════════════════════════╗
   *   ║┌───────────┐                      ║
   *   ║│  GroupB   │                      ║
   *   ║└───────────┘                      ║
   *   ║      ▲                            ║
   *   ║      │              ┌───────────┐ ║
   *   ║      ├────"from A"──│ IngressB  │ ║
   *   ║      │              └───────────┘ ║
   *   ║      │              ┌───────────┐ ║
   *   ║      ├─────"to B"───│  EgressA  │ ║
   *   ║      │              └───────────┘ ║
   *   ║      │              ┌───────────┐ ║
   *   ║      └─────"to B"───│  EgressC  │ ║  <-- oops
   *   ║                     └───────────┘ ║
   *   ╚═══════════════════════════════════╝
   */

  protected determineRuleScope(
    peer: IPeer,
    connection: Port,
    fromTo: 'from' | 'to',
    remoteRule?: boolean): RuleScope {
    if (remoteRule && SecurityGroupBase.isSecurityGroup(peer) && differentStacks(this, peer)) {
      // Reversed
      const reversedFromTo = fromTo === 'from' ? 'to' : 'from';
      return { scope: peer, id: `${this.uniqueId}:${connection} ${reversedFromTo}` };
    } else {
      // Regular (do old ID escaping to in order to not disturb existing deployments)
      return { scope: this, id: `${fromTo} ${this.renderPeer(peer)}:${connection}`.replace('/', '_') };
    }
  }

  private renderPeer(peer: IPeer) {
    if (Token.isUnresolved(peer.uniqueId)) {
      // Need to return a unique value each time a peer
      // is an unresolved token, else the duplicate skipper
      // in `sg.addXxxRule` can detect unique rules as duplicates
      return this.peerAsTokenCount++ ? `'{IndirectPeer${this.peerAsTokenCount}}'` : '{IndirectPeer}';
    } else {
      return peer.uniqueId;
    }
  }
}

/**
 * The scope and id in which a given SecurityGroup rule should be defined.
 */
export interface RuleScope {
  /**
   * The SecurityGroup in which a rule should be scoped.
   */
  readonly scope: ISecurityGroup;
  /**
   * The construct ID to use for the rule.
   */
  readonly id: string;
}

function differentStacks(group1: SecurityGroupBase, group2: SecurityGroupBase) {
  return Stack.of(group1) !== Stack.of(group2);
}

export interface SecurityGroupProps {
  /**
   * The name of the security group. For valid values, see the GroupName
   * parameter of the CreateSecurityGroup action in the Amazon EC2 API
   * Reference.
   *
   * It is not recommended to use an explicit group name.
   *
   * @default If you don't specify a GroupName, AWS CloudFormation generates a
   * unique physical ID and uses that ID for the group name.
   */
  readonly securityGroupName?: string;

  /**
   * A description of the security group.
   *
   * @default The default name will be the construct's CDK path.
   */
  readonly description?: string;

  /**
   * The VPC in which to create the security group.
   */
  readonly vpc: IVpc;

  /**
   * Whether to allow all outbound traffic by default.
   *
   * If this is set to true, there will only be a single egress rule which allows all
   * outbound traffic. If this is set to false, no outbound traffic will be allowed by
   * default and all egress traffic must be explicitly authorized.
   *
   * To allow all ipv6 traffic use allowAllIpv6Outbound
   *
   * @default true
   */
  readonly allowAllOutbound?: boolean;

  /**
   * Whether to allow all outbound ipv6 traffic by default.
   *
   * If this is set to true, there will only be a single egress rule which allows all
   * outbound ipv6 traffic. If this is set to false, no outbound traffic will be allowed by
   * default and all egress ipv6 traffic must be explicitly authorized.
   *
   * To allow all ipv4 traffic use allowAllOutbound
   *
   * @default false
   */
  readonly allowAllIpv6Outbound?: boolean;

  /**
   * Whether to disable inline ingress and egress rule optimization.
   *
   * If this is set to true, ingress and egress rules will not be declared under the
   * SecurityGroup in cloudformation, but will be separate elements.
   *
   * Inlining rules is an optimization for producing smaller stack templates. Sometimes
   * this is not desirable, for example when security group access is managed via tags.
   *
   * The default value can be overridden globally by setting the context variable
   * '@aws-cdk/aws-ec2.securityGroupDisableInlineRules'.
   *
   * @default false
   */
  readonly disableInlineRules?: boolean;
}

/**
 * Additional options for imported security groups
 */
export interface SecurityGroupImportOptions {
  /**
   * Mark the SecurityGroup as having been created allowing all outbound traffic
   *
   * Only if this is set to false will egress rules be added to this security
   * group. Be aware, this would undo any potential "all outbound traffic"
   * default.
   *
   *
   * @default true
   */
  readonly allowAllOutbound?: boolean;

  /**
   * Mark the SecurityGroup as having been created allowing all outbound ipv6 traffic
   *
   * Only if this is set to false will egress rules for ipv6 be added to this security
   * group. Be aware, this would undo any potential "all outbound traffic"
   * default.
   *
   * @default false
   */
  readonly allowAllIpv6Outbound?: boolean;

  /**
   * If a SecurityGroup is mutable CDK can add rules to existing groups
   *
   * Beware that making a SecurityGroup immutable might lead to issue
   * due to missing ingress/egress rules for new resources.
   *
   *
   * @default true
   */
  readonly mutable?: boolean;
}

/**
 * Creates an Amazon EC2 security group within a VPC.
 *
 * Security Groups act like a firewall with a set of rules, and are associated
 * with any AWS resource that has or creates Elastic Network Interfaces (ENIs).
 * A typical example of a resource that has a security group is an Instance (or
 * Auto Scaling Group of instances)
 *
 * If you are defining new infrastructure in CDK, there is a good chance you
 * won't have to interact with this class at all. Like IAM Roles, Security
 * Groups need to exist to control access between AWS resources, but CDK will
 * automatically generate and populate them with least-privilege permissions
 * for you so you can concentrate on your business logic.
 *
 * All Constructs that require Security Groups will create one for you if you
 * don't specify one at construction. After construction, you can selectively
 * allow connections to and between constructs via--for example-- the `instance.connections`
 * object. Think of it as "allowing connections to your instance", rather than
 * "adding ingress rules a security group". See the [Allowing
 * Connections](https://docs.aws.amazon.com/cdk/api/latest/docs/aws-cdk-lib.aws_ec2-readme.html#allowing-connections)
 * section in the library documentation for examples.
 *
 * Direct manipulation of the Security Group through `addIngressRule` and
 * `addEgressRule` is possible, but mutation through the `.connections` object
 * is recommended. If you peer two constructs with security groups this way,
 * appropriate rules will be created in both.
 *
 * If you have an existing security group you want to use in your CDK application,
 * you would import it like this:
 *
 * ```ts
 * const securityGroup = ec2.SecurityGroup.fromSecurityGroupId(this, 'SG', 'sg-12345', {
 *   mutable: false
 * });
 * ```
 */
@propertyInjectable
@noBoxStackTraces
export class SecurityGroup extends SecurityGroupBase {
  /**
   * Uniquely identifies this class.
   */
  public static readonly PROPERTY_INJECTION_ID: string = 'aws-cdk-lib.aws-ec2.SecurityGroup';

  /**
   * Look up a security group by id.
   *
   * @deprecated Use `fromLookupById()` instead
   */
  public static fromLookup(scope: Construct, id: string, securityGroupId: string) {
    return this.fromLookupAttributes(scope, id, { securityGroupId });
  }

  /**
   * Look up a security group by id.
   */
  public static fromLookupById(scope: Construct, id: string, securityGroupId: string) {
    return this.fromLookupAttributes(scope, id, { securityGroupId });
  }

  /**
   * Look up a security group by name.
   */
  public static fromLookupByName(scope: Construct, id: string, securityGroupName: string, vpc: IVpc) {
    return this.fromLookupAttributes(scope, id, { securityGroupName, vpc });
  }

  /**
   * Import an existing security group into this app.
   *
   * This method will assume that the Security Group has a rule in it which allows
   * all outbound traffic, and so will not add egress rules to the imported Security
   * Group (only ingress rules).
   *
   * If your existing Security Group needs to have egress rules added, pass the
   * `allowAllOutbound: false` option on import.
   */
  public static fromSecurityGroupId(scope: Construct, id: string, securityGroupId: string, options: SecurityGroupImportOptions = {}): ISecurityGroup {
    class MutableImport extends SecurityGroupBase {
      public securityGroupId = securityGroupId;
      public allowAllOutbound = options.allowAllOutbound ?? true;
      public allowAllIpv6Outbound = options.allowAllIpv6Outbound ?? false;

      public addEgressRule(peer: IPeer, connection: Port, description?: string, remoteRule?: boolean) {
        // Only if allowAllOutbound has been disabled
        if (options.allowAllOutbound === false) {
          super.addEgressRule(peer, connection, description, remoteRule);
        }
      }
    }

    class ImmutableImport extends SecurityGroupBase {
      public securityGroupId = securityGroupId;
      public allowAllOutbound = options.allowAllOutbound ?? true;
      public allowAllIpv6Outbound = options.allowAllIpv6Outbound ?? false;

      public addEgressRule(_peer: IPeer, _connection: Port, _description?: string, _remoteRule?: boolean) {
        // do nothing
      }

      public addIngressRule(_peer: IPeer, _connection: Port, _description?: string, _remoteRule?: boolean) {
        // do nothing
      }
    }

    return options.mutable !== false
      ? new MutableImport(scope, id)
      : new ImmutableImport(scope, id);
  }

  /**
   * Look up a security group.
   */
  private static fromLookupAttributes(scope: Construct, id: string, options: SecurityGroupLookupOptions) {
    if (Token.isUnresolved(options.securityGroupId) || Token.isUnresolved(options.securityGroupName) || Token.isUnresolved(options.vpc?.vpcId)) {
      throw new ValidationError(lit`ArgumentsLookUpSecurityGroup`, 'All arguments to look up a security group must be concrete (no Tokens)', scope);
    }

    const attributes: cxapi.SecurityGroupContextResponse = ContextProvider.getValue(scope, {
      provider: cxschema.ContextProvider.SECURITY_GROUP_PROVIDER,
      props: {
        securityGroupId: options.securityGroupId,
        securityGroupName: options.securityGroupName,
        vpcId: options.vpc?.vpcId,
      },
      dummyValue: {
        securityGroupId: 'sg-12345678',
        allowAllOutbound: true,
      } as cxapi.SecurityGroupContextResponse,
    }).value;

    return SecurityGroup.fromSecurityGroupId(scope, id, attributes.securityGroupId, {
      allowAllOutbound: attributes.allowAllOutbound,
      mutable: true,
    });
  }

  /**
   * An attribute that represents the security group name.
   *
   * @attribute
   * @deprecated returns the security group ID, rather than the name.
   */
  public readonly securityGroupName: string;

  /**
   * The ID of the security group
   *
   * @attribute
   */
  public readonly securityGroupId: string;

  /**
   * The VPC ID this security group is part of.
   *
   * @attribute
   */
  public readonly securityGroupVpcId: string;

  /**
   * Whether the SecurityGroup has been configured to allow all outbound traffic
   */
  public readonly allowAllOutbound: boolean;

  /**
   * Whether the SecurityGroup has been configured to allow all outbound ipv6 traffic
   */
  public readonly allowAllIpv6Outbound: boolean;

  private readonly securityGroup: CfnSecurityGroup;
  private readonly directIngressRules: IArrayBox<CfnSecurityGroup.IngressProperty>;
  private readonly directEgressRules: IArrayBox<CfnSecurityGroup.EgressProperty>;

  /**
   * Whether to disable optimization for inline security group rules.
   */
  private readonly disableInlineRules: boolean;

  constructor(scope: Construct, id: string, props: SecurityGroupProps) {
    super(scope, id, {
      physicalName: props.securityGroupName,
    });
    // Enhanced CDK Analytics Telemetry
    addConstructMetadata(this, props);

    const groupDescription = props.description || this.node.path;

    this.allowAllOutbound = props.allowAllOutbound !== false;
    this.allowAllIpv6Outbound = props.allowAllIpv6Outbound ?? false;

    this.disableInlineRules = props.disableInlineRules !== undefined ?
      !!props.disableInlineRules :
      !!this.node.tryGetContext(SECURITY_GROUP_DISABLE_INLINE_RULES_CONTEXT_KEY);

    this.directIngressRules = Box.fromArray();
    this.directEgressRules = Box.fromArray();

    this.securityGroup = new CfnSecurityGroup(this, 'Resource', {
      groupName: this.physicalName,
      groupDescription,
      securityGroupIngress: this.directIngressRules,
      securityGroupEgress: this.directEgressRules,
      vpcId: props.vpc.vpcId,
    });

    this.securityGroupId = this.securityGroup.attrGroupId;
    this.securityGroupVpcId = this.securityGroup.attrVpcId;
    this.securityGroupName = this.securityGroup.ref;

    this.addDefaultEgressRule();
    this.addDefaultIpv6EgressRule();
  }

  @MethodMetadata()
  public addIngressRule(peer: IPeer, connection: Port, description?: string, remoteRule?: boolean) {
    if (!peer.canInlineRule || !connection.canInlineRule || this.disableInlineRules) {
      super.addIngressRule(peer, connection, description, remoteRule);
      return;
    }

    if (description === undefined) {
      description = `from ${peer.uniqueId}:${connection}`;
    }

    this.addDirectIngressRule({
      ...peer.toIngressRuleConfig(),
      ...connection.toRuleJson(),
      description,
    });
  }

  @MethodMetadata()
  public addEgressRule(peer: IPeer, connection: Port, description?: string, remoteRule?: boolean) {
    const isIpv6 = peer.toEgressRuleConfig().hasOwnProperty('cidrIpv6');

    if (!isIpv6 && this.allowAllOutbound) {
      // In the case of "allowAllOutbound", we don't add any more rules. There
      // is only one rule which allows all traffic and that subsumes any other
      // rule.
      if (!remoteRule) { // Warn only if addEgressRule() was explicitly called
        Annotations.of(this).addWarningV2('@aws-cdk/aws-ec2:ipv4IgnoreEgressRule', 'Ignoring Egress rule since \'allowAllOutbound\' is set to true; To add customized rules, set allowAllOutbound=false on the SecurityGroup');
      }
      return;
    } else if (!isIpv6 && !this.allowAllOutbound) {
      // Otherwise, if the bogus rule exists we can now remove it because the
      // presence of any other rule will get rid of EC2's implicit "all
      // outbound" rule anyway.
      this.removeNoTrafficRule();
    }

    if (isIpv6 && this.allowAllIpv6Outbound) {
      // In the case of "allowAllIpv6Outbound", we don't add any more rules. There
      // is only one rule which allows all traffic and that subsumes any other
      // rule.
      if (!remoteRule) { // Warn only if addEgressRule() was explicitly called
        Annotations.of(this).addWarningV2('@aws-cdk/aws-ec2:ipv6IgnoreEgressRule', 'Ignoring Egress rule since \'allowAllIpv6Outbound\' is set to true; To add customized rules, set allowAllIpv6Outbound=false on the SecurityGroup');
      }
      return;
    }

    if (!peer.canInlineRule || !connection.canInlineRule || this.disableInlineRules) {
      super.addEgressRule(peer, connection, description, remoteRule);
      return;
    }

    if (description === undefined) {
      description = `from ${peer.uniqueId}:${connection}`;
    }

    const rule = {
      ...peer.toEgressRuleConfig(),
      ...connection.toRuleJson(),
      description,
    };

    if (isAllTrafficRule(rule)) {
      // We cannot allow this; if someone adds the rule in this way, it will be
      // removed again if they add other rules. We also can't automatically switch
      // to "allOutbound=true" mode, because we might have already emitted
      // EgressRule objects (which count as rules added later) and there's no way
      // to recall those. Better to prevent this for now.
      throw new ValidationError(lit`CannotAddTrafficEgressRule`, 'Cannot add an "all traffic" egress rule in this way; set allowAllOutbound=true (for ipv4) or allowAllIpv6Outbound=true (for ipv6) on the SecurityGroup instead.', this);
    }

    this.addDirectEgressRule(rule);
  }

  /**
   * Add a direct ingress rule
   */
  private addDirectIngressRule(rule: CfnSecurityGroup.IngressProperty) {
    if (!this.hasIngressRule(rule)) {
      this.directIngressRules.push(rule);
    }
  }

  /**
   * Return whether the given ingress rule exists on the group
   */
  private hasIngressRule(rule: CfnSecurityGroup.IngressProperty): boolean {
    return this.directIngressRules.findIndex(r => ingressRulesEqual(r, rule)) > -1;
  }

  /**
   * Add a direct egress rule
   */
  private addDirectEgressRule(rule: CfnSecurityGroup.EgressProperty) {
    if (!this.hasEgressRule(rule)) {
      this.directEgressRules.push(rule);
    }
  }

  /**
   * Return whether the given egress rule exists on the group
   */
  private hasEgressRule(rule: CfnSecurityGroup.EgressProperty): boolean {
    return this.directEgressRules.findIndex(r => egressRulesEqual(r, rule)) > -1;
  }

  /**
   * Add the default egress rule to the securityGroup
   *
   * This depends on allowAllOutbound:
   *
   * - If allowAllOutbound is true, we *TECHNICALLY* don't need to do anything, because
   *   EC2 is going to create this default rule anyway. But, for maximum readability
   *   of the template, we will add one anyway.
   * - If allowAllOutbound is false, we add a bogus rule that matches no traffic in
   *   order to get rid of the default "all outbound" rule that EC2 creates by default.
   *   If other rules happen to get added later, we remove the bogus rule again so
   *   that it doesn't clutter up the template too much (even though that's not
   *   strictly necessary).
   */
  private addDefaultEgressRule() {
    if (this.disableInlineRules) {
      const peer = this.allowAllOutbound ? ALL_TRAFFIC_PEER : NO_TRAFFIC_PEER;
      const port = this.allowAllOutbound ? ALL_TRAFFIC_PORT : NO_TRAFFIC_PORT;
      const description = this.allowAllOutbound ? ALLOW_ALL_RULE.description : MATCH_NO_TRAFFIC.description;
      super.addEgressRule(peer, port, description, false);
    } else {
      const rule = this.allowAllOutbound ? ALLOW_ALL_RULE : MATCH_NO_TRAFFIC;
      this.directEgressRules.push(rule);
    }
  }

  /**
   * Add a allow all ipv6 egress rule to the securityGroup
   *
   * This depends on allowAllIpv6Outbound:
   *
   * - If allowAllIpv6Outbound is true, we will add an allow all rule.
   * - If allowAllOutbound is false, we don't do anything since EC2 does not add
   *   a default allow all ipv6 rule.
   */
  private addDefaultIpv6EgressRule() {
    const description = 'Allow all outbound ipv6 traffic by default';
    const peer = Peer.anyIpv6();
    if (this.allowAllIpv6Outbound) {
      if (this.disableInlineRules) {
        super.addEgressRule(peer, Port.allTraffic(), description, false);
      } else {
        this.directEgressRules.push({
          ipProtocol: '-1',
          cidrIpv6: peer.uniqueId,
          description,
        });
      }
    }
  }

  /**
   * Remove the bogus rule if it exists
   */
  private removeNoTrafficRule() {
    if (this.disableInlineRules) {
      const { scope, id } = this.determineRuleScope(
        NO_TRAFFIC_PEER,
        NO_TRAFFIC_PORT,
        'to',
        false,
      );
      scope.node.tryRemoveChild(id);
    } else {
      const i = this.directEgressRules.findIndex(r => egressRulesEqual(r, MATCH_NO_TRAFFIC));
      if (i > -1) {
        this.directEgressRules.splice(i, 1);
      }
    }
  }
}

/**
 * Egress rule that purposely matches no traffic
 *
 * This is used in order to disable the "all traffic" default of Security Groups.
 *
 * No machine can ever actually have the 255.255.255.255 IP address, but
 * in order to lock it down even more we'll restrict to a nonexistent
 * ICMP traffic type.
 */
const MATCH_NO_TRAFFIC = {
  cidrIp: '255.255.255.255/32',
  description: 'Disallow all traffic',
  ipProtocol: 'icmp',
  fromPort: 252,
  toPort: 86,
};

const NO_TRAFFIC_PEER = Peer.ipv4(MATCH_NO_TRAFFIC.cidrIp);
const NO_TRAFFIC_PORT = Port.icmpTypeAndCode(MATCH_NO_TRAFFIC.fromPort, MATCH_NO_TRAFFIC.toPort);

/**
 * Egress rule that matches all traffic
 */
const ALLOW_ALL_RULE = {
  cidrIp: '0.0.0.0/0',
  description: 'Allow all outbound traffic by default',
  ipProtocol: '-1',
};

const ALL_TRAFFIC_PEER = Peer.anyIpv4();
const ALL_TRAFFIC_PORT = Port.allTraffic();

export interface ConnectionRule {
  /**
   * The IP protocol name (tcp, udp, icmp) or number (see Protocol Numbers).
   * Use -1 to specify all protocols. If you specify -1, or a protocol number
   * other than tcp, udp, icmp, or 58 (ICMPv6), traffic on all ports is
   * allowed, regardless of any ports you specify. For tcp, udp, and icmp, you
   * must specify a port range. For protocol 58 (ICMPv6), you can optionally
   * specify a port range; if you don't, traffic for all types and codes is
   * allowed.
   *
   * @default tcp
   */
  readonly protocol?: string;

  /**
   * Start of port range for the TCP and UDP protocols, or an ICMP type number.
   *
   * If you specify icmp for the IpProtocol property, you can specify
   * -1 as a wildcard (i.e., any ICMP type number).
   */
  readonly fromPort: number;

  /**
   * End of port range for the TCP and UDP protocols, or an ICMP code.
   *
   * If you specify icmp for the IpProtocol property, you can specify -1 as a
   * wildcard (i.e., any ICMP code).
   *
   * @default If toPort is not specified, it will be the same as fromPort.
   */
  readonly toPort?: number;

  /**
   * Description of this connection. It is applied to both the ingress rule
   * and the egress rule.
   *
   * @default No description
   */
  readonly description?: string;
}

/**
 * Compare two ingress rules for equality the same way CloudFormation would (discarding description)
 */
function ingressRulesEqual(a: CfnSecurityGroup.IngressProperty, b: CfnSecurityGroup.IngressProperty) {
  return a.cidrIp === b.cidrIp
    && a.cidrIpv6 === b.cidrIpv6
    && a.fromPort === b.fromPort
    && a.toPort === b.toPort
    && a.ipProtocol === b.ipProtocol
    && a.sourceSecurityGroupId === b.sourceSecurityGroupId
    && a.sourceSecurityGroupName === b.sourceSecurityGroupName
    && a.sourceSecurityGroupOwnerId === b.sourceSecurityGroupOwnerId;
}

/**
 * Compare two egress rules for equality the same way CloudFormation would (discarding description)
 */
function egressRulesEqual(a: CfnSecurityGroup.EgressProperty, b: CfnSecurityGroup.EgressProperty) {
  return a.cidrIp === b.cidrIp
    && a.cidrIpv6 === b.cidrIpv6
    && a.fromPort === b.fromPort
    && a.toPort === b.toPort
    && a.ipProtocol === b.ipProtocol
    && a.destinationPrefixListId === b.destinationPrefixListId
    && a.destinationSecurityGroupId === b.destinationSecurityGroupId;
}

/**
 * Whether this rule refers to all traffic
 */
function isAllTrafficRule(rule: any) {
  return (rule.cidrIp === '0.0.0.0/0' || rule.cidrIpv6 === '::/0') && rule.ipProtocol === '-1';
}

/**
 * Properties for looking up an existing SecurityGroup.
 *
 * Either `securityGroupName` or `securityGroupId` has to be specified.
 */
interface SecurityGroupLookupOptions {
  /**
   * The name of the security group
   *
   * If given, will import the SecurityGroup with this name.
   *
   * @default Don't filter on securityGroupName
   */
  readonly securityGroupName?: string;

  /**
   * The ID of the security group
   *
   * If given, will import the SecurityGroup with this ID.
   *
   * @default Don't filter on securityGroupId
   */
  readonly securityGroupId?: string;

  /**
   * The VPC of the security group
   *
   * If given, will filter the SecurityGroup based on the VPC.
   *
   * @default Don't filter on VPC
   */
  readonly vpc?: IVpc;
}

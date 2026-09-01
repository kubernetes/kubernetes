import type { Construct } from 'constructs';
import type { ICluster } from './cluster';
import { CfnAddon } from './eks.generated';
import type { IResource, RemovalPolicy } from '../../core';
import { ArnFormat, Resource, Stack, Fn } from '../../core';
import { memoizedGetter } from '../../core/lib/helpers-internal';
import { addConstructMetadata } from '../../core/lib/metadata-resource';
import { propertyInjectable } from '../../core/lib/prop-injectable';
import type { AddonReference, IAddonRef } from '../../interfaces/generated/aws-eks-interfaces.generated';

/**
 * Represents an Amazon EKS Add-On.
 */
export interface IAddon extends IResource, IAddonRef {
  /**
   * Name of the Add-On.
   * @attribute
   */
  readonly addonName: string;
  /**
   * ARN of the Add-On.
   * @attribute
   */
  readonly addonArn: string;
}

/**
 * Properties for creating an Amazon EKS Add-On.
 */
export interface AddonProps {
  /**
   * Name of the Add-On.
   */
  readonly addonName: string;
  /**
   * Version of the Add-On. You can check all available versions with describe-addon-versions.
   * For example, this lists all available versions for the `eks-pod-identity-agent` addon:
   * $ aws eks describe-addon-versions --addon-name eks-pod-identity-agent \
   * --query 'addons[*].addonVersions[*].addonVersion'
   *
   * @default the latest version.
   */
  readonly addonVersion?: string;
  /**
   * The EKS cluster the Add-On is associated with.
   */
  readonly cluster: ICluster;
  /**
   * Specifying this option preserves the add-on software on your cluster but Amazon EKS stops managing any settings for the add-on.
   * If an IAM account is associated with the add-on, it isn't removed.
   *
   * @default true
   */
  readonly preserveOnDelete?: boolean;

  /**
   * The configuration values for the Add-on.
   *
   * @default - Use default configuration.
   */
  readonly configurationValues?: Record<string, any>;

  /**
   * The removal policy applied to the EKS add-on.
   *
   * The removal policy controls what happens to the resource if it stops being managed by CloudFormation.
   * This can happen in one of three situations:
   *
   * - The resource is removed from the template, so CloudFormation stops managing it
   * - A change to the resource is made that requires it to be replaced, so CloudFormation stops managing it
   * - The stack is deleted, so CloudFormation stops managing all resources in it
   *
   * @default RemovalPolicy.DESTROY
   */
  readonly removalPolicy?: RemovalPolicy;
}

/**
 * Represents the attributes of an addon for an Amazon EKS cluster.
 */
export interface AddonAttributes {
  /**
   * The name of the addon.
   */
  readonly addonName: string;

  /**
   * The name of the Amazon EKS cluster the addon is associated with.
   */
  readonly clusterName: string;
}

/**
 * Represents an Amazon EKS Add-On.
 * @resource AWS::EKS::Addon
 */
@propertyInjectable
export class Addon extends Resource implements IAddon {
  /** Uniquely identifies this class. */
  public static readonly PROPERTY_INJECTION_ID: string = 'aws-cdk-lib.aws-eks-v2.Addon';

  /**
   * Creates an `IAddon` instance from the given addon attributes.
   *
   * @param scope - The parent construct.
   * @param id - The construct ID.
   * @param attrs - The attributes of the addon, including the addon name and the cluster name.
   * @returns An `IAddon` instance.
   */
  public static fromAddonAttributes(scope: Construct, id: string, attrs: AddonAttributes): IAddon {
    class Import extends Resource implements IAddon {
      private readonly clusterName = attrs.clusterName;
      public readonly addonName = attrs.addonName;
      public readonly addonArn = Stack.of(scope).formatArn({
        service: 'eks',
        resource: 'addon',
        resourceName: `${attrs.clusterName}/${attrs.addonName}`,
      });

      public get addonRef(): AddonReference {
        return {
          addonArn: this.addonArn,
          addonName: this.addonName,
          clusterName: this.clusterName,
        };
      }
    }
    return new Import(scope, id);
  }
  /**
   * Creates an `IAddon` from an existing addon ARN.
   *
   * @param scope - The parent construct.
   * @param id - The ID of the construct.
   * @param addonArn - The ARN of the addon.
   * @returns An `IAddon` implementation.
   */
  public static fromAddonArn(scope: Construct, id: string, addonArn: string): IAddon {
    const parsedArn = Stack.of(scope).splitArn(addonArn, ArnFormat.COLON_RESOURCE_NAME);
    const splitResourceName = Fn.split('/', parsedArn.resourceName!);
    class Import extends Resource implements IAddon {
      public readonly addonName = Fn.select(1, splitResourceName);
      public readonly addonArn = addonArn;

      public get addonRef(): AddonReference {
        return {
          addonArn: this.addonArn,
          addonName: this.addonName,
          get clusterName(): string {
            // eslint-disable-next-line @cdklabs/no-throw-default-error
            throw new Error('Cannot access clusterName, addon has been created without knowledge of its cluster');
          },
        };
      }
    }

    return new Import(scope, id);
  }

  private readonly clusterName: string;
  private resource: CfnAddon;

  /**
   * Creates a new Amazon EKS Add-On.
   * @param scope The parent construct.
   * @param id The construct ID.
   * @param props The properties for the Add-On.
   */
  constructor(scope: Construct, id: string, props: AddonProps) {
    super(scope, id, {
      physicalName: props.addonName,
    });
    // Enhanced CDK Analytics Telemetry
    addConstructMetadata(this, props);

    this.clusterName = props.cluster.clusterName;

    this.resource = new CfnAddon(this, 'Resource', {
      addonName: props.addonName,
      clusterName: this.clusterName,
      addonVersion: props.addonVersion,
      preserveOnDelete: props.preserveOnDelete,
      configurationValues: this.stack.toJsonString(props.configurationValues),
    });

    if (props.removalPolicy) {
      this.resource.applyRemovalPolicy(props.removalPolicy);
    }
  }

  /**
   * Name of the addon.
   */
  @memoizedGetter
  public get addonName(): string {
    return this.getResourceNameAttribute(this.resource.ref);
  }

  @memoizedGetter
  public get addonArn(): string {
    return this.getResourceArnAttribute(this.resource.attrArn, {
      service: 'eks',
      resource: 'addon',
      resourceName: `${this.clusterName}/${this.addonName}/`,
    });
  }

  public get addonRef(): AddonReference {
    return {
      addonArn: this.addonArn,
      addonName: this.addonName,
      clusterName: this.clusterName,
    };
  }
}

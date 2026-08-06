import { Construct } from 'constructs';
import * as ec2 from '../../../aws-ec2';
import * as iam from '../../../aws-iam';
import type * as kms from '../../../aws-kms';
import type { Size } from '../../../core';
import { Token, ValidationError } from '../../../core';
import { lit } from '../../../core/lib/private/literal-string';
import type { BaseMountPoint, ContainerDefinition } from '../container-definition';

/**
 * Represents the Volume configuration for an ECS service.
 */
export interface ServiceManagedVolumeProps {
  /**
   * The name of the volume. This corresponds to the name provided in the ECS TaskDefinition.
   */
  readonly name: string;

  /**
   * Configuration for an Amazon Elastic Block Store (EBS) volume managed by ECS.
   *
   * @default - undefined
   */
  readonly managedEBSVolume?: ServiceManagedEBSVolumeConfiguration;
}

/**
 * Represents the configuration for an ECS Service managed EBS volume.
 */
export interface ServiceManagedEBSVolumeConfiguration {
  /**
   * An IAM role that allows ECS to make calls to EBS APIs on your behalf.
   * This role is required to create and manage the Amazon EBS volume.
   *
   * @default - automatically generated role.
   */
  readonly role?: iam.IRole;

  /**
   * Indicates whether the volume should be encrypted.
   *
   * @default - Default Amazon EBS encryption.
   */
  readonly encrypted?: boolean;

  /**
   * AWS Key Management Service key to use for Amazon EBS encryption.
   *
   * @default - When `encryption` is turned on and no `kmsKey` is specified,
   * the default AWS managed key for Amazon EBS volumes is used.
   */
  readonly kmsKeyId?: kms.IKey;

  /**
   * The volume type.
   *
   * @default - ec2.EbsDeviceVolumeType.GP2
   */
  readonly volumeType?: ec2.EbsDeviceVolumeType;

  /**
   * The size of the volume in GiB.
   *
   * You must specify either `size` or `snapshotId`.
   * You can optionally specify a volume size greater than or equal to the snapshot size.
   *
   * The following are the supported volume size values for each volume type.
   *   - gp2 and gp3: 1-16,384
   *   - io1 and io2: 4-16,384
   *   - st1 and sc1: 125-16,384
   *   - standard: 1-1,024
   *
   * @default - The snapshot size is used for the volume size if you specify `snapshotId`,
   * otherwise this parameter is required.
   */
  readonly size?: Size;

  /**
   * The snapshot that Amazon ECS uses to create the volume.
   *
   * You must specify either `size` or `snapshotId`.
   *
   * @default - No snapshot.
   */
  readonly snapShotId?: string;

  /**
   * The number of I/O operations per second (IOPS).
   *
   * For gp3, io1, and io2 volumes, this represents the number of IOPS that are provisioned
   * for the volume. For gp2 volumes, this represents the baseline performance of the volume
   * and the rate at which the volume accumulates I/O credits for bursting.
   *
   * The following are the supported values for each volume type.
   *   - gp3: 3,000 - 16,000 IOPS
   *   - io1: 100 - 64,000 IOPS
   *   - io2: 100 - 256,000 IOPS
   *
   * This parameter is required for io1 and io2 volume types. The default for gp3 volumes is
   * 3,000 IOPS. This parameter is not supported for st1, sc1, or standard volume types.
   *
   * @default - undefined
   */
  readonly iops?: number;

  /**
   * The throughput to provision for a volume, in MiB/s, with a maximum of 1,000 MiB/s.
   *
   * This parameter is only supported for the gp3 volume type.
   *
   * @default - No throughput.
   */
  readonly throughput?: number;

  /**
   * The Linux filesystem type for the volume.
   *
   * For volumes created from a snapshot, you must specify the same filesystem type that
   * the volume was using when the snapshot was created.
   * The available filesystem types are ext3, ext4, and xfs.
   *
   * @default - FileSystemType.XFS
   */
  readonly fileSystemType?: FileSystemType;

  /**
   * Specifies the tags to apply to the volume and whether to propagate those tags to the volume.
   *
   * @default - No tags are specified.
   */
  readonly tagSpecifications?: EBSTagSpecification[];

  /**
   * Specifies the Amazon EBS Provisioned Rate for Volume Initialization (volume initialization rate),
   * at which to download the snapshot blocks from Amazon S3 to the volume.
   *
   * Valid range is between 100 and 300 MiB/s.
   *
   * @default undefined - The volume initialization rate is not set.
   */
  readonly volumeInitializationRate?: Size;
}

/**
 *  Tag Specification for EBS volume.
 */
export interface EBSTagSpecification {
  /**
   * The tags to apply to the volume.
   *
   * @default - No tags
   */
  readonly tags?: {[key: string]: string};

  /**
   * Specifies whether to propagate the tags from the task definition or the service to the task.
   * Valid values are: PropagatedTagSource.SERVICE, PropagatedTagSource.TASK_DEFINITION
   *
   * @default - undefined
   */
  readonly propagateTags?: EbsPropagatedTagSource;
}

/**
 * FileSystemType for Service Managed EBS Volume Configuration.
 */
export enum FileSystemType {
  /**
   * ext3 type
   */
  EXT3 = 'ext3',
  /**
   * ext4 type
   */
  EXT4 = 'ext4',
  /**
   * xfs type
   */
  XFS = 'xfs',
  /**
   * ntfs type
   */
  NTFS = 'ntfs',
}

/**
 * Propagate tags for EBS Volume Configuration from either service or task definition.
 */
export enum EbsPropagatedTagSource {
  /**
   * SERVICE
   */
  SERVICE = 'SERVICE',
  /**
   * TASK_DEFINITION
   */
  TASK_DEFINITION = 'TASK_DEFINITION',
}

/**
 * Defines the mount point details for attaching a volume to a container.
 */
export interface ContainerMountPoint extends BaseMountPoint {
}

/**
 * Represents a service-managed volume and always configured at launch.
 */
export class ServiceManagedVolume extends Construct {
  /**
   * Name of the volume, referenced by task definition and mount point.
   */
  public readonly name: string;

  /**
   * Volume configuration
   */
  public readonly config?: ServiceManagedEBSVolumeConfiguration;

  /**
   * configuredAtLaunch indicates volume at launch time, referenced by taskdefinition volume.
   */
  public readonly configuredAtLaunch: boolean = true;

  /**
   * An IAM role that allows ECS to make calls to EBS APIs.
   * If not provided, a new role with appropriate permissions will be created by default.
   */
  public readonly role: iam.IRole;

  constructor(scope: Construct, id: string, props: ServiceManagedVolumeProps) {
    super(scope, id);
    this.validateEbsVolumeConfiguration(props.managedEBSVolume);
    this.name = props.name;
    this.role = props.managedEBSVolume?.role ?? new iam.Role(this, 'EBSRole', {
      assumedBy: new iam.ServicePrincipal('ecs.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('service-role/AmazonECSInfrastructureRolePolicyForVolumes'),
      ],
    });
    this.config = {
      ...props.managedEBSVolume,
      role: this.role,
    };
  }

  /**
   *  Mounts the service managed volume to a specified container at a defined mount point.
   * @param container The container to mount the volume on.
   * @param mountPoint The mounting point details within the container.
   */
  public mountIn(container: ContainerDefinition, mountPoint: ContainerMountPoint) {
    container.addMountPoints({
      sourceVolume: this.name,
      ...mountPoint,
    });
  }

  private validateEbsVolumeConfiguration(volumeConfig?: ServiceManagedEBSVolumeConfiguration) {
    if (!volumeConfig) return;

    const { volumeType = ec2.EbsDeviceVolumeType.GP2, iops, size, throughput, snapShotId, volumeInitializationRate } = volumeConfig;

    if (volumeInitializationRate !== undefined && !Token.isUnresolved(volumeInitializationRate)) {
      if (snapShotId === undefined) {
        throw new ValidationError(lit`VolumeInitializationRateOnlySpecifiedSnapshotId`, '\'volumeInitializationRate\' can only be specified when \'snapShotId\' is provided.', this);
      }
      if (volumeInitializationRate.toMebibytes() < 100 || volumeInitializationRate.toMebibytes() > 300) {
        throw new ValidationError(lit`MustBeVolumeInitializationRateBetweenMibS`, `'volumeInitializationRate' must be between 100 and 300 MiB/s, got ${volumeInitializationRate.toMebibytes()} MiB/s.`, this);
      }
    }

    // Validate if both size and snapShotId are not specified.
    if (size === undefined && snapShotId === undefined) {
      throw new ValidationError(lit`MustBeSizeSnapshotIdSpecified`, '\'size\' or \'snapShotId\' must be specified', this);
    }

    if (snapShotId && !Token.isUnresolved(snapShotId) && !/^snap-[0-9a-fA-F]+$/.test(snapShotId)) {
      throw new ValidationError(lit`SnapshotIdDoesMatchExpected`, `'snapshotId' does match expected pattern. Expected 'snap-<hexadecmial value>' (ex: 'snap-05abe246af') or Token, got: ${snapShotId}`, this);
    }

    // https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-ecs-service-servicemanagedebsvolumeconfiguration.html#cfn-ecs-service-servicemanagedebsvolumeconfiguration-sizeingib
    const sizeInGiBRanges = {
      [ec2.EbsDeviceVolumeType.GP2]: { minSize: 1, maxSize: 16384 },
      [ec2.EbsDeviceVolumeType.GP3]: { minSize: 1, maxSize: 65536 },
      [ec2.EbsDeviceVolumeType.IO1]: { minSize: 4, maxSize: 16384 },
      [ec2.EbsDeviceVolumeType.IO2]: { minSize: 4, maxSize: 65536 },
      [ec2.EbsDeviceVolumeType.SC1]: { minSize: 125, maxSize: 16384 },
      [ec2.EbsDeviceVolumeType.ST1]: { minSize: 125, maxSize: 16384 },
      [ec2.EbsDeviceVolumeType.STANDARD]: { minSize: 1, maxSize: 1024 },
    };

    // Validate volume size ranges.
    if (size !== undefined) {
      const { minSize, maxSize } = sizeInGiBRanges[volumeType];
      if (size.toGibibytes() < minSize || size.toGibibytes() > maxSize) {
        throw new ValidationError(lit`VolumesSizeBetween`, `'${volumeType}' volumes must have a size between ${minSize} and ${maxSize} GiB, got ${size.toGibibytes()} GiB`, this);
      }
    }

    // Validate throughput.
    // https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-ecs-service-servicemanagedebsvolumeconfiguration.html#cfn-ecs-service-servicemanagedebsvolumeconfiguration-throughput
    if (throughput !== undefined) {
      if (volumeType !== ec2.EbsDeviceVolumeType.GP3) {
        throw new ValidationError(lit`ThroughputOnlyConfiguredVolume`, `'throughput' can only be configured with gp3 volume type, got ${volumeType}`, this);
      } else if (!Token.isUnresolved(throughput) && throughput > 2000) {
        throw new ValidationError(lit`MustBeThroughputLessThan`, `'throughput' must be less than or equal to 2000 MiB/s, got ${throughput} MiB/s`, this);
      }
    }

    // Check if IOPS is not supported for the volume type.
    // https://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_CreateVolume.html
    if ([ec2.EbsDeviceVolumeType.SC1, ec2.EbsDeviceVolumeType.ST1, ec2.EbsDeviceVolumeType.STANDARD,
      ec2.EbsDeviceVolumeType.GP2].includes(volumeType) && iops !== undefined) {
      throw new ValidationError(lit`IopsCannotSpecified`, `'iops' cannot be specified with sc1, st1, gp2 and standard volume types, got ${volumeType}`, this);
    }

    // Check if IOPS is required but not provided.
    if ([ec2.EbsDeviceVolumeType.IO1, ec2.EbsDeviceVolumeType.IO2].includes(volumeType) && iops === undefined) {
      throw new ValidationError(lit`MustBeIopsSpecifiedVolume`, `'iops' must be specified with io1 or io2 volume types, got ${volumeType}`, this);
    }

    // Validate IOPS range if specified.
    const iopsRanges: { [key: string]: { min: number; max: number } } = {};
    iopsRanges[ec2.EbsDeviceVolumeType.GP3]= { min: 3000, max: 80000 };
    iopsRanges[ec2.EbsDeviceVolumeType.IO1]= { min: 100, max: 64000 };
    iopsRanges[ec2.EbsDeviceVolumeType.IO2]= { min: 100, max: 256000 };
    if (iops !== undefined && !Token.isUnresolved(iops)) {
      const { min, max } = iopsRanges[volumeType];
      if ((iops < min || iops > max)) {
        throw new ValidationError(lit`IopsOutOfRange`, `'${volumeType}' volumes must have 'iops' between ${min} and ${max}, got ${iops}`, this);
      }
    }
  }
}

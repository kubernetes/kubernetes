import type { Construct } from 'constructs';
import { Annotations, SizeRoundingBehavior, ValidationError } from '../../../core';
import { lit } from '../../../core/lib/private/literal-string';
import type { CfnInstance, CfnLaunchTemplate } from '../ec2.generated';
import type { BlockDevice } from '../volume';
import { EbsDeviceVolumeType } from '../volume';

export function instanceBlockDeviceMappings(construct: Construct, blockDevices: BlockDevice[]): CfnInstance.BlockDeviceMappingProperty[] {
  for (const blockDevice of blockDevices) {
    if (blockDevice.volume.ebsDevice?.throughput !== undefined) {
      Annotations.of(construct).addWarningV2('@aws-cdk/aws-ec2:throughputNotSupported',
        'The throughput property is not supported on EC2 instances. Use a Launch Template instead. ' +
          'See https://github.com/aws/aws-cdk/issues/34033 for more information.',
      );
    }
    if (blockDevice.volume.ebsDevice?.volumeInitializationRate !== undefined) {
      Annotations.of(construct).addWarningV2('@aws-cdk/aws-ec2:volumeInitializationRateNotSupported',
        'The volumeInitializationRate is not supported on EC2 instances. Use a Launch Template instead.',
      );
    }
  }
  return synthesizeBlockDeviceMappings<CfnInstance.BlockDeviceMappingProperty, object>(construct, blockDevices, {});
}

export function launchTemplateBlockDeviceMappings(construct: Construct, blockDevices: BlockDevice[]): CfnLaunchTemplate.BlockDeviceMappingProperty[] {
  return synthesizeBlockDeviceMappings<CfnLaunchTemplate.BlockDeviceMappingProperty, string>(construct, blockDevices, '');
}

/**
 * Synthesize an array of block device mappings from a list of block device
 *
 * @param construct the instance/asg construct, used to host any warning
 * @param blockDevices list of block devices
 */
function synthesizeBlockDeviceMappings<RT, NDT>(construct: Construct, blockDevices: BlockDevice[], noDeviceValue: NDT): RT[] {
  return blockDevices.map<RT>(({ deviceName, volume, mappingEnabled }): RT => {
    const { virtualName, ebsDevice: ebs } = volume;

    let finalEbs: CfnLaunchTemplate.EbsProperty | CfnInstance.EbsProperty | undefined;

    if (ebs) {
      const { iops, throughput, volumeType, kmsKey, volumeInitializationRate, ...rest } = ebs;

      if (throughput) {
        if (volumeType !== EbsDeviceVolumeType.GP3) {
          throw new ValidationError(lit`RequiresThroughputRequiresVolumetype`, `'throughput' requires 'volumeType': ${EbsDeviceVolumeType.GP3}, got: ${volumeType}.`, construct);
        }

        if (!Number.isInteger(throughput)) {
          throw new ValidationError(lit`MustBeThroughputInteger`, `'throughput' must be an integer, got: ${throughput}.`, construct);
        }

        if (throughput < 125 || throughput > 2000) {
          throw new ValidationError(lit`MustBeThroughputBetween2000`, `'throughput' must be between 125 and 2000, got ${throughput}.`, construct);
        }

        const maximumThroughputRatio = 0.25;
        if (iops) {
          const iopsRatio = (throughput / iops);
          if (iopsRatio > maximumThroughputRatio) {
            throw new ValidationError(lit`ThroughputMiBpsIopsRatio`, `Throughput (MiBps) to iops ratio of ${iopsRatio} is too high; maximum is ${maximumThroughputRatio} MiBps per iops`, construct);
          }
        }
      }

      if (volumeInitializationRate !== undefined && !volumeInitializationRate.isUnresolved()) {
        const rateMiBs = volumeInitializationRate.toMebibytes({ rounding: SizeRoundingBehavior.NONE });
        if (rateMiBs < 100 || rateMiBs > 300) {
          throw new ValidationError(lit`VolumeInitializationRateOutOfRange`, `volumeInitializationRate must be between 100 and 300 MiB/s, got: ${rateMiBs} MiB/s`, construct);
        }
      }

      if (!iops) {
        if (volumeType === EbsDeviceVolumeType.IO1 || volumeType === EbsDeviceVolumeType.IO2) {
          throw new ValidationError(lit`IopsPropertyRequiredVolumeType`, 'iops property is required with volumeType: EbsDeviceVolumeType.IO1 and EbsDeviceVolumeType.IO2', construct);
        }
      } else if (volumeType !== EbsDeviceVolumeType.IO1 && volumeType !== EbsDeviceVolumeType.IO2 && volumeType !== EbsDeviceVolumeType.GP3) {
        Annotations.of(construct).addWarningV2('@aws-cdk/aws-ec2:iopsIgnored', 'iops will be ignored without volumeType: IO1, IO2, or GP3');
      }

      /**
       * Because the Ebs properties of the L2 Constructs do not match the Ebs properties of the Cfn Constructs,
       * we have to do some transformation and handle all destructed properties
       */

      finalEbs = {
        ...rest,
        iops,
        throughput,
        volumeInitializationRate: volumeInitializationRate?.toMebibytes(),
        volumeType,
        kmsKeyId: kmsKey?.keyArn,
      };
    } else {
      finalEbs = undefined;
    }

    const noDevice = mappingEnabled === false ? noDeviceValue : undefined;
    return { deviceName, ebs: finalEbs, virtualName, noDevice } as any;
  });
}

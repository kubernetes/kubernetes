import * as cdk from 'aws-cdk-lib';
import * as integ from '@aws-cdk/integ-tests-alpha';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import { SnapshotProvider } from './snapshot-provider';

const app = new cdk.App();

const stack = new cdk.Stack(app, 'launchtemplate-volume-initialization-rate-stack');

const az = stack.availabilityZones[0];

// Custom resource that creates a snapshot dynamically during deployment.
// This is needed because:
// 1. The test validates volumeInitializationRate, which requires a real snapshot
// 2. Snapshot IDs are region/account-specific and cannot be hardcoded
// 3. The snapshot must exist before the Volume resource is created
//
// The SnapshotProvider creates a temporary volume, snapshots it, deletes the volume,
// and returns the snapshot ID. On stack deletion, it cleans up the snapshot.
const snapshotProvider = new SnapshotProvider(stack, 'SnapshotProvider', {
  availabilityZone: az,
});

// Integration test case focusing on volumeInitializationRate with a snapshot-backed EBS volume
new ec2.LaunchTemplate(stack, 'LTWithVolumeInitRate', {
  blockDevices: [{
    deviceName: '/dev/xvda',
    volume: ec2.BlockDeviceVolume.ebsFromSnapshot(snapshotProvider.snapshotId, {
      volumeType: ec2.EbsDeviceVolumeType.GP3,
      volumeInitializationRate: cdk.Size.mebibytes(300),
    }),
  }],
});

new integ.IntegTest(app, 'launchtemplate-volume-initialization-rate-integ', {
  testCases: [stack],
});

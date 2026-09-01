import * as cdk from '../../core';
import { HostedZone } from '../lib';

describe('hosted zone provider', () => {
  describe('Hosted Zone Provider', () => {
    test('HostedZoneProvider will return context values if available', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'TestStack', {
        env: { account: '12345', region: 'us-east-1' },
      });
      const filter = { domainName: 'test.com' };

      HostedZone.fromLookup(stack, 'Ref', filter);

      const assembly = app.synth().getStackArtifact(stack.artifactId);
      const missing = assembly.assembly.manifest.missing!;
      expect(missing && missing.length === 1).toEqual(true);

      const fakeZoneId = '111111111111';
      const fakeZone = {
        Id: `/hostedzone/${fakeZoneId}`,
        Name: 'example.com.',
        CallerReference: 'TestLates-PublicZo-OESZPDFV7G6A',
        Config: {
          Comment: 'CDK created',
          PrivateZone: false,
        },
        ResourceRecordSetCount: 3,
      };

      const stack2 = new cdk.Stack(undefined, 'TestStack', {
        env: { account: '12345', region: 'us-east-1' },
      });
      stack2.node.setContext(missing[0].key, fakeZone);

      // WHEN
      const zoneRef = HostedZone.fromLookup(stack2, 'MyZoneProvider', filter);

      // THEN
      expect(zoneRef.hostedZoneId).toEqual(fakeZoneId);
    });

    test('HostedZoneProvider will return context values if available when using plain hosted zone id', () => {
      // GIVEN
      const app = new cdk.App();
      const stack = new cdk.Stack(app, 'TestStack', {
        env: { account: '12345', region: 'us-east-1' },
      });
      const filter = { domainName: 'test.com' };

      HostedZone.fromLookup(stack, 'Ref', filter);

      const assembly = app.synth().getStackArtifact(stack.artifactId);
      const missing = assembly.assembly.manifest.missing!;
      expect(missing && missing.length === 1).toEqual(true);

      const fakeZoneId = '111111111111';
      const fakeZone = {
        Id: `/hostedzone/${fakeZoneId}`,
        Name: 'example.com.',
        CallerReference: 'TestLates-PublicZo-OESZPDFV7G6A',
        Config: {
          Comment: 'CDK created',
          PrivateZone: false,
        },
        ResourceRecordSetCount: 3,
      };

      const stack2 = new cdk.Stack(undefined, 'TestStack', {
        env: { account: '12345', region: 'us-east-1' },
      });
      stack2.node.setContext(missing[0].key, fakeZone);

      const zone = HostedZone.fromLookup(stack2, 'MyZoneProvider', filter);

      // WHEN
      const zoneId = zone.hostedZoneId;

      // THEN
      expect(fakeZoneId).toEqual(zoneId);
    });
  });
});

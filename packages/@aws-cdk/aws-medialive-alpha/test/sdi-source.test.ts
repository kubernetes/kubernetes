import { App, Stack } from 'aws-cdk-lib';
import { Template, Match } from 'aws-cdk-lib/assertions';
import { SdiSource, SdiType, SdiMode } from '../lib';

let app: App;
let stack: Stack;

beforeEach(() => {
  app = new App();
  stack = new Stack(app, 'TestStack', {
    env: { account: '123456789012', region: 'us-east-1' },
  });
});

describe('SdiSource', () => {
  test('creates a minimal single SDI source', () => {
    new SdiSource(stack, 'Sdi', {
      sdiSourceName: 'cam-1',
      type: SdiType.SINGLE,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::SdiSource', {
      Name: 'cam-1',
      Type: 'SINGLE',
    });
  });

  test.each([
    SdiMode.INTERLEAVE,
    SdiMode.QUADRANT,
  ])('renders a QUAD source with mode %s', (mode) => {
    new SdiSource(stack, 'Sdi', {
      sdiSourceName: 'quad-cam',
      type: SdiType.QUAD,
      mode,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::SdiSource', {
      Type: 'QUAD',
      Mode: mode.value,
    });
  });

  test('allows mode when type is QUAD via the of() escape hatch', () => {
    new SdiSource(stack, 'Sdi', {
      sdiSourceName: 'quad-cam',
      type: SdiType.of('QUAD'),
      mode: SdiMode.INTERLEAVE,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::SdiSource', {
      Type: 'QUAD',
      Mode: 'INTERLEAVE',
    });
  });

  test('fails when mode is set but type is not QUAD', () => {
    expect(() => new SdiSource(stack, 'Sdi', {
      sdiSourceName: 'single-cam',
      type: SdiType.SINGLE,
      mode: SdiMode.INTERLEAVE,
    })).toThrow('mode is only valid when type is QUAD');
  });

  test('omits mode for a SINGLE source', () => {
    new SdiSource(stack, 'Sdi', {
      sdiSourceName: 'single-cam',
      type: SdiType.SINGLE,
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::SdiSource', {
      Mode: Match.absent(),
    });
  });

  test('renders tags', () => {
    new SdiSource(stack, 'Sdi', {
      sdiSourceName: 'tagged-cam',
      type: SdiType.SINGLE,
      tags: { env: 'prod' },
    });

    Template.fromStack(stack).hasResourceProperties('AWS::MediaLive::SdiSource', {
      Tags: [{ Key: 'env', Value: 'prod' }],
    });
  });

  test('fromSdiSourceAttributes wires the provided attributes', () => {
    const imported = SdiSource.fromSdiSourceAttributes(stack, 'Imported', {
      sdiSourceArn: 'arn:aws:medialive:us-east-1:123456789012:sdiSource:sdi-123',
      sdiSourceId: 'sdi-123',
    });

    expect(imported.sdiSourceArn).toBe('arn:aws:medialive:us-east-1:123456789012:sdiSource:sdi-123');
    expect(imported.sdiSourceId).toBe('sdi-123');
    expect(imported.sdiSourceInputs).toBeUndefined();
    expect(imported.sdiSourceState).toBeUndefined();
    expect(imported.sdiSourceRef).toEqual({
      sdiSourceId: 'sdi-123',
      sdiSourceArn: 'arn:aws:medialive:us-east-1:123456789012:sdiSource:sdi-123',
    });
  });

  test('sdiSourceRef resolves from the construct', () => {
    const source = new SdiSource(stack, 'Sdi', {
      sdiSourceName: 'ref-cam',
      type: SdiType.SINGLE,
    });

    expect(source.sdiSourceRef).toEqual({
      sdiSourceId: source.sdiSourceId,
      sdiSourceArn: source.sdiSourceArn,
    });
  });
});

import { Lazy } from 'aws-cdk-lib';
import { Segment, PixelAspectRatio, Framerate } from '../lib';

describe('Segment', () => {
  test('seconds stores value and units', () => {
    const s = Segment.seconds(6);
    expect(s._length()).toBe(6);
    expect(s._units()).toBe('SECONDS');
  });

  test('milliseconds stores value and units', () => {
    const s = Segment.milliseconds(4000);
    expect(s._length()).toBe(4000);
    expect(s._units()).toBe('MILLISECONDS');
  });

  test('_toSeconds converts whole-second milliseconds', () => {
    expect(Segment.milliseconds(4000)._toSeconds()).toBe(4);
    expect(Segment.seconds(6)._toSeconds()).toBe(6);
  });

  test('_toSeconds fails for sub-second milliseconds', () => {
    expect(() => Segment.milliseconds(4500)._toSeconds()).toThrow(/whole number of seconds/);
  });

  test.each([-1, 2.5])('fails for invalid value %p', (value) => {
    expect(() => Segment.seconds(value)).toThrow(/non-negative integer/);
  });

  test('accepts zero', () => {
    expect(() => Segment.seconds(0)).not.toThrow();
  });

  test('does not validate tokenized values', () => {
    expect(() => Segment.seconds(Lazy.number({ produce: () => 6 }))).not.toThrow();
  });
});

describe('PixelAspectRatio', () => {
  test('of stores numerator and denominator', () => {
    const par = PixelAspectRatio.of(16, 9);
    expect(par._numerator()).toBe(16);
    expect(par._denominator()).toBe(9);
    expect(par.toString()).toBe('16:9');
  });

  test('SQUARE is 1:1', () => {
    expect(PixelAspectRatio.SQUARE.toString()).toBe('1:1');
  });

  test.each([0, -1, 1.5])('fails for invalid numerator %p', (value) => {
    expect(() => PixelAspectRatio.of(value, 1)).toThrow(/numerator must be a positive integer/);
  });

  test.each([0, -1, 1.5])('fails for invalid denominator %p', (value) => {
    expect(() => PixelAspectRatio.of(1, value)).toThrow(/denominator must be a positive integer/);
  });

  test('does not validate tokenized values', () => {
    expect(() => PixelAspectRatio.of(
      Lazy.number({ produce: () => 16 }),
      Lazy.number({ produce: () => 9 }),
    )).not.toThrow();
  });
});

describe('Framerate', () => {
  test('of stores numerator and denominator', () => {
    const fr = Framerate.of(30000, 1001);
    expect(fr._numerator()).toBe(30000);
    expect(fr._denominator()).toBe(1001);
    expect(fr.toString()).toBe('30000/1001');
  });

  test('FPS_29_97 constant renders as 30000/1001', () => {
    expect(Framerate.FPS_29_97.toString()).toBe('30000/1001');
  });

  test.each([0, -1, 1.5])('fails for invalid numerator %p', (value) => {
    expect(() => Framerate.of(value, 1)).toThrow(/numerator must be a positive integer/);
  });

  test.each([0, -1, 1.5])('fails for invalid denominator %p', (value) => {
    expect(() => Framerate.of(1, value)).toThrow(/denominator must be a positive integer/);
  });

  test('does not validate tokenized values', () => {
    expect(() => Framerate.of(
      Lazy.number({ produce: () => 30 }),
      Lazy.number({ produce: () => 1 }),
    )).not.toThrow();
  });
});

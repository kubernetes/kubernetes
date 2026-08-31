import { Lazy } from 'aws-cdk-lib';
import { GopSize } from '../lib';

describe('GopSize', () => {
  test('seconds stores value and units', () => {
    const g = GopSize.seconds(2);
    expect(g._value).toBe(2);
    expect(g._units).toBe('SECONDS');
  });

  test('frames stores value and units', () => {
    const g = GopSize.frames(60);
    expect(g._value).toBe(60);
    expect(g._units).toBe('FRAMES');
  });

  test('seconds may be fractional', () => {
    expect(() => GopSize.seconds(1.5)).not.toThrow();
    expect(GopSize.seconds(1.5)._value).toBe(1.5);
  });

  test.each([0, -1])('seconds fails for non-positive value %p', (value) => {
    expect(() => GopSize.seconds(value)).toThrow(/greater than zero/);
  });

  test.each([0, -1])('frames fails for non-positive value %p', (value) => {
    expect(() => GopSize.frames(value)).toThrow(/greater than zero/);
  });

  test.each([1.5, 59.9])('frames fails for fractional value %p', (value) => {
    expect(() => GopSize.frames(value)).toThrow(/whole number/);
  });

  test('does not validate tokenized values', () => {
    expect(() => GopSize.seconds(Lazy.number({ produce: () => 2 }))).not.toThrow();
    expect(() => GopSize.frames(Lazy.number({ produce: () => 60 }))).not.toThrow();
  });
});

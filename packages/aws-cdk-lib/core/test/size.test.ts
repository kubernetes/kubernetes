import { Size, SizeRoundingBehavior, Stack, Token, Lazy } from '../lib';

describe('size', () => {
  test('negative amount', () => {
    expect(() => Size.kibibytes(-1)).toThrow(/negative/);
  });

  test('unresolved amount', () => {
    const stack = new Stack();
    const lazySize = Size.kibibytes(Token.asNumber({ resolve: () => 1337 }));
    expect(stack.resolve(lazySize.toKibibytes())).toEqual(1337);
    expect(
      () => stack.resolve(lazySize.toMebibytes())).toThrow(
      /Size must be specified as 'Size.mebibytes\(\)' here/,
    );
  });

  test('Size in bytes', () => {
    const size = Size.bytes(1_099_511_627_776);

    expect(size.toBytes()).toEqual(1_099_511_627_776);
    expect(size.toKibibytes()).toEqual(1_073_741_824);
    expect(size.toMebibytes()).toEqual(1_048_576);
    expect(size.toGibibytes()).toEqual(1024);
    expect(size.toTebibytes()).toEqual(1);
    expect(() => size.toPebibytes()).toThrow(/'1099511627776 bytes' cannot be converted into a whole number/);
    floatEqual(size.toPebibytes({ rounding: SizeRoundingBehavior.NONE }), 1_099_511_627_776 / (1024 * 1024 * 1024 * 1024 * 1024));

    expect(Size.bytes(4 * 1024 * 1024 * 1024).toGibibytes()).toEqual(4);
  });

  test('Size in kibibytes', () => {
    const size = Size.kibibytes(4_294_967_296);

    expect(size.toKibibytes()).toEqual(4_294_967_296);
    expect(size.toMebibytes()).toEqual(4_194_304);
    expect(size.toGibibytes()).toEqual(4_096);
    expect(size.toTebibytes()).toEqual(4);
    expect(() => size.toPebibytes()).toThrow(/'4294967296 kibibytes' cannot be converted into a whole number/);
    floatEqual(size.toPebibytes({ rounding: SizeRoundingBehavior.NONE }), 4_294_967_296 / (1024 * 1024 * 1024 * 1024));

    expect(Size.kibibytes(4 * 1024 * 1024).toGibibytes()).toEqual(4);
  });

  test('Size in mebibytes', () => {
    const size = Size.mebibytes(4_194_304);

    expect(size.toKibibytes()).toEqual(4_294_967_296);
    expect(size.toMebibytes()).toEqual(4_194_304);
    expect(size.toGibibytes()).toEqual(4_096);
    expect(size.toTebibytes()).toEqual(4);
    expect(() => size.toPebibytes()).toThrow(/'4194304 mebibytes' cannot be converted into a whole number/);
    floatEqual(size.toPebibytes({ rounding: SizeRoundingBehavior.NONE }), 4_194_304 / (1024 * 1024 * 1024));

    expect(Size.mebibytes(4 * 1024).toGibibytes()).toEqual(4);
  });

  test('Size in gibibyte', () => {
    const size = Size.gibibytes(5);

    expect(size.toKibibytes()).toEqual(5_242_880);
    expect(size.toMebibytes()).toEqual(5_120);
    expect(size.toGibibytes()).toEqual(5);
    expect(() => size.toTebibytes()).toThrow(/'5 gibibytes' cannot be converted into a whole number/);
    floatEqual(size.toTebibytes({ rounding: SizeRoundingBehavior.NONE }), 5 / 1024);
    expect(() => size.toPebibytes()).toThrow(/'5 gibibytes' cannot be converted into a whole number/);
    floatEqual(size.toPebibytes({ rounding: SizeRoundingBehavior.NONE }), 5 / (1024 * 1024));

    expect(Size.gibibytes(4096).toTebibytes()).toEqual(4);
  });

  test('Size in tebibyte', () => {
    const size = Size.tebibytes(5);

    expect(size.toKibibytes()).toEqual(5_368_709_120);
    expect(size.toMebibytes()).toEqual(5_242_880);
    expect(size.toGibibytes()).toEqual(5_120);
    expect(size.toTebibytes()).toEqual(5);
    expect(() => size.toPebibytes()).toThrow(/'5 tebibytes' cannot be converted into a whole number/);
    floatEqual(size.toPebibytes({ rounding: SizeRoundingBehavior.NONE }), 5 / 1024);

    expect(Size.tebibytes(4096).toPebibytes()).toEqual(4);
  });

  test('Size in pebibytes', () => {
    const size = Size.pebibytes(5);

    expect(size.toKibibytes()).toEqual(5_497_558_138_880);
    expect(size.toMebibytes()).toEqual(5_368_709_120);
    expect(size.toGibibytes()).toEqual(5_242_880);
    expect(size.toTebibytes()).toEqual(5_120);
    expect(size.toPebibytes()).toEqual(5);
  });

  test('rounding behavior', () => {
    const size = Size.mebibytes(5_200);

    expect(() => size.toGibibytes()).toThrow(/cannot be converted into a whole number/);
    expect(() => size.toGibibytes({ rounding: SizeRoundingBehavior.FAIL })).toThrow(/cannot be converted into a whole number/);

    expect(size.toGibibytes({ rounding: SizeRoundingBehavior.FLOOR })).toEqual(5);
    expect(size.toTebibytes({ rounding: SizeRoundingBehavior.FLOOR })).toEqual(0);
    floatEqual(size.toKibibytes({ rounding: SizeRoundingBehavior.FLOOR }), 5_324_800);

    expect(size.toGibibytes({ rounding: SizeRoundingBehavior.NONE })).toEqual(5.078125);
    expect(size.toTebibytes({ rounding: SizeRoundingBehavior.NONE })).toEqual(5200 / (1024 * 1024));
    expect(size.toKibibytes({ rounding: SizeRoundingBehavior.NONE })).toEqual(5_324_800);
  });

  test('size is unresolved', () => {
    const lazySize = Size.pebibytes(Lazy.number({ produce: () => 10 }));
    expect(lazySize.isUnresolved()).toEqual(true);
    expect(Size.mebibytes(10).isUnresolved()).toEqual(false);
  });

  test('stringification of Size objects', () => {
    expect(Size.bytes(123).toHumanString()).toEqual('123 bytes');
    expect(Size.bytes(1000000).toHumanString()).toEqual('976.56 KiB');
    expect(Size.bytes(1024 * 1024 * 1024).toHumanString()).toEqual('1 GiB');
    expect(Size.gibibytes(1).toHumanString()).toEqual('1 GiB');
  });

  test('stringification of Size objects', () => {
    expect(String(Size.bytes(123))).toEqual('Size.bytes(123)');
    expect(String(Size.bytes(1000000))).toEqual('Size.bytes(1000000)');
    expect(String(Size.bytes(1024 * 1024 * 1024))).toEqual('Size.bytes(1073741824)');
    expect(String(Size.gibibytes(1))).toEqual('Size.gibibytes(1)');
  });
});

function floatEqual(actual: number, expected: number) {
  expect(
    // Floats are subject to rounding errors up to Number.ESPILON
    actual >= expected - Number.EPSILON && actual <= expected + Number.EPSILON,
  ).toEqual(true);
}

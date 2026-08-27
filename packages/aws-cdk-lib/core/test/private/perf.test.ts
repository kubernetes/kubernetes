import * as fs from 'fs';
import type { RecordPerformanceOptions } from '../../lib/private/perf';
import { profileClass, profileFn, profileObj, profileSpan, readPerfCounters, recordPerformanceEntry, resetCounters } from '../../lib/private/perf';

beforeEach(() => {
  resetCounters();
});

test('functions can be instrumented', () => {
  // GIVEN
  const fn = profileFn('fn')(() => {});

  // WHEN
  fn();

  // THEN
  expect(readPerfCounters()).toMatchObject({
    fn: {
      count: 1,
      total: expect.anything(),
    },
  });
});

test('instrumented functions have the same name', () => {
  // GIVEN
  const orig = function someFunction() { };

  // WHEN
  const fn = profileFn('fn')(orig);

  // THEN
  expect(fn.name).toEqual(orig.name);
});

test('only top-level profiled function calls are recorded', () => {
  // GIVEN
  const inner = profileFn('inner')(() => {});
  const outer = profileFn('outer')(() => {
    inner();
  });

  // WHEN
  outer();

  // THEN
  const ctrs = readPerfCounters();
  expect(ctrs).toMatchObject({ outer: expect.anything() });
  expect(ctrs).not.toMatchObject({ inner: expect.anything() });
});

test('spans can be recorded, and counts can be skipped', () => {
  // GIVEN
  {
    using _ = profileSpan('x');
  }

  {
    using _ = profileSpan('x', { skipCount: true });
  }

  // THEN
  const ctrs = readPerfCounters();
  expect(ctrs).toMatchObject({
    x: {
      total: expect.anything(),
      count: 1,
    },
  });
});

function recordCounter(key: string, count: number, options?: RecordPerformanceOptions) {
  recordPerformanceEntry(key, { count, ...options });
}

test('arbitrary counters can be recorded, aggregated, and selected for telemetry', () => {
  // WHEN
  recordCounter('yes', 2, { telemetry: true });
  recordCounter('yes', 3, { telemetry: true });
  recordCounter('zero', 0, { telemetry: true });
  recordCounter('no', 7);

  // THEN
  const ctrs = readPerfCounters({ telemetry: true });
  expect(ctrs).toMatchObject({
    yes: { count: 5, total: 0 },
    zero: { count: 0, total: 0 },
  });
  expect(ctrs).not.toHaveProperty('no');
});

/**
 * This test can only exist once, after monkey patching 'fs' it cannot be monkey patched
 * again and we don't have an "uninstall" function.
 */
test('fs can be monkey-patched', () => {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  profileObj('fs')(require('fs'));

  // GIVEN
  fs.readFileSync(__filename, 'utf-8');

  // THEN -- patching works
  const ctrs = readPerfCounters();
  expect(ctrs).toMatchObject({ 'fs.readFileSync': expect.anything() });

  // THEN -- fs.realpath.native is still a function (fs-extra requires this, otherwise it will complain)
  expect(typeof fs.realpath.native).toEqual('function');
});

test('classes can be instrumented', () => {
  const InstrumentedClass = profileClass(class SomeClass {
    public method() {
    }
  });

  new InstrumentedClass().method();

  // THEN
  const ctrs = readPerfCounters();
  expect(ctrs).toMatchObject({ 'SomeClass.method': expect.anything() });
});

test('readPerfCounters can select only telemetry profilings', () => {
  // GIVEN
  const yes = profileFn('yes', { telemetry: true })(() => {});
  const no = profileFn('no')(() => {});

  // WHEN
  yes();
  no();

  // THEN
  const ctrs = readPerfCounters({ telemetry: true });
  expect(ctrs).toMatchObject({ yes: expect.anything() });
  expect(ctrs).not.toMatchObject({ no: expect.anything() });
});

test('readPerfCounters can select all profilings', () => {
  // GIVEN
  const yes = profileFn('yes', { telemetry: true })(() => {});
  const no = profileFn('no')(() => {});

  // WHEN
  yes();
  no();

  // THEN
  const ctrs = readPerfCounters();
  expect(ctrs).toMatchObject({
    yes: expect.anything(),
    no: expect.anything(),
  });
});

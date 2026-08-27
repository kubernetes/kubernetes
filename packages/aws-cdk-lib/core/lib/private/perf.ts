import { performance } from 'perf_hooks';
import './dispose-polyfill';

/**
 * Global state for this module.
 *
 * This is to safeguard against multiple loads of the same module, we want
 * them to share the same global state.
 */
interface GlobalState {
  /**
   * Whether we hooked the built-ins already.
   */
  builtinsHooked: boolean;

  /**
   * Global counter to keep track of nesting
   */
  nesting: number;

  telemetryCounters: PerfCounters;

  allCounters: PerfCounters;
}

const STATE: GlobalState = ((globalThis as any)[Symbol.for('@aws-cdk/core:performanceProfilingState')] ??= {
  builtinsHooked: false,
  nesting: 0,
  telemetryCounters: {},
  allCounters: {},
} satisfies GlobalState);

export interface ProfileOptions {
  /**
   * Whether to include the given profiling information in the report that the CLI sends for telemetry.
   *
   * @default false
   */
  readonly telemetry?: boolean;

  /**
   * If set to true, do not increment the counter by `1`
   *
   * This is a trick for if multiple distinct spans should be counted together as time for a single
   * conceptual invocation. Used for bundling, use sparingly.
   *
   * @default false
   */
  readonly skipCount?: boolean;
}

/**
 * Make a decorator that will profile a given function.
 *
 * Emits a measurement to the performance timeline, potentially with `{ telemetry: true }` in
 * the `detail` field.
 *
 * If a profiled function is called from another profiled function, only the
 * top-level call is accounted for.
 */
export function profileFn(key: string, options?: ProfileOptions) {
  return function <A extends Function>(fn: A): A {
    const ret = function(this: any) {
      STATE.nesting++;
      const start = performance.now();
      try {
        return fn.apply(this, arguments);
      } finally {
        const end = performance.now();
        STATE.nesting--;

        if (STATE.nesting === 0) {
          recordPerformanceEntry(key, {
            durationMs: end - start,
            count: options?.skipCount ? 0 : 1,
            telemetry: !!options?.telemetry,
          });
        }
      }
    };

    // Copy members, notably 'name' and sometimes add-on members
    Object.defineProperties(ret, Object.getOwnPropertyDescriptors(fn));

    return ret as any;
  };
}

/**
 * Profile a block by returning a disposable that will emit a (non-exclusive) counter when disposed
 *
 * Recommended way of using this:
 *
 * ```ts
 * using _span = profileSpan('span-name', { telemetry: true });
 * ```
 */
export function profileSpan(key: string, options?: ProfileOptions): Disposable {
  const start = performance.now();
  return {
    [Symbol.dispose]() {
      recordPerformanceEntry(key, {
        durationMs: performance.now() - start,
        count: options?.skipCount ? 0 : 1,
        telemetry: !!options?.telemetry,
      });
    },
  };
}

export interface RecordPerformanceOptions extends Pick<ProfileOptions, 'telemetry'> {
  /**
   * How much time to add to the timer
   *
   * @default 0
   */
  readonly durationMs?: number;

  /**
   * How many occurrences of this timer we've missed
   *
   * @default 1
   */
  readonly count?: number;
}

/**
 * Record this performance timer/counter in the global state
 *
 * This *used* to be implemented as `performance.measure()`, but that would eat
 * too much memory in case of a large application. In this function, instead we
 * immediately aggregate into the smaller in-memory state.
 *
 * `durationMs` and `count` are allowed to be `0` in order record only times or
 * counters, if desired.
 */
export function recordPerformanceEntry(key: string, options: RecordPerformanceOptions) {
  bumpPerfCounter(STATE.allCounters, key, options?.durationMs, options?.count);
  if (options?.telemetry) {
    bumpPerfCounter(STATE.telemetryCounters, key, options?.durationMs, options?.count);
  }
}

/**
 * Reset all counters
 */
export function resetCounters() {
  STATE.allCounters = {};
  STATE.telemetryCounters = {};
}

/**
 * Increase an entry in a single set of perf counters
 */
function bumpPerfCounter(counters: PerfCounters, key: string, durationMs?: number, count?: number) {
  const ctr = counters[key];
  if (ctr) {
    ctr.count += count ?? 1;
    ctr.total += durationMs ?? 0;
  } else {
    counters[key] = {
      count: count ?? 1,
      total: durationMs ?? 0,
    };
  }
}

/**
 * Make all functions on this given object (exclusively) profiled
 */
export function profileObj(objName: string, options?: ProfileOptions) {
  return function<A extends object>(obj: A): A {
    for (const [name, { value }] of Object.entries(Object.getOwnPropertyDescriptors(obj))) {
      if (typeof value === 'function') {
        const key = `${objName}.${name}`;
        (obj as any)[name] = profileFn(key, options)(value);
      }
    }
    return obj;
  };
}

type ArbitraryConstructor = { new (...args: any[]): {} };

/**
 * Make all functions on this given class (exclusively) profiled
 */
export function profileClass<A extends ArbitraryConstructor>(cls: A, options?: ProfileOptions): A {
  profileObj(cls.name, options)(cls.prototype);
  return cls;
}

export interface PerfCounter {
  count: number;
  total: number;
}

export type PerfCounters = Record<string, PerfCounter>;

export interface ReadCountersOptions {
  /**
   * Whether to read only counters emitted for telemetry.
   *
   * If set to `false`, all counters are read.
   *
   * @default false
   */
  readonly telemetry?: boolean;

  /**
   * Return at most the top N entries.
   *
   * @default - All entries
   */
  readonly n?: number;
}

/**
 * Read measurements from the performance timeline and return them as counters
 *
 * Returns an ordered map, with the counters with the highest durations first.
 */
export function readPerfCounters(options?: ReadCountersOptions): PerfCounters {
  const counters: PerfCounters = options?.telemetry
    ? STATE.telemetryCounters
    : STATE.allCounters;

  let ret = Object.entries(counters)
    .map(([key, { count, total }]) => [key, { count, total: Math.floor(total) }] as const)
    .sort((a, b) => b[1].total - a[1].total);
  if (options?.n !== undefined) {
    ret = ret.slice(0, options.n);
  }
  return Object.fromEntries(ret);
}

/**
 * Print a table of the performance counters
 *
 * This can be used for local debugging and performance analysis.
 */
export function printPerfCounters(options?: ReadCountersOptions) {
  line('MEASUREMENT', 'TIME(MS)', 'CALLS');
  for (const [key, ctr] of Object.entries(readPerfCounters(options))) {
    line(key, `${ctr.total}`, `${ctr.count}`);
  }

  function line(a: string, b: string, c: string) {
    process.stdout.write(`${a.padEnd(40, ' ')}  ${b.padStart(8, ' ')}  ${c}\n`);
  }
}

/**
 * Initialize profiling for global Node built-ins
 *
 * Be sure to only do this once, even if this library is loaded multiple times.
 * Also don't do it if we are currently testing, as this monkey patching
 * is undoubtedly going to wreak havoc on mocking of globals in test suites.
 */
function initializeBuiltinProfiling() {
  const envYes = ['1', 'yes', 'true'] as Array<string | undefined>;
  if (STATE.builtinsHooked || process.env.NODE_ENV === 'test' || envYes.includes(process.env.CDK_NO_PERF_BUILTINS)) {
    return;
  }
  STATE.builtinsHooked = true;
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  profileObj('fs', { telemetry: true })(require('fs'));
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  profileObj('child_process', { telemetry: true })(require('child_process'));
}
initializeBuiltinProfiling();

import type { PerformanceEntry } from 'perf_hooks';
import { performance } from 'perf_hooks';
import './dispose-polyfill';

/**
 * Global state for this module.
 *
 * This is to safeguard against multiple loads of the same module, we don't
 * want them stepping over each other.
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
}

const STATE: GlobalState = ((global as any)[Symbol.for('@aws-cdk/core:performanceProfilingState')] ??= {
  builtinsHooked: false,
  nesting: 0,
} satisfies GlobalState);

/**
 * The field in `detail` that needs to be set to `true` in order to send this to telemetry.
 */
export const TELEMETRY_FIELD = 'telemetry';

/**
 * The field in `detail` that needs to be set to `true` in order to skip this for counting
 */
export const SKIPCOUNT_FIELD = 'skipCount';

/**
 * The field in `detail` that overrides the amount added to the counter.
 */
export const COUNT_FIELD = 'count';

interface PerformanceMeasureEntry extends PerformanceEntry {
  detail?: Record<string, unknown>;
}

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
 * Emits a measurementto the performance timeline, potentially with `{ telemetry: true }` in
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
          performance.measure(key, {
            start,
            end,
            detail: {
              [TELEMETRY_FIELD]: !!options?.telemetry,
              [SKIPCOUNT_FIELD]: !!options?.skipCount,
            },
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
      performance.measure(key, {
        start,
        detail: {
          [TELEMETRY_FIELD]: !!options?.telemetry,
          [SKIPCOUNT_FIELD]: !!options?.skipCount,
        },
      });
    },
  };
}

/**
 * Add an arbitrary value to a performance counter without recording a duration.
 */
export function recordCounter(key: string, count: number, options?: Pick<ProfileOptions, 'telemetry'>): void {
  const now = performance.now();
  performance.measure(key, {
    start: now,
    end: now,
    detail: {
      [TELEMETRY_FIELD]: !!options?.telemetry,
      [COUNT_FIELD]: count,
    },
  });
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
  // Do all perfs
  const counters: PerfCounters = {};
  for (const entry of performance.getEntriesByType('measure') as PerformanceMeasureEntry[]) {
    if (options?.telemetry && !(entry.detail)?.[TELEMETRY_FIELD]) {
      continue;
    }

    const recordedCount = entry.detail?.[COUNT_FIELD];
    const count = typeof recordedCount === 'number'
      ? recordedCount
      : entry.detail?.[SKIPCOUNT_FIELD] ? 0 : 1;

    const ctr = counters[entry.name];
    if (ctr) {
      ctr.count += count;
      ctr.total += entry.duration;
    } else {
      counters[entry.name] = {
        count,
        total: entry.duration,
      };
    }
  }

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

import { performance } from 'perf_hooks';
import type { Construct, IConstruct } from 'constructs';
import * as fs from 'fs-extra';
import { readPerfCounters, recordPerformanceEntry, resetCounters } from './helpers-internal';
import { PRIVATE_CONTEXT_DEFAULT_STACK_SYNTHESIZER } from './private/private-context';
import type { ICustomSynthesis } from './private/synthesis';
import { addCustomSynthesis } from './private/synthesis';
import type { IPropertyInjector } from './prop-injectors';
import { PropertyInjectors } from './prop-injectors';
import type { IReusableStackSynthesizer } from './stack-synthesizers';
import type { StageSynthesisOptions } from './stage';
import { Stage } from './stage';
import type { IPolicyValidationPluginBeta1 } from './validation/validation';
import * as cxapi from '../../cx-api';
import type * as public_cxapi from '../../cx-api';
import { appOf, APP_TYPE } from './private/core-construct-finders';

/**
 * Can hold a function to globally initialize Apps.
 *
 * Intended for testing, no stability guarantees on this behavior.
 */
const APP_INIT_HOOK_SYMBOL = Symbol.for('@aws-cdk/core.App#initHook');

/**
 * Report performance counters if synthesis time exceeds this
 */
const DEFAULT_SLOW_SYNTH_PER_STACK_THRESHOLD_MS = 10_000;

/**
 * A context key that can be set to control the emission threshold
 */
const SLOW_SYNTH_THRESHOLD_CTX = '@aws-cdk/core.slowSynthThreshold';

/**
 * Initialization props for apps.
 */
export interface AppProps {
  /**
   * Automatically call `synth()` before the program exits.
   *
   * If you set this, you don't have to call `synth()` explicitly. Note that
   * this feature is only available for certain programming languages, and
   * calling `synth()` is still recommended.
   *
   * @default true if running via CDK CLI (`CDK_OUTDIR` is set), `false`
   * otherwise
   */
  readonly autoSynth?: boolean;

  /**
   * The output directory into which to emit synthesized artifacts.
   *
   * You should never need to set this value. By default, the value you pass to
   * the CLI's `--output` flag will be used, and if you change it to a different
   * directory the CLI will fail to pick up the generated Cloud Assembly.
   *
   * This property is intended for internal and testing use.
   *
   * @default - If this value is _not_ set, considers the environment variable `CDK_OUTDIR`.
   *            If `CDK_OUTDIR` is not defined, uses a temp directory.
   */
  readonly outdir?: string;

  /**
   * Include construct creation stack trace in the `aws:cdk:trace` metadata key of all constructs.
   * @default true stack traces are included unless `aws:cdk:disable-stack-trace` is set in the context.
   */
  readonly stackTraces?: boolean;

  /**
   * Include runtime versioning information in the Stacks of this app
   *
   * @deprecated use `analyticsReporting` instead
   * @default Value of 'aws:cdk:version-reporting' context key
   */
  readonly runtimeInfo?: boolean;

  /**
   * Include runtime versioning information in the Stacks of this app
   *
   * @default Value of 'aws:cdk:version-reporting' context key
   */
  readonly analyticsReporting?: boolean;

  /**
   * Produce a performance counter report if supported by the CLI
   *
   * The performance report will be produced if the total synthesis time
   * exceeds 10 seconds/stack, unless this property is used to switch the
   * report off altogether (set to `false`).
   *
   * @default Value of 'aws:cdk:performance-reporting' context key
   */
  readonly performanceReporting?: boolean;

  /**
   * Additional context values for the application.
   *
   * Context set by the CLI or the `context` key in `cdk.json` has precedence.
   *
   * Context can be read from any construct using `node.getContext(key)`.
   *
   * @default - no additional context
   */
  readonly context?: { [key: string]: any };

  /**
   * Additional context values for the application.
   *
   * Context provided here has precedence over context set by:
   *
   * - The CLI via --context
   * - The `context` key in `cdk.json`
   * - The `AppProps.context` property
   *
   * This property is recommended over the `AppProps.context` property since you
   * can make final decision over which context value to take in your app.
   *
   * Context can be read from any construct using `node.getContext(key)`.
   *
   * @example
   * // context from the CLI and from `cdk.json` are stored in the
   * // CDK_CONTEXT env variable
   * const cliContext = JSON.parse(process.env.CDK_CONTEXT!);
   *
   * // determine whether to take the context passed in the CLI or not
   * const determineValue = process.env.PROD ? cliContext.SOMEKEY : 'my-prod-value';
   * new App({
   *   postCliContext: {
   *     SOMEKEY: determineValue,
   *   },
   * });
   *
   * @default - no additional context
   */
  readonly postCliContext?: { [key: string]: any };

  /**
   * Include construct tree metadata as part of the Cloud Assembly.
   *
   * @default true
   */
  readonly treeMetadata?: boolean;

  /**
   * The stack synthesizer to use by default for all Stacks in the App
   *
   * The Stack Synthesizer controls aspects of synthesis and deployment,
   * like how assets are referenced and what IAM roles to use. For more
   * information, see the README of the main CDK package.
   *
   * @default - A `DefaultStackSynthesizer` with default settings
   */
  readonly defaultStackSynthesizer?: IReusableStackSynthesizer;

  /**
   * Validation plugins to run after synthesis
   *
   * @default - no validation plugins
   * @deprecated Use `Validations.of(app).addPlugins()` instead.
   */
  readonly policyValidationBeta1?: IPolicyValidationPluginBeta1[];

  /**
   * A list of IPropertyInjector attached to this App.
   * @default - no PropertyInjectors
   */
  readonly propertyInjectors?: IPropertyInjector[];
}

/**
 * A construct which represents an entire CDK app. This construct is normally
 * the root of the construct tree.
 *
 * You would normally define an `App` instance in your program's entrypoint,
 * then define constructs where the app is used as the parent scope.
 *
 * After all the child constructs are defined within the app, you should call
 * `app.synth()` which will emit a "cloud assembly" from this app into the
 * directory specified by `outdir`. Cloud assemblies includes artifacts such as
 * CloudFormation templates and assets that are needed to deploy this app into
 * the AWS cloud.
 *
 * @see https://docs.aws.amazon.com/cdk/latest/guide/apps.html
 */
export class App extends Stage {
  /**
   * Return the app that is the root of the construct tree, if available.
   */
  public static of(construct: IConstruct): Stage | undefined {
    // This cannot return `App` because we inherit this method from `Stage` and jsii doesn't allow us
    // to change the return type (even though it is static T_T)
    return appOf(construct);
  }

  /**
   * Checks if an object is an instance of the `App` class.
   * @returns `true` if `obj` is an `App`.
   * @param obj The object to evaluate
   */
  public static isApp(obj: any): obj is App {
    return APP_TYPE.isMarked(obj);
  }

  /**
   * Include construct tree metadata as part of the Cloud Assembly.
   *
   * @internal
   */
  public readonly _treeMetadata: boolean;

  private readonly initMark: number;
  private readonly performanceReporting: boolean;

  private alreadySynthed = false;

  /**
   * Initializes a CDK application.
   * @param props initialization properties
   */
  constructor(props: AppProps = {}) {
    super(undefined as any, '', {
      outdir: props.outdir ?? process.env[cxapi.OUTDIR_ENV],
    });

    this.initMark = performance.now();
    if (!PERF_STATE.loadTimeMeasured) {
      // Measure the load time of the application -- up until the construction of the first App
      // object is considered "Load Time" (executing all require()s).
      recordPerformanceEntry('phase:Load', {
        durationMs: this.initMark,
        telemetry: true,
      });
      PERF_STATE.loadTimeMeasured = true;
    }

    if (props.propertyInjectors) {
      const injectors = PropertyInjectors.of(this);
      injectors.add(...props.propertyInjectors);
    }

    if (props.policyValidationBeta1) {
      this._addValidationPlugins(...props.policyValidationBeta1);
    }

    APP_TYPE.mark(this);

    this.loadContext(props.context, props.postCliContext);

    if (props.stackTraces === false) {
      this.node.setContext(cxapi.DISABLE_METADATA_STACK_TRACE, true);
    }

    if (props.defaultStackSynthesizer) {
      this.node.setContext(PRIVATE_CONTEXT_DEFAULT_STACK_SYNTHESIZER, props.defaultStackSynthesizer);
    }

    const analyticsReporting = props.analyticsReporting ?? props.runtimeInfo;

    if (analyticsReporting !== undefined) {
      this.node.setContext(cxapi.ANALYTICS_REPORTING_ENABLED_CONTEXT, analyticsReporting);
    }

    const autoSynth = props.autoSynth ?? cxapi.OUTDIR_ENV in process.env;
    if (autoSynth) {
      // synth() guarantees it will only execute once, so a default of 'true'
      // doesn't bite manual calling of the function.
      process.once('beforeExit', () => this.synth({ errorOnDuplicateSynth: false }));
    }

    this._treeMetadata = props.treeMetadata ?? true;

    this.performanceReporting = props.performanceReporting ?? this.node.tryGetContext(cxapi.PERFORMANCE_REPORTING_ENABLED_CONTEXT) ?? true;

    if ((globalThis as any)[APP_INIT_HOOK_SYMBOL]) {
      (globalThis as any)[APP_INIT_HOOK_SYMBOL](this);
    }
  }

  private loadContext(defaults: { [key: string]: string } = { }, final: { [key: string]: string } = {}) {
    // prime with defaults passed through constructor
    for (const [k, v] of Object.entries(defaults)) {
      this.node.setContext(k, v);
    }

    // reconstructing the context from the two possible sources:
    const context = {
      ...this.readContextFromEnvironment(),
      ...this.readContextFromTempFile(),
    };

    for (const [k, v] of Object.entries(context)) {
      this.node.setContext(k, v);
    }

    // finalContext passed through constructor overwrites
    for (const [k, v] of Object.entries(final)) {
      this.node.setContext(k, v);
    }
  }

  /**
   * Synthesize this App into a cloud assembly.
   *
   * Once an assembly has been synthesized, it cannot be modified. Subsequent
   * calls will return the same assembly.
   */
  public synth(options: StageSynthesisOptions = { }): public_cxapi.CloudAssembly {
    // Synth may be called multiple times, we do not emit performance counters the
    // second time. super.synth() already does caching.
    if (this.alreadySynthed) {
      return super.synth(options);
    }
    this.alreadySynthed = true;

    const startSynthMark = performance.now();

    recordPerformanceEntry('phase:Construction', {
      durationMs: startSynthMark - this.initMark,
      telemetry: true,
    });
    const ret = super.synth(options);
    recordPerformanceEntry('phase:Synthesis', {
      durationMs: performance.now() - startSynthMark,
      telemetry: true,
    });

    const totalAppTimeMs = performance.now() - this.initMark;
    const stackCount = ret.stacksRecursively.length;
    if (this.shouldReportSlowSynth(totalAppTimeMs / stackCount)) {
      emitPerformanceCountersFile();
    }
    resetCounters();

    return ret;
  }

  private readContextFromTempFile() {
    const location = process.env[cxapi.CONTEXT_OVERFLOW_LOCATION_ENV];
    return location ? fs.readJSONSync(location) : {};
  }

  private readContextFromEnvironment() {
    const contextJson = process.env[cxapi.CONTEXT_ENV];
    return contextJson ? JSON.parse(contextJson) : {};
  }

  private shouldReportSlowSynth(perStackTime: number): boolean {
    if (!this.performanceReporting) {
      return false;
    }
    const threshold = parseAsNumber(this.node.tryGetContext(SLOW_SYNTH_THRESHOLD_CTX)) ?? DEFAULT_SLOW_SYNTH_PER_STACK_THRESHOLD_MS;
    return perStackTime >= threshold;
  }
}

/**
 * Add a custom synthesis for the given construct
 *
 * When the construct is being synthesized, this allows it to add additional items
 * into the Cloud Assembly output.
 *
 * This feature is intended for use by official AWS CDK libraries only; 3rd party
 * library authors and CDK users should not use this function. That's why it's not
 * exposed via jsii.
 */
export function attachCustomSynthesis(construct: Construct, synthesis: ICustomSynthesis): void {
  // synthesis.ts where the implementation lives is not exported. So
  // this function is just a re-export of that function.
  addCustomSynthesis(construct, synthesis);
}

/**
 * Emit the performance counters file if requested by the CLI
 *
 * The file will look like this:
 *
 * ```
 * {
 *   "counters": {
 *     "counter1": 3582,
 *     "counter1(cnt)": 3
 *     ...
 *   }
 * }
 * ```
 */
function emitPerformanceCountersFile() {
  const filename = process.env[cxapi.PERF_COUNTERS_FILE_ENV];
  if (!filename) {
    return;
  }

  const counters: Record<string, number> = {};
  for (const [name, ctr] of Object.entries(readPerfCounters({ telemetry: true }))) {
    counters[name] = ctr.total;
    counters[`${name}(cnt)`] = ctr.count;
  }

  fs.writeFileSync(filename, JSON.stringify({ counters }, undefined, 2), 'utf-8');
}

/**
 * Global state, made unique across multiple instances, to read perf counters
 */
interface AppPerfState {
  loadTimeMeasured: boolean;
}

const PERF_STATE: AppPerfState = ((global as any)[Symbol.for('@aws-cdk/core.AppPerfState')] ??= {
  loadTimeMeasured: false,
} satisfies AppPerfState);

/**
 * Froces a string or number to a number, or return undefined
 */
function parseAsNumber(x: unknown) {
  if (typeof x === 'number') {
    return x;
  }
  const r = parseInt(`${x}`, 10);
  return isNaN(r) ? undefined : r;
}

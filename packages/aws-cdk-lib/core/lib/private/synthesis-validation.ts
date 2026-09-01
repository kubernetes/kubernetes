import * as crypto from 'crypto';
import * as fs from 'fs';
import * as path from 'path';
import type * as private_cxapi from '@aws-cdk/cloud-assembly-api';
import type { IConstruct } from 'constructs';
import { AnnotationPlugin, collectAnnotationReport } from './annotation-plugin';
import { collectAcknowledgedRuleIds } from './collect-acknowledged-rule-ids';
import { lit } from './literal-string';
import * as cxapi from '../../../cx-api';
import { _convertCloudAssemblyBuilder } from '../../../cx-api/lib/legacy-moved';
import { App } from '../app';
import { _aspectTreeRevisionReader } from '../aspect';
import { AssumptionError, UnscopedValidationError } from '../errors';
import { FeatureFlags } from '../feature-flags';
import type { Stage } from '../stage';
import type { IPolicyValidationPlugin, PolicyValidationPluginReport } from '../validation';
import { STAGE_TYPE } from './core-construct-finders';
import { profileSpan } from './perf';
import { CloudFormationValidatePlugin } from '../validation/cloudformation-validate-plugin';
import { ConstructTree } from '../validation/private/construct-tree';
import { formatValidationReports, humanFriendlyFilename } from '../validation/private/modern-formatter';
import type { NamedValidationPluginReport, SuppressedViolation } from '../validation/private/report';
import { isPluginFailure, isSuppressibleViolation, mkPluginFailure, PolicyValidationReportFormatter } from '../validation/private/report';

const LEGACY_POLICY_VALIDATION_FILE_PATH = 'policy-validation-report.json';

/**
 * Invoke validation plugins for all stages in an App.
 */
export function validateTemplates(root: IConstruct, outdir: string, assembly: private_cxapi.CloudAssembly) {
  if (!App.isApp(root)) return;

  using _span = profileSpan('validateTemplates', { telemetry: true });
  const assemblies = getAssemblies(root, assembly);
  const stacksByPlugin: Map<IPolicyValidationPlugin, Set<private_cxapi.CloudFormationStackArtifact>> = new Map();

  visitAssemblies(root, 'post', construct => {
    if (STAGE_TYPE.isMarked(construct)) {
      const stageAssembly = assemblies.get(construct.artifactId);
      if (!stageAssembly) throw new AssumptionError(lit`ValidationFailed`, `Validation failed, cannot find cloud assembly for stage ${construct.stageName}`);

      const plugins = pluginsToEvaluate(root, construct);
      for (const plugin of plugins) {
        if (!stacksByPlugin.has(plugin)) {
          stacksByPlugin.set(plugin, new Set());
        }
        for (const stack of stageAssembly.stacksRecursively) {
          stacksByPlugin.get(plugin)!.add(stack);
        }
      }
    }
  });

  if (stacksByPlugin.size === 0) return;

  const reports: NamedValidationPluginReport[] = doInvokeValidationPlugins(outdir, Array.from(stacksByPlugin), root);

  // When the default validation plugin is not explicitly opted-in, downgrade
  // its errors to warnings so synthesis does not fail.
  const validateFlagExplicitlyEnabled = getBooleanContext(root, cxapi.VALIDATE_AGAINST_DEFAULT_RULES, false);
  let warningifiedAnyErrors = false;
  if (!validateFlagExplicitlyEnabled) {
    warningifiedAnyErrors = downgradeCfnValidateErrorsToWarnings(reports);
  }

  const suppressedByReport: Map<number, SuppressedViolation[]> = collectSuppressions(root, reports);

  const formatter = new PolicyValidationReportFormatter(new ConstructTree(root));
  const reportJson = formatter.formatJson(reports, assembly.version, suppressedByReport);

  // Always write validation report to disk
  const reportFile = path.join(assembly.directory, cxapi.VALIDATION_REPORT_FILE);
  fs.writeFileSync(reportFile, JSON.stringify(reportJson, undefined, 2));

  // Write legacy report if requested
  if (getBooleanContext(root, cxapi.VALIDATION_REPORT_JSON_CONTEXT, false)) {
    fs.writeFileSync(
      path.join(assembly.directory, LEGACY_POLICY_VALIDATION_FILE_PATH),
      JSON.stringify(formatter.formatLegacyJson(reports), undefined, 2),
    );
  }

  // We are ready to report on validation results here. This is somewhat complicated because the CDK library
  // can operate in a variety of modes and reporting requirements.
  //
  // ENVIRONMENT
  // -----------
  // 1. The CDK library is running as an app in a subprocess (primary reporting method: stderr, exit code)
  // 2. The CDK library is running in (unit) tests (primary reporting method: exception)
  // 3. The CDK library is running as an app in-memory in the toolkit-lib (primary reporting method: stderr, exception)
  // (4. We don't know where we are running -- could be legacy CLI; we treat this the same as 3).
  //
  // FAILURE CHECKS
  // --------------
  // It can be that the CDK app itself is responsible for checking the result of validation and reporting/failing,
  // or it can be that the CDK app just reports validation failures to a file and the caller reads the report
  // and does its own error checking (signaled by a context variable).
  //
  // WARNINGS
  // --------
  // If we only report warnings but we are supposed to report via an exception, then we have no choice but
  // to report via stderr after all, because there is no exception to throw.

  // Construct library strict mode -- everything is errors. This is intended for construct library testing.
  // Its behavior is not intended to be user-facing, and behavior may change to suit our needs.
  const cdkAppHandlesValidationReporting = getBooleanContext(root, cxapi.FAIL_SYNTH_ON_VALIDATION_ERRORS_CONTEXT, true);
  // Try to determine whether we are running unit tests from common environment variable. Best effort guess.
  const appMode = cdkAppMode(root);

  // See if we should fail synthesis. If any of our reports failed, or we are running in "strict mode"
  // with warnings, we fail.
  const constructLibStrictMode = getBooleanContext(root, cxapi.STRICT_CFN_VALIDATE_ERRORS, false);
  const validationFails = reports.some(r => !r.success) || (constructLibStrictMode && reports.some(r => r.violations.some(v => v.severity === 'warning')));
  const reportText = formatValidationReports(process.cwd(), reportJson.pluginReports);
  const reportPath = humanFriendlyFilename(process.cwd(), reportFile);

  let preamble = '';
  if (warningifiedAnyErrors) {
    if (constructLibStrictMode) {
      preamble = '[Warning] Template validation found issues in your templates. Construct library strict mode considers these errors.';
    } else {
      preamble = '[Warning] Template validation found issues in your templates (reported as warnings).'
        + `\nSet feature flag "${cxapi.VALIDATE_AGAINST_DEFAULT_RULES}" to true to turn these into errors.`;
    }
  }

  // Execution on the settings before starts here
  if (reportText.length === 0) {
    return;
  }

  if (cdkAppHandlesValidationReporting) {
    // The CDK app handles validation output (default true). The CLI can set
    // this to false to take over the responsibility of printing the validation
    // report and setting the exit code.

    // In this case, we report either via an exception or via stderr+process.exitCode;
    // or we print to stderr but do not fail if there are only warnings.
    const throwException = validationFails && (appMode === 'inmemory' || appMode === 'unknown');
    const exceptionContainsReport = appMode === 'inmemory';

    const fullMessage: string[] = [];
    if (preamble) {
      fullMessage.push(preamble);
    }
    fullMessage.push(...reportText);
    fullMessage.push(`A copy of this report can be found in: ${reportPath}`);

    if (throwException && exceptionContainsReport) {
      // The exception contains all information.
      throw new UnscopedValidationError(lit`ValidationFailed`, stripAnsi(fullMessage.join('\n\n')));
    } else if (throwException && !exceptionContainsReport) {
      // We are running in some unclear mode. We get the best fidelity results
      // from printing the report (in color) but also making sure to exit with an
      // exception (= most reliable error reporting).

      // eslint-disable-next-line no-console
      console.error([
        ...preamble ? [preamble] : [],
        ...reportText,
      ].join('\n\n'));
      throw new UnscopedValidationError(lit`ValidationFailed`, `Validation failed. A copy of this report can be found in: ${reportPath}`);
    } else {
      // eslint-disable-next-line no-console
      console.error(fullMessage.join('\n\n'));

      if (validationFails) {
        process.exitCode = 1;
      }
    }
  } else {
    // The CLI handles validation. However, we don't have a good place to put the preamble
    // (there is no field to put it into the report), so we just print it  to stderr here.
    // TODO: Add a field to the report for this preamble, so the CLI can print it in a better place.

    if (preamble) {
      // eslint-disable-next-line no-console
      console.error(preamble);
    }
  }
}

/**
 * Find all the assemblies in the app, including all levels of nested assemblies
 * and return a map where the assemblyId is the key
 */
function getAssemblies(root: App, rootAssembly: private_cxapi.CloudAssembly): Map<string, private_cxapi.CloudAssembly> {
  const assemblies = new Map<string, private_cxapi.CloudAssembly>();
  assemblies.set(root.artifactId, rootAssembly);
  visitAssemblies(root, 'pre', construct => {
    const stage = construct as Stage;
    if (stage.parentStage && assemblies.has(stage.parentStage.artifactId)) {
      assemblies.set(
        stage.artifactId,
        assemblies.get(stage.parentStage.artifactId)!.getNestedAssembly(stage.artifactId),
      );
    }
  });
  return assemblies;
}

/**
 * Return the list of plugins to invoke for the given stage
 */
function pluginsToEvaluate(root: IConstruct, stage: Stage): IPolicyValidationPlugin[] {
  const ret: IPolicyValidationPlugin[] = [];

  // 1. User-registered plugins
  ret.push(...stage._validationPlugins);

  // 2. Default validation engine (runs unless the user registered one explicitly,
  // or disabled validation via the CDK_VALIDATION environment variable)
  if (defaultValidationEnabled() && !hasUserRegisteredCloudFormationValidatePlugin(root)) {
    ret.push(CloudFormationValidatePlugin._singletonInstance());
  }

  // 3. Construct annotations (as a plugin, only if there are annotations to report and only on the root)
  if (stage === root && FeatureFlags.of(root).isEnabled(cxapi.ANNOTATIONS_IN_VALIDATION_REPORT)) {
    const annotationsPlugin = collectAnnotationReport(stage);
    if (annotationsPlugin) {
      ret.push(annotationsPlugin);
    }
  }

  return ret;
}

/**
 * Whether the default (auto-registered) CloudFormation validation engine should run.
 *
 * Users can disable template validation by setting the `CDK_VALIDATION` environment
 * variable to 'false'. This is the same environment variable that backs the CLI's
 * `--no-validation` option, so the two validation layers are controlled consistently.
 *
 * Explicitly user-registered validation plugins are not affected by this setting.
 */
function defaultValidationEnabled(): boolean {
  return process.env.CDK_VALIDATION !== 'false';
}

function downgradeCfnValidateErrorsToWarnings(reports: NamedValidationPluginReport[]) {
  let warningifiedAnyErrors = false;
  for (const report of reports) {
    if (report.pluginName !== CloudFormationValidatePlugin.PLUGIN_NAME) {
      continue;
    }
    for (const v of report.violations) {
      if (v.severity === 'error' || v.severity === 'fatal') {
        mutable(v).severity = 'warning';
        warningifiedAnyErrors = true;
      }
    }
    mutable(report).success = true;
  }
  return warningifiedAnyErrors;
}

/**
 * Filter out suppressed violations. Collect all acknowledged rule IDs
 * from construct metadata across the tree, then remove matching violations
 * from reports. Fatal violations cannot be suppressed.
 *
 * Rule matching: violations are matched as <pluginName>::<ruleName> with
 * spaces replaced by dashes. Users suppress with:
 *   Validations.of(x).acknowledge({ id: '<plugin-name>::<rule-id>' })
 */
function collectSuppressions(root: App, reports: NamedValidationPluginReport[]) {
  const suppressedByReport: Map<number, SuppressedViolation[]> = new Map();
  const acknowledgedRules = collectAcknowledgedRuleIds(root);

  if (acknowledgedRules.size > 0) {
    for (let i = 0; i < reports.length; i++) {
      const pluginName = reports[i].pluginName;
      const active: typeof reports[0]['violations'] = [];
      const suppressed: SuppressedViolation[] = [];
      for (const v of reports[i].violations) {
        if (!isSuppressibleViolation(v)) {
          active.push(v);
          continue;
        }

        const ackIds: string[] = [];
        if (v.ruleName.includes('::')) {
          ackIds.push(v.ruleName);

          // Annotations are special; we renamed the suppression namespace at one point
          // from "Construct-Annotations" to just "Annotation", and we also want
          // to support the naked-rule-name form for backwards compatibility.
          if (pluginName === AnnotationPlugin.NAME) {
            const unnamespacedPart = v.ruleName.split('::').slice(1).join('::');
            ackIds.push(`${pluginName}::${unnamespacedPart}`);
          }
        } else {
          ackIds.push(`${pluginName}::${v.ruleName}`);
        }

        const ack = firstThat(ackIds.map(hyphenify), id => acknowledgedRules.get(id));

        if (ack) {
          suppressed.push({
            ...v,
            acknowledgedId: ack.key,
            reason: ack.value.reason,
            acknowledgedAt: ack.value.constructPath,
            acknowledgedStackTrace: ack.value.stackTrace,
          });
        } else {
          active.push(v);
        }
      }
      if (suppressed.length > 0) {
        suppressedByReport.set(i, suppressed);
        reports[i] = {
          ...reports[i],
          violations: active,
          success: active.every(v => v.severity !== 'error' && v.severity !== 'fatal'),
        };
      }
    }
  }
  return suppressedByReport;
}

/**
 * Invoke all validation plugins, make sure they don't accidentally modify any files in the output directory (so they are strictly readonly).
 */
function doInvokeValidationPlugins(
  outdir: string,
  plugins: Array<[IPolicyValidationPlugin, Set<private_cxapi.CloudFormationStackArtifact>]>,
  root: App,
) {
  const untrustedPlugins = new Set(Array.from(plugins.values())
    .map(p => p[0])
    .filter(p => !isTrustedPlugin(p)));

  const preExistingFileHashes = untrustedPlugins.size > 0 ? snapshotFileHashes(outdir) : undefined;

  const ret = plugins.flatMap(([plugin, stacks]) => invokeSinglePlugin(plugin, Array.from(stacks)));

  if (preExistingFileHashes) {
    if (hasModifiedPreExistingFiles(preExistingFileHashes)) {
      const pluginNames = Array.from(untrustedPlugins).map(p => p.name);
      throw new AssumptionError(lit`IllegalPluginOperation`, `One of the validation plugins (${pluginNames.join(', ')}) modified the cloud assembly`);
    }
  }

  return ret;

  function invokeSinglePlugin(
    plugin: IPolicyValidationPlugin,
    stackArtifacts: private_cxapi.CloudFormationStackArtifact[],
  ): NamedValidationPluginReport[] {
    const stacksByEnv = groupStacksByEnvironment(stackArtifacts);

    const reports = stacksByEnv.map(({ accountId, region, stacks }) => {
      try {
        const report = makeTemplatePathsRelative(plugin.validate({
          // path.resolve() because templateFullPath might not be as full as you'd expect
          templatePaths: stacks.map(s => s.templateFullPath),
          stackTemplates: stacks.map(s => ({ stackConstructPath: s.hierarchicalId, templatePath: s.templateFullPath })),
          appConstruct: root,
          accountId: accountId !== cxapi.UNKNOWN_ACCOUNT ? accountId : undefined,
          region: region !== cxapi.UNKNOWN_REGION ? region : undefined,
        }));

        return { ...report, pluginName: plugin.name, pluginVersion: plugin.version } satisfies NamedValidationPluginReport;
      } catch (e: any) {
        if (e instanceof AssumptionError && e.name === 'IllegalPluginOperation') {
          throw e;
        }
        return mkPluginFailure(plugin, e);
      }
    });

    return mergeReports(reports);
  }

  /**
   * We gave the validation plugins absolute template paths.
   *
   * The most logical thing to do would be for them to stick the absolute paths in the report, but
   * we want root-assembly-relative paths, otherwise assemblies are not self-contained.
   *
   * Ideally the API should have been designed to pass relative paths, but we
   * can't change that anymore. So we have to fix up the report after the fact.
   */
  function makeTemplatePathsRelative(report: PolicyValidationPluginReport) {
    for (const v of report.violations) {
      for (const r of v.violatingResources) {
        if (r.templatePath) {
          mutable(r).templatePath = path.relative(outdir, path.resolve(r.templatePath));
        }
      }
    }

    return report;
  }
}

function isTrustedPlugin(x: IPolicyValidationPlugin) {
  return x instanceof CloudFormationValidatePlugin;
}

interface StacksByEnvironment {
  readonly accountId: string | undefined;
  readonly region: string | undefined;
  readonly stacks: private_cxapi.CloudFormationStackArtifact[];
}

function groupStacksByEnvironment(stacks: private_cxapi.CloudFormationStackArtifact[]): StacksByEnvironment[] {
  const ret = new Map<string, StacksByEnvironment>();

  for (const stack of stacks) {
    const key = `${stack.environment.account || ''}::${stack.environment.region || ''}`;
    if (!ret.has(key)) {
      ret.set(key, { accountId: stack.environment.account, region: stack.environment.region, stacks: [] });
    }
    ret.get(key)!.stacks.push(stack);
  }

  return Array.from(ret.values());
}

/**
 * Visit the given construct tree in either pre or post order, only looking at Assemblies
 */
export function visitAssemblies(root: IConstruct, order: 'pre' | 'post', cb: (x: IConstruct) => void) {
  if (order === 'pre') {
    cb(root);
  }

  for (const child of root.node.children) {
    if (!STAGE_TYPE.isMarked(child)) { continue; }
    visitAssemblies(child, order, cb);
  }

  if (order === 'post') {
    cb(root);
  }
}

function getBooleanContext(root: IConstruct, key: string, defaultValue: boolean): boolean {
  const raw = root.node.tryGetContext(key);
  if (raw === undefined) return defaultValue;
  return raw !== false && raw !== 'false';
}

function snapshotFileHashes(dir: string): Map<string, string> {
  const hashes = new Map<string, string>();
  for (const filePath of collectFilePaths(dir)) {
    hashes.set(filePath, hashFile(filePath));
  }
  return hashes;
}

function hasModifiedPreExistingFiles(snapshot: Map<string, string>): boolean {
  for (const [filePath, originalHash] of snapshot) {
    if (!fileOrSymlinkExists(filePath) || hashFile(filePath) !== originalHash) {
      return true;
    }
  }
  return false;
}

function collectFilePaths(dir: string): string[] {
  const results: string[] = [];
  function walk(current: string) {
    for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
      const full = path.join(current, entry.name);
      if (entry.isDirectory()) {
        // `isDirectory()` is false for a symlink-to-directory, so we never recurse
        // through symlinks (avoids following links out of the cloud assembly / cycles).
        walk(full);
      } else if (entry.isFile() || entry.isSymbolicLink()) {
        // Collect regular files and symlinks (including symlink-to-directory). The
        // symlink is hashed by its target path in hashFile(), never dereferenced.
        results.push(full);
      }
    }
  }
  walk(dir);
  return results;
}

function hashFile(filePath: string): string {
  // Dereferencing a symlink-to-directory with readFileSync() throws EISDIR, so hash
  // the link target string instead of the (non-existent) file contents.
  const content = fs.lstatSync(filePath).isSymbolicLink()
    ? Buffer.from(fs.readlinkSync(filePath))
    : fs.readFileSync(filePath);
  return crypto.createHash('sha256').update(content).digest('hex');
}

/**
 * Like fs.existsSync(), but does not follow symlinks: a symlink (even a dangling
 * one) counts as existing. This prevents preserved symlinks in the cloud assembly
 * from being misreported as "deleted by a validation plugin".
 */
function fileOrSymlinkExists(p: string): boolean {
  try {
    fs.lstatSync(p);
    return true;
  } catch {
    return false;
  }
}

/**
 * Validate that CloudFormationValidatePlugin is registered correctly.
 * Returns true if the user registered exactly one valid instance, false if none.
 * Throws if the plugin is registered on a nested Stage or more than once.
 */
function hasUserRegisteredCloudFormationValidatePlugin(root: IConstruct): boolean {
  let count = 0;
  visitAssemblies(root, 'post', construct => {
    if (!STAGE_TYPE.isMarked(construct)) return;
    for (const plugin of construct._validationPlugins) {
      if (!(plugin instanceof CloudFormationValidatePlugin)) continue;
      if (construct !== root) {
        throw new UnscopedValidationError(lit`CloudFormationValidatePluginNotAtApp`, 'CloudFormationValidatePlugin can only be registered at the App level, not on a Stage');
      }
      count++;
    }
  });
  if (count > 1) {
    throw new UnscopedValidationError(lit`DuplicateCloudFormationValidatePlugin`, 'only one instance of CloudFormationValidatePlugin can be registered');
  }
  return count === 1;
}

function mutable<A extends object>(obj: A): { -readonly [P in keyof A]: A[P] } {
  return obj as any;
}

/**
 * Merge the reports from multiple invocations of the same plugin into a single report.
 *
 * All non-errors are combined into a single report, and errors are combined by error message.
 */
function mergeReports(reports: NamedValidationPluginReport[]): NamedValidationPluginReport[] {
  const nonErrors = reports.filter(r => isPluginFailure(r) === undefined);
  const errors = reports.filter(r => isPluginFailure(r) !== undefined);

  const ret: NamedValidationPluginReport[] = [];
  if (nonErrors.length > 0) {
    const merged: NamedValidationPluginReport = nonErrors[0];
    for (const candidate of nonErrors.slice(1)) {
      merged.violations.push(...candidate.violations);
      mutable(merged).metadata = { ...merged.metadata, ...candidate.metadata };
      mutable(merged).success = merged.success && candidate.success;
    }
    ret.push(merged);
  }

  if (errors.length > 0) {
    const errorMap = new Map<string, NamedValidationPluginReport>();
    for (const candidate of errors) {
      const errorMessage = isPluginFailure(candidate);
      if (!errorMessage) continue;
      if (!errorMap.has(errorMessage)) {
        errorMap.set(errorMessage, candidate);
      }
    }
    ret.push(...errorMap.values());
  }

  return ret;
}

function cdkAppMode(root: IConstruct): 'process' | 'inmemory' | 'unknown' {
  const contextMode = root.node.tryGetContext(cxapi.CDK_APP_MODE_CONTEXT);
  if (contextMode === 'process' || contextMode === 'inmemory') {
    return contextMode;
  }

  const envMode = process.env[cxapi.CDK_APP_MODE_ENV];
  if (envMode === 'process' || envMode === 'inmemory') {
    return envMode;
  }

  // Try to guess whether we are running in a unit testing environment. In that case,
  // default to inmemory mode.
  if (process.env.NODE_ENV === 'test' || process.env.PYTEST_VERSION || process.env.PYTEST_CURRENT_TEST) {
    return 'inmemory';
  }

  // Unknown mode, either a legacy CLI or running via toolkit-lib.
  return 'unknown';
}

function stripAnsi(x: string) {
  const pattern = [
    '[\\u001B\\u009B][[\\]()#;?]*(?:(?:(?:(?:;[-a-zA-Z\\d\\/#&.:=?%@~_]+)*|[a-zA-Z\\d]+(?:;[-a-zA-Z\\d\\/#&.:=?%@~_]*)*)?\\u0007)',
    '(?:(?:\\d{1,4}(?:;\\d{0,4})*)?[\\dA-PR-TZcf-ntqry=><~]))',
  ].join('|');

  const re = new RegExp(pattern, 'g');
  return x.replaceAll(re, '');
}

function firstThat<A, B>(xs: A[], predicate: (x: A) => B | undefined): { key: A; value: B } | undefined {
  for (const x of xs) {
    const value = predicate(x);
    if (value !== undefined) {
      return { key: x, value };
    }
  }
  return undefined;
}

function hyphenify(x: string) {
  return x.replace(/ /g, '-');
}

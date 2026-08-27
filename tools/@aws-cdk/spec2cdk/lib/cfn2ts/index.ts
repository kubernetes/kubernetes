import { generate as generateModules, loadPatchedSpec } from '../generate';
import type { ModuleMap, ModuleMapScope } from '../module-topology';
import { readModuleMap } from '../module-topology';
import * as naming from '../naming';
import { jsii } from '../util';
import { CANNED_METRICS_SUFFIX } from './submodules';
import { getAllScopes } from '../util/db';

/**
 * Configuration options for the generateAll function
 */
export interface GenerateAllOptions {
  /**
   * Path of the file containing the map of module names to their CFN Scopes
   */
  readonly scopeMapPath: string;

  /**
   * List of service names to be skipped it will be in format AWS::Service like AWS::S3
   */
  readonly skippedServices?: string[];

  /**
   * Automatically generate service suffixes
   *
   * @default false
   */
  readonly autoGenerateSuffixes?: boolean;

  /**
   * Whether the modules being generated are considered stable
   *
   * @default true
   */
  readonly isStable?: boolean;

  /**
   * If specified, only generate code for the service that corresponds to this module name.
   * Takes precedence over `skippedServices`.
   *
   * @default - generate all modules
   */
  readonly singleModule?: string;
}

/**
 * Generates L1s for all submodules of a monomodule. Modules to generate are
 * chosen based on the contents of the `scopeMapPath` file.
 *
 * This is intended for use in generated L1s in aws-cdk-lib, and gets called whenever
 * `yarn gen` is run.
 *
 * Code-generates all L1s, and writes the necessary index files.
 *
 * @param outPath The root directory to generate L1s in
 * @param alphaRootDir The root directory to generate grant modules for grants configured in alpha packages
 * @returns       A ModuleMap containing the ModuleDefinition and CFN scopes for each generated module.
 */
export async function generateAll(
  outPath: string,
  { scopeMapPath, skippedServices, isStable, singleModule }: GenerateAllOptions,
): Promise<ModuleMap> {
  const db = await loadPatchedSpec();
  const allScopes = getAllScopes(db, 'cloudFormationNamespace');

  let scopes: ModuleMapScope[];
  if (singleModule != null) {
    scopes = allScopes.filter((scope) => {
      const moduleName = naming.modulePartsFromNamespace(scope.namespace).moduleName;
      return moduleName === singleModule;
    });
  } else {
    scopes = skippedServices ? allScopes.filter((scope) => !skippedServices.includes(scope.namespace)) : allScopes;
  }

  const moduleMap = singleModule == null ? readModuleMap(scopeMapPath): {};

  // Make sure all scopes have their own dedicated package/namespace.
  // Adds new submodules for new namespaces.
  for (const scope of scopes) {
    const moduleName = naming.modulePartsFromNamespace(scope.namespace).moduleName;
    const currentScopes = moduleMap[moduleName]?.scopes ?? [];
    // remove dupes, give precedence to data from module map
    const newScopes = Array.from(
      new Map([scope, ...currentScopes].map(s => [s.namespace, s])).values(),
    );

    // Add new modules to module map and return to caller
    moduleMap[moduleName] = {
      name: moduleName,
      scopes: newScopes,
      definition: moduleMap[moduleName]?.definition ?? jsii.namespaceToModuleDefinition(scope.namespace),
      targets: moduleMap[moduleName]?.targets ?? undefined,
      resources: {},
      files: [],
    };
  }

  const moduleGenerationRequests = Object.fromEntries(
    Object.entries(moduleMap).map(([moduleName, { scopes: services }]) => [
      moduleName,
      { services },
    ]),
  );

  const generated = await generateModules(
    moduleGenerationRequests,
    {
      outputPath: outPath,
      clearOutput: false,
      isStable,
      builderProps: {
        inCdkLib: true,
        filePatterns: {
          resources: '%moduleName%/lib/%serviceShortName%.generated.ts',
          augmentations: '%moduleName%/lib/%serviceShortName%-augmentations.generated.ts',
          cannedMetrics: `%moduleName%/lib/%serviceShortName%${CANNED_METRICS_SUFFIX}`,
          grants: `${isStable ? '%moduleName%/' : ''}lib/%serviceShortName%-grants.generated.ts`,
        },
      },
    },
  );

  return Object.fromEntries(Object.entries(generated.modules).map(([moduleName, moduleInfo]) => [
    moduleName,
    {
      files: moduleInfo.outputFiles,
      name: moduleName,
      resources: moduleInfo.resources,
      scopes: moduleMap[moduleName]?.scopes ?? [],
      definition: moduleMap[moduleName]?.definition,
      targets: moduleMap[moduleName]?.targets,
    },
  ]));
}


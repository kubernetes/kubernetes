import type { GenerateModuleMap, GenerateOptions as Spec2CdkOptions } from '../generate';
import { generate, loadPatchedSpec } from '../generate';
import { LogsDeliveryBuilder } from './builder';
import { type GeneratorResult, loadModuleMap, type ModuleMap } from '../module-topology';
import type { PackageBaseNames } from '../util/jsii';

export interface LogsGenerateOptions extends Pick<Spec2CdkOptions<typeof LogsDeliveryBuilder>, 'outputPath' | 'clearOutput' | 'debug'> {
  readonly packageBases: PackageBaseNames;
}

/**
 * Services for which log delivery mixins are not generated.
 *
 * Keyed by service (module) name, e.g. `aws-logs`. Resources in these services may
 * still declare `vendedLogs` in the service spec — listing the service here suppresses
 * the generated mixin file, its `mixins.ts` barrel, and its `index.ts` export.
 */
const EXCLUDED_SERVICES = new Set([
  'aws-apigateway',
  'aws-aps',
  'aws-b2bi',
  'aws-backupgateway',
  'aws-cleanrooms',
  'aws-cognito',
  'aws-connect',
  'aws-iotfleetwise',
  'aws-ivschat',
  'aws-kafkaconnect',
  'aws-logs',
  'aws-m2',
  'aws-msk',
  'aws-osis',
  'aws-pipes',
  'aws-route53globalresolver',
  'aws-route53profiles',
  'aws-rum',
  'aws-sagemaker',
  'aws-stepfunctions',
  'aws-transfer',
  'aws-vpclattice',
  'aws-wafv2',
]);

export async function generateAll(options: LogsGenerateOptions): Promise<GeneratorResult> {
  const db = await loadPatchedSpec();
  const services = await db.all('service');
  const moduleMap: ModuleMap = loadModuleMap({
    packageBases: options.packageBases,
    respectOverrides: false,
  });
  const moduleRequests: GenerateModuleMap = {};

  for (const service of services) {
    if (moduleMap[service.name] && !EXCLUDED_SERVICES.has(service.name)) {
      moduleRequests[service.name] = {
        services: [{ namespace: service.cloudFormationNamespace }],
      };
    }
  }

  const generated = await generate<typeof LogsDeliveryBuilder>(moduleRequests, {
    ...options,
    db,
    astBuilder: LogsDeliveryBuilder,
  });

  return {
    moduleMap: Object.fromEntries(Object.entries(generated.modules).map(([moduleName, moduleInfo]) => [
      moduleName,
      {
        files: moduleInfo.outputFiles,
        name: moduleName,
        resources: moduleInfo.resources,
        scopes: moduleMap[moduleName]?.scopes ?? [],
        definition: moduleMap[moduleName]?.definition,
        targets: moduleMap[moduleName]?.targets,
      },
    ])),
    contributions: [{
      barrelFile: 'mixins.ts',
      exportLines: ["export * from './logs-delivery-mixins.generated';"],
      jsiircNamespace: 'mixins',
    }],
  };
}

import type { IConstruct } from 'constructs';
import type { Architecture } from './architecture';
import type { IFunction } from './function-base';
import { ValidationError } from '../../core/lib/errors';
import { lit } from '../../core/lib/private/literal-string';
import { Stack } from '../../core/lib/stack';
import { Token } from '../../core/lib/token';
import { RegionInfo } from '../../region-info';
import { FactName } from '../../region-info/lib/fact';

/**
 * The type of ADOT Lambda layer
 */
enum AdotLambdaLayerType {
  /**
   * The Lambda layer for ADOT Java instrumentation library. This layer only auto-instruments AWS
   * SDK libraries.
   */
  JAVA_SDK = 'JAVA_SDK',

  /**
   * The Lambda layer for ADOT Java Auto-Instrumentation Agent. This layer automatically instruments
   * a large number of libraries and frameworks out of the box and has notable impact on startup
   * performance.
   */
  JAVA_AUTO_INSTRUMENTATION = 'JAVA_AUTO_INSTRUMENTATION',

  /**
   * The Lambda layer for ADOT Collector, OpenTelemetry for JavaScript and supported libraries.
   */
  JAVASCRIPT_SDK = 'JAVASCRIPT_SDK',

  /**
   * The Lambda layer for ADOT Collector, OpenTelemetry for Python and supported libraries.
   */
  PYTHON_SDK = 'PYTHON_SDK',

  /**
   * The generic Lambda layer that contains only ADOT Collector, used for manual instrumentation
   * use cases (such as Go or DotNet runtimes).
   */
  GENERIC = 'GENERIC',
}

/**
 * Config returned from `AdotLambdaLayerVersion._bind`
 */
interface AdotLambdaLayerBindConfig {
  /**
   * ARN of the ADOT Lambda layer version
   */
  readonly arn: string;
}

/**
 * Return the ARN of an ADOT Lambda layer given its properties. If the region name is unknown
 * at synthesis time, it will generate a map in the CloudFormation template and perform the look
 * up at deployment time.
 *
 * @param scope the parent Construct that will use the imported layer.
 * @param type the type of the ADOT Lambda layer
 * @param version The version of the ADOT Lambda layer
 * @param architecture the architecture of the Lambda layer ('amd64' or 'arm64')
 */
function getLayerArn(scope: IConstruct, type: string, version: string, architecture: string): string {
  const scopeStack = Stack.of(scope);
  const region = scopeStack.region;

  // Region is defined, look up the arn, or throw an error if the version isn't supported by a region
  if (region !== undefined && !Token.isUnresolved(region)) {
    const arn = RegionInfo.get(region).adotLambdaLayerArn(type, version, architecture);
    if (arn === undefined) {
      throw new ValidationError(
        lit`AdotLayerArnNotFound`,
        `Could not find the ARN information for the ADOT Lambda Layer of type ${type} and version ${version} in ${region}`, scope,
      );
    }
    return arn;
  }

  // Otherwise, need to add a mapping to be looked up at deployment time
  return scopeStack.regionalFact(FactName.adotLambdaLayer(type, version, architecture));
}

/**
 * Properties for an ADOT instrumentation in Lambda.
 *
 * **Note:** This uses the legacy ADOT Lambda layers that bundle an embedded collector.
 * For the recommended optimized ADOT Lambda layers (the `AWSOpenTelemetryDistro*` family),
 * add the layer via `Function.addLayers()` and set the `AWS_LAMBDA_EXEC_WRAPPER`
 * environment variable to `/opt/otel-instrument` instead.
 *
 * @see https://aws-otel.github.io/docs/getting-started/lambda
 */
export interface AdotInstrumentationConfig {
  /**
   * The ADOT Lambda layer.
   */
  readonly layerVersion: AdotLayerVersion;

  /**
   * The startup script to run, see ADOT documentation to pick the right script for your use case: https://aws-otel.github.io/docs/getting-started/lambda
   */
  readonly execWrapper: AdotLambdaExecWrapper;
}

/**
 * An ADOT Lambda layer version that's specific to a lambda layer type and an architecture.
 *
 * **Note:** These resolve to the legacy ADOT Lambda layers with an embedded collector.
 * For the recommended optimized layers, use `LayerVersion.fromLayerVersionArn()` with
 * an `AWSOpenTelemetryDistro*` layer ARN instead.
 */
export abstract class AdotLayerVersion {
  /**
   * The ADOT Lambda layer for Java SDK
   *
   * @param version The version of the Lambda layer to use
   */
  public static fromJavaSdkLayerVersion(version: AdotLambdaLayerJavaSdkVersion): AdotLayerVersion {
    return AdotLayerVersion.fromAdotVersion(version);
  }

  /**
   * The ADOT Lambda layer for Java auto instrumentation
   *
   * @param version The version of the Lambda layer to use
   */
  public static fromJavaAutoInstrumentationLayerVersion(
    version: AdotLambdaLayerJavaAutoInstrumentationVersion,
  ): AdotLayerVersion {
    return AdotLayerVersion.fromAdotVersion(version);
  }

  /**
   * The ADOT Lambda layer for JavaScript SDK
   *
   * @param version The version of the Lambda layer to use
   */
  public static fromJavaScriptSdkLayerVersion(version: AdotLambdaLayerJavaScriptSdkVersion): AdotLayerVersion {
    return AdotLayerVersion.fromAdotVersion(version);
  }

  /**
   * The ADOT Lambda layer for Python SDK
   *
   * @param version The version of the Lambda layer to use
   */
  public static fromPythonSdkLayerVersion(version: AdotLambdaLayerPythonSdkVersion): AdotLayerVersion {
    return AdotLayerVersion.fromAdotVersion(version);
  }

  /**
   * The ADOT Lambda layer for generic use cases
   *
   * @param version The version of the Lambda layer to use
   */
  public static fromGenericLayerVersion(version: AdotLambdaLayerGenericVersion): AdotLayerVersion {
    return AdotLayerVersion.fromAdotVersion(version);
  }

  private static fromAdotVersion(adotVersion: AdotLambdaLayerVersion): AdotLayerVersion {
    return new (class extends AdotLayerVersion {
      _bind(_function: IFunction): AdotLambdaLayerBindConfig {
        return {
          arn: adotVersion.layerArn(_function.stack, _function.architecture),
        };
      }
    })();
  }

  /**
   * Produce a `AdotLambdaLayerBindConfig` instance from this `AdotLayerVersion` instance.
   *
   * @internal
   */
  public abstract _bind(_function: IFunction): AdotLambdaLayerBindConfig;
}

/**
 * The wrapper script to be used for the Lambda function in order to enable auto instrumentation
 * with ADOT.
 *
 * **Note:** These wrappers are for the legacy ADOT Lambda layers with an embedded collector.
 * The recommended optimized ADOT Lambda layers use `/opt/otel-instrument` set via the
 * `AWS_LAMBDA_EXEC_WRAPPER` environment variable directly.
 */
export enum AdotLambdaExecWrapper {
  /**
   * Wrapping regular Lambda handlers.
   */
  REGULAR_HANDLER = '/opt/otel-handler',

  /**
   * Wrapping regular handlers (implementing RequestHandler) proxied through API Gateway, enabling
   * HTTP context propagation.
   */
  PROXY_HANDLER = '/opt/otel-proxy-handler',

  /**
   * Wrapping streaming handlers (implementing RequestStreamHandler), enabling HTTP context
   * propagation for HTTP requests.
   */
  STREAM_HANDLER = '/opt/otel-stream-handler',

  /**
   * Wrapping python lambda handlers see https://aws-otel.github.io/docs/getting-started/lambda/lambda-python
   */
  INSTRUMENT_HANDLER = '/opt/otel-instrument',

  /**
   * Wrapping SQS-triggered function handlers (implementing RequestHandler)
   */
  SQS_HANDLER = '/opt/otel-sqs-handler',
}

abstract class AdotLambdaLayerVersion {
  protected constructor(protected readonly type: AdotLambdaLayerType, protected readonly version: string) {}

  /**
   * The ARN of the Lambda layer
   *
   * @param scope The binding scope. Usually this is the stack where the Lambda layer is bound to
   * @param architecture The architecture of the Lambda layer (either X86_64 or ARM_64)
   */
  public layerArn(scope: IConstruct, architecture: Architecture): string {
    return getLayerArn(scope, this.type, this.version, architecture.name);
  }
}

/**
 * The collection of versions of the ADOT Lambda Layer for Java SDK.
 *
 * **Note:** These are legacy ADOT Lambda layers with an embedded collector.
 */
export class AdotLambdaLayerJavaSdkVersion extends AdotLambdaLayerVersion {
  /**
   * Version 1.32.0
   */
  public static readonly V1_32_0_1 = new AdotLambdaLayerJavaSdkVersion('1.32.0-1');

  /**
   * Version 1.32.0
   */
  public static readonly V1_32_0 = new AdotLambdaLayerJavaSdkVersion('1.32.0');

  /**
   * Version 1.31.0
   */
  public static readonly V1_31_0 = new AdotLambdaLayerJavaSdkVersion('1.31.0');

  /**
   * Version 1.30.0
   */
  public static readonly V1_30_0 = new AdotLambdaLayerJavaSdkVersion('1.30.0');

  /**
   * Version 1.28.1
   */
  public static readonly V1_28_1 = new AdotLambdaLayerJavaSdkVersion('1.28.1');

  /**
   * Version 1.19.0
   */
  public static readonly V1_19_0 = new AdotLambdaLayerJavaSdkVersion('1.19.0');

  /**
   * The latest layer version available in this CDK version. New versions could
   * introduce incompatible changes. Make sure to test them before deploying to production.
   */
  public static readonly LATEST = this.V1_32_0_1;

  private constructor(protected readonly layerVersion: string) {
    super(AdotLambdaLayerType.JAVA_SDK, layerVersion);
  }
}

/**
 * The collection of versions of the ADOT Lambda Layer for Java auto-instrumentation.
 *
 * **Note:** These are legacy ADOT Lambda layers with an embedded collector.
 */
export class AdotLambdaLayerJavaAutoInstrumentationVersion extends AdotLambdaLayerVersion {
  /**
   * Version 1.32.0
   */
  public static readonly V1_32_0_1 = new AdotLambdaLayerJavaAutoInstrumentationVersion('1.32.0-1');

  /**
   * Version 1.32.0
   */
  public static readonly V1_32_0 = new AdotLambdaLayerJavaAutoInstrumentationVersion('1.32.0');

  /**
   * Version 1.31.0
   */
  public static readonly V1_31_0 = new AdotLambdaLayerJavaAutoInstrumentationVersion('1.31.0');

  /**
   * Version 1.30.0
   */
  public static readonly V1_30_0 = new AdotLambdaLayerJavaAutoInstrumentationVersion('1.30.0');

  /**
   * Version 1.28.1
   */
  public static readonly V1_28_1 = new AdotLambdaLayerJavaAutoInstrumentationVersion('1.28.1');

  /**
   * Version 1.19.2
   */
  public static readonly V1_19_2 = new AdotLambdaLayerJavaAutoInstrumentationVersion('1.19.2');

  /**
   * The latest layer version available in this CDK version. New versions could
   * introduce incompatible changes. Make sure to test them before deploying to production.
   */
  public static readonly LATEST = this.V1_32_0_1;

  private constructor(protected readonly layerVersion: string) {
    super(AdotLambdaLayerType.JAVA_AUTO_INSTRUMENTATION, layerVersion);
  }
}

/**
 * The collection of versions of the ADOT Lambda Layer for Python SDK.
 *
 * **Note:** These are legacy ADOT Lambda layers with an embedded collector.
 */
export class AdotLambdaLayerPythonSdkVersion extends AdotLambdaLayerVersion {
  /**
   * Version 1.29.0
   */
  public static readonly V1_29_0 = new AdotLambdaLayerPythonSdkVersion('1.29.0');

  /**
   * Version 1.25.0
   */
  public static readonly V1_25_0 = new AdotLambdaLayerPythonSdkVersion('1.25.0');

  /**
   * Version 1.24.0
   */
  public static readonly V1_24_0 = new AdotLambdaLayerPythonSdkVersion('1.24.0');

  /**
   * Version 1.21.0
   */
  public static readonly V1_21_0 = new AdotLambdaLayerPythonSdkVersion('1.21.0');

  /**
   * Version 1.20.0
   */
  public static readonly V1_20_0_1 = new AdotLambdaLayerPythonSdkVersion('1.20.0-1');

  /**
   * Version 1.20.0
   */
  public static readonly V1_20_0 = new AdotLambdaLayerPythonSdkVersion('1.20.0');

  /**
   * Version 1.19.0
   */
  public static readonly V1_19_0_1 = new AdotLambdaLayerPythonSdkVersion('1.19.0-1');

  /**
   * Version 1.18.0
   */
  public static readonly V1_18_0 = new AdotLambdaLayerPythonSdkVersion('1.18.0');

  /**
   * Version 1.17.0
   */
  public static readonly V1_17_0 = new AdotLambdaLayerPythonSdkVersion('1.17.0');

  /**
   * Version 1.16.0
   */
  public static readonly V1_16_0 = new AdotLambdaLayerPythonSdkVersion('1.16.0');

  /**
   * Version 1.15.0
   */
  public static readonly V1_15_0 = new AdotLambdaLayerPythonSdkVersion('1.15.0');

  /**
   * Version 1.14.0
   */
  public static readonly V1_14_0 = new AdotLambdaLayerPythonSdkVersion('1.14.0');

  /**
   * Version 1.13.0
   */
  public static readonly V1_13_0 = new AdotLambdaLayerPythonSdkVersion('1.13.0');

  /**
   * The latest layer version available in this CDK version. New versions could
   * introduce incompatible changes. Make sure to test them before deploying to production.
   */
  public static readonly LATEST = this.V1_29_0;

  private constructor(protected readonly layerVersion: string) {
    super(AdotLambdaLayerType.PYTHON_SDK, layerVersion);
  }
}

/**
 * The collection of versions of the ADOT Lambda Layer for JavaScript SDK.
 *
 * **Note:** These are legacy ADOT Lambda layers with an embedded collector.
 */
export class AdotLambdaLayerJavaScriptSdkVersion extends AdotLambdaLayerVersion {
  /**
   * Version 1.30.0
   */
  public static readonly V1_30_0 = new AdotLambdaLayerJavaScriptSdkVersion('1.30.0');

  /**
   * Version 1.18.1
   */
  public static readonly V1_18_1 = new AdotLambdaLayerJavaScriptSdkVersion('1.18.1');

  /**
   * Version 1.17.1
   */
  public static readonly V1_17_1 = new AdotLambdaLayerJavaScriptSdkVersion('1.17.1');

  /**
   * Version 1.16.0
   */
  public static readonly V1_16_0 = new AdotLambdaLayerJavaScriptSdkVersion('1.16.0');

  /**
   * Version 1.15.0
   */
  public static readonly V1_15_0_1 = new AdotLambdaLayerJavaScriptSdkVersion('1.15.0-1');

  /**
   * Version 1.7.0
   */
  public static readonly V1_7_0 = new AdotLambdaLayerJavaScriptSdkVersion('1.7.0');

  /**
   * The latest layer version available in this CDK version. New versions could
   * introduce incompatible changes. Make sure to test them before deploying to production.
   */
  public static readonly LATEST = this.V1_30_0;

  private constructor(protected readonly layerVersion: string) {
    super(AdotLambdaLayerType.JAVASCRIPT_SDK, layerVersion);
  }
}

/**
 * The collection of versions of the ADOT Lambda Layer for generic purpose.
 *
 * **Note:** These are legacy ADOT Lambda layers with an embedded collector.
 */
export class AdotLambdaLayerGenericVersion extends AdotLambdaLayerVersion {
  /**
   * Version 0.115.0
   */
  public static readonly V0_115_0 = new AdotLambdaLayerGenericVersion('0.115.0');

  /**
   * Version 0.102.1
   */
  public static readonly V0_102_1 = new AdotLambdaLayerGenericVersion('0.102.1');

  /**
   * Version 0.98.0
   */
  public static readonly V0_98_0 = new AdotLambdaLayerGenericVersion('0.98.0');

  /**
   * Version 0.90.1
   */
  public static readonly V0_90_1 = new AdotLambdaLayerGenericVersion('0.90.1');

  /**
   * Version 0.88.0
   */
  public static readonly V0_88_0 = new AdotLambdaLayerGenericVersion('0.88.0');

  /**
   * Version 0.84.0
   */
  public static readonly V0_84_0 = new AdotLambdaLayerGenericVersion('0.84.0');

  /**
   * Version 0.82.0
   */
  public static readonly V0_82_0 = new AdotLambdaLayerGenericVersion('0.82.0');

  /**
   * Version 0.62.1
   */
  public static readonly V0_62_1 = new AdotLambdaLayerGenericVersion('0.62.1');

  /**
   * The latest layer version available in this CDK version. New versions could
   * introduce incompatible changes. Make sure to test them before deploying to production.
   */
  public static readonly LATEST = this.V0_115_0;

  private constructor(protected readonly layerVersion: string) {
    super(AdotLambdaLayerType.GENERIC, layerVersion);
  }
}

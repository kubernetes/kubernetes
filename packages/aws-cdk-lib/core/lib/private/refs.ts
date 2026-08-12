// ----------------------------------------------------
// CROSS REFERENCES
// ----------------------------------------------------

import type { IConstruct } from 'constructs';
import { Construct } from 'constructs';
import { CfnReference } from './cfn-reference';
import { Intrinsic } from './intrinsic';
import { findTokens } from './resolve';
import { makeUniqueId } from './uniqueid';
import * as cxapi from '../../../cx-api';
import { Annotations } from '../annotations';
import { ArnFormat } from '../arn';
import { CfnElement } from '../cfn-element';
import { Fn } from '../cfn-fn';
import { CfnOutput } from '../cfn-output';
import { CfnParameter } from '../cfn-parameter';
import { ExportWriter } from '../custom-resource-provider/cross-region-export-providers/export-writer-provider';
import { AssumptionError, UnscopedValidationError } from '../errors';
import { Lazy } from '../lazy';
import { Names } from '../names';
import type { Reference } from '../reference';
import type { IResolvable, IResolveContext } from '../resolvable';
import type { Stack } from '../stack';
import { Token, Tokenization } from '../token';
import { ResolutionTypeHint } from '../type-hints';
import { iterateDfsPreorder } from './construct-iteration';
import { CfnResource } from '../cfn-resource';
import { lit } from './literal-string';
import type {
  PolicyReference,
  RoleReference,
} from '../../../interfaces/generated/aws-iam-interfaces.generated';
import { ReferenceStrength } from '../cross-stack-reference-strength';
import { stackOf } from './core-construct-finders';

export const STRING_LIST_REFERENCE_DELIMITER = '||';

function crossStackReferenceStrength(scope: IConstruct): ReferenceStrength | undefined {
  const value = scope.node.tryGetContext(cxapi.DEFAULT_CROSS_STACK_REFERENCES);
  if (value === undefined || value === null) {
    return undefined;
  }
  if (Object.values(ReferenceStrength).includes(value)) {
    return value;
  }
  throw new UnscopedValidationError(
    lit`InvalidReferenceStrength`,
    `Invalid value for ${cxapi.DEFAULT_CROSS_STACK_REFERENCES}: "${value}". Must be "strong", "weak", or "both".`,
  );
}

const WEAK_REFS_WARNING_EMITTED = Symbol.for('@aws-cdk/core.WeakRefsWarningEmitted');

const OVERRIDDEN_REFERENCE_SYMBOL = Symbol.for('@aws-cdk/core.CustomCoupledReference');

/**
 * A token wrapper that carries a per-usage reference strength override.
 *
 * When the resolution loop encounters this token, it resolves the underlying
 * CfnReference using the overridden strength instead of the default lookup chain,
 * and stores the result on this wrapper (not on the singleton CfnReference).
 */
export class CustomCoupledReference extends Intrinsic {
  public static isCustomCoupledReference(x: IResolvable): x is CustomCoupledReference {
    return OVERRIDDEN_REFERENCE_SYMBOL in x;
  }

  public readonly reference: CfnReference;
  public readonly strength: ReferenceStrength;
  private resolvedValue?: IResolvable;

  constructor(reference: CfnReference, strength: ReferenceStrength) {
    super(reference, { typeHint: reference.typeHint });
    this.reference = reference;
    this.strength = strength;
    Object.defineProperty(this, OVERRIDDEN_REFERENCE_SYMBOL, { value: true });
  }

  public assignValue(value: IResolvable): void {
    this.resolvedValue = value;
  }

  public resolve(context: IResolveContext): any {
    if (this.resolvedValue) {
      return this.resolvedValue.resolve(context);
    }
    return this.reference.resolve(context);
  }
}

/**
 * This is called from the App level to resolve all references defined. Each
 * reference is resolved based on its consumption context.
 */
export function resolveReferences(scope: IConstruct): void {
  const { refs, overrides } = findAllReferences(scope);

  for (const { source, value } of refs) {
    const consumer = stackOf(source);

    // resolve the value in the context of the consumer
    if (!value.hasValueForStack(consumer)) {
      const resolved = resolveValue(consumer, value);
      value.assignValueForStack(consumer, resolved);
    }
  }

  for (const { source, override } of overrides) {
    const consumer = stackOf(source);
    const resolved = resolveValue(consumer, override.reference, override.strength);
    override.assignValue(resolved);
  }
}

/**
 * Resolves the value for `reference` in the context of `consumer`.
 */
function resolveValue(consumer: Stack, reference: CfnReference, strengthOverride?: ReferenceStrength): IResolvable {
  const producer = stackOf(reference.target);
  const producerAccount = !Token.isUnresolved(producer.account) ? producer.account : cxapi.UNKNOWN_ACCOUNT;
  const producerRegion = !Token.isUnresolved(producer.region) ? producer.region : cxapi.UNKNOWN_REGION;
  const consumerAccount = !Token.isUnresolved(consumer.account) ? consumer.account : cxapi.UNKNOWN_ACCOUNT;
  const consumerRegion = !Token.isUnresolved(consumer.region) ? consumer.region : cxapi.UNKNOWN_REGION;

  // Priority: per-usage override > per-resource override > global consumer context > default
  const resourceStrength = CfnResource.isCfnResource(reference.target)
    ? reference.target._crossStackReferenceStrengthOverride
    : undefined;
  const strength = strengthOverride
    ?? resourceStrength
    ?? crossStackReferenceStrength(consumer)
    ?? 'strong';

  // produce and consumer stacks are the same, we can just return the value itself.
  if (producer === consumer) {
    return reference;
  }

  // unsupported: stacks from different apps
  if (producer.node.root !== consumer.node.root) {
    throw new UnscopedValidationError(lit`CannotReferenceAcrossApps`, 'Cannot reference across apps. Consuming and producing stacks must be defined within the same CDK app.');
  }

  // ----------------------------------------------------------------------
  // consumer is nested in the producer (directly or indirectly)
  // ----------------------------------------------------------------------

  // if the consumer is nested within the producer (directly or indirectly),
  // wire through a CloudFormation parameter and then resolve the reference with
  // the parent stack as the consumer.
  if (consumer.nestedStackParent && isNested(consumer, producer)) {
    const parameterValue = resolveValue(consumer.nestedStackParent, reference);
    return createNestedStackParameter(consumer, reference, parameterValue);
  }

  // ----------------------------------------------------------------------
  // producer is a nested stack
  // ----------------------------------------------------------------------

  // if the producer is nested, always publish the value through a
  // cloudformation output and resolve recursively with the Fn::GetAtt
  // of the output in the parent stack.

  // one might ask, if the consumer is not a parent of the producer,
  // why not just use export/import? the reason is that we cannot
  // generate an "export name" from a nested stack because the export
  // name must contain the stack name to ensure uniqueness, and we
  // don't know the stack name of a nested stack before we deploy it.
  // therefore, we can only export from a top-level stack.
  if (producer.nested) {
    const outputValue = createNestedStackOutput(producer, reference);
    const resolvedValue = resolveValue(consumer, outputValue);

    if (reference.typeHint === ResolutionTypeHint.STRING_LIST) {
      return Tokenization.reverseList(Fn.split(STRING_LIST_REFERENCE_DELIMITER, Token.asString(resolvedValue))) as IResolvable;
    } else {
      return resolvedValue;
    }
  }

  // ----------------------------------------------------------------------
  // export/import
  // ----------------------------------------------------------------------

  // Emit a once-per-app warning nudging users toward weak references
  const appRoot = consumer.node.root;
  if (!(appRoot as any)[WEAK_REFS_WARNING_EMITTED]) {
    const contextStrength = crossStackReferenceStrength(consumer);
    if (contextStrength === undefined) {
      (appRoot as any)[WEAK_REFS_WARNING_EMITTED] = true;
      Annotations.of(consumer).addWarningV2(
        '@aws-cdk/core:crossStackReferencesDefaultStrong',
        `No cross-stack-reference strength configured, defaulting to "strong". We recommend you set feature flag "${cxapi.DEFAULT_CROSS_STACK_REFERENCES}" to "both", then deploy everywhere, then set it to "weak". Alternatively, set it to "strong" explicitly to lock in the current producer-protecting behavior. ` +
        '(See: https://github.com/aws/aws-cdk/blob/main/packages/aws-cdk-lib/README.md#reference-strength)',
      );
    } else if (contextStrength === 'both') {
      (appRoot as any)[WEAK_REFS_WARNING_EMITTED] = true;
      Annotations.of(consumer).addWarningV2(
        '@aws-cdk/core:crossStackReferencesBothTransitional',
        `Feature flag "${cxapi.DEFAULT_CROSS_STACK_REFERENCES}" currently set to "both". This is a transitory state. After you have finished deploying this application everywhere, set it to "weak". ` +
        '(See: https://github.com/aws/aws-cdk/blob/main/packages/aws-cdk-lib/README.md#reference-strength)',
      );
    }
  }

  // stacks are not in the same account
  if (producerAccount !== consumerAccount) {
    if (strength === 'strong') {
      Annotations.of(consumer).addWarningV2(
        '@aws-cdk/core:crossAccountRefsAreAlwaysWeak',
        'Strong references requested, but cross-account references can only be weak. ' +
        `Acknowledge this warning or set "${cxapi.DEFAULT_CROSS_STACK_REFERENCES}" to "weak" to remove this message.`,
      );
      // Fall through to weak behavior since strong is not possible for cross-account
    }

    // "weak" or "both" fallback — use Fn::GetStackOutput with cross-account role
    if (consumer.synthesizer.cloudFormationExecutionRole == null) {
      throw new UnscopedValidationError(lit`NoCfnExecutionRoleForCrossAccountRefs`,
        `Stack "${consumer.node.path}" cannot reference ${renderReference(reference)} in stack "${producer.node.path}". ` +
        'Could not find a CloudFormation execution role for the consumer stack. Use a stack synthesizer that provides a ' +
        'CloudFormation execution role, such as DefaultStackSynthesizer (that uses the role from the bootstrap stack), ' +
        'or one that you can customize, such as BootstraplessSynthesizer.',
      );
    }

    if (producerRegion === cxapi.UNKNOWN_REGION || consumerRegion === cxapi.UNKNOWN_REGION) {
      throw new UnscopedValidationError(lit`CrossRegionReferencesRequireExplicitRegion`,
        `Stack "${consumer.node.path}" cannot reference ${renderReference(reference)} in stack "${producer.node.path}". ` +
        'Cross stack/region references are only supported for stacks with an explicit region defined. ');
    }

    const producerStackArn = stackOf(reference.target).formatArn({
      service: 'cloudformation',
      resource: 'stack',
      resourceName: `${producer.stackName}/*`,
      region: producerRegion,
      account: producerAccount,
    });

    return createGetStackOutput(reference, {
      consumerRoleArn: consumer.synthesizer.cloudFormationExecutionRole,
      producerAccount,
      producerRegion,
      producerStackArn,
    });
  }

  // Stacks are in the same account, but different regions
  if (producerRegion !== consumerRegion) {
    if (producerRegion === cxapi.UNKNOWN_REGION || consumerRegion === cxapi.UNKNOWN_REGION) {
      throw new UnscopedValidationError(lit`CrossRegionReferencesRequireExplicitRegion`,
        `Stack "${consumer.node.path}" cannot reference ${renderReference(reference)} in stack "${producer.node.path}". ` +
        'Cross stack/region references are only supported for stacks with an explicit region defined. ');
    }
    consumer.addStackDependency(producer,
      `${consumer.node.path} -> ${reference.target.node.path}.${reference.displayName}`);

    if (strength === 'strong') {
      return createCrossRegionImportValue(reference, consumer);
    }

    if (strength === 'both') {
      // Generate the ExportWriter (strong side) but don't create the ExportReader
      createCrossRegionExportOnly(reference, consumer);
    }

    const producerStackArn = producer.formatArn({
      service: 'cloudformation',
      resource: 'stack',
      resourceName: `${producer.stackName}/*`,
      region: producerRegion,
      account: producerAccount,
    });

    return createGetStackOutput(reference, {
      producerStackArn,
    });
  }

  // add a dependency between the producer and the consumer. dependency logic
  // will take care of applying the dependency at the right level (e.g. the
  // top-level stacks).
  consumer.addStackDependency(producer,
    `${consumer.node.path} -> ${reference.target.node.path}.${reference.displayName}`);

  if (strength === 'strong') {
    return createImportValue(reference);
  }

  if (strength === 'both') {
    // Create the Import/Export pair, but drop the Import side.
    createImportValue(reference);
  }

  // strength === 'weak'
  return createGetStackOutput(reference, {});
}

/**
 * Return a human readable version of this reference
 */
function renderReference(ref: CfnReference) {
  return `{${ref.target.node.path}[${ref.displayName}]}`;
}

/**
 * Finds all the CloudFormation references in a construct tree.
 */
function findAllReferences(root: IConstruct) {
  const refs = new Array<{ source: CfnElement; value: CfnReference }>();
  const overrides = new Array<{ source: CfnElement; override: CustomCoupledReference }>();

  for (const consumer of iterateDfsPreorder(root)) {
    // include only CfnElements (i.e. resources)
    if (!CfnElement.isCfnElement(consumer)) {
      continue;
    }

    try {
      const tokens = findTokens(consumer, () => consumer._toCloudFormation());

      // iterate over all the tokens (e.g. intrinsic functions, lazies, etc) that
      // were found in the cloudformation representation of this resource.
      for (const token of tokens) {
        if (CustomCoupledReference.isCustomCoupledReference(token)) {
          overrides.push({ source: consumer, override: token });
        } else if (CfnReference.isCfnReference(token)) {
          refs.push({ source: consumer, value: token });
        }
      }
    } catch (e: any) {
      // Note: it might be that the properties of the CFN object aren't valid.
      // This will usually be preventatively caught in a construct's validate()
      // and turned into a nicely descriptive error, but we're running prepare()
      // before validate(). Swallow errors that occur because the CFN layer
      // doesn't validate completely.
      //
      // This does make the assumption that the error will not be rectified,
      // but the error will be thrown later on anyway. If the error doesn't
      // get thrown down the line, we may miss references.
      if (e.type === 'CfnSynthesisError') {
        continue;
      }

      throw e;
    }
  }

  return { refs, overrides };
}

// ------------------------------------------------------------------------------------------------
// export/import
// ------------------------------------------------------------------------------------------------

/**
 * Imports a value from another stack by creating an "Output" with an "ExportName"
 * and returning an "Fn::ImportValue" token.
 */
function createImportValue(reference: Reference): Intrinsic {
  const exportingStack = stackOf(reference.target);
  let importExpr;

  if (reference.typeHint === ResolutionTypeHint.STRING_LIST) {
    importExpr = exportingStack.exportStringListValue(reference);
    // I happen to know this returns a Fn.split() which implements Intrinsic.
    return Tokenization.reverseList(importExpr) as Intrinsic;
  }

  importExpr = exportingStack.exportValue(reference);
  // I happen to know this returns a Fn.importValue() which implements Intrinsic.
  return Tokenization.reverseCompleteString(importExpr) as Intrinsic;
}

function getOrCreateExportWriter(reference: Reference, importStack: Stack): { exportWriter: ExportWriter; exportName: string } {
  const referenceStack = stackOf(reference.target);
  const exportingStack = referenceStack.nestedStackParent ?? referenceStack;

  const exportable = getExportable(exportingStack, reference);
  const id = JSON.stringify(exportingStack.resolve(exportable));
  const exportName = generateExportName(importStack, reference, id);
  if (Token.isUnresolved(exportName)) {
    throw new UnscopedValidationError(lit`UnresolvedTokenInExportName`, `unresolved token in generated export name: ${JSON.stringify(exportingStack.resolve(exportName))}`);
  }

  const writerConstructName = makeUniqueId(['ExportsWriter', importStack.region]);
  const exportWriter = ExportWriter.getOrCreate(exportingStack, writerConstructName, {
    region: importStack.region,
  });

  return { exportWriter, exportName };
}

/**
 * Imports a value from another stack in a different region using ExportWriter/ExportReader.
 */
function createCrossRegionImportValue(reference: Reference, importStack: Stack): Intrinsic {
  const { exportWriter, exportName } = getOrCreateExportWriter(reference, importStack);

  const exported = exportWriter.exportValue(exportName, reference, importStack);
  if (importStack.nestedStackParent) {
    return createNestedStackParameter(importStack, (exported as CfnReference), exported);
  }
  return exported;
}

/**
 * Creates the ExportWriter in the producing stack without creating the ExportReader
 * in the consuming stack. Used during the "both" transitional state.
 */
function createCrossRegionExportOnly(reference: Reference, importStack: Stack): void {
  const { exportWriter, exportName } = getOrCreateExportWriter(reference, importStack);
  exportWriter.exportValueWriteOnly(exportName, reference, importStack);
}

/**
 * Generate a unique physical name for the export
 */
function generateExportName(importStack: Stack, reference: Reference, id: string): string {
  const referenceStack = stackOf(reference.target);

  const components = [
    referenceStack.stackName ?? '',
    referenceStack.region,
    id,
  ];
  const prefix = `${importStack.nestedStackParent?.stackName ?? importStack.stackName}/`;
  const localPart = makeUniqueId(components);
  // max name length for a system manager parameter is 1011 characters
  // including the arn, i.e.
  // arn:aws:ssm:us-east-2:111122223333:parameter/cdk/exports/${stackName}/${name}
  const maxLength = 900;
  return prefix + localPart.slice(Math.max(0, localPart.length - maxLength + prefix.length));
}

interface GetStackOutputOptions {
  consumerRoleArn?: string;
  producerRegion?: string;
  producerAccount?: string;
  producerStackArn?: string;
}

interface GetStackOutputRoleProps {
  readonly consumerRoleArn: string;
  readonly producerAccount?: string;
}

const ROLE_CONSUMERS = new WeakMap<CfnResource, Set<string>>();

function createGetStackOutputRole(scope: Construct, id: string, props: GetStackOutputRoleProps): { resource: CfnResource; roleRef: RoleReference } {
  const consumers = new Set<string>([props.consumerRoleArn]);
  const resource = new CfnResource(scope, id, {
    type: 'AWS::IAM::Role',
    properties: {
      AssumeRolePolicyDocument: {
        Version: '2012-10-17',
        Statement: [
          {
            Effect: 'Allow',
            Principal: {
              AWS: Lazy.any({
                produce: () => {
                  const arns = [...consumers].map(arn => ({ 'Fn::Sub': arn }));
                  return arns.length === 1 ? arns[0] : arns;
                },
              }),
            },
            Action: [
              'sts:AssumeRole',
            ],
          },
        ],
      },
    },
  });
  ROLE_CONSUMERS.set(resource, consumers);

  const roleName = Names.uniqueResourceName(resource, {
    maxLength: 64,
  });
  resource.addPropertyOverride('RoleName', roleName);

  const roleArn = stackOf(scope).formatArn({
    service: 'iam',
    resource: 'role',
    resourceName: roleName,
    account: props.producerAccount,
    arnFormat: ArnFormat.SLASH_RESOURCE_NAME,
    region: '',
  });

  return { resource, roleRef: { roleArn, roleName } };
}

function addConsumerToRole(roleResource: CfnResource, consumerRoleArn: string): void {
  const consumers = ROLE_CONSUMERS.get(roleResource);
  if (consumers) {
    consumers.add(consumerRoleArn);
  }
}

interface GetStackOutputPolicyProps {
  readonly role: CfnResource;
  readonly producerStackArn?: string;
}

function createGetStackOutputPolicy(
  scope: Construct,
  id: string,
  props: GetStackOutputPolicyProps,
): { resource: CfnResource; policyRef: PolicyReference } {
  const resource = new CfnResource(scope, id, {
    type: 'AWS::IAM::Policy',
    properties: {
      Roles: [Fn.ref(props.role.logicalId)],
      PolicyDocument: {
        Version: '2012-10-17',
        Statement: [
          {
            Effect: 'Allow',
            Action: 'cloudformation:DescribeStacks',
            Resource: props.producerStackArn,
          },
        ],
      },
    },
  });

  const policyName = Names.uniqueResourceName(resource, {
    maxLength: 128,
  });
  resource.addPropertyOverride('PolicyName', policyName);

  return { resource, policyRef: { policyId: policyName } };
}

function createGetStackOutput(reference: Reference, options: GetStackOutputOptions = {}): Intrinsic {
  const exportingStack = stackOf(reference.target);

  const resolved = JSON.stringify(exportingStack.resolve(reference));
  const outputId = 'Output' + resolved;
  const roleId = 'GetStackOutputRole';
  const policyId = 'GetStackOutputPolicy';

  function createScope(stack: Stack) {
    const scopeName = 'Publish';
    let scope = stack.node.tryFindChild(scopeName) as Construct;
    if (scope === undefined) {
      scope = new Construct(stack, scopeName);
    }

    return scope;
  }

  const scope = createScope(exportingStack);

  let output = scope.node.tryFindChild(outputId) as CfnOutput;
  if (output == null) {
    if (reference.typeHint === ResolutionTypeHint.STRING_LIST) {
      output = new CfnOutput(scope, outputId, { value: Fn.join(STRING_LIST_REFERENCE_DELIMITER, Token.asList(reference)) });
    } else {
      output = new CfnOutput(scope, outputId, {
        value: Token.asString(reference),
      });
    }
  }

  let roleArn: string | undefined = undefined;
  if (options.consumerRoleArn) {
    let roleResource = scope.node.tryFindChild(roleId) as CfnResource;
    if (roleResource == null) {
      const { resource, roleRef } = createGetStackOutputRole(scope, roleId, {
        consumerRoleArn: options.consumerRoleArn,
        producerAccount: options.producerAccount,
      });
      roleResource = resource;
      roleArn = roleRef.roleArn;
    } else {
      addConsumerToRole(roleResource, options.consumerRoleArn);
      const roleName = Names.uniqueResourceName(roleResource, { maxLength: 64 });
      roleArn = exportingStack.formatArn({
        service: 'iam',
        resource: 'role',
        resourceName: roleName,
        account: options.producerAccount,
        arnFormat: ArnFormat.SLASH_RESOURCE_NAME,
        region: '',
      });
    }

    let policy = scope.node.tryFindChild(policyId) as CfnResource;
    if (policy == null) {
      createGetStackOutputPolicy(scope, policyId, {
        role: roleResource,
        producerStackArn: options.producerStackArn,
      });
    }
  }

  const getStackOutput = Fn.getStackOutput(exportingStack.stackName, output.logicalId, exportingStack.region, roleArn);

  if (reference.typeHint === ResolutionTypeHint.STRING_LIST) {
    return Tokenization.reverseList(Fn.split(STRING_LIST_REFERENCE_DELIMITER, getStackOutput)) as Intrinsic;
  }

  return Tokenization.reverseCompleteString(getStackOutput) as Intrinsic;
}

export function getExportable(stack: Stack, reference: Reference): Intrinsic {
  // could potentially be changed by a call to overrideLogicalId. This would cause our Export/Import
  // to have an incorrect id. For a better user experience, lock the logicalId and throw an error
  // if the user tries to override the id _after_ calling exportValue
  if (CfnElement.isCfnElement(reference.target)) {
    reference.target._lockLogicalId();
  }

  // "teleport" the value here, in case it comes from a nested stack. This will also
  // ensure the value is from our own scope.
  return referenceNestedStackValueInParent(reference, stack);
}

// ------------------------------------------------------------------------------------------------
// nested stacks
// ------------------------------------------------------------------------------------------------

/**
 * Adds a CloudFormation parameter to a nested stack and assigns it with the
 * value of the reference.
 */
function createNestedStackParameter(nested: Stack, reference: CfnReference, value: IResolvable) {
  const paramId = generateUniqueId(nested, reference, 'reference-to-');
  let param = nested.node.tryFindChild(paramId) as CfnParameter;
  if (!param) {
    param = new CfnParameter(nested, paramId, { type: 'String' });

    // Ugly little hack until we move NestedStack to this module.
    if (!('setParameter' in nested)) {
      throw new UnscopedValidationError(lit`NestedStackMustHaveSetParameter`, 'assertion failed: nested stack should have a "setParameter" method');
    }

    (nested as any).setParameter(param.logicalId, Token.asString(value));
  }

  return param.value as CfnReference;
}

/**
 * Adds a CloudFormation output to a nested stack and returns an "Fn::GetAtt"
 * intrinsic that can be used to reference this output in the parent stack.
 */
function createNestedStackOutput(producer: Stack, reference: Reference): CfnReference {
  const outputId = generateUniqueId(producer, reference);
  let output = producer.node.tryFindChild(outputId) as CfnOutput;
  if (!output) {
    if (reference.typeHint === ResolutionTypeHint.STRING_LIST) {
      output = new CfnOutput(producer, outputId, { value: Fn.join(STRING_LIST_REFERENCE_DELIMITER, Token.asList(reference)) });
    } else {
      output = new CfnOutput(producer, outputId, { value: Token.asString(reference) });
    }
  }

  if (!producer.nestedStackResource) {
    throw new AssumptionError(lit`AssertionFailed`, 'assertion failed');
  }

  return producer.nestedStackResource.getAtt(`Outputs.${output.logicalId}`) as CfnReference;
}

/**
 * Translate a Reference into a nested stack into a value in the parent stack
 *
 * Will create Outputs along the chain of Nested Stacks, and return the final `{ Fn::GetAtt }`.
 */
export function referenceNestedStackValueInParent(reference: Reference, targetStack: Stack): Intrinsic {
  let currentStack = stackOf(reference.target);
  if (currentStack !== targetStack && !isNested(currentStack, targetStack)) {
    throw new UnscopedValidationError(lit`ReferencedResourceMustBeInTargetStack`, `Referenced resource must be in stack '${targetStack.node.path}', got '${reference.target.node.path}'`);
  }

  const isNestedListReference = currentStack !== targetStack && reference.typeHint === ResolutionTypeHint.STRING_LIST;

  while (currentStack !== targetStack) {
    reference = createNestedStackOutput(stackOf(reference.target), reference);
    currentStack = stackOf(reference.target);
  }

  if (isNestedListReference) {
    return Tokenization.reverseList(Fn.split(STRING_LIST_REFERENCE_DELIMITER, Token.asString(reference))) as Intrinsic;
  }

  return reference;
}

/**
 * @returns true if this stack is a direct or indirect parent of the nested
 * stack `nested`.
 *
 * If `child` is not a nested stack, always returns `false` because it can't
 * have a parent, dah.
 */
function isNested(nested: Stack, parent: Stack): boolean {
  // if the parent is a direct parent
  if (nested.nestedStackParent === parent) {
    return true;
  }

  // we reached a top-level (non-nested) stack without finding the parent
  if (!nested.nestedStackParent) {
    return false;
  }

  // recurse with the child's direct parent
  return isNested(nested.nestedStackParent, parent);
}

/**
 * Generates a unique id for a `Reference`
 * @param stack A stack used to resolve tokens
 * @param ref The reference
 * @param prefix Optional prefix for the id
 * @returns A unique id
 */
function generateUniqueId(stack: Stack, ref: Reference, prefix = '') {
  // we call "resolve()" to ensure that tokens do not creep in (for example, if the reference display name includes tokens)
  return stack.resolve(`${prefix}${Names.nodeUniqueId(ref.target.node)}${ref.displayName}`);
}

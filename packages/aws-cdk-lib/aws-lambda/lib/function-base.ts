import { createHash } from 'crypto';
import type { Construct, Node } from 'constructs';
import type { AliasOptions } from './alias';
import type { Architecture } from './architecture';
import type { EventInvokeConfigOptions } from './event-invoke-config';
import { EventInvokeConfig } from './event-invoke-config';
import type { IEventSource } from './event-source';
import type { EventSourceMappingOptions } from './event-source-mapping';
import { EventSourceMapping } from './event-source-mapping';
import type { FunctionUrlOptions } from './function-url';
import { FunctionUrl, FunctionUrlAuthType } from './function-url';
import type { IVersion } from './lambda-version';
import type { FunctionReference, IFunctionRef, VersionReference } from './lambda.generated';
import { CfnPermission } from './lambda.generated';
import type { Permission } from './permission';
import type { TenancyConfig } from './tenancy-config';
import { addAlias, flatMap } from './util';
import type * as cloudwatch from '../../aws-cloudwatch';
import type * as ec2 from '../../aws-ec2';
import * as iam from '../../aws-iam';
import type { IResource } from '../../core';
import { Annotations, ArnFormat, FeatureFlags, Resource, Stack, Token } from '../../core';
import { ValidationError } from '../../core/lib/errors';
import { MethodMetadata } from '../../core/lib/metadata-resource';
import { lit } from '../../core/lib/private/literal-string';
import * as cxapi from '../../cx-api';

export interface IFunction extends IResource, ec2.IConnectable, iam.IGrantable, IFunctionRef {

  /**
   * The name of the function.
   *
   * @attribute
   */
  readonly functionName: string;

  /**
   * The ARN of the function.
   *
   * @attribute
   */
  readonly functionArn: string;

  /**
   * The IAM role associated with this function.
   */
  readonly role?: iam.IRole;

  /**
   * Whether or not this Lambda function was bound to a VPC
   *
   * If this is `false`, trying to access the `connections` object will fail.
   */
  readonly isBoundToVpc: boolean;

  /**
   * The `$LATEST` version of this function.
   *
   * Note that this is reference to a non-specific AWS Lambda version, which
   * means the function this version refers to can return different results in
   * different invocations.
   *
   * To obtain a reference to an explicit version which references the current
   * function configuration, use `lambdaFunction.currentVersion` instead.
   */
  readonly latestVersion: IVersion;

  /**
   * The construct node where permissions are attached.
   */
  readonly permissionsNode: Node;

  /**
   * The tenancy configuration for this function.
   */
  readonly tenancyConfig?: TenancyConfig;

  /**
   * The system architectures compatible with this lambda function.
   */
  readonly architecture: Architecture;

  /**
   * The ARN(s) to put into the resource field of the generated IAM policy for grantInvoke().
   *
   * This property is for cdk modules to consume only. You should not need to use this property.
   * Instead, use grantInvoke() directly.
   */
  readonly resourceArnsForGrantInvoke: string[];

  /**
   * Adds an event source that maps to this AWS Lambda function.
   * @param id construct ID
   * @param options mapping options
   */
  addEventSourceMapping(id: string, options: EventSourceMappingOptions): EventSourceMapping;

  /**
   * Adds a permission to the Lambda resource policy.
   * @param id The id for the permission construct
   * @param permission The permission to grant to this Lambda function. @see Permission for details.
   */
  addPermission(id: string, permission: Permission): void;

  /**
   * Adds a statement to the IAM role assumed by the instance.
   */
  addToRolePolicy(statement: iam.PolicyStatement): void;

  /**
   * Grant the given identity permissions to invoke this Lambda
   */
  grantInvoke(identity: iam.IGrantable): iam.Grant;

  /**
   * Grant the given identity permissions to invoke the $LATEST version or
   * unqualified version of this Lambda
   */
  grantInvokeLatestVersion(identity: iam.IGrantable): iam.Grant;

  /**
   * Grant the given identity permissions to invoke the given version of this Lambda
   */
  grantInvokeVersion(identity: iam.IGrantable, version: IVersion): iam.Grant;

  /**
   * Grant the given identity permissions to invoke this Lambda Function URL
   */
  grantInvokeUrl(identity: iam.IGrantable): iam.Grant;

  /**
   * Grant multiple principals the ability to invoke this Lambda via CompositePrincipal
   */
  grantInvokeCompositePrincipal(compositePrincipal: iam.CompositePrincipal): iam.Grant[];

  /**
   * Return the given named metric for this Lambda
   */
  metric(metricName: string, props?: cloudwatch.MetricOptions): cloudwatch.Metric;

  /**
   * Metric for the Duration of this Lambda
   *
   * @default average over 5 minutes
   */
  metricDuration(props?: cloudwatch.MetricOptions): cloudwatch.Metric;

  /**
   * Metric for the number of invocations of this Lambda
   *
   * @default sum over 5 minutes
   */
  metricInvocations(props?: cloudwatch.MetricOptions): cloudwatch.Metric;

  /**
   * Metric for the number of throttled invocations of this Lambda
   *
   * @default sum over 5 minutes
   */
  metricThrottles(props?: cloudwatch.MetricOptions): cloudwatch.Metric;

  /**
   * Adds an event source to this function.
   *
   * Event sources are implemented in the aws-cdk-lib/aws-lambda-event-sources module.
   *
   * The following example adds an SQS Queue as an event source:
   * ```
   * import { SqsEventSource } from 'aws-cdk-lib/aws-lambda-event-sources';
   * myFunction.addEventSource(new SqsEventSource(myQueue));
   * ```
   */
  addEventSource(source: IEventSource): void;

  /**
   * Configures options for asynchronous invocation.
   */
  configureAsyncInvoke(options: EventInvokeConfigOptions): void;

  /**
   * Adds a url to this lambda function.
   */
  addFunctionUrl(options?: FunctionUrlOptions): FunctionUrl;
}

/**
 * Represents a Lambda function defined outside of this stack.
 */
export interface FunctionAttributes {
  /**
   * The ARN of the Lambda function.
   *
   * Format: arn:<partition>:lambda:<region>:<account-id>:function:<function-name>
   */
  readonly functionArn: string;

  /**
   * The IAM execution role associated with this function.
   *
   * If the role is not specified, any role-related operations will no-op.
   */
  readonly role?: iam.IRole;

  /**
   * Id of the security group of this Lambda, if in a VPC.
   *
   * This needs to be given in order to support allowing connections
   * to this Lambda.
   *
   * @deprecated use `securityGroup` instead
   */
  readonly securityGroupId?: string;

  /**
   * The security group of this Lambda, if in a VPC.
   *
   * This needs to be given in order to support allowing connections
   * to this Lambda.
   */
  readonly securityGroup?: ec2.ISecurityGroup;

  /**
   * Setting this property informs the CDK that the imported function is in the same environment as the stack.
   * This affects certain behaviours such as, whether this function's permission can be modified.
   * When not configured, the CDK attempts to auto-determine this. For environment agnostic stacks, i.e., stacks
   * where the account is not specified with the `env` property, this is determined to be false.
   *
   * Set this to property *ONLY IF* the imported function is in the same account as the stack
   * it's imported in.
   * @default - depends: true, if the Stack is configured with an explicit `env` (account and region) and the account is the same as this function.
   * For environment-agnostic stacks this will default to `false`.
   */
  readonly sameEnvironment?: boolean;

  /**
   * Setting this property informs the CDK that the imported function ALREADY HAS the necessary permissions
   * for what you are trying to do. When not configured, the CDK attempts to auto-determine whether or not
   * additional permissions are necessary on the function when grant APIs are used. If the CDK tried to add
   * permissions on an imported lambda, it will fail.
   *
   * Set this property *ONLY IF* you are committing to manage the imported function's permissions outside of
   * CDK. You are acknowledging that your CDK code alone will have insufficient permissions to access the
   * imported function.
   *
   * @default false
   */
  readonly skipPermissions?: boolean;

  /**
   * The architecture of this Lambda Function (this is an optional attribute and defaults to X86_64).
   * @default - Architecture.X86_64
   */
  readonly architecture?: Architecture;

  /**
   * The tenancy configuration of this Lambda Function.
   * @default - Tenant isolation is not enabled
   */
  readonly tenancyConfig?: TenancyConfig;
}

export abstract class FunctionBase extends Resource implements IFunction, ec2.IClientVpnConnectionHandler {
  /**
   * The principal this Lambda Function is running as
   */
  public abstract readonly grantPrincipal: iam.IPrincipal;

  /**
   * The name of the function.
   */
  public abstract readonly functionName: string;

  /**
   * The ARN fo the function.
   */
  public abstract readonly functionArn: string;

  /**
   * The IAM role associated with this function.
   *
   * Undefined if the function was imported without a role.
   */
  public abstract readonly role?: iam.IRole;

  /**
   * The construct node where permissions are attached.
   */
  public abstract readonly permissionsNode: Node;

  /**
   * The architecture of this Lambda Function.
   */
  public abstract readonly architecture: Architecture;

  /**
   * The tenancy configuration for this function.
   */
  public abstract readonly tenancyConfig?: TenancyConfig;

  /**
   * Whether the addPermission() call adds any permissions
   *
   * True for new Lambdas, false for version $LATEST and imported Lambdas
   * from different accounts.
   */
  protected abstract readonly canCreatePermissions: boolean;

  /**
   * The ARN(s) to put into the resource field of the generated IAM policy for grantInvoke()
   */
  public abstract readonly resourceArnsForGrantInvoke: string[];

  /**
   * Whether the user decides to skip adding permissions.
   * The only use case is for cross-account, imported lambdas
   * where the user commits to modifying the permisssions
   * on the imported lambda outside CDK.
   * @internal
   */
  protected readonly _skipPermissions?: boolean;

  /**
   * Actual connections object for this Lambda
   *
   * May be unset, in which case this Lambda is not configured use in a VPC.
   * @internal
   */
  protected _connections?: ec2.Connections;

  private _latestVersion?: LatestVersion;

  /**
   * Flag to delay adding a warning message until current version is invoked.
   * @internal
   */
  protected _warnIfCurrentVersionCalled: boolean = false;

  /**
   * Mapping of invocation principals to grants. Used to de-dupe `grantInvoke()` calls.
   * @internal
   */
  protected _invocationGrants: Record<string, iam.Grant> = {};

  /**
   * Mapping of function URL invocation principals to grants. Used to de-dupe `grantInvokeUrl()` calls.
   * @internal
   */
  protected _functionUrlInvocationGrants: Record<string, iam.Grant> = {};

  /**
   * The number of permissions added to this function
   * @internal
   */
  private _policyCounter: number = 0;

  /**
   * Track whether we've added statements with literal resources to the role's default policy
   * @internal
   */
  private _hasAddedLiteralStatements: boolean = false;

  /**
   * Track whether we've added statements with array token resources to the role's default policy
   * @internal
   */
  private _hasAddedArrayTokenStatements: boolean = false;

  public get functionRef(): FunctionReference {
    return {
      functionName: this.functionName,
      functionArn: this.functionArn,
    };
  }

  /**
   * A warning will be added to functions under the following conditions:
   * - permissions that include `lambda:InvokeFunction` are added to the unqualified function.
   * - function.currentVersion is invoked before or after the permission is created.
   *
   * This applies only to permissions on Lambda functions, not versions or aliases.
   * This function is overridden as a noOp for QualifiedFunctionBase.
   */
  public considerWarningOnInvokeFunctionPermissions(scope: Construct, action: string) {
    const affectedPermissions = ['lambda:InvokeFunction', 'lambda:*', 'lambda:Invoke*'];
    if (affectedPermissions.includes(action)) {
      if (scope.node.tryFindChild('CurrentVersion')) {
        this.warnInvokeFunctionPermissions(scope);
      } else {
        this._warnIfCurrentVersionCalled = true;
      }
    }
  }

  protected warnInvokeFunctionPermissions(scope: Construct): void {
    Annotations.of(scope).addWarningV2('@aws-cdk/aws-lambda:addPermissionsToVersionOrAlias', [
      "AWS Lambda has changed their authorization strategy, which may cause client invocations using the 'Qualifier' parameter of the lambda function to fail with Access Denied errors.",
      "If you are using a lambda Version or Alias, make sure to call 'grantInvoke' or 'addPermission' on the Version or Alias, not the underlying Function",
      'See: https://github.com/aws/aws-cdk/issues/19273',
    ].join('\n'));
  }

  /**
   * Adds a permission to the Lambda resource policy.
   * @param id The id for the permission construct
   * @param permission The permission to grant to this Lambda function. @see Permission for details.
   */
  public addPermission(id: string, permission: Permission) {
    if (!this.canCreatePermissions) {
      if (!this._skipPermissions) {
        Annotations.of(this).addWarningV2('UnclearLambdaEnvironment', `addPermission() has no effect on a Lambda Function with region=${this.env.region}, account=${this.env.account}, in a Stack with region=${Stack.of(this).region}, account=${Stack.of(this).account}. Suppress this warning if this is intentional, or pass sameEnvironment=true to fromFunctionAttributes() if you would like to add the permissions.`);
      }
      return;
    }

    let principal = this.parsePermissionPrincipal(permission.principal);

    let { sourceArn, sourceAccount, principalOrgID } = this.validateConditionCombinations(permission.principal) ?? {};

    const action = permission.action ?? 'lambda:InvokeFunction';
    const scope = permission.scope ?? this;

    this.considerWarningOnInvokeFunctionPermissions(scope, action);

    new CfnPermission(scope, id, {
      action,
      principal,
      functionName: this.functionArn,
      eventSourceToken: permission.eventSourceToken,
      sourceAccount: permission.sourceAccount ?? sourceAccount,
      sourceArn: permission.sourceArn ?? sourceArn,
      principalOrgId: permission.organizationId ?? principalOrgID,
      functionUrlAuthType: permission.functionUrlAuthType,
      invokedViaFunctionUrl: permission.invokedViaFunctionUrl,
    });
  }

  /**
   * Adds a statement to the IAM role assumed by the instance.
   */
  public addToRolePolicy(statement: iam.PolicyStatement) {
    const useCreateNewPolicies = FeatureFlags.of(this).isEnabled(cxapi.LAMBDA_CREATE_NEW_POLICIES_WITH_ADDTOROLEPOLICY);
    if (!this.role) {
      return;
    }

    const hasArrayTokens = this.statementHasArrayTokens(statement);

    // Check if mixing different token types would cause CloudFormation resolution conflicts
    // Array tokens and literal resources cannot be safely merged in the same policy document
    const wouldCauseConflict = (hasArrayTokens && this._hasAddedLiteralStatements) ||
      (!hasArrayTokens && this._hasAddedArrayTokenStatements);

    if (useCreateNewPolicies || wouldCauseConflict) {
      // Create separate policy to avoid CloudFormation token resolution conflicts
      const policyToAdd = new iam.Policy(this, `inlinePolicyAddedToExecutionRole-${this._policyCounter++}`, {
        statements: [statement],
      });
      this.role.attachInlinePolicy(policyToAdd);
    } else {
      // Safe to merge - track what type of resources we're adding to the default policy
      this.role.addToPrincipalPolicy(statement);

      if (hasArrayTokens) {
        this._hasAddedArrayTokenStatements = true;
      } else {
        this._hasAddedLiteralStatements = true;
      }
    }
  }

  /**
   * Access the Connections object
   *
   * Will fail if not a VPC-enabled Lambda Function
   */
  public get connections(): ec2.Connections {
    if (!this._connections) {
      throw new ValidationError(lit`OnlyVpcAssociatedLambdaFunctionsHaveSecurityGroups`, 'Only VPC-associated Lambda Functions have security groups to manage. Supply the "vpc" parameter when creating the Lambda, or "securityGroupId" when importing it.', this);
    }
    return this._connections;
  }

  public get latestVersion(): IVersion {
    if (!this._latestVersion) {
      this._latestVersion = new LatestVersion(this);
    }
    return this._latestVersion;
  }

  /**
   * Whether or not this Lambda function was bound to a VPC
   *
   * If this is `false`, trying to access the `connections` object will fail.
   */
  public get isBoundToVpc(): boolean {
    return !!this._connections;
  }

  public addEventSourceMapping(id: string, options: EventSourceMappingOptions): EventSourceMapping {
    return new EventSourceMapping(this, id, {
      target: this,
      ...options,
    });
  }

  /**
   * Grant the given identity permissions to invoke this Lambda
   *
   * [disable-awslint:no-grants]
   */
  public grantInvoke(grantee: iam.IGrantable): iam.Grant {
    const hash = createHash('sha256')
      .update(JSON.stringify({
        principal: grantee.grantPrincipal.toString(),
        conditions: grantee.grantPrincipal.policyFragment.conditions,
      }), 'utf8')
      .digest('base64');
    const identifier = `Invoke${hash}`;

    // Memoize the result so subsequent grantInvoke() calls are idempotent
    let grant = this._invocationGrants[identifier];
    if (!grant) {
      grant = this.grant(grantee, identifier, 'lambda:InvokeFunction', this.resourceArnsForGrantInvoke);
      this._invocationGrants[identifier] = grant;
    }
    return grant;
  }

  /**
   * Grant the given identity permissions to invoke the $LATEST version or
   * unqualified version of this Lambda
   *
   * [disable-awslint:no-grants]
   */
  public grantInvokeLatestVersion(grantee: iam.IGrantable): iam.Grant {
    return this.grantInvokeVersion(grantee, this.latestVersion);
  }

  /**
   * Grant the given identity permissions to invoke the given version of this Lambda
   *
   * [disable-awslint:no-grants]
   */
  public grantInvokeVersion(grantee: iam.IGrantable, version: IVersion): iam.Grant {
    const hash = createHash('sha256')
      .update(JSON.stringify({
        principal: grantee.grantPrincipal.toString(),
        conditions: grantee.grantPrincipal.policyFragment.conditions,
        version: version.version,
      }), 'utf8')
      .digest('base64');
    const identifier = `Invoke${hash}`;

    // Memoize the result so subsequent grantInvoke() calls are idempotent
    let grant = this._invocationGrants[identifier];
    if (!grant) {
      let resourceArns = [`${this.functionArn}:${version.version}`];
      if (version == this.latestVersion) {
        resourceArns.push(this.functionArn);
      }
      grant = this.grant(grantee, identifier, 'lambda:InvokeFunction', resourceArns);
      this._invocationGrants[identifier] = grant;
    }
    return grant;
  }

  /**
   * Grant the given identity permissions to invoke this Lambda Function URL
   *
   * [disable-awslint:no-grants]
   */
  public grantInvokeUrl(grantee: iam.IGrantable): iam.Grant {
    const identifier = `InvokeFunctionUrl${grantee.grantPrincipal}`; // calls the .toString() of the principal
    const identifierDualAuth = `${identifier}-DualAuth`;

    // Memoize the result so subsequent grantInvoke() calls are idempotent
    let grant = this._functionUrlInvocationGrants[identifier];
    if (!grant) {
      // Build conditions for function URL with AWS_IAM auth type
      const functionUrlConditions: Record<string, Record<string, unknown>> = {
        StringEquals: {
          'lambda:FunctionUrlAuthType': FunctionUrlAuthType.AWS_IAM,
        },
      };

      grant = this.grant(
        grantee,
        identifier,
        'lambda:InvokeFunctionUrl',
        [this.functionArn],
        {
          functionUrlAuthType: FunctionUrlAuthType.AWS_IAM,
        },
        functionUrlConditions,
      );

      // Build conditions for dual auth (invoked via function URL)
      const dualAuthConditions: Record<string, Record<string, unknown>> = {
        Bool: {
          'lambda:InvokedViaFunctionUrl': true,
        },
      };

      // proceed to grant invokefunction for FURL Dual auth
      grant = this.grant(
        grantee,
        identifierDualAuth,
        'lambda:InvokeFunction',
        [this.functionArn],
        {
          invokedViaFunctionUrl: true,
        },
        dualAuthConditions,
      );

      this._functionUrlInvocationGrants[identifier] = grant;
    }
    return grant;
  }

  /**
   * Grant multiple principals the ability to invoke this Lambda via CompositePrincipal
   *
   * [disable-awslint:no-grants]
   */
  public grantInvokeCompositePrincipal(compositePrincipal: iam.CompositePrincipal): iam.Grant[] {
    return compositePrincipal.principals.map((principal) => this.grantInvoke(principal));
  }

  public addEventSource(source: IEventSource) {
    if (this.tenancyConfig?.tenancyConfigProperty?.tenantIsolationMode !== undefined && source.constructor.name !== 'ApiEventSource') {
      throw new ValidationError(lit`EventSourceNotSupportedForTenantIsolation`, `Event source ${source.constructor.name} is not supported for functions with tenant isolation mode`, this);
    }
    source.bind(this);
  }

  public configureAsyncInvoke(options: EventInvokeConfigOptions): void {
    if (this.node.tryFindChild('EventInvokeConfig') !== undefined) {
      throw new ValidationError(lit`EventInvokeConfigAlreadyConfigured`, `An EventInvokeConfig has already been configured for the function at ${this.node.path}`, this);
    }

    new EventInvokeConfig(this, 'EventInvokeConfig', {
      function: this,
      ...options,
    });
  }

  public addFunctionUrl(options?: FunctionUrlOptions): FunctionUrl {
    return new FunctionUrl(this, 'FunctionUrl', {
      function: this,
      ...options,
    });
  }

  /**
   * Returns the construct tree node that corresponds to the lambda function.
   * For use internally for constructs, when the tree is set up in non-standard ways. Ex: SingletonFunction.
   * @internal
   */
  protected _functionNode(): Node {
    return this.node;
  }

  /**
   * Given the function arn, check if the account id matches this account
   *
   * Function ARNs look like this:
   *
   *   arn:aws:lambda:region:account-id:function:function-name
   *
   * ..which means that in order to extract the `account-id` component from the ARN, we can
   * split the ARN using ":" and select the component in index 4.
   *
   * @returns true if account id of function matches the account specified on the stack, false otherwise.
   *
   * @internal
   */
  protected _isStackAccount(): boolean {
    if (Token.isUnresolved(this.stack.account) || Token.isUnresolved(this.functionArn)) {
      return false;
    }
    return this.stack.splitArn(this.functionArn, ArnFormat.SLASH_RESOURCE_NAME).account === this.stack.account;
  }

  private grant(
    grantee: iam.IGrantable,
    identifier:string,
    action: string,
    resourceArns: string[],
    permissionOverrides?: Partial<Permission>,
    conditions?: Record<string, Record<string, unknown>>,
  ): iam.Grant {
    const grant = iam.Grant.addToPrincipalOrResource({
      grantee,
      actions: [action],
      resourceArns,
      conditions,

      // Fake resource-like object on which to call addToResourcePolicy(), which actually
      // calls addPermission()
      resource: {
        addToResourcePolicy: (_statement) => {
          // Couldn't add permissions to the principal, so add them locally.
          this.addPermission(identifier, {
            principal: grantee.grantPrincipal!,
            action: action,
            ...permissionOverrides,
          });

          const permissionNode = this._functionNode().tryFindChild(identifier);
          if (!permissionNode && !this._skipPermissions) {
            throw new ValidationError(lit`CannotModifyLambdaPermission`, 'Cannot modify permission to lambda function. Function is either imported or $LATEST version.\n'
              + 'If the function is imported from the same account use `fromFunctionAttributes()` API with the `sameEnvironment` flag.\n'
              + 'If the function is imported from a different account and already has the correct permissions use `fromFunctionAttributes()` API with the `skipPermissions` flag.', this);
          }
          return { statementAdded: true, policyDependable: permissionNode };
        },
        env: this.env,
      },
    });

    return grant;
  }

  /**
   * Translate IPrincipal to something we can pass to AWS::Lambda::Permissions
   *
   * Do some nasty things because `Permission` supports a subset of what the
   * full IAM principal language supports, and we may not be able to parse strings
   * outright because they may be tokens.
   *
   * Try to recognize some specific Principal classes first, then try a generic
   * fallback.
   */
  private parsePermissionPrincipal(principal: iam.IPrincipal | { readonly wrapped: iam.IPrincipal }) {
    // Try some specific common classes first.
    // use duck-typing, not instance of
    if ('wrapped' in principal) {
      // eslint-disable-next-line dot-notation
      principal = principal['wrapped'];
    }

    if ('accountId' in principal) {
      return (principal as iam.AccountPrincipal).accountId;
    }

    if ('service' in principal) {
      return (principal as iam.ServicePrincipal).service;
    }

    if ('arn' in principal) {
      return (principal as iam.ArnPrincipal).arn;
    }

    const stringEquals = matchSingleKey('StringEquals', principal.policyFragment.conditions);
    if (stringEquals) {
      const orgId = matchSingleKey('aws:PrincipalOrgID', stringEquals);
      if (orgId) {
        // we will move the organization id to the `principalOrgId` property of `Permissions`.
        return '*';
      }
    }

    // Try a best-effort approach to support simple principals that are not any of the predefined
    // classes, but are simple enough that they will fit into the Permission model. Main target
    // here: imported Roles, Users, Groups.
    //
    // The principal cannot have conditions and must have a single { AWS: [arn] } entry.
    const json = principal.policyFragment.principalJson;
    if (Object.keys(principal.policyFragment.conditions).length === 0 && json.AWS) {
      if (typeof json.AWS === 'string') { return json.AWS; }
      if (Array.isArray(json.AWS) && json.AWS.length === 1 && typeof json.AWS[0] === 'string') {
        return json.AWS[0];
      }
    }

    throw new ValidationError(lit`InvalidPrincipalTypeForLambdaPermission`, `Invalid principal type for Lambda permission statement: ${principal.constructor.name}. ` +
      'Supported: AccountPrincipal, ArnPrincipal, ServicePrincipal, OrganizationPrincipal', this);

    /**
     * Returns the value at the key if the object contains the key and nothing else. Otherwise,
     * returns undefined.
     */
    function matchSingleKey(key: string, obj: Record<string, any>): any | undefined {
      if (Object.keys(obj).length !== 1) { return undefined; }

      return obj[key];
    }
  }

  private validateConditionCombinations(principal: iam.IPrincipal): {
    sourceArn: string | undefined;
    sourceAccount: string | undefined;
    principalOrgID: string | undefined;
  } | undefined {
    const conditions = this.validateConditions(principal);

    if (!conditions) { return undefined; }

    const sourceArn = requireString(requireObject(conditions.ArnLike)?.['aws:SourceArn']);
    const sourceAccount = requireString(requireObject(conditions.StringEquals)?.['aws:SourceAccount']);
    const principalOrgID = requireString(requireObject(conditions.StringEquals)?.['aws:PrincipalOrgID']);

    // PrincipalOrgID cannot be combined with any other conditions
    if (principalOrgID && (sourceArn || sourceAccount)) {
      throw new ValidationError(lit`PrincipalOrgIdCannotBeCombinedWithOtherConditions`, 'PrincipalWithConditions had unsupported condition combinations for Lambda permission statement: principalOrgID cannot be set with other conditions.', this);
    }

    return {
      sourceArn,
      sourceAccount,
      principalOrgID,
    };
  }

  private validateConditions(principal: iam.IPrincipal): iam.Conditions | undefined {
    if (this.isPrincipalWithConditions(principal)) {
      const conditions: iam.Conditions = principal.policyFragment.conditions;
      const conditionPairs = flatMap(
        Object.entries(conditions),
        ([operator, conditionObjs]) => Object.keys(conditionObjs as object).map(key => { return { operator, key }; }),
      );

      // These are all the supported conditions. Some combinations are not supported,
      // like only 'aws:SourceArn' or 'aws:PrincipalOrgID' and 'aws:SourceAccount'.
      // These will be validated through `this.validateConditionCombinations`.
      const supportedPrincipalConditions = [{
        operator: 'ArnLike',
        key: 'aws:SourceArn',
      }, {
        operator: 'StringEquals',
        key: 'aws:SourceAccount',
      }, {
        operator: 'StringEquals',
        key: 'aws:PrincipalOrgID',
      }];

      const unsupportedConditions = conditionPairs.filter(
        (condition) => !supportedPrincipalConditions.some(
          (supportedCondition) => supportedCondition.operator === condition.operator && supportedCondition.key === condition.key,
        ),
      );

      if (unsupportedConditions.length == 0) {
        return conditions;
      } else {
        throw new ValidationError(lit`UnsupportedConditionsForLambdaPermission`, `PrincipalWithConditions had unsupported conditions for Lambda permission statement: ${JSON.stringify(unsupportedConditions)}. ` +
          `Supported operator/condition pairs: ${JSON.stringify(supportedPrincipalConditions)}`, this);
      }
    }

    return undefined;
  }

  private isPrincipalWithConditions(principal: iam.IPrincipal): boolean {
    return Object.keys(principal.policyFragment.conditions).length > 0;
  }

  /**
   * Check if a policy statement contains array tokens that would cause CloudFormation
   * resolution conflicts when mixed with literal arrays in the same policy document.
   *
   * Array tokens are created by CloudFormation intrinsic functions that return arrays,
   * such as Fn::Split, Fn::GetAZs, etc. These cannot be safely merged with literal
   * resource arrays due to CloudFormation's token resolution limitations.
   *
   * Individual string tokens within literal arrays (e.g., `["arn:${token}:..."]`) are
   * safe and do not cause conflicts, so they are not detected by this method.
   * @internal
   */
  private statementHasArrayTokens(statement: iam.PolicyStatement): boolean {
    const resources = statement.resources;
    if (!resources) {
      return false;
    }

    // Detect when the entire resources property is an unresolved token that will
    // resolve to an array at CloudFormation deployment time.
    return Token.isUnresolved(resources);
  }
}

export abstract class QualifiedFunctionBase extends FunctionBase {
  /** The underlying `IFunction` */
  public abstract readonly lambda: IFunction;

  public readonly permissionsNode = this.node;

  /**
   * The qualifier of the version or alias of this function.
   * A qualifier is the identifier that's appended to a version or alias ARN.
   * @see https://docs.aws.amazon.com/lambda/latest/dg/API_GetFunctionConfiguration.html#API_GetFunctionConfiguration_RequestParameters
   */
  protected abstract readonly qualifier: string;

  public get latestVersion() {
    return this.lambda.latestVersion;
  }

  public get tenancyConfig() {
    return this.lambda.tenancyConfig;
  }

  public get resourceArnsForGrantInvoke() {
    return [this.functionArn];
  }

  public configureAsyncInvoke(options: EventInvokeConfigOptions): void {
    if (this.node.tryFindChild('EventInvokeConfig') !== undefined) {
      throw new ValidationError(lit`EventInvokeConfigAlreadyConfiguredForQualifiedFunction`, `An EventInvokeConfig has already been configured for the qualified function at ${this.node.path}`, this);
    }

    new EventInvokeConfig(this, 'EventInvokeConfig', {
      function: this.lambda,
      qualifier: this.qualifier,
      ...options,
    });
  }

  public considerWarningOnInvokeFunctionPermissions(_scope: Construct, _action: string): void {
    // noOp
    return;
  }
}

/**
 * The $LATEST version of a function, useful when attempting to create aliases.
 */
class LatestVersion extends FunctionBase implements IVersion {
  public readonly lambda: IFunction;
  public readonly version = '$LATEST';
  public readonly permissionsNode = this.node;

  protected readonly canCreatePermissions = false;

  constructor(lambda: FunctionBase) {
    super(lambda, '$LATEST');
    this.lambda = lambda;
  }

  public get versionRef(): VersionReference {
    return {
      functionArn: this.functionArn,
    };
  }

  public get functionRef(): FunctionReference {
    return {
      functionArn: this.functionArn,
      functionName: this.functionName,
    };
  }

  public get functionArn() {
    return `${this.lambda.functionArn}:${this.version}`;
  }

  public get functionName() {
    return `${this.lambda.functionName}:${this.version}`;
  }

  public get architecture() {
    return this.lambda.architecture;
  }

  public get grantPrincipal() {
    return this.lambda.grantPrincipal;
  }

  public get latestVersion() {
    return this;
  }

  public get role() {
    return this.lambda.role;
  }

  public get tenancyConfig() {
    return this.lambda.tenancyConfig;
  }

  public get edgeArn(): never {
    throw new ValidationError(lit`LatestVersionCannotBeUsedForLambdaEdge`, '$LATEST function version cannot be used for Lambda@Edge', this);
  }

  public get resourceArnsForGrantInvoke() {
    return [this.functionArn];
  }

  @MethodMetadata()
  public addAlias(aliasName: string, options: AliasOptions = {}) {
    return addAlias(this, this, aliasName, options);
  }
}

function requireObject(x: unknown): Record<string, unknown> | undefined {
  return x && typeof x === 'object' && !Array.isArray(x) ? x as any : undefined;
}

function requireString(x: unknown): string | undefined {
  return x && typeof x === 'string' ? x : undefined;
}

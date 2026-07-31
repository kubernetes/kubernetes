import type { IConstruct } from 'constructs';
import { Group } from './group';
import type { IPrincipal, PrincipalPolicyFragment, ServicePrincipalOpts } from './principals';
import {
  AccountPrincipal, AccountRootPrincipal, AnyPrincipal, ArnPrincipal, CanonicalUserPrincipal,
  FederatedPrincipal, PrincipalBase, ServicePrincipal, validateConditionObject,
} from './principals';
import { normalizeStatement } from './private/postprocess-policy-document';
import { LITERAL_STRING_KEY, mergePrincipal, sum } from './private/util';
import * as cdk from '../../core';
import { UnscopedValidationError } from '../../core';
import { lit } from '../../core/lib/private/literal-string';

/**
 * Options for resource-based policy validation
 */
export interface ResourcePolicyValidationOptions {
  /**
   * Whether to skip resource validation for policies where resources are implicit
   * (e.g., ECR repository policies where the resource is the repository itself)
   *
   * @default false
   */
  readonly skipResourceValidation?: boolean;
}

const ensureArrayOrUndefined = (field: any) => {
  if (field === undefined) {
    return undefined;
  }
  if (typeof (field) !== 'string' && !Array.isArray(field)) {
    throw new UnscopedValidationError(lit`MustBeFieldsEitherString`, 'Fields must be either a string or an array of strings');
  }
  if (Array.isArray(field) && !!field.find((f: any) => typeof (f) !== 'string')) {
    throw new UnscopedValidationError(lit`MustBeFieldsEitherString`, 'Fields must be either a string or an array of strings');
  }
  return Array.isArray(field) ? field : [field];
};

/**
 * An estimate on how long ARNs typically are
 *
 * This is used to decide when to start splitting statements into new Managed Policies.
 * Because we often can't know the length of an ARN (it may be a token and only
 * available at deployment time) we'll have to estimate it.
 *
 * The estimate can be overridden by setting the `@aws-cdk/aws-iam.arnSizeEstimate` context key.
 */
const DEFAULT_ARN_SIZE_ESTIMATE = 150;

/**
 * Context key which can be used to override the estimated length of unresolved ARNs.
 */
const ARN_SIZE_ESTIMATE_CONTEXT_KEY = '@aws-cdk/aws-iam.arnSizeEstimate';

/**
 * Represents a statement in an IAM policy document.
 */
export class PolicyStatement {
  /**
   * Uniquely identifies this class.
   */
  public static readonly PROPERTY_INJECTION_ID: string = 'aws-cdk-lib.aws-iam.PolicyStatement';

  /**
   * Creates a new PolicyStatement based on the object provided.
   * This will accept an object created from the `.toJSON()` call
   * @param obj the PolicyStatement in object form.
   */
  public static fromJson(obj: any) {
    const ret = new PolicyStatement({
      sid: obj.Sid,
      actions: ensureArrayOrUndefined(obj.Action),
      resources: ensureArrayOrUndefined(obj.Resource),
      conditions: obj.Condition,
      effect: obj.Effect,
      notActions: ensureArrayOrUndefined(obj.NotAction),
      notResources: ensureArrayOrUndefined(obj.NotResource),
      principals: obj.Principal ? [new JsonPrincipal(obj.Principal)] : undefined,
      notPrincipals: obj.NotPrincipal ? [new JsonPrincipal(obj.NotPrincipal)] : undefined,
    });

    // validate that the PolicyStatement has the correct shape
    const errors = ret.validateForAnyPolicy();
    if (errors.length > 0) {
      throw new UnscopedValidationError(lit`IncorrectPolicyStatement`, 'Incorrect Policy Statement: ' + errors.join('\n'));
    }

    return ret;
  }

  private readonly _action = new OrderedSet<string>();
  private readonly _notAction = new OrderedSet<string>();
  private readonly _principal: { [key: string]: any[] } = {};
  private readonly _notPrincipal: { [key: string]: any[] } = {};
  private readonly _resource = new OrderedSet<string>();
  private readonly _notResource = new OrderedSet<string>();
  private readonly _condition: { [key: string]: any } = {};
  private _sid?: string;
  private _effect: Effect;
  private principalConditionsJson?: string;

  // Hold on to those principals
  private readonly _principals = new OrderedSet<IPrincipal>();
  private readonly _notPrincipals = new OrderedSet<IPrincipal>();
  private _frozen = false;

  constructor(props: PolicyStatementProps = {}) {
    this._sid = props.sid;
    this._effect = props.effect || Effect.ALLOW;

    this.addActions(...props.actions || []);
    this.addNotActions(...props.notActions || []);
    this.addPrincipals(...props.principals || []);
    this.addNotPrincipals(...props.notPrincipals || []);
    this.addResources(...props.resources || []);
    this.addNotResources(...props.notResources || []);
    if (props.conditions !== undefined) {
      this.addConditions(props.conditions);
    }
  }

  /**
   * Statement ID for this statement
   */
  public get sid(): string | undefined {
    return this._sid;
  }

  /**
   * Set Statement ID for this statement
   */
  public set sid(sid: string | undefined) {
    this.assertNotFrozen('sid');
    this._sid = sid;
  }

  /**
   * Whether to allow or deny the actions in this statement
   */
  public get effect(): Effect {
    return this._effect;
  }

  /**
   * Set effect for this statement
   */
  public set effect(effect: Effect) {
    this.assertNotFrozen('effect');
    this._effect = effect;
  }

  //
  // Actions
  //

  /**
   * Specify allowed actions into the "Action" section of the policy statement.
   *
   * @see https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_elements_action.html
   *
   * @param actions actions that will be allowed.
   */
  public addActions(...actions: string[]) {
    this.assertNotFrozen('addActions');
    if (actions.length > 0 && this._notAction.length > 0) {
      throw new UnscopedValidationError(lit`CannotAddActionsPolicyStatement`, 'Cannot add \'Actions\' to policy statement if \'NotActions\' have been added');
    }
    this.validatePolicyActions(actions);
    this._action.push(...actions);
  }

  /**
   * Explicitly allow all actions except the specified list of actions into the "NotAction" section
   * of the policy document.
   *
   * @see https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_elements_notaction.html
   *
   * @param notActions actions that will be denied. All other actions will be permitted.
   */
  public addNotActions(...notActions: string[]) {
    this.assertNotFrozen('addNotActions');
    if (notActions.length > 0 && this._action.length > 0) {
      throw new UnscopedValidationError(lit`CannotAddActionsPolicyStatement`, 'Cannot add \'NotActions\' to policy statement if \'Actions\' have been added');
    }
    this.validatePolicyActions(notActions);
    this._notAction.push(...notActions);
  }

  //
  // Principal
  //

  /**
   * Indicates if this permission has a "Principal" section.
   */
  public get hasPrincipal() {
    return this._principals.length + this._notPrincipals.length > 0;
  }

  /**
   * Adds principals to the "Principal" section of a policy statement.
   *
   * @see https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_elements_principal.html
   *
   * @param principals IAM principals that will be added
   */
  public addPrincipals(...principals: IPrincipal[]) {
    this.assertNotFrozen('addPrincipals');
    if (principals.length > 0 && this._notPrincipals.length > 0) {
      throw new UnscopedValidationError(lit`CannotAddPrincipalsPolicyStatement`, 'Cannot add \'Principals\' to policy statement if \'NotPrincipals\' have been added');
    }
    for (const principal of principals) {
      this.validatePolicyPrincipal(principal);
    }

    const added = this._principals.push(...principals);
    for (const principal of added) {
      const fragment = principal.policyFragment;
      mergePrincipal(this._principal, fragment.principalJson);
      this.addPrincipalConditions(fragment.conditions);
    }
  }

  /**
   * Specify principals that is not allowed or denied access to the "NotPrincipal" section of
   * a policy statement.
   *
   * @see https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_elements_notprincipal.html
   *
   * @param notPrincipals IAM principals that will be denied access
   */
  public addNotPrincipals(...notPrincipals: IPrincipal[]) {
    this.assertNotFrozen('addNotPrincipals');
    if (notPrincipals.length > 0 && this._principals.length > 0) {
      throw new UnscopedValidationError(lit`CannotAddPrincipalsPolicyStatement`, 'Cannot add \'NotPrincipals\' to policy statement if \'Principals\' have been added');
    }
    for (const notPrincipal of notPrincipals) {
      this.validatePolicyPrincipal(notPrincipal);
    }

    const added = this._notPrincipals.push(...notPrincipals);
    for (const notPrincipal of added) {
      const fragment = notPrincipal.policyFragment;
      mergePrincipal(this._notPrincipal, fragment.principalJson);
      this.addPrincipalConditions(fragment.conditions);
    }
  }

  private validatePolicyActions(actions: string[]) {
    // In case of an unresolved list of actions return early
    if (cdk.Token.isUnresolved(actions)) return;
    for (const action of actions || []) {
      if (!cdk.Token.isUnresolved(action) && !/^(\*|[a-zA-Z0-9-]+:[a-zA-Z0-9*]+)$/.test(action)) {
        throw new UnscopedValidationError(lit`ActionInvalid`, `Action '${action}' is invalid. An action string consists of a service namespace, a colon, and the name of an action. Action names can include wildcards.`);
      }
    }
  }

  private validatePolicyPrincipal(principal: IPrincipal) {
    if (principal instanceof Group) {
      throw new UnscopedValidationError(lit`CannotUseGroupAsPrincipal`, 'Cannot use an IAM Group as the \'Principal\' or \'NotPrincipal\' in an IAM Policy');
    }
  }

  /**
   * Specify AWS account ID as the principal entity to the "Principal" section of a policy statement.
   */
  public addAwsAccountPrincipal(accountId: string) {
    this.addPrincipals(new AccountPrincipal(accountId));
  }

  /**
   * Specify a principal using the ARN  identifier of the principal.
   * You cannot specify IAM groups and instance profiles as principals.
   *
   * @param arn ARN identifier of AWS account, IAM user, or IAM role (i.e. arn:aws:iam::123456789012:user/user-name)
   */
  public addArnPrincipal(arn: string) {
    this.addPrincipals(new ArnPrincipal(arn));
  }

  /**
   * Adds a service principal to this policy statement.
   *
   * @param service the service name for which a service principal is requested (e.g: `s3.amazonaws.com`).
   * @param opts    options for adding the service principal (such as specifying a principal in a different region)
   */
  public addServicePrincipal(service: string, opts?: ServicePrincipalOpts) {
    this.addPrincipals(new ServicePrincipal(service, opts));
  }

  /**
   * Adds a federated identity provider such as Amazon Cognito to this policy statement.
   *
   * @param federated federated identity provider (i.e. 'cognito-identity.amazonaws.com')
   * @param conditions The conditions under which the policy is in effect.
   *   See [the IAM documentation](https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_elements_condition.html).
   */
  public addFederatedPrincipal(federated: any, conditions: Conditions) {
    this.addPrincipals(new FederatedPrincipal(federated, conditions));
  }

  /**
   * Adds an AWS account root user principal to this policy statement
   */
  public addAccountRootPrincipal() {
    this.addPrincipals(new AccountRootPrincipal());
  }

  /**
   * Adds a canonical user ID principal to this policy document
   *
   * @param canonicalUserId unique identifier assigned by AWS for every account
   */
  public addCanonicalUserPrincipal(canonicalUserId: string) {
    this.addPrincipals(new CanonicalUserPrincipal(canonicalUserId));
  }

  /**
   * Adds all identities in all accounts ("*") to this policy statement
   */
  public addAnyPrincipal() {
    this.addPrincipals(new AnyPrincipal());
  }

  //
  // Resources
  //

  /**
   * Specify resources that this policy statement applies into the "Resource" section of
   * this policy statement.
   *
   * @see https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_elements_resource.html
   *
   * @param arns Amazon Resource Names (ARNs) of the resources that this policy statement applies to
   */
  public addResources(...arns: string[]) {
    this.assertNotFrozen('addResources');
    if (arns.length > 0 && this._notResource.length > 0) {
      throw new UnscopedValidationError(lit`CannotAddResourcesPolicyStatement`, 'Cannot add \'Resources\' to policy statement if \'NotResources\' have been added');
    }
    this._resource.push(...arns);
  }

  /**
   * Specify resources that this policy statement will not apply to in the "NotResource" section
   * of this policy statement. All resources except the specified list will be matched.
   *
   * @see https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_elements_notresource.html
   *
   * @param arns Amazon Resource Names (ARNs) of the resources that this policy statement does not apply to
   */
  public addNotResources(...arns: string[]) {
    this.assertNotFrozen('addNotResources');
    if (arns.length > 0 && this._resource.length > 0) {
      throw new UnscopedValidationError(lit`CannotAddResourcesPolicyStatement`, 'Cannot add \'NotResources\' to policy statement if \'Resources\' have been added');
    }
    this._notResource.push(...arns);
  }

  /**
   * Adds a ``"*"`` resource to this statement.
   */
  public addAllResources() {
    this.addResources('*');
  }

  /**
   * Indicates if this permission has at least one resource associated with it.
   */
  public get hasResource() {
    return this._resource && this._resource.length > 0;
  }

  //
  // Condition
  //

  /**
   * Add a condition to the Policy
   *
   * If multiple calls are made to add a condition with the same operator and field, only
   * the last one wins. For example:
   *
   * ```ts
   * declare const stmt: iam.PolicyStatement;
   *
   * stmt.addCondition('StringEquals', { 'aws:SomeField': '1' });
   * stmt.addCondition('StringEquals', { 'aws:SomeField': '2' });
   * ```
   *
   * Will end up with the single condition `StringEquals: { 'aws:SomeField': '2' }`.
   *
   * If you meant to add a condition to say that the field can be *either* `1` or `2`, write
   * this:
   *
   * ```ts
   * declare const stmt: iam.PolicyStatement;
   *
   * stmt.addCondition('StringEquals', { 'aws:SomeField': ['1', '2'] });
   * ```
   */
  public addCondition(key: string, value: Condition) {
    this.assertNotFrozen('addCondition');
    validateConditionObject(value);

    const existingValue = this._condition[key];
    this._condition[key] = existingValue ? { ...existingValue, ...value } : value;
  }

  /**
   * Add multiple conditions to the Policy
   *
   * See the `addCondition` function for a caveat on calling this method multiple times.
   */
  public addConditions(conditions: Conditions) {
    Object.keys(conditions).map(key => {
      this.addCondition(key, conditions[key]);
    });
  }

  /**
   * Add a `StringEquals` condition that limits to a given account from `sts:ExternalId`.
   *
   * This method can only be called once: subsequent calls will overwrite earlier calls.
   *
   * @see https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_create_for-user_externalid.html
   */
  public addAccountCondition(accountId: string) {
    this.addCondition('StringEquals', { 'sts:ExternalId': accountId });
  }

  /**
   * Add an `StringEquals` condition that limits to a given account from `aws:SourceAccount`.
   *
   * This method can only be called once: subsequent calls will overwrite earlier calls.
   *
   * @see https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_condition-keys.html#condition-keys-sourceaccount
   */
  public addSourceAccountCondition(accountId: string) {
    this.addCondition('StringEquals', { 'aws:SourceAccount': accountId });
  }

  /**
   * Add an `ArnEquals` condition that limits to a given resource arn from `aws:SourceArn`.
   *
   * This method can only be called once: subsequent calls will overwrite earlier calls.
   *
   * @see https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_condition-keys.html#condition-keys-sourcearn
   */
  public addSourceArnCondition(arn: string) {
    this.addCondition('ArnEquals', { 'aws:SourceArn': arn });
  }

  /**
   * Create a new `PolicyStatement` with the same exact properties
   * as this one, except for the overrides
   */
  public copy(overrides: PolicyStatementProps = {}) {
    return new PolicyStatement({
      sid: overrides.sid ?? this.sid,
      effect: overrides.effect ?? this.effect,
      actions: overrides.actions ?? this.actions,
      notActions: overrides.notActions ?? this.notActions,

      principals: overrides.principals ?? this.principals,
      notPrincipals: overrides.notPrincipals ?? this.notPrincipals,

      resources: overrides.resources ?? this.resources,
      notResources: overrides.notResources ?? this.notResources,

      conditions: overrides.conditions ?? this.conditions,
    });
  }

  /**
   * JSON-ify the policy statement
   *
   * Used when JSON.stringify() is called
   */
  public toStatementJson(): any {
    return normalizeStatement({
      Action: this._action.direct(),
      NotAction: this._notAction.direct(),
      Condition: this._condition,
      Effect: this.effect,
      Principal: this._principal,
      NotPrincipal: this._notPrincipal,
      Resource: this._resource.direct(),
      NotResource: this._notResource.direct(),
      Sid: this.sid,
    });
  }

  /**
   * String representation of this policy statement
   */
  public toString() {
    return cdk.Token.asString(this, {
      displayHint: 'PolicyStatement',
    });
  }

  /**
   * JSON-ify the statement
   *
   * Used when JSON.stringify() is called
   */
  public toJSON() {
    return this.toStatementJson();
  }

  /**
   * Add a principal's conditions
   *
   * For convenience, principals have been modeled as both a principal
   * and a set of conditions. This makes it possible to have a single
   * object represent e.g. an "SNS Topic" (SNS service principal + aws:SourcArn
   * condition) or an Organization member (* + aws:OrgId condition).
   *
   * However, when using multiple principals in the same policy statement,
   * they must all have the same conditions or the OR samentics
   * implied by a list of principals cannot be guaranteed (user needs to
   * add multiple statements in that case).
   */
  private addPrincipalConditions(conditions: Conditions) {
    // Stringifying the conditions is an easy way to do deep equality
    const theseConditions = JSON.stringify(conditions);
    if (this.principalConditionsJson === undefined) {
      // First principal, anything goes
      this.principalConditionsJson = theseConditions;
    } else {
      if (this.principalConditionsJson !== theseConditions) {
        throw new UnscopedValidationError(lit`PrincipalsPolicystatementSameConditions`, `All principals in a PolicyStatement must have the same Conditions (got '${this.principalConditionsJson}' and '${theseConditions}'). Use multiple statements instead.`);
      }
    }
    this.addConditions(conditions);
  }

  /**
   * Validate that the policy statement satisfies base requirements for a policy.
   *
   * @returns An array of validation error messages, or an empty array if the statement is valid.
   */
  public validateForAnyPolicy(): string[] {
    const errors = new Array<string>();
    if (this._action.length === 0 && this._notAction.length === 0) {
      errors.push('A PolicyStatement must specify at least one \'action\' or \'notAction\'.');
    }
    return errors;
  }

  /**
   * Validate that the policy statement satisfies all requirements for a resource-based policy.
   *
   * @param options Optional validation options
   * @returns An array of validation error messages, or an empty array if the statement is valid.
   */
  public validateForResourcePolicy(options?: ResourcePolicyValidationOptions): string[] {
    const errors = this.validateForAnyPolicy();
    if (this._principals.length === 0 && this._notPrincipals.length === 0) {
      errors.push('A PolicyStatement used in a resource-based policy must specify at least one IAM principal.');
    }
    // Only validate resources if not explicitly skipped (for services like ECR where resources are implicit)
    if (!options?.skipResourceValidation && this._resource.length === 0 && this._notResource.length === 0) {
      errors.push('A PolicyStatement used in a resource-based policy must specify at least one resource.');
    }
    return errors;
  }

  /**
   * Validate that the policy statement satisfies all requirements for a trust policy (assume role policy).
   *
   * Trust policies are a special type of resource-based policy where the resource is implicit (the role itself).
   *
   * @returns An array of validation error messages, or an empty array if the statement is valid.
   */
  public validateForTrustPolicy(): string[] {
    const errors = this.validateForAnyPolicy();
    if (this._principals.length === 0 && this._notPrincipals.length === 0) {
      errors.push('A PolicyStatement used in a trust policy must specify at least one IAM principal.');
    }
    // Note: Trust policies do not require resources because the resource is implicit (the role itself)
    return errors;
  }

  /**
   * Validate that the policy statement satisfies all requirements for an identity-based policy.
   *
   * @returns An array of validation error messages, or an empty array if the statement is valid.
   */
  public validateForIdentityPolicy(): string[] {
    const errors = this.validateForAnyPolicy();
    if (this._principals.length > 0 || this._notPrincipals.length > 0) {
      errors.push('A PolicyStatement used in an identity-based policy cannot specify any IAM principals.');
    }
    if (this._resource.length === 0 && this._notResource.length === 0) {
      errors.push('A PolicyStatement used in an identity-based policy must specify at least one resource.');
    }
    if (this._sid !== undefined && !cdk.Token.isUnresolved(this._sid) && !/^[0-9A-Za-z]*$/.test(this._sid)) {
      errors.push(`Statement ID (sid) '${this._sid}' must be alphanumeric (A-Z, a-z, 0-9) when used in an identity-based policy. See https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_elements_sid.html`);
    }
    return errors;
  }

  /**
   * The Actions added to this statement
   */
  public get actions() {
    return this._action.copy();
  }

  /**
   * The NotActions added to this statement
   */
  public get notActions() {
    return this._notAction.copy();
  }

  /**
   * The Principals added to this statement
   */
  public get principals(): IPrincipal[] {
    return this._principals.copy();
  }

  /**
   * The NotPrincipals added to this statement
   */
  public get notPrincipals(): IPrincipal[] {
    return this._notPrincipals.copy();
  }

  /**
   * The Resources added to this statement
   */
  public get resources() {
    return this._resource.copy();
  }

  /**
   * The NotResources added to this statement
   */
  public get notResources() {
    return this._notResource.copy();
  }

  /**
   * The conditions added to this statement
   */
  public get conditions(): any {
    return { ...this._condition };
  }

  /**
   * Make the PolicyStatement immutable
   *
   * After calling this, any of the `addXxx()` methods will throw an exception.
   *
   * Libraries that lazily generate statement bodies can override this method to
   * fill the actual PolicyStatement fields. Be aware that this method may be called
   * multiple times.
   */
  public freeze(): PolicyStatement {
    this._frozen = true;
    return this;
  }

  /**
   * Whether the PolicyStatement has been frozen
   *
   * The statement object is frozen when `freeze()` is called.
   */
  public get frozen(): boolean {
    return this._frozen;
  }

  /**
   * Estimate the size of this policy statement
   *
   * By necessity, this will not be accurate. We'll do our best to overestimate
   * so we won't have nasty surprises.
   *
   * @internal
   */
  public _estimateSize(options: EstimateSizeOptions): number {
    let ret = 0;

    const { actionEstimate, arnEstimate } = options;

    ret += `"Effect": "${this.effect}",`.length;

    count('Action', this.actions, actionEstimate);
    count('NotAction', this.notActions, actionEstimate);
    count('Resource', this.resources, arnEstimate);
    count('NotResource', this.notResources, arnEstimate);

    ret += this.principals.length * arnEstimate;
    ret += this.notPrincipals.length * arnEstimate;

    ret += JSON.stringify(this.conditions).length;
    return ret;

    function count(key: string, values: string[], tokenSize: number) {
      if (values.length > 0) {
        ret += key.length + 5 /* quotes, colon, brackets */ +
          sum(values.map(v => (cdk.Token.isUnresolved(v) ? tokenSize : v.length) + 3 /* quotes, separator */));
      }
    }
  }

  /**
   * Throw an exception when the object is frozen
   */
  private assertNotFrozen(method: string) {
    if (this._frozen) {
      throw new UnscopedValidationError(lit`FreezeCalledPolicystatementPreviously`, `${method}: freeze() has been called on this PolicyStatement previously, so it can no longer be modified`);
    }
  }
}

/**
 * The Effect element of an IAM policy
 *
 * @see https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_elements_effect.html
 */
export enum Effect {
  /**
   * Allows access to a resource in an IAM policy statement. By default, access to resources are denied.
   */
  ALLOW = 'Allow',

  /**
   * Explicitly deny access to a resource. By default, all requests are denied implicitly.
   *
   * @see https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_evaluation-logic.html
   */
  DENY = 'Deny',
}

/**
 * Condition for when an IAM policy is in effect. Maps from the keys in a request's context to
 * a string value or array of string values. See the Conditions interface for more details.
 */
export type Condition = unknown;

// NOTE! We would have liked to have typed this as `Record<string, unknown>`, but in some places
// of the code we are assuming we can pass a `CfnJson` object into where a `Condition` is expected,
// and that wouldn't typecheck anymore.
//
// Needs to be `unknown` instead of `any` so that the type of `Conditions` is
// `Record<string, unknown>`; if it had been `Record<string, any>`, TypeScript would have allowed
// passing an array into `conditions` arguments (where it needs to be a map).

/**
 * Conditions for when an IAM Policy is in effect, specified in the following structure:
 *
 * `{ "Operator": { "keyInRequestContext": "value" } }`
 *
 * The value can be either a single string value or an array of string values.
 *
 * For more information, including which operators are supported, see [the IAM
 * documentation](https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_elements_condition.html).
 */
export type Conditions = Record<string, Condition>;

/**
 * Interface for creating a policy statement
 */
export interface PolicyStatementProps {
  /**
   * The Sid (statement ID) is an optional identifier that you provide for the
   * policy statement. You can assign a Sid value to each statement in a
   * statement array. In services that let you specify an ID element, such as
   * SQS and SNS, the Sid value is just a sub-ID of the policy document's ID. In
   * IAM, the Sid value must be unique within a JSON policy. In IAM identity
   * policies, the Sid value must contain only alphanumeric characters.
   *
   * @default - no sid
   */
  readonly sid?: string;

  /**
   * List of actions to add to the statement
   *
   * @default - no actions
   */
  readonly actions?: string[];

  /**
   * List of not actions to add to the statement
   *
   * @default - no not-actions
   */
  readonly notActions?: string[];

  /**
   * List of principals to add to the statement
   *
   * @default - no principals
   */
  readonly principals?: IPrincipal[];

  /**
   * List of not principals to add to the statement
   *
   * @default - no not principals
   */
  readonly notPrincipals?: IPrincipal[];

  /**
   * Resource ARNs to add to the statement
   *
   * @default - no resources
   */
  readonly resources?: string[];

  /**
   * NotResource ARNs to add to the statement
   *
   * @default - no not-resources
   */
  readonly notResources?: string[];

  /**
   * Conditions to add to the statement
   *
   * @default - no condition
   */
  readonly conditions?: { [key: string]: any };

  /**
   * Whether to allow or deny the actions in this statement
   *
   * @default Effect.ALLOW
   */
  readonly effect?: Effect;
}

class JsonPrincipal extends PrincipalBase {
  public readonly policyFragment: PrincipalPolicyFragment;

  constructor(json: any = {}) {
    super();

    // special case: if principal is a string, turn it into a "LiteralString" principal,
    // so we render the exact same string back out.
    if (typeof (json) === 'string') {
      json = { [LITERAL_STRING_KEY]: [json] };
    }
    if (typeof(json) !== 'object') {
      throw new UnscopedValidationError(lit`ExpectedPrincipalObject`, `JSON IAM principal should be an object, got ${JSON.stringify(json)}`);
    }

    this.policyFragment = {
      principalJson: json,
      conditions: {},
    };
  }

  public dedupeString(): string | undefined {
    return JSON.stringify(this.policyFragment);
  }
}

/**
 * Options for _estimateSize
 *
 * These can optionally come from context, but it's too expensive to look
 * them up every time so we bundle them into a struct first.
 *
 * @internal
 */
export interface EstimateSizeOptions {
  /**
   * Estimated size of an unresolved ARN
   */
  readonly arnEstimate: number;

  /**
   * Estimated size of an unresolved action
   */
  readonly actionEstimate: number;
}

/**
 * Derive the size estimation options from context
 *
 * @internal
 */
export function deriveEstimateSizeOptions(scope: IConstruct): EstimateSizeOptions {
  const actionEstimate = 20;
  const arnEstimate = scope.node.tryGetContext(ARN_SIZE_ESTIMATE_CONTEXT_KEY) ?? DEFAULT_ARN_SIZE_ESTIMATE;
  if (typeof arnEstimate !== 'number') {
    throw new UnscopedValidationError(lit`ExpectedContextNumber`, `Context value ${ARN_SIZE_ESTIMATE_CONTEXT_KEY} should be a number, got ${JSON.stringify(arnEstimate)}`);
  }

  return { actionEstimate, arnEstimate };
}

/**
 * A class that behaves both as a set and an array
 *
 * Used for the elements of a PolicyStatement. In practice they behave as sets,
 * but we have thousands of snapshot tests in existence that will rely on a
 * particular order so we can't just change the type to `Set<>` wholesale without
 * causing a lot of churn.
 */
class OrderedSet<A> {
  private readonly set = new Set<A>();
  private readonly array = new Array<A>();

  /**
   * Add new elements to the set
   *
   * @param xs the elements to be added
   *
   * @returns the elements actually added
   */
  public push(...xs: readonly A[]): A[] {
    const ret = new Array<A>();
    for (const x of xs) {
      if (this.set.has(x)) {
        continue;
      }
      this.set.add(x);
      this.array.push(x);
      ret.push(x);
    }
    return ret;
  }

  public get length() {
    return this.array.length;
  }

  public copy(): A[] {
    return [...this.array];
  }

  /**
   * Direct (read-only) access to the underlying array
   *
   * (Saves a copy)
   */
  public direct(): readonly A[] {
    return this.array;
  }
}

import type { IConstruct } from 'constructs';
import type { ResourcePolicyValidationOptions } from './policy-statement';
import { PolicyStatement, deriveEstimateSizeOptions } from './policy-statement';
import { mergeStatements } from './private/merge-statements';
import { PostProcessPolicyDocument } from './private/postprocess-policy-document';
import * as cdk from '../../core';
import { lit } from '../../core/lib/private/literal-string';
import * as cxapi from '../../cx-api';

/**
 * Properties for a new PolicyDocument
 */
export interface PolicyDocumentProps {
  /**
   * Automatically assign Statement Ids to all statements
   *
   * @default false
   */
  readonly assignSids?: boolean;

  /**
   * Initial statements to add to the policy document
   *
   * @default - No statements
   */
  readonly statements?: PolicyStatement[];

  /**
   * Try to minimize the policy by merging statements
   *
   * To avoid overrunning the maximum policy size, combine statements if they produce
   * the same result. Merging happens according to the following rules:
   *
   * - The Effect of both statements is the same
   * - Neither of the statements have a 'Sid'
   * - Combine Principals if the rest of the statement is exactly the same.
   * - Combine Resources if the rest of the statement is exactly the same.
   * - Combine Actions if the rest of the statement is exactly the same.
   * - We will never combine NotPrincipals, NotResources or NotActions, because doing
   *   so would change the meaning of the policy document.
   *
   * @default - false, unless the feature flag `@aws-cdk/aws-iam:minimizePolicies` is set
   */
  readonly minimize?: boolean;
}

/**
 * A PolicyDocument is a collection of statements
 */
export class PolicyDocument implements cdk.IResolvable {
  /**
   * Creates a new PolicyDocument based on the object provided.
   * This will accept an object created from the `.toJSON()` call
   * @param obj the PolicyDocument in object form.
   */
  public static fromJson(obj: any): PolicyDocument {
    const newPolicyDocument = new PolicyDocument();
    const statement = obj.Statement ?? [];
    if (statement && !Array.isArray(statement)) {
      throw new cdk.UnscopedValidationError(lit`StatementMustBeArray`, 'Statement must be an array');
    }
    newPolicyDocument.addStatements(...obj.Statement.map((s: any) => PolicyStatement.fromJson(s)));
    return newPolicyDocument;
  }

  public readonly creationStack: string[] = ['Token stack traces are no longer captured'];
  private readonly statements = new Array<PolicyStatement>();
  private readonly autoAssignSids: boolean;
  private readonly minimize?: boolean;

  constructor(props: PolicyDocumentProps = {}) {
    this.autoAssignSids = !!props.assignSids;
    this.minimize = props.minimize;

    this.addStatements(...props.statements || []);
  }

  public resolve(context: cdk.IResolveContext): any {
    this.freezeStatements();
    this._maybeMergeStatements(context.scope);

    // In the previous implementation of 'merge', sorting of actions/resources on
    // a statement always happened, even  on singular statements. In the new
    // implementation of 'merge', sorting only happens when actually combining 2
    // statements. This affects all test snapshots, so we need to put in mechanisms
    // to avoid having to update all snapshots.
    //
    // To do sorting in a way compatible with the previous implementation of merging,
    // (so we don't have to update snapshots) do it after rendering, but only when
    // merging is enabled.
    const sort = this.shouldMerge(context.scope);
    context.registerPostProcessor(new PostProcessPolicyDocument(this.autoAssignSids, sort));
    return this.render();
  }

  /**
   * Whether the policy document contains any statements.
   */
  public get isEmpty(): boolean {
    return this.statements.length === 0;
  }

  /**
   * The number of statements already added to this policy.
   * Can be used, for example, to generate unique "sid"s within the policy.
   */
  public get statementCount(): number {
    return this.statements.length;
  }

  /**
   * Adds a statement to the policy document.
   *
   * @param statement the statement to add.
   */
  public addStatements(...statement: PolicyStatement[]) {
    this.statements.push(...statement);
  }

  /**
   * Encode the policy document as a string
   */
  public toString() {
    return cdk.Token.asString(this, {
      displayHint: 'PolicyDocument',
    });
  }

  /**
   * JSON-ify the document
   *
   * Used when JSON.stringify() is called
   */
  public toJSON() {
    return this.render();
  }

  /**
   * Validate that all policy statements in the policy document satisfies the
   * requirements for any policy.
   *
   * @see https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies.html#access_policies-json
   *
   * @returns An array of validation error messages, or an empty array if the document is valid.
   */
  public validateForAnyPolicy(): string[] {
    const errors = new Array<string>();
    for (const statement of this.statements) {
      errors.push(...statement.validateForAnyPolicy());
    }
    return errors;
  }

  /**
   * Validate that all policy statements in the policy document satisfies the
   * requirements for a resource-based policy.
   *
   * @see https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies.html#access_policies-json
   *
   * @param options Optional validation options
   * @returns An array of validation error messages, or an empty array if the document is valid.
   */
  public validateForResourcePolicy(options?: ResourcePolicyValidationOptions): string[] {
    const errors = new Array<string>();
    for (const statement of this.statements) {
      errors.push(...statement.validateForResourcePolicy(options));
    }
    return errors;
  }

  /**
   * Validate that all policy statements in the policy document satisfies the
   * requirements for a trust policy (assume role policy).
   *
   * Trust policies are a special type of resource-based policy where the resource is implicit (the role itself).
   *
   * @see https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_elements_principal.html
   *
   * @returns An array of validation error messages, or an empty array if the document is valid.
   */
  public validateForTrustPolicy(): string[] {
    const errors = new Array<string>();
    for (const statement of this.statements) {
      errors.push(...statement.validateForTrustPolicy());
    }
    return errors;
  }

  /**
   * Validate that all policy statements in the policy document satisfies the
   * requirements for an identity-based policy.
   *
   * @see https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies.html#access_policies-json
   *
   * @returns An array of validation error messages, or an empty array if the document is valid.
   */
  public validateForIdentityPolicy(): string[] {
    const errors = new Array<string>();
    for (const statement of this.statements) {
      errors.push(...statement.validateForIdentityPolicy());
    }
    return errors;
  }

  /**
   * Perform statement merging (if enabled and not done yet)
   *
   * @internal
   */
  public _maybeMergeStatements(scope: IConstruct): void {
    if (this.shouldMerge(scope)) {
      const result = mergeStatements(this.statements, { scope });
      this.statements.splice(0, this.statements.length, ...result.mergedStatements);
    }
  }

  /**
   * Split the statements of the PolicyDocument into multiple groups, limited by their size
   *
   * We do a round of size-limited merging first (making sure to not produce statements too
   * large to fit into standalone policies), so that we can most accurately estimate total
   * policy size. Another final round of minimization will be done just before rendering to
   * end up with minimal policies that look nice to humans.
   *
   * Return a map of the final set of policy documents, mapped to the ORIGINAL (pre-merge)
   * PolicyStatements that ended up in the given PolicyDocument.
   *
   * @internal
   */
  public _splitDocument(scope: IConstruct, selfMaximumSize: number, splitMaximumSize: number): Map<PolicyDocument, PolicyStatement[]> {
    const self = this;
    const newDocs: PolicyDocument[] = [];

    // Maps final statements to original statements
    this.freezeStatements();
    let statementsToOriginals = new Map(this.statements.map(s => [s, [s]]));

    // We always run 'mergeStatements' to minimize the policy before splitting.
    // However, we only 'merge' when the feature flag is on. If the flag is not
    // on, we only combine statements that are *exactly* the same. We must do
    // this before splitting, otherwise we may end up with the statement set [X,
    // X, X, X, X] being split off into [[X, X, X], [X, X]] before being reduced
    // to [[X], [X]] (but should have been just [[X]]).
    const doActualMerging = this.shouldMerge(scope);
    const result = mergeStatements(this.statements, { scope, limitSize: true, mergeIfCombinable: doActualMerging });
    this.statements.splice(0, this.statements.length, ...result.mergedStatements);
    statementsToOriginals = result.originsMap;

    const sizeOptions = deriveEstimateSizeOptions(scope);

    // Cache statement sizes to avoid recomputing them based on the fields
    const statementSizes = new Map<PolicyStatement, number>(this.statements.map(s => [s, s._estimateSize(sizeOptions)]));

    // Keep some size counters so we can avoid recomputing them based on the statements in each
    let selfSize = 0;
    const polSizes = new Map<PolicyDocument, number>();
    // Getter with a default to save some syntactic noise
    const polSize = (x: PolicyDocument) => polSizes.get(x) ?? 0;

    let i = 0;
    while (i < this.statements.length) {
      const statement = this.statements[i];

      const statementSize = statementSizes.get(statement) ?? 0;
      if (selfSize + statementSize < selfMaximumSize) {
        // Fits in self
        selfSize += statementSize;
        i++;
        continue;
      }

      // Split off to new PolicyDocument. Find the PolicyDocument we can add this to,
      // or add a fresh one.
      const addToDoc = findDocWithSpace(statementSize);
      addToDoc.addStatements(statement);
      polSizes.set(addToDoc, polSize(addToDoc) + statementSize);
      this.statements.splice(i, 1);
    }

    // Return the set of all policy document and original statements
    const ret = new Map<PolicyDocument, PolicyStatement[]>();
    ret.set(this, this.statements.flatMap(s => statementsToOriginals.get(s) ?? [s]));
    for (const newDoc of newDocs) {
      ret.set(newDoc, newDoc.statements.flatMap(s => statementsToOriginals.get(s) ?? [s]));
    }
    return ret;

    function findDocWithSpace(size: number) {
      let j = 0;
      while (j < newDocs.length && polSize(newDocs[j]) + size > splitMaximumSize) {
        j++;
      }
      if (j < newDocs.length) {
        return newDocs[j];
      }

      const newDoc = new PolicyDocument({
        assignSids: self.autoAssignSids,
        minimize: self.minimize,
      });
      newDocs.push(newDoc);
      return newDoc;
    }
  }

  private render(): any {
    if (this.isEmpty) {
      return undefined;
    }

    const doc = {
      Statement: this.statements.map(s => s.toStatementJson()),
      Version: '2012-10-17',
    };

    return doc;
  }

  private shouldMerge(scope: IConstruct) {
    return this.minimize ?? cdk.FeatureFlags.of(scope).isEnabled(cxapi.IAM_MINIMIZE_POLICIES) ?? false;
  }

  /**
   * Freeze all statements
   */
  private freezeStatements() {
    for (const statement of this.statements) {
      statement.freeze();
    }
  }
}

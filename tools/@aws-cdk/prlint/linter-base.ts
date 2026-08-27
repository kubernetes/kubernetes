
import type { Octokit } from '@octokit/rest';
import type { CheckRun, GitHubComment, GitHubPr, Review } from './github';

export const PR_FROM_MAIN_ERROR = 'Pull requests from `main` branch of a fork cannot be accepted. Please reopen this contribution from another branch on your fork. For more information, see https://github.com/aws/aws-cdk/blob/main/CONTRIBUTING.md#step-4-pull-request.';

/**
 * Props used to perform linting against the pull request.
 */
export interface PullRequestLinterBaseProps {
  /**
   * GitHub client scoped to pull requests. Imported via @actions/github.
   */
  readonly client: Octokit;

  /**
   * Repository owner.
   */
  readonly owner: string;

  /**
   * Repository name.
   */
  readonly repo: string;

  /**
   * Pull request number.
   */
  readonly number: number;

  /**
   * For cases where the linter needs to know its own username
   */
  readonly linterLogin: string;
}

/**
 * Base "interacting with GitHub" functionality, devoid of any specific validation logic.
 *
 * Inheritance is not great, but this was the easiest way to factor this out for now.
 */
export class PullRequestLinterBase {
  /**
   * Find an open PR for the given commit.
   * @param sha the commit sha to find the PR of
   */
  public static async getPRFromCommit(client: Octokit, owner: string, repo: string, sha: string): Promise<GitHubPr | undefined> {
    const prs = await client.search.issuesAndPullRequests({
      q: sha,
    });
    console.log('Found PRs: ', prs);
    const foundPr = prs.data.items.find(pr => pr.state === 'open');
    if (foundPr) {
      // need to do this because the list PR response does not have
      // all the necessary information
      const pr = (await client.pulls.get({
        owner,
        repo,
        pull_number: foundPr.number,
      })).data;
      console.log(`PR: ${foundPr.number}: `, pr);
      // only process latest commit
      if (pr.head.sha === sha) {
        return pr;
      }
    }
    return;
  }

  protected readonly client: Octokit;
  protected readonly prParams: { owner: string; repo: string; pull_number: number };
  protected readonly issueParams: { owner: string; repo: string; issue_number: number };
  protected readonly linterLogin: string;

  private _pr: GitHubPr | undefined;

  constructor(readonly props: PullRequestLinterBaseProps) {
    this.client = props.client;
    this.prParams = { owner: props.owner, repo: props.repo, pull_number: props.number };
    this.issueParams = { owner: props.owner, repo: props.repo, issue_number: props.number };
    this.linterLogin = props.linterLogin;
  }

  public async pr(): Promise<GitHubPr> {
    if (!this._pr) {
      const r = await this.client.pulls.get(this.prParams);
      this._pr = r.data;
    }
    return this._pr;
  }

  /**
   * Execute the given set of actions
   */
  public async executeActions(actions: LinterActions) {
    const pr = await this.pr();

    for (const label of actions.removeLabels ?? []) {
      await this.removeLabel(label, pr);
    }

    for (const label of actions.addLabels ?? []) {
      await this.addLabel(label, pr);
    }

    for (const comment of actions.deleteComments ?? []) {
      await this.deleteComment(comment);
    }

    if (actions.dismissPreviousReview || actions.requestChanges) {
      if (actions.dismissPreviousReview && actions.requestChanges) {
        throw new Error(`It does not make sense to supply both dismissPreviousReview and requestChanges: ${JSON.stringify(actions)}`);
      }

      const existingReviews = await this.findExistingPRLinterReviews();

      if (actions.dismissPreviousReview) {
        await this.dismissPRLinterReviews(existingReviews, 'passing');
      }
      if (actions.requestChanges) {
        await this.createOrUpdatePRLinterReview(actions.requestChanges.failures, actions.requestChanges.exemptionRequest, existingReviews);
      }
    }
  }

  /**
   * For code that requires an exception
   */
  public actionsToException(actions: LinterActions) {
    if (actions.requestChanges) {
      // The tests check for exactly these messages
      const messages = [...actions.requestChanges.failures];
      if (actions.requestChanges.exemptionRequest) {
        messages.push('A exemption request has been requested. Please wait for a maintainer\'s review.');
      }
      throw new LinterError(messages.join('\n'));
    }
  }

  /**
   * Dismisses previous reviews by aws-cdk-automation when the pull request succeeds the linter.
   * @param existingReview The review created by a previous run of the linter
   */
  private async dismissPRLinterReviews(existingReviews: Review[], reason: 'passing' | 'stale'): Promise<void> {
    let message: string;
    switch (reason) {
      case 'passing':
        message = '✅ Updated pull request passes all PRLinter validations. Dismissing previous PRLinter review.';
        break;
      case 'stale':
        message = 'Dismissing outdated PRLinter review.';
        break;
    }

    for (const existingReview of existingReviews ?? []) {
      try {
        console.log(`👉 Dismissing review ${existingReview.id}`);
        // Make sure to update the old review texts. Dismissing won't do that,
        // and having multiple messages in place is confusing.
        await this.client.pulls.updateReview({
          ...this.prParams,
          review_id: existingReview.id,
          body: '(This review is outdated)',
        });
        await this.client.pulls.dismissReview({
          ...this.prParams,
          review_id: existingReview.id,
          message,
        });
      } catch (e: any) {
        if (githubResponseErrors(e).includes('Can not dismiss a dismissed pull request review')) {
          // Can happen because of race conditions. We already achieved the result we wanted, so ignore.
          continue;
        }

        // This can fail with a "not found" for some reason
        throw new Error(`Dismissing review failed, user is probably not authorized: ${JSON.stringify(e, undefined, 2)}`);
      }
    }
  }

  /**
   * Finds existing review, if present
   * @returns Existing review, if present
   */
  private async findExistingPRLinterReviews(): Promise<Review[]> {
    let reviews: Review[];
    try {
      reviews = await this.client.paginate(this.client.pulls.listReviews, this.prParams);
    } catch (e: any) {
      // If the GitHub API returns a 404 when listing reviews, there are
      // effectively no reviews to dismiss. This can happen for cross-repo
      // PRs or when the PROJEN_GITHUB_TOKEN doesn't have access to the
      // source repository.
      if (e.status === 404) {
        console.log('Reviews endpoint returned 404, assuming no existing prlint reviews');
        return [];
      }
      throw e;
    }
    return reviews.filter((review) => review.user?.login === this.linterLogin && review.state !== 'DISMISSED');
  }

  /**
   * Return all existing comments made by the PR linter
   */
  public async findExistingPRLinterComments(): Promise<GitHubComment[]> {
    const comments = await this.client.paginate(this.client.issues.listComments, this.issueParams);
    return comments.filter((x) => x.user?.login === this.linterLogin && x.body?.includes('pull request linter'));
  }

  /**
   * Creates a new review and comment for first run with failure or creates a new comment with new failures for existing reviews.
   *
   * We assume the PR linter only ever creates "Changes Requested" reviews, or dismisses
   * their own "Changes Requested" reviews.
   *
   * @param failureMessages The failures received by the pr linter validation checks.
   * @param existingReview The review created by a previous run of the linter.
   */
  private async createOrUpdatePRLinterReview(failureMessages: string[], exemptionRequest?: boolean, existingReviews?: Review[]): Promise<void> {
    // FIXME: this function is doing too much at once. We should split this out into separate
    // actions on `LinterActions`.

    const paras = [
      'The pull request linter fails with the following errors:',
      this.formatErrors(failureMessages),
      'If you believe this pull request should receive an exemption, please comment and provide a justification. ' +
      'A comment requesting an exemption should contain the text `Exemption Request`. ' +
      'Additionally, if clarification is needed, add `Clarification Request` to a comment.',
    ];

    if (exemptionRequest) {
      paras.push('✅ A exemption request has been requested. Please wait for a maintainer\'s review.');
    }
    const body = paras.join('\n\n');

    // Dismiss every review except the last one
    await this.dismissPRLinterReviews((existingReviews ?? []).slice(0, -1), 'stale');

    // Update the last review
    const existingReview = (existingReviews ?? []).slice(-1)[0];

    if (!existingReview) {
      console.log('👉 Creating new request-changes review');
      await this.client.pulls.createReview({
        ...this.prParams,
        event: 'REQUEST_CHANGES',
        body,
      });
    } else if (existingReview.body !== body && existingReview.state === 'CHANGES_REQUESTED') {
      // State is good but body is wrong
      console.log(`👉 Updating existing request-changes review: ${existingReview.id}`);
      await this.client.pulls.updateReview({
        ...this.prParams,
        review_id: existingReview.id,
        body,
      });
    }

    // Closing the PR if it is opened from main branch of author's fork
    if (failureMessages.includes(PR_FROM_MAIN_ERROR)) {
      const errorMessageBody = 'Your pull request must be based off of a branch in a personal account '
      + '(not an organization owned account, and not the main branch). You must also have the setting '
      + 'enabled that <a href="https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/allowing-changes-to-a-pull-request-branch-created-from-a-fork">allows the CDK team to push changes to your branch</a> '
      + '(this setting is enabled by default for personal accounts, and cannot be enabled for organization owned accounts). '
      + 'The reason for this is that our automation needs to synchronize your branch with our main after it has been approved, '
      + 'and we cannot do that if we cannot push to your branch.';

      console.log('👉 Closing pull request');
      await this.client.issues.createComment({
        ...this.issueParams,
        body: errorMessageBody,
      });

      await this.client.pulls.update({
        ...this.prParams,
        state: 'closed',
      });
    }
  }

  /**
   * Return the check run results for the given SHA
   *
   * It *seems* like this only returns completed check runs. That is, in-progress
   * runs are not included.
   */
  public async checkRuns(sha: string): Promise<{[name: string]: CheckRun}> {
    const response = await this.client.paginate(this.client.checks.listForRef, {
      owner: this.prParams.owner,
      repo: this.prParams.repo,
      ref: sha,
    });

    const ret: {[name: string]: CheckRun} = {};
    for (const c of response) {
      if (!c.started_at) {
        continue;
      }

      // Insert if this is the first time seeing this check, or the current DT is newer
      if (!ret[c.name] || ret[c.name].started_at!.localeCompare(c.started_at) > 1) {
        ret[c.name] = c;
      }
    }
    return ret;
  }

  /**
   * Delete a comment
   */
  private async deleteComment(comment: GitHubComment) {
    console.log(`👉 Deleting comment ${comment.id}`);
    await this.client.issues.deleteComment({
      ...this.issueParams,
      comment_id: comment.id,
    });
  }

  private formatErrors(errors: string[]) {
    return errors.map(e => `    ❌ ${e}`).join('\n');
  }

  private async addLabel(label: string, pr: Pick<GitHubPr, 'labels' | 'number'>) {
    // already has label, so no-op
    if (pr.labels.some(l => l.name === label)) { return; }

    console.log(`👉 Add label ${label}`);
    await this.client.issues.addLabels({
      issue_number: pr.number,
      owner: this.prParams.owner,
      repo: this.prParams.repo,
      labels: [
        label,
      ],
    });
  }

  private async removeLabel(label: string, pr: Pick<GitHubPr, 'labels' | 'number'>) {
    // does not have label, so no-op
    if (!pr.labels.some(l => l.name === label)) { return; }

    console.log(`👉 Remove label ${label}`);
    await this.client.issues.removeLabel({
      issue_number: pr.number,
      owner: this.prParams.owner,
      repo: this.prParams.repo,
      name: label,
    });
  }
}

export class LinterError extends Error {
  constructor(message: string) {
    super(message);
  }
}

/**
 * Actions that the PR linter should carry out
 */
export interface LinterActions {
  /**
   * Add labels to the PR
   */
  addLabels?: string[];

  /**
   * Remove labels from the PR
   */
  removeLabels?: string[];

  /**
   * Delete the given comments
   */
  deleteComments?: GitHubComment[];

  /**
   * Post a "request changes" review
   */
  requestChanges?: {
    failures: string[];
    exemptionRequest?: boolean;
  };

  /**
   * Dismiss the PR linter's previous review.
   */
  dismissPreviousReview?: boolean;
}

export function mergeLinterActions(a: LinterActions, b: LinterActions): LinterActions {
  return {
    addLabels: nonEmpty([...(a.addLabels ?? []), ...(b.addLabels ?? [])]),
    removeLabels: nonEmpty([...(a.removeLabels ?? []), ...(b.removeLabels ?? [])]),
    requestChanges: b.requestChanges ?? a.requestChanges,
    deleteComments: nonEmpty([...(a.deleteComments ?? []), ...(b.deleteComments ?? [])]),
    dismissPreviousReview: b.dismissPreviousReview ?? a.dismissPreviousReview,
  };
}

function nonEmpty<A>(xs: A[]): A[] | undefined {
  return xs.length > 0 ? xs : undefined;
}

/**
 * Return error messages from an exception thrown by GitHub
 */
function githubResponseErrors(e: Error): string[] {
  return (e as any).response?.data?.errors ?? [];
}

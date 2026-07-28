import { getChangedSnapshots } from './utils';

/**
 * The GitHub team that can approve integration test deployments
 */
const CDK_TEAM = 'aws-cdk-team';
const CDK_ORG = 'aws';

/**
 * Result of the preflight check
 */
export interface PreflightResult {
  /** Whether the integration tests should run */
  shouldRun: boolean;
  /** Reason for the decision */
  reason: string;
}

/**
 * Check if a user is a member of the CDK team
 */
async function isCdkTeamMember(githubToken: string, username: string): Promise<boolean> {
  // Check team membership using GitHub API
  // GET /orgs/{org}/teams/{team_slug}/memberships/{username}
  const response = await fetch(
    `https://api.github.com/orgs/${CDK_ORG}/teams/${CDK_TEAM}/memberships/${username}`,
    {
      headers: {
        Authorization: `token ${githubToken}`,
        Accept: 'application/vnd.github.v3+json',
      },
    },
  );

  if (response.status === 200) {
    const membership = await response.json() as { state: string };
    return membership.state === 'active';
  }

  // 404 means not a member
  if (response.status === 404) {
    return false;
  }

  // Authentication/authorization failures should not silently bypass security checks
  if (response.status === 401 || response.status === 403) {
    throw new Error(`GitHub authentication/authorization failed when checking team membership for ${username}. Status: ${response.status}. Verify the token has the required permissions.`);
  }

  // Other errors should also fail explicitly to avoid silent security bypass
  throw new Error(`Failed to verify team membership for ${username}. GitHub API returned status: ${response.status}`);
}

/**
 * Check if integration tests should run based on PR state.
 *
 * Integration tests should run when:
 * 1. PR has snapshot changes AND
 * 2. PR is approved by a CDK team member (@aws/aws-cdk-team)
 */
export async function shouldRunIntegTests(props: {
  githubToken: string;
  owner: string;
  repo: string;
  prNumber: number;
}): Promise<PreflightResult> {
  const { githubToken, owner, repo, prNumber } = props;

  // Check if PR has snapshot changes
  const changedSnapshots = await getChangedSnapshots();
  if (changedSnapshots.length === 0) {
    return {
      shouldRun: false,
      reason: 'No snapshot changes detected in this PR',
    };
  }

  // Fetch reviews to check for CDK team approval
  const reviewsResponse = await fetch(`https://api.github.com/repos/${owner}/${repo}/pulls/${prNumber}/reviews`, {
    headers: {
      Authorization: `token ${githubToken}`,
      Accept: 'application/vnd.github.v3+json',
    },
  });

  if (!reviewsResponse.ok) {
    throw new Error(`Failed to fetch PR reviews: ${reviewsResponse.status} ${reviewsResponse.statusText}`);
  }

  const reviews = await reviewsResponse.json() as Array<{
    state: string;
    author_association: string;
    user: { login: string } | null;
  }>;

  // Get all approving reviewers
  const approvingReviewers = reviews
    .filter(review => review.state === 'APPROVED' && review.user?.login)
    .map(review => review.user!.login);

  // Check if any approving reviewer is a CDK team member
  for (const reviewer of approvingReviewers) {
    const isTeamMember = await isCdkTeamMember(githubToken, reviewer);
    if (isTeamMember) {
      return {
        shouldRun: true,
        reason: `PR has snapshot changes and is approved by CDK team member @${reviewer}`,
      };
    }
  }

  return {
    shouldRun: false,
    reason: `PR has snapshot changes but is not yet approved by a CDK team member (@${CDK_ORG}/${CDK_TEAM}).`,
  };
}

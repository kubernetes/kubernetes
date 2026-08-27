#!/usr/bin/env node
/**
 * Verifies that a PR is in an approved state by a CDK team member.
 *
 * Usage: node verify-approval.js <pr_number> <approved_sha>
 *
 * Env: PROJEN_GITHUB_TOKEN - token for GitHub API (team membership check)
 *      GITHUB_REPOSITORY - owner/repo (e.g. "aws/aws-cdk")
 *
 * Exit code: 0 if approved by a CDK team member (non-stale), 1 otherwise
 */
'use strict';

/**
 * Makes a GitHub API request. Returns { status, data }.
 */
async function githubFetch(path, token) {
  const headers = {
    'Accept': 'application/vnd.github.v3+json',
    'User-Agent': 'aws-cdk-integ-test-trigger',
  };
  if (token) {
    headers['Authorization'] = `token ${token}`;
  }
  const response = await fetch(`https://api.github.com${path}`, { headers });
  const text = await response.text();
  const data = text ? JSON.parse(text) : {};
  return { status: response.status, data };
}

/**
 * Core verification logic. Returns { exitCode, message }.
 */
async function verifyApproval({ prNumber, approvedSha, token, repository, apiFn }) {
  if (!prNumber) {
    return { exitCode: 1, message: 'Usage: verify-approval.js <pr_number> <approved_sha>' };
  }

  if (!approvedSha) {
    return { exitCode: 1, message: 'Usage: verify-approval.js <pr_number> <approved_sha>' };
  }

  if (!token || !repository) {
    return { exitCode: 1, message: 'PROJEN_GITHUB_TOKEN and GITHUB_REPOSITORY env vars required' };
  }

  const [owner, repo] = repository.split('/');

  // Assertion: this script verifies CDK maintainer approvals which are scoped
  // to the aws/aws-cdk repository and the aws-cdk-team within the aws org.
  if (owner !== 'aws' || repo !== 'aws-cdk') {
    return { exitCode: 1, message: `Unexpected repository: ${repository} (expected aws/aws-cdk)` };
  }

  // Verify the PR's current head matches the approved SHA.
  // If someone pushed after approval, the PR head will differ and we reject.
  const { status: prStatus, data: pr } = await apiFn(`/repos/${owner}/${repo}/pulls/${prNumber}`, token);

  if (prStatus === 404) {
    return { exitCode: 1, message: `PR #${prNumber} not found` };
  }
  if (prStatus !== 200) {
    return { exitCode: 1, message: `Failed to fetch PR #${prNumber}: HTTP ${prStatus}` };
  }
  if (pr.state !== 'open') {
    return { exitCode: 1, message: `PR #${prNumber} is ${pr.state}, skipping (only open PRs are eligible)` };
  }
  if (pr.head.sha !== approvedSha) {
    return { exitCode: 1, message: `PR #${prNumber} head has changed (expected ${approvedSha.slice(0, 7)}, got ${pr.head.sha.slice(0, 7)}). New approval needed.` };
  }

  // List reviews
  const { status: reviewsStatus, data: reviews } = await apiFn(
    `/repos/${owner}/${repo}/pulls/${prNumber}/reviews`,
    token,
  );

  if (reviewsStatus !== 200) {
    return { exitCode: 1, message: `Failed to fetch reviews for PR #${prNumber}: HTTP ${reviewsStatus}` };
  }

  const approvals = reviews.filter(r => r.state === 'APPROVED');

  if (approvals.length === 0) {
    return { exitCode: 1, message: `PR #${prNumber} has no approvals` };
  }

  // Check each approval: must be non-stale and from a CDK team member
  for (const review of approvals) {
    if (review.commit_id !== approvedSha) {
      continue;
    }

    // Check team membership
    // Returns { state: "active"|"pending", role: "member"|"maintainer" } on 200,
    // or 404 if user is not a member of the team.
    const { status: memberStatus, data: membership } = await apiFn(
      `/orgs/${owner}/teams/aws-cdk-team/memberships/${review.user.login}`,
      token,
    );
    if (memberStatus === 200 && membership.state === 'active') {
      return { exitCode: 0, message: `PR #${prNumber} is approved by CDK team member: ${review.user.login}` };
    }
  }

  return { exitCode: 1, message: `PR #${prNumber} has no valid (non-stale) approval from a CDK team member` };
}

// Only execute when run directly from command line (skipped when imported by tests)
if (require.main === module) {
  (async () => {
    const token = process.env.PROJEN_GITHUB_TOKEN;

    const result = await verifyApproval({
      prNumber: process.argv[2],
      approvedSha: process.argv[3],
      token,
      repository: process.env.GITHUB_REPOSITORY,
      apiFn: (path, authToken) => githubFetch(path, authToken),
    });

    if (result.exitCode === 0) {
      console.log(result.message);
    } else {
      console.error(result.message);
    }
    process.exit(result.exitCode);
  })().catch((err) => {
    console.error(err.message);
    process.exit(1);
  });
}

module.exports = { verifyApproval, githubFetch };

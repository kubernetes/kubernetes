'use strict';

const { verifyApproval } = require('../../verify-approval');

// ─── Helpers ────────────────────────────────────────────────────────────────────

const VALID_TOKEN = 'ghp_test_token';
const VALID_REPO = 'aws/aws-cdk';
const APPROVED_SHA = 'abc1234567890abcdef1234567890abcdef123456';
const STALE_SHA = 'def0000000000000000000000000000000000000';

function openPr(sha = APPROVED_SHA) {
  return { state: 'open', head: { sha } };
}

function approval(login, commitId = APPROVED_SHA) {
  return { state: 'APPROVED', commit_id: commitId, user: { login } };
}

function makeApiFn(responses) {
  return async (path) => {
    for (const [pattern, response] of Object.entries(responses)) {
      if (path.includes(pattern)) {
        return response;
      }
    }
    return { status: 404, data: { message: 'Not Found' } };
  };
}

// ─── Tests ──────────────────────────────────────────────────────────────────────

describe('verify-approval', () => {
  test('Missing prNumber arg → exit 1', async () => {
    const result = await verifyApproval({
      prNumber: undefined,
      approvedSha: APPROVED_SHA,
      token: VALID_TOKEN,
      repository: VALID_REPO,
      apiFn: makeApiFn({}),
    });
    expect(result.exitCode).toBe(1);
    expect(result.message).toContain('Usage');
  });

  test('Missing approvedSha arg → exit 1', async () => {
    const result = await verifyApproval({
      prNumber: '123',
      approvedSha: undefined,
      token: VALID_TOKEN,
      repository: VALID_REPO,
      apiFn: makeApiFn({}),
    });
    expect(result.exitCode).toBe(1);
    expect(result.message).toContain('Usage');
  });

  test('Missing PROJEN_GITHUB_TOKEN → exit 1', async () => {
    const result = await verifyApproval({
      prNumber: '123',
      approvedSha: APPROVED_SHA,
      token: undefined,
      repository: VALID_REPO,
      apiFn: makeApiFn({}),
    });
    expect(result.exitCode).toBe(1);
    expect(result.message).toContain('PROJEN_GITHUB_TOKEN');
  });

  test('Missing GITHUB_REPOSITORY → exit 1', async () => {
    const result = await verifyApproval({
      prNumber: '123',
      approvedSha: APPROVED_SHA,
      token: VALID_TOKEN,
      repository: undefined,
      apiFn: makeApiFn({}),
    });
    expect(result.exitCode).toBe(1);
    expect(result.message).toContain('PROJEN_GITHUB_TOKEN');
  });

  test('Wrong repository (not aws/aws-cdk) → exit 1', async () => {
    const result = await verifyApproval({
      prNumber: '123',
      approvedSha: APPROVED_SHA,
      token: VALID_TOKEN,
      repository: 'foo/bar',
      apiFn: makeApiFn({}),
    });
    expect(result.exitCode).toBe(1);
    expect(result.message).toContain('Unexpected repository');
  });

  test('PR returns 404 → exit 1', async () => {
    const result = await verifyApproval({
      prNumber: '123',
      approvedSha: APPROVED_SHA,
      token: VALID_TOKEN,
      repository: VALID_REPO,
      apiFn: makeApiFn({
        '/pulls/123': { status: 404, data: { message: 'Not Found' } },
      }),
    });
    expect(result.exitCode).toBe(1);
    expect(result.message).toContain('not found');
  });

  test('PR returns 500 → exit 1', async () => {
    const result = await verifyApproval({
      prNumber: '123',
      approvedSha: APPROVED_SHA,
      token: VALID_TOKEN,
      repository: VALID_REPO,
      apiFn: makeApiFn({
        '/pulls/123': { status: 500, data: { message: 'Internal Server Error' } },
      }),
    });
    expect(result.exitCode).toBe(1);
    expect(result.message).toContain('HTTP 500');
  });

  test('PR is closed → exit 1', async () => {
    const result = await verifyApproval({
      prNumber: '123',
      approvedSha: APPROVED_SHA,
      token: VALID_TOKEN,
      repository: VALID_REPO,
      apiFn: makeApiFn({
        '/pulls/123': { status: 200, data: { state: 'closed', head: { sha: APPROVED_SHA } } },
      }),
    });
    expect(result.exitCode).toBe(1);
    expect(result.message).toContain('closed');
  });

  test('PR is merged → exit 1', async () => {
    const result = await verifyApproval({
      prNumber: '123',
      approvedSha: APPROVED_SHA,
      token: VALID_TOKEN,
      repository: VALID_REPO,
      apiFn: makeApiFn({
        '/pulls/123': { status: 200, data: { state: 'merged', head: { sha: APPROVED_SHA } } },
      }),
    });
    expect(result.exitCode).toBe(1);
    expect(result.message).toContain('merged');
  });

  test('PR head has changed (new commits pushed) → exit 1', async () => {
    const result = await verifyApproval({
      prNumber: '123',
      approvedSha: APPROVED_SHA,
      token: VALID_TOKEN,
      repository: VALID_REPO,
      apiFn: makeApiFn({
        '/pulls/123': { status: 200, data: { state: 'open', head: { sha: 'different_sha_after_push' } } },
      }),
    });
    expect(result.exitCode).toBe(1);
    expect(result.message).toContain('head has changed');
    expect(result.message).toContain('New approval needed');
  });

  test('Reviews returns 500 → exit 1', async () => {
    const result = await verifyApproval({
      prNumber: '123',
      approvedSha: APPROVED_SHA,
      token: VALID_TOKEN,
      repository: VALID_REPO,
      apiFn: makeApiFn({
        '/pulls/123/reviews': { status: 500, data: { message: 'Error' } },
        '/pulls/123': { status: 200, data: openPr() },
      }),
    });
    expect(result.exitCode).toBe(1);
    expect(result.message).toContain('HTTP 500');
  });

  test('No approvals → exit 1', async () => {
    const result = await verifyApproval({
      prNumber: '123',
      approvedSha: APPROVED_SHA,
      token: VALID_TOKEN,
      repository: VALID_REPO,
      apiFn: makeApiFn({
        '/pulls/123/reviews': { status: 200, data: [] },
        '/pulls/123': { status: 200, data: openPr() },
      }),
    });
    expect(result.exitCode).toBe(1);
    expect(result.message).toContain('no approvals');
  });

  test('All approvals stale → exit 1', async () => {
    const result = await verifyApproval({
      prNumber: '123',
      approvedSha: APPROVED_SHA,
      token: VALID_TOKEN,
      repository: VALID_REPO,
      apiFn: makeApiFn({
        '/pulls/123/reviews': { status: 200, data: [approval('user1', STALE_SHA)] },
        '/pulls/123': { status: 200, data: openPr() },
      }),
    });
    expect(result.exitCode).toBe(1);
    expect(result.message).toContain('no valid');
  });

  test('Approval from non-team member (membership returns 404) → exit 1', async () => {
    const result = await verifyApproval({
      prNumber: '123',
      approvedSha: APPROVED_SHA,
      token: VALID_TOKEN,
      repository: VALID_REPO,
      apiFn: makeApiFn({
        '/pulls/123/reviews': { status: 200, data: [approval('outsider')] },
        '/pulls/123': { status: 200, data: openPr() },
        '/memberships/outsider': { status: 404, data: { message: 'Not Found' } },
      }),
    });
    expect(result.exitCode).toBe(1);
    expect(result.message).toContain('no valid');
  });

  test('Valid approval from team member → exit 0', async () => {
    const result = await verifyApproval({
      prNumber: '123',
      approvedSha: APPROVED_SHA,
      token: VALID_TOKEN,
      repository: VALID_REPO,
      apiFn: makeApiFn({
        '/pulls/123/reviews': { status: 200, data: [approval('team-member')] },
        '/pulls/123': { status: 200, data: openPr() },
        '/memberships/team-member': { status: 200, data: { state: 'active', role: 'member' } },
      }),
    });
    expect(result.exitCode).toBe(0);
    expect(result.message).toContain('approved by CDK team member');
    expect(result.message).toContain('team-member');
  });

  test('Multiple approvals: one stale, one valid → exit 0', async () => {
    const result = await verifyApproval({
      prNumber: '123',
      approvedSha: APPROVED_SHA,
      token: VALID_TOKEN,
      repository: VALID_REPO,
      apiFn: makeApiFn({
        '/pulls/123/reviews': {
          status: 200,
          data: [
            approval('stale-reviewer', STALE_SHA),
            approval('fresh-reviewer', APPROVED_SHA),
          ],
        },
        '/pulls/123': { status: 200, data: openPr() },
        '/memberships/fresh-reviewer': { status: 200, data: { state: 'active', role: 'member' } },
      }),
    });
    expect(result.exitCode).toBe(0);
    expect(result.message).toContain('fresh-reviewer');
  });

  test('Multiple approvals: one non-team, one team → exit 0', async () => {
    const apiFn = async (path) => {
      if (path.includes('/pulls/123/reviews')) {
        return { status: 200, data: [approval('outsider'), approval('insider')] };
      }
      if (path.includes('/pulls/123')) {
        return { status: 200, data: openPr() };
      }
      if (path.includes('/memberships/outsider')) {
        return { status: 404, data: { message: 'Not Found' } };
      }
      if (path.includes('/memberships/insider')) {
        return { status: 200, data: { state: 'active', role: 'member' } };
      }
      return { status: 404, data: { message: 'Not Found' } };
    };

    const result = await verifyApproval({
      prNumber: '123',
      approvedSha: APPROVED_SHA,
      token: VALID_TOKEN,
      repository: VALID_REPO,
      apiFn,
    });
    expect(result.exitCode).toBe(0);
    expect(result.message).toContain('insider');
  });

  test('Membership API returns 403 (rate limited) → treated as non-member', async () => {
    const result = await verifyApproval({
      prNumber: '123',
      approvedSha: APPROVED_SHA,
      token: VALID_TOKEN,
      repository: VALID_REPO,
      apiFn: makeApiFn({
        '/pulls/123/reviews': { status: 200, data: [approval('rate-limited-user')] },
        '/pulls/123': { status: 200, data: openPr() },
        '/memberships/rate-limited-user': { status: 403, data: { message: 'API rate limit exceeded' } },
      }),
    });
    expect(result.exitCode).toBe(1);
    expect(result.message).toContain('no valid');
  });

  test('Membership state is pending (invited but not accepted) → treated as non-member', async () => {
    const result = await verifyApproval({
      prNumber: '123',
      approvedSha: APPROVED_SHA,
      token: VALID_TOKEN,
      repository: VALID_REPO,
      apiFn: makeApiFn({
        '/pulls/123/reviews': { status: 200, data: [approval('pending-user')] },
        '/pulls/123': { status: 200, data: openPr() },
        '/memberships/pending-user': { status: 200, data: { state: 'pending', role: 'member' } },
      }),
    });
    expect(result.exitCode).toBe(1);
    expect(result.message).toContain('no valid');
  });
});

describe('find-pr logic', () => {
  const testSha = 'aaa1111222233334444555566667777888899990000';

  function findPr(prs, headSha) {
    const matching = prs.filter(p => p.head.sha === headSha);
    if (matching.length === 0) {
      return { success: false, error: `No open PR found with head ${headSha}` };
    }
    if (matching.length > 1) {
      return { success: false, error: `Multiple PRs (${matching.length}) found for SHA ${headSha}, aborting` };
    }
    return {
      success: true,
      outputs: {
        pr_number: String(matching[0].number),
        head_sha: matching[0].head.sha,
        base_sha: matching[0].base.sha,
      },
    };
  }

  test('0 PRs match SHA → fail', () => {
    const prs = [
      { number: 1, head: { sha: 'other-sha-1' }, base: { sha: 'base1' } },
      { number: 2, head: { sha: 'other-sha-2' }, base: { sha: 'base2' } },
    ];
    const result = findPr(prs, testSha);
    expect(result.success).toBe(false);
    expect(result.error).toContain('No open PR found');
    expect(result.error).toContain(testSha);
  });

  test('1 PR matches SHA → success, outputs correct', () => {
    const prs = [
      { number: 42, head: { sha: testSha }, base: { sha: 'base-sha-42' } },
      { number: 99, head: { sha: 'different' }, base: { sha: 'base-sha-99' } },
    ];
    const result = findPr(prs, testSha);
    expect(result.success).toBe(true);
    expect(result.outputs.pr_number).toBe('42');
    expect(result.outputs.head_sha).toBe(testSha);
    expect(result.outputs.base_sha).toBe('base-sha-42');
  });

  test('2+ PRs match SHA → fail', () => {
    const prs = [
      { number: 10, head: { sha: testSha }, base: { sha: 'base10' } },
      { number: 20, head: { sha: testSha }, base: { sha: 'base20' } },
      { number: 30, head: { sha: testSha }, base: { sha: 'base30' } },
    ];
    const result = findPr(prs, testSha);
    expect(result.success).toBe(false);
    expect(result.error).toContain('Multiple PRs');
    expect(result.error).toContain('3');
  });

  test('Empty PR list → fail', () => {
    const result = findPr([], testSha);
    expect(result.success).toBe(false);
    expect(result.error).toContain('No open PR found');
  });
});

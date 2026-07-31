/**
 * Monitors open `beginning-contributor` PRs and tracks the ones that need
 * maintainer action in a weekly tracking issue.
 *
 * A PR needs attention when either:
 *   - CI is pending approval (a workflow run for the PR's head commit has
 *     conclusion `action_required`), or
 *   - the build never started (no build workflow run for the head commit and
 *     no commit statuses -- CI approval was skipped or never triggered, even
 *     if non-build `pull_request_target` workflows ran).
 *
 * Each week gets its own tracking issue, identified by the marker label
 * (`ci-pending-tracking`) plus an ISO week key in the title (e.g.
 * `2026-July13-W29`, where `July13` is the Monday the week starts on). The
 * issue lists only the PRs whose pending state began during that week (by
 * `pendingSince` ISO week), not every currently-pending PR.
 * On the first run of a week the issue is created; subsequent runs in the same
 * week update its body with that week's pending PRs. Issues from previous weeks
 * are left open for maintainers to review and close manually.
 *
 * Supports a DRY_RUN mode (set env DRY_RUN=true) to log intended issue
 * creation/updates without writing anything.
 */

const CONFIG = {
  label: 'beginning-contributor',
  trackingLabel: 'ci-pending-tracking',
  // Stable identifiers (workflow file paths) for the PR build workflows, since the
  // display `name:` field can be renamed at any time.
  buildWorkflowPaths: [
    '.github/workflows/pr-build.yml',
    '.github/workflows/codebuild-pr-build.yml',
  ],
};

const REASON_DESCRIPTIONS = {
  pending_approval: { short: '🔒 Pending CI approval' },
  no_ci_activity: { short: '⚠️ No build CI activity' },
};

const DRY_RUN = process.env.DRY_RUN === 'true';

/**
 * Returns the week key for a date, combining the ISO 8601 week number with
 * the date of the Monday that week starts on (e.g. `2026-July13-W29`). The
 * year reflects the Monday's calendar year.
 */
function isoWeekKey(d) {
  const date = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate()));
  // Shift to the Thursday of the current ISO week (ISO weeks are keyed by their Thursday).
  const day = date.getUTCDay() || 7;
  date.setUTCDate(date.getUTCDate() + 4 - day);
  const yearStart = new Date(Date.UTC(date.getUTCFullYear(), 0, 1));
  const week = Math.ceil(((date - yearStart) / 86400000 + 1) / 7);
  // Monday of the ISO week (3 days before its Thursday). The year prefix uses
  // the Monday's calendar year (not the ISO week-year) so the date part of the
  // key always reads as a real calendar date, e.g. `2025-December29-W01`.
  const monday = new Date(date);
  monday.setUTCDate(monday.getUTCDate() - 3);
  const monthName = monday.toLocaleString('en-US', { month: 'long', timeZone: 'UTC' });
  return `${monday.getUTCFullYear()}-${monthName}${monday.getUTCDate()}-W${String(week).padStart(2, '0')}`;
}

/**
 * Fetches all open, non-draft PRs with the configured label.
 */
async function fetchLabeledOpenPrs({ github, owner, repo, label }) {
  const prs = await github.paginate(github.rest.issues.listForRepo, {
    owner,
    repo,
    state: 'open',
    labels: label,
    per_page: 100,
  });
  // issues.listForRepo also returns plain issues; keep PRs only.
  return prs.filter((item) => item.pull_request);
}

/**
 * Determines whether a PR needs maintainer attention (pending CI approval or
 * no build workflow activity at all), based on its head commit's workflow
 * runs and combined commit status. For flagged PRs, also returns
 * `pendingSince`: the time the PR entered the pending state (the last push --
 * approximated by the earliest action_required run's creation time, or the
 * head commit date when no build run exists).
 */
async function needsAttention({ github, owner, repo, pr }) {
  // per_page is the API maximum: a head SHA on this repo can accumulate 20+
  // runs, and missing an action_required run here would silently drop a
  // pending PR from the report.
  const { data: workflowRuns } = await github.rest.actions.listWorkflowRunsForRepo({
    owner,
    repo,
    head_sha: pr.head.sha,
    per_page: 100,
  });
  const runs = workflowRuns.workflow_runs;

  const actionRequiredRuns = runs.filter((run) => run.conclusion === 'action_required');
  if (actionRequiredRuns.length > 0) {
    const pendingSince = actionRequiredRuns
      .map((run) => run.created_at)
      .sort()[0];
    return { needsAttention: true, reason: 'pending_approval', pendingSince };
  }

  const buildRun = runs.find((run) => CONFIG.buildWorkflowPaths.includes(run.path));

  if (buildRun && (buildRun.status === 'queued' || buildRun.status === 'in_progress')) {
    return { needsAttention: false, reason: 'build_in_progress' };
  }

  if (!buildRun) {
    const { data: status } = await github.rest.repos.getCombinedStatusForRef({
      owner,
      repo,
      ref: pr.head.sha,
    });
    if (status.state === 'pending' && status.total_count === 0) {
      // Non-build workflows (pull_request_target: PR Linter, prioritization,
      // etc.) may have run on the head commit, but the build workflow never
      // started and is not awaiting approval, and nothing reported a commit
      // status either -- CI approval was skipped or never triggered.
      const { data: commit } = await github.rest.repos.getCommit({
        owner,
        repo,
        ref: pr.head.sha,
      });
      return { needsAttention: true, reason: 'no_ci_activity', pendingSince: commit.commit.committer.date };
    }
  }

  return { needsAttention: false, reason: 'ok' };
}

/**
 * Escapes PR title text for safe embedding in a Markdown table cell:
 * backslashes are escaped first (so they can't neutralize the pipe escaping),
 * pipes would break the table layout, and stray whitespace/newlines are
 * collapsed.
 */
function sanitizeTitle(title) {
  return title.replace(/\s+/g, ' ').replace(/\\/g, '\\\\').replace(/\|/g, '\\|').trim();
}

/**
 * Renders the tracking issue body as a Markdown table. PR titles are
 * sanitized before embedding; PR links use absolute URLs so they render
 * correctly regardless of where the issue lives.
 */
function renderIssueBody({ owner, repo, weekKey, flaggedPrs }) {
  const workflowUrl = `https://github.com/${owner}/${repo}/blob/main/.github/workflows/pending-maintainer-action-check.yml`;
  const counts = {};
  for (const { reason } of flaggedPrs) {
    counts[reason] = (counts[reason] ?? 0) + 1;
  }
  const summary = Object.entries(counts)
    .map(([reason, n]) => `${REASON_DESCRIPTIONS[reason]?.short ?? reason}: ${n}`)
    .join(' · ');

  const lines = [
    `Beginning-contributor PRs that entered pending maintainer CI action during week **${weekKey}**.`,
    '',
    `_Last updated: ${new Date().toISOString()} (refreshed daily by the [Pending Maintainer Action Check workflow](${workflowUrl}))_`,
    '',
  ];

  if (flaggedPrs.length === 0) {
    lines.push(`No PRs entered pending maintainer CI action during week **${weekKey}**. :tada:`);
  } else {
    lines.push(`**${flaggedPrs.length} PR(s)** — ${summary}`);
    for (const reason of Object.keys(counts)) {
      // Oldest pending first, so the most stale PRs surface at the top.
      const group = flaggedPrs
        .filter((f) => f.reason === reason)
        .sort((a, b) => a.pendingSince.localeCompare(b.pendingSince));
      lines.push(
        '',
        `### ${REASON_DESCRIPTIONS[reason]?.short ?? reason} (${group.length})`,
        '',
        '| PR | Title | Author | Pending since |',
        '|----|-------|--------|---------------|',
      );
      for (const { pr, pendingSince } of group) {
        const link = `[#${pr.number}](https://github.com/${owner}/${repo}/pull/${pr.number})`;
        const days = Math.floor((Date.now() - new Date(pendingSince)) / 86400000);
        lines.push(`| ${link} | ${sanitizeTitle(pr.title)} | @${pr.user.login} | ${pendingSince.slice(0, 10)} (${days}d) |`);
      }
    }
  }

  lines.push('', 'Close this issue once all listed PRs have been handled.');
  return lines.join('\n');
}

/**
 * Finds this week's tracking issue (marker label + week key in title), if it
 * already exists.
 */
async function findTrackingIssue({ github, owner, repo, weekKey }) {
  const issues = await github.paginate(github.rest.issues.listForRepo, {
    owner,
    repo,
    state: 'open',
    labels: CONFIG.trackingLabel,
    per_page: 100,
  });
  return issues.find((issue) => !issue.pull_request && issue.title.includes(weekKey));
}

module.exports = async ({ github, context }) => {
  const { owner, repo } = context.repo;
  const { label } = CONFIG;

  if (DRY_RUN) {
    console.log('DRY_RUN=true -- no issue will be created or updated, only logged.');
  }

  const prs = await fetchLabeledOpenPrs({ github, owner, repo, label });
  console.log(`Found ${prs.length} open "${label}" PR(s) on ${owner}/${repo}`);

  const flaggedPrs = [];

  for (const item of prs) {
    const prNumber = item.number;

    try {
      const { data: pr } = await github.rest.pulls.get({
        owner,
        repo,
        pull_number: prNumber,
      });

      if (pr.draft) {
        console.log(`PR #${prNumber}: draft, skipping`);
        continue;
      }

      const { needsAttention: flag, reason, pendingSince } = await needsAttention({ github, owner, repo, pr });

      if (!flag) {
        console.log(`PR #${prNumber}: no action needed (${reason})`);
        continue;
      }

      console.log(`PR #${prNumber}: needs attention (${reason}, pending since ${pendingSince})`);
      flaggedPrs.push({ pr, reason, pendingSince });
    } catch (error) {
      console.error(`Error processing PR #${prNumber}:`, error.message);
    }
  }

  const weekKey = isoWeekKey(new Date());
  const title = `CI pending maintainer action: week ${weekKey}`;
  const existingIssue = await findTrackingIssue({ github, owner, repo, weekKey });

  // Keep only PRs whose pending state began this ISO week. Comparing week keys
  // (rather than a separate date window) stays in lockstep with the title and
  // reuses the ISO year-boundary handling isoWeekKey already encodes.
  const weeklyPrs = flaggedPrs.filter(
    ({ pendingSince }) => isoWeekKey(new Date(pendingSince)) === weekKey,
  );
  console.log(`${weeklyPrs.length} of ${flaggedPrs.length} pending PR(s) entered the pending state during week ${weekKey}`);

  if (weeklyPrs.length === 0 && !existingIssue) {
    console.log(`No PRs entered the pending state during week ${weekKey} and no tracking issue exists; nothing to do.`);
    return;
  }

  const body = renderIssueBody({ owner, repo, weekKey, flaggedPrs: weeklyPrs });

  if (existingIssue) {
    if (DRY_RUN) {
      console.log(`[DRY RUN] would update tracking issue #${existingIssue.number} with ${weeklyPrs.length} PR(s)`);
    } else {
      await github.rest.issues.update({
        owner,
        repo,
        issue_number: existingIssue.number,
        body,
      });
      console.log(`Updated tracking issue #${existingIssue.number} with ${weeklyPrs.length} PR(s)`);
    }
  } else {
    if (DRY_RUN) {
      console.log(`[DRY RUN] would create tracking issue "${title}" with ${weeklyPrs.length} PR(s)`);
    } else {
      const { data: issue } = await github.rest.issues.create({
        owner,
        repo,
        title,
        body,
        labels: [CONFIG.trackingLabel],
      });
      console.log(`Created tracking issue #${issue.number} ("${title}") with ${weeklyPrs.length} PR(s)`);
    }
  }
};

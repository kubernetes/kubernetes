package main

import (
	"bytes"
	"fmt"
	"regexp"
	"strings"
	"text/template"
)

var (
	// AllCommitValidators holds all registered checks.
	AllCommitValidators = []func(Commit) []string{
		ValidateCommitAuthor,

		// vendor/* -> commit meesgae
		ValidatePatchHasUpstreamCommitMessage,
		ValidateBumpHasBumpCommitMessage,
		ValidateNoBumpAndPatchesTogether,

		// commit message -> vendor/*
		ValidateUpstreamCommit,
		ValidateBumpCommit,
	}
)

func ValidateCommitAuthor(commit Commit) []string {
	var allErrors []string

	if strings.HasPrefix(commit.Email, "root@") {
		allErrors = append(allErrors, fmt.Sprintf("Commit %s has invalid email %q", commit.Sha, commit.Email))
	}

	return allErrors
}

func ValidatePatchHasUpstreamCommitMessage(commit Commit) []string {
	if !commit.HasPatches() {
		return nil
	}

	var allErrors []string

	if !commit.MatchesUpstreamSummaryPattern() {
		tmpl, _ := template.New("problems").Parse(`
UPSTREAM commit {{ .Commit.Sha }} has invalid summary {{ .Commit.Summary }}.

UPSTREAM commits are validated against the following regular expression:
  {{ .Pattern }}

UPSTREAM commit summaries should look like:

  UPSTREAM: <PR number|carry|drop>: description

UPSTREAM commits which revert previous UPSTREAM commits should look like:

  UPSTREAM: revert: <normal upstream format>

Examples of valid summaries:

  UPSTREAM: 12345: A kube fix
  UPSTREAM: <carry>: A carried kube change
  UPSTREAM: <drop>: A dropped kube change
  UPSTREAM: revert: 12345: A kube revert
`)
		data := struct {
			Pattern *regexp.Regexp
			Commit  Commit
		}{
			Pattern: UpstreamSummaryPattern,
			Commit:  commit,
		}
		buffer := &bytes.Buffer{}
		err := tmpl.Execute(buffer, data)
		if err != nil {
			allErrors = append(allErrors, err.Error())
			return allErrors
		}

		allErrors = append(allErrors, buffer.String())

		return allErrors
	}

	return allErrors
}

func ValidateBumpHasBumpCommitMessage(commit Commit) []string {
	if !commit.HasBumpedFiles() {
		return nil
	}

	var allErrors []string

	if !commit.MatchesBumpSummaryPattern() {
		allErrors = append(allErrors, fmt.Sprintf("Commit %s bumps dependencies but summary %q doesn't match the bump summary pattern %q", commit.Sha, commit.Summary, BumpSummaryPattern))
	}

	return allErrors
}

// ValidateNoBumpAndPatchesTogether is also covered by requiring non-intersecting commit messages for bumps and patches but it gives nicer error message
func ValidateNoBumpAndPatchesTogether(commit Commit) []string {
	var allErrors []string

	if commit.HasBumpedFiles() && commit.HasPatches() {
		allErrors = append(allErrors, fmt.Sprintf("Commit %s (%q) bumps dependencies and also modifies patched paths. This is not allowed! Your dependency manager cache might be stale or the publisher bot is broken. Try cleaning the cache and bumping again. If it doesn't work and you are convinced the bot is broken, contact the master team. Inside vendor/ directories for patches are identified by matching any of the following patterns: %q", commit.Sha, commit.Summary, strings.Join(RegexpsToStrings(PatchRegexps), ", ")))
	}

	return allErrors
}

func ValidateUpstreamCommit(commit Commit) []string {
	if !commit.MatchesUpstreamSummaryPattern() {
		return nil
	}

	var allErrors []string

	if commit.HasPatches() {
		patchedRepos, err := commit.PatchedRepos()
		if err != nil {
			allErrors = append(allErrors, err.Error())
		} else if len(patchedRepos) == 0 {
			allErrors = append(allErrors, fmt.Errorf("commit %s (%q): failed to detect patched repositories", commit.Sha, commit.Summary).Error())
		} else if len(patchedRepos) > 1 {
			allErrors = append(allErrors, fmt.Sprintf("Commit %s (%q) modifies more then 1 repository: %q", commit.Sha, commit.Summary, strings.Join(patchedRepos, ", ")))
		} else {
			commitRepo, err := commit.DeclaredUpstreamRepo()
			if err != nil {
				allErrors = append(allErrors, err.Error())
			}

			if commitRepo != patchedRepos[0] {
				allErrors = append(allErrors, fmt.Sprintf("Commit %s (%q) declares to modify repository %q but modifies repository %q", commit.Sha, commit.Summary, commitRepo, patchedRepos[0]))
			}
		}
	} else {
		allErrors = append(allErrors, fmt.Sprintf("Upstream commit %s (%q) is missing changes to patch paths. Inside vendor/ directories for patches are identified by matching any of the following patterns: %q", commit.Sha, commit.Summary, strings.Join(RegexpsToStrings(PatchRegexps), ", ")))
	}

	if commit.HasBumpedFiles() {
		allErrors = append(allErrors, fmt.Sprintf("Upstream commit %s (%q) is not allowed to bump dependencies.", commit.Sha, commit.Summary))
	}

	if commit.HasNonVendoredCodeChanges() {
		allErrors = append(allErrors, fmt.Sprintf("Upstream commit %s (%q) is not allowed to have non-vendor code changes.", commit.Sha, commit.Summary))
	}

	return allErrors
}

func ValidateBumpCommit(commit Commit) []string {
	if !commit.MatchesBumpSummaryPattern() {
		return nil
	}

	var allErrors []string

	if !commit.HasBumpedFiles() {
		allErrors = append(allErrors, fmt.Sprintf("Bump commit %s (%q) is missing changes to dependencies.", commit.Sha, commit.Summary))
	}

	if commit.HasPatches() {
		allErrors = append(allErrors, fmt.Sprintf("Bump commit %s (%q) is not allowed to change patched paths. Inside vendor/ directories for patches are identified by matching any of the following patterns: %q", commit.Sha, commit.Summary, strings.Join(RegexpsToStrings(PatchRegexps), ", ")))
	}

	if commit.HasNonVendoredCodeChanges() {
		allErrors = append(allErrors, fmt.Sprintf("Bump commit %s (%q) is not allowed to have non-vendor code changes. If you are modifying vendoring files like glide.yaml, glide.lock, go.mod, go.sum, ... these belong into a separate commit. It allows for easily checking what's beeing bumped and automation verifying the content of vendor folder matches that description.", commit.Sha, commit.Summary))
	}

	return allErrors
}

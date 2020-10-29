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

		// Local commit messages must be prefixed with UPSTREAM as per
		// README.openshift.md to aid in rebasing on upstream kube.
		ValidateCommitMessage,
	}
)

func ValidateCommitAuthor(commit Commit) []string {
	var allErrors []string

	if strings.HasPrefix(commit.Email, "root@") {
		allErrors = append(allErrors, fmt.Sprintf("Commit %s has invalid email %q", commit.Sha, commit.Email))
	}

	return allErrors
}

func ValidateCommitMessage(commit Commit) []string {
	if commit.MatchesMergeSummaryPattern() {
		// Ignore merges
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

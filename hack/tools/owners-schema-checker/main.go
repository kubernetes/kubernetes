package main

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"sigs.k8s.io/yaml"
)

type ownersFile struct {
	Filters           map[string]filtersInfo `json:"filters,omitempty"`
	Approvers         []string               `json:"approvers,omitempty"`
	Reviewers         []string               `json:"reviewers,omitempty"`
	RequiredReviewers []string               `json:"required_reviewers,omitempty"`
	Labels            []string               `json:"labels,omitempty"`
	EmeritusApprovers []string               `json:"emeritus_approvers,omitempty"`
	EmeritusReviewers []string               `json:"emeritus_reviewers,omitempty"`
	Options           dirOptions             `json:"options,omitempty"`
}

type filtersInfo struct {
	Approvers         []string `json:"approvers,omitempty"`
	Reviewers         []string `json:"reviewers,omitempty"`
	Labels            []string `json:"labels,omitempty"`
	EmeritusApprovers []string `json:"emeritus_approvers,omitempty"`
	EmeritusReviewers []string `json:"emeritus_reviewers,omitempty"`
	RequiredReviewers []string `json:"required_reviewers,omitempty"`
}

type dirOptions struct {
	NoParentOwners bool `json:"no_parent_owners,omitempty"`
}

type ownersAliasesFile struct {
	Aliases map[string][]string `json:"aliases,omitempty"`
}

func main() {
	root := "."
	if len(os.Args) > 2 {
		fmt.Fprintf(os.Stderr, "usage: %s [repo-root]\n", filepath.Base(os.Args[0]))
		os.Exit(2)
	}
	if len(os.Args) == 2 {
		root = os.Args[1]
	}

	files, err := listOwnersFiles(root)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: %v\n", err)
		os.Exit(1)
	}

	var failed bool
	for _, file := range files {
		if err := validateFile(filepath.Join(root, file)); err != nil {
			fmt.Fprintf(os.Stderr, "ERROR: %s: %v\n", file, err)
			failed = true
		}
	}

	if failed {
		os.Exit(1)
	}
}

func listOwnersFiles(root string) ([]string, error) {
	cmd := exec.Command(
		"git", "-C", root, "ls-files", "--",
		"OWNERS*",
		"**/OWNERS*",
		":(exclude)vendor/*/OWNERS*",
	)
	output, err := cmd.Output()
	if err != nil {
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) {
			return nil, fmt.Errorf("unable to list OWNERS files: %s", strings.TrimSpace(string(exitErr.Stderr)))
		}
		return nil, fmt.Errorf("unable to list OWNERS files: %w", err)
	}

	lines := strings.Split(strings.TrimSpace(string(output)), "\n")
	if len(lines) == 1 && lines[0] == "" {
		return nil, nil
	}
	return lines, nil
}

func validateFile(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	if isCommentsOnly(data) {
		return errors.New("file is empty or contains only comments")
	}

	switch filepath.Base(path) {
	case "OWNERS":
		var parsed ownersFile
		if err := yaml.UnmarshalStrict(data, &parsed); err != nil {
			return err
		}
	case "OWNERS_ALIASES":
		var parsed ownersAliasesFile
		if err := yaml.UnmarshalStrict(data, &parsed); err != nil {
			return err
		}
	default:
		return nil
	}

	return nil
}

func isCommentsOnly(data []byte) bool {
	for _, line := range bytes.Split(data, []byte("\n")) {
		trimmed := strings.TrimSpace(string(line))
		if trimmed == "" || strings.HasPrefix(trimmed, "#") {
			continue
		}
		return false
	}
	return true
}

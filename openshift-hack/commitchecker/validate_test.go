package main

import (
	"reflect"
	"testing"
)

func TestValidateCommitAuthor(t *testing.T) {
	tt := []struct {
		name         string
		commit       Commit
		expectedErrs []string
	}{
		{
			name: "fails on root@locahost",
			commit: Commit{
				Sha:     "aaa0000",
				Summary: "a summary",
				Files: []File{
					"README.md",
				},
				Email: "root@localhost",
			},
			expectedErrs: []string{
				"Commit aaa0000 has invalid email \"root@localhost\"",
			},
		},
		{
			name: "succeeds for deads2k@redhat.com",
			commit: Commit{
				Sha:     "aaa0000",
				Summary: "a summary",
				Files: []File{
					"README.md",
				},
				Email: "deads2k@redhat.com",
			},
			expectedErrs: nil,
		},
	}

	for _, tc := range tt {
		t.Run(tc.name, func(t *testing.T) {
			gotErrs := ValidateCommitAuthor(tc.commit)
			if !reflect.DeepEqual(tc.expectedErrs, gotErrs) {
				t.Errorf("expected %#v, got %#v", tc.expectedErrs, gotErrs)
			}
		})
	}
}

func TestValidateCommitMessage(t *testing.T) {
	tt := []struct {
		name         string
		commit       Commit
		expectedErrs []string
	}{
		{
			name: "modifying k8s without UPSTREAM commit fails",
			commit: Commit{
				Sha:     "aaa0000",
				Summary: "wrong summary",
				Files: []File{
					"README.md",
					"pkg/controller/deployment/deployment.go",
				},
			},
			expectedErrs: []string{
				"\nUPSTREAM commit aaa0000 has invalid summary wrong summary.\n\nUPSTREAM commits are validated against the following regular expression:\n  ^UPSTREAM: (revert: )?(([\\w\\.-]+\\/[\\w-\\.-]+)?: )?(\\d+:|<carry>:|<drop>:)\n\nUPSTREAM commit summaries should look like:\n\n  UPSTREAM: <PR number|carry|drop>: description\n\nUPSTREAM commits which revert previous UPSTREAM commits should look like:\n\n  UPSTREAM: revert: <normal upstream format>\n\nExamples of valid summaries:\n\n  UPSTREAM: 12345: A kube fix\n  UPSTREAM: <carry>: A carried kube change\n  UPSTREAM: <drop>: A dropped kube change\n  UPSTREAM: revert: 12345: A kube revert\n",
			},
		},
		{
			name: "modifying k8s with UPSTREAM commit succeeds",
			commit: Commit{
				Sha:     "aaa0000",
				Summary: "UPSTREAM: 42: Fix kube",
				Files: []File{
					"README.md",
					"pkg/controller/deployment/deployment.go",
				},
			},
			expectedErrs: nil,
		},
	}

	for _, tc := range tt {
		t.Run(tc.name, func(t *testing.T) {
			gotErrs := ValidateCommitMessage(tc.commit)
			if !reflect.DeepEqual(tc.expectedErrs, gotErrs) {
				t.Errorf("expected %#v, got %#v", tc.expectedErrs, gotErrs)
			}
		})
	}
}

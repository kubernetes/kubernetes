package main

import (
	"reflect"
	"testing"

	"github.com/openshift/origin/tools/rebasehelpers/util"
)

func TestValidateCommitAuthor(t *testing.T) {
	tt := []struct {
		name         string
		commit       util.Commit
		expectedErrs []string
	}{
		{
			name: "fails on root@locahost",
			commit: util.Commit{
				Sha:     "aaa0000",
				Summary: "a summary",
				Files: []util.File{
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
			commit: util.Commit{
				Sha:     "aaa0000",
				Summary: "a summary",
				Files: []util.File{
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

func TestValidatePatchHasUpstreamCommitMessage(t *testing.T) {
	tt := []struct {
		name         string
		commit       util.Commit
		expectedErrs []string
	}{
		{
			name: "modifying k8s without UPSTREAM commit fails",
			commit: util.Commit{
				Sha:     "aaa0000",
				Summary: "wrong summary",
				Files: []util.File{
					"README.md",
					"vendor/k8s.io/kubernetes/kubernetes/pkg/controller/deployment/deployment.go",
				},
			},
			expectedErrs: []string{
				"\nUPSTREAM commit aaa0000 has invalid summary wrong summary.\n\nUPSTREAM commits are validated against the following regular expression:\n  ^UPSTREAM: (revert: )?(([\\w\\.-]+\\/[\\w-\\.-]+)?: )?(\\d+:|<carry>:|<drop>:)\n\nUPSTREAM commit summaries should look like:\n\n  UPSTREAM: <PR number|carry|drop>: description\n\nUPSTREAM commits which revert previous UPSTREAM commits should look like:\n\n  UPSTREAM: revert: <normal upstream format>\n\nExamples of valid summaries:\n\n  UPSTREAM: 12345: A kube fix\n  UPSTREAM: <carry>: A carried kube change\n  UPSTREAM: <drop>: A dropped kube change\n  UPSTREAM: revert: 12345: A kube revert\n",
			},
		},
		{
			name: "modifying k8s with UPSTREAM commit succeeds",
			commit: util.Commit{
				Sha:     "aaa0000",
				Summary: "UPSTREAM: 42: Fix kube",
				Files: []util.File{
					"README.md",
					"vendor/k8s.io/kubernetes/kubernetes/pkg/controller/deployment/deployment.go",
				},
			},
			expectedErrs: nil,
		},
		{
			name: "bumping deps doesn't require UPSTREAM commit",
			commit: util.Commit{
				Sha:     "aaa0000",
				Summary: "bump(*)",
				Files: []util.File{
					"vendor/github.com/somethingrandom/42/cmd/main.go",
				},
			},
			expectedErrs: nil,
		},
	}

	for _, tc := range tt {
		t.Run(tc.name, func(t *testing.T) {
			gotErrs := ValidatePatchHasUpstreamCommitMessage(tc.commit)
			if !reflect.DeepEqual(tc.expectedErrs, gotErrs) {
				t.Errorf("expected %#v, got %#v", tc.expectedErrs, gotErrs)
			}
		})
	}
}

func TestValidateBumpHasBumpCommitMessage(t *testing.T) {
	tt := []struct {
		name         string
		commit       util.Commit
		expectedErrs []string
	}{
		{
			name: "modifying a dependency without bump commit fails",
			commit: util.Commit{
				Sha:     "aaa0000",
				Summary: "wrong summary",
				Files: []util.File{
					"README.md",
					"vendor/github.com/somethingrandom/42/cmd/main.go",
				},
			},
			expectedErrs: []string{
				"Commit aaa0000 bumps dependencies but summary \"wrong summary\" doesn't match the bump summary pattern \"^bump[\\\\(\\\\w].*\"",
			},
		},
		{
			name: "modifying a dependency with bump commit succeeds",
			commit: util.Commit{
				Sha:     "aaa0000",
				Summary: "bump(*)",
				Files: []util.File{
					"README.md",
					"vendor/github.com/somethingrandom/42/cmd/main.go",
				},
			},
			expectedErrs: nil,
		},
		{
			name: "UPSTREAM patches don't require bump commit message",
			commit: util.Commit{
				Sha:     "aaa0000",
				Summary: "UPSTREAM: 42: Fix kube",
				Files: []util.File{
					"README.md",
					"vendor/k8s.io/kubernetes/kubernetes/pkg/controller/deployment/deployment.go",
				},
			},
			expectedErrs: nil,
		},
	}

	for _, tc := range tt {
		t.Run(tc.name, func(t *testing.T) {
			gotErrs := ValidateBumpHasBumpCommitMessage(tc.commit)
			if !reflect.DeepEqual(tc.expectedErrs, gotErrs) {
				t.Errorf("expected %#v, got %#v", tc.expectedErrs, gotErrs)
			}
		})
	}
}

func TestValidateNoBumpAndPatchesTogether(t *testing.T) {
	tt := []struct {
		name         string
		commit       util.Commit
		expectedErrs []string
	}{
		{
			name: "bumping dependency together with a patch fails",
			commit: util.Commit{
				Sha:     "aaa0000",
				Summary: "bump(*)",
				Files: []util.File{
					"vendor/github.com/somethingrandom/42/cmd/main.go",
					"vendor/k8s.io/kubernetes/kubernetes/pkg/controller/deployment/deployment.go",
				},
			},
			expectedErrs: []string{
				"Commit aaa0000 (\"bump(*)\") bumps dependencies and also modifies patched paths. This is not allowed! Your dependency manager cache might be stale or the publisher bot is broken. Try cleaning the cache and bumping again. If it doesn't work and you are convinced the bot is broken, contact the master team. Inside vendor/ directories for patches are identified by matching any of the following patterns: \"^k8s.io/kubernetes/.*\"",
			},
		},
		{
			name: "patch which bumps a dependency fail",
			commit: util.Commit{
				Sha:     "aaa0000",
				Summary: "UPSTREAM: 42: Fix kube",
				Files: []util.File{
					"vendor/github.com/somethingrandom/42/cmd/main.go",
					"vendor/k8s.io/kubernetes/kubernetes/pkg/controller/deployment/deployment.go",
				},
			},
			expectedErrs: []string{
				"Commit aaa0000 (\"UPSTREAM: 42: Fix kube\") bumps dependencies and also modifies patched paths. This is not allowed! Your dependency manager cache might be stale or the publisher bot is broken. Try cleaning the cache and bumping again. If it doesn't work and you are convinced the bot is broken, contact the master team. Inside vendor/ directories for patches are identified by matching any of the following patterns: \"^k8s.io/kubernetes/.*\"",
			},
		},
		{
			name: "bumping dependencies succeeds",
			commit: util.Commit{
				Sha:     "aaa0000",
				Summary: "bump(*)",
				Files: []util.File{
					"vendor/github.com/somethingrandom/42/cmd/main.go",
				},
			},
			expectedErrs: nil,
		},
		{
			name: "UPSTREAM patches succeed",
			commit: util.Commit{
				Sha:     "aaa0000",
				Summary: "UPSTREAM: 42: Fix kube",
				Files: []util.File{
					"vendor/k8s.io/kubernetes/kubernetes/pkg/controller/deployment/deployment.go",
				},
			},
			expectedErrs: nil,
		},
	}

	for _, tc := range tt {
		t.Run(tc.name, func(t *testing.T) {
			gotErrs := ValidateNoBumpAndPatchesTogether(tc.commit)
			if !reflect.DeepEqual(tc.expectedErrs, gotErrs) {
				t.Errorf("expected %#v, got %#v", tc.expectedErrs, gotErrs)
			}
		})
	}
}

func TestValidateUpstreamCommit(t *testing.T) {
	tt := []struct {
		name         string
		commit       util.Commit
		expectedErrs []string
	}{
		{
			name: "UPSTREAM commit modifying a dependency fails",
			commit: util.Commit{
				Sha:     "aaa0000",
				Summary: "UPSTREAM: 42: Fix kube",
				Files: []util.File{
					"vendor/k8s.io/kubernetes/kubernetes/pkg/controller/deployment/deployment.go",
					"vendor/github.com/somethingrandom/42/cmd/main.go",
				},
			},
			expectedErrs: []string{
				"Upstream commit aaa0000 (\"UPSTREAM: 42: Fix kube\") is not allowed to bump dependencies.",
			},
		},
		{
			name: "UPSTREAM commit modifying regular file fails",
			commit: util.Commit{
				Sha:     "aaa0000",
				Summary: "UPSTREAM: 42: Fix kube",
				Files: []util.File{
					"vendor/k8s.io/kubernetes/kubernetes/pkg/controller/deployment/deployment.go",
					"README.md",
				},
			},
			expectedErrs: []string{
				"Upstream commit aaa0000 (\"UPSTREAM: 42: Fix kube\") is not allowed to have non-vendor code changes.",
			},
		},
		/*
			// We only publish from 1 repository at this time
			// TODO: enable this when we publish from more then 1 repository
			{
				name: "UPSTREAM commit modifying 2 repositories fails",
				commit: util.Commit{
					Sha:     "aaa0000",
					Summary: "UPSTREAM: 42: Fix kube",
					Files: []util.File{
						"vendor/k8s.io/kubernetes/kubernetes/pkg/controller/deployment/deployment.go",
						// TODO: bellow come the second repo
						"vendor/k8s.io/client-go/pkg/tools/cache/cache.go",
					},
				},
				expectedErrs: []string{
					"Commit aaa0000 (\"UPSTREAM: 42: Fix kube\") modifies more then 1 repository: \"k8s.io/kubernetes, k8s.io/client-go\"",
				},
			},
		*/
		{
			name: "UPSTREAM commit modifying different repository then it claims to fails",
			commit: util.Commit{
				Sha:     "aaa0000",
				Summary: "UPSTREAM: k8s.io/client-go: 42: Fix kube",
				Files: []util.File{
					"vendor/k8s.io/kubernetes/staging/src/k8s.io/client-go/REDAME.md",
				},
			},
			expectedErrs: []string{
				"Commit aaa0000 (\"UPSTREAM: k8s.io/client-go: 42: Fix kube\") declares to modify repository \"k8s.io/client-go\" but modifies repository \"k8s.io/kubernetes\"",
			},
		},
		{
			name: "UPSTREAM commit modifying the same repository it claims succeeds",
			commit: util.Commit{
				Sha:     "aaa0000",
				Summary: "UPSTREAM: k8s.io/kubernetes: 42: Fix kube",
				Files: []util.File{
					"vendor/k8s.io/kubernetes/kubernetes/pkg/controller/deployment/deployment.go",
				},
			},
			expectedErrs: nil,
		},
		{
			name: "UPSTREAM commit not modifying any files fails",
			commit: util.Commit{
				Sha:     "aaa0000",
				Summary: "UPSTREAM: 42: Fix kube",
				Files:   []util.File{},
			},
			expectedErrs: []string{
				"Upstream commit aaa0000 (\"UPSTREAM: 42: Fix kube\") is missing changes to patch paths. Inside vendor/ directories for patches are identified by matching any of the following patterns: \"^k8s.io/kubernetes/.*\"",
			},
		},
	}

	for _, tc := range tt {
		t.Run(tc.name, func(t *testing.T) {
			gotErrs := ValidateUpstreamCommit(tc.commit)
			if !reflect.DeepEqual(tc.expectedErrs, gotErrs) {
				t.Errorf("expected %#v, got %#v", tc.expectedErrs, gotErrs)
			}
		})
	}
}

func TestValidateBumpCommit(t *testing.T) {
	tt := []struct {
		name         string
		commit       util.Commit
		expectedErrs []string
	}{
		{
			name: "bump commit updating only dependencies succeeds",
			commit: util.Commit{
				Sha:     "aaa0000",
				Summary: "bump(*)",
				Files: []util.File{
					"vendor/github.com/somethingrandom/42/cmd/main.go",
					"vendor/github.com/somethingrandom/84/cmd/main.go",
					"vendor/github.com/anotherrandom/0/cmd/main.go",
				},
			},
			expectedErrs: nil,
		},
		{
			name: "bump commit not updating dependencies fails",
			commit: util.Commit{
				Sha:     "aaa0000",
				Summary: "bump(*)",
				Files:   []util.File{},
			},
			expectedErrs: []string{
				"Bump commit aaa0000 (\"bump(*)\") is missing changes to dependencies.",
			},
		},
		{
			name: "bump commit not updating dependencies but only other files fails",
			commit: util.Commit{
				Sha:     "aaa0000",
				Summary: "bump(*)",
				Files: []util.File{
					"README.md",
				},
			},
			expectedErrs: []string{
				"Bump commit aaa0000 (\"bump(*)\") is missing changes to dependencies.", "Bump commit aaa0000 (\"bump(*)\") is not allowed to have non-vendor code changes. If you are modifying vendoring files like glide.yaml, glide.lock, go.mod, go.sum, ... these belong into a separate commit. It allows for easily checking what's beeing bumped and automation verifying the content of vendor folder matches that description.",
			},
		},
		{
			name: "bump commit not updating dependencies but only patched files fails",
			commit: util.Commit{
				Sha:     "aaa0000",
				Summary: "bump(*)",
				Files: []util.File{
					"vendor/k8s.io/kubernetes/kubernetes/pkg/controller/deployment/deployment.go",
				},
			},
			expectedErrs: []string{
				"Bump commit aaa0000 (\"bump(*)\") is missing changes to dependencies.", "Bump commit aaa0000 (\"bump(*)\") is not allowed to change patched paths. Inside vendor/ directories for patches are identified by matching any of the following patterns: \"^k8s.io/kubernetes/.*\"",
			},
		},
		{
			name: "bump commit updating dependencies and patched files fails",
			commit: util.Commit{
				Sha:     "aaa0000",
				Summary: "bump(*)",
				Files: []util.File{
					"vendor/k8s.io/kubernetes/kubernetes/pkg/controller/deployment/deployment.go",
					"vendor/github.com/somethingrandom/42/cmd/main.go",
				},
			},
			expectedErrs: []string{
				"Bump commit aaa0000 (\"bump(*)\") is not allowed to change patched paths. Inside vendor/ directories for patches are identified by matching any of the following patterns: \"^k8s.io/kubernetes/.*\"",
			},
		},
		{
			name: "bump commit including glide.lock fails",
			commit: util.Commit{
				Sha:     "aaa0000",
				Summary: "bump(*)",
				Files: []util.File{
					"glide.lock",
				},
			},
			expectedErrs: []string{
				"Bump commit aaa0000 (\"bump(*)\") is missing changes to dependencies.", "Bump commit aaa0000 (\"bump(*)\") is not allowed to have non-vendor code changes. If you are modifying vendoring files like glide.yaml, glide.lock, go.mod, go.sum, ... these belong into a separate commit. It allows for easily checking what's beeing bumped and automation verifying the content of vendor folder matches that description.",
			},
		},
		{
			name: "bump commit including go.sum fails",
			commit: util.Commit{
				Sha:     "aaa0000",
				Summary: "bump(*)",
				Files: []util.File{
					"go.sum",
				},
			},
			expectedErrs: []string{
				"Bump commit aaa0000 (\"bump(*)\") is missing changes to dependencies.", "Bump commit aaa0000 (\"bump(*)\") is not allowed to have non-vendor code changes. If you are modifying vendoring files like glide.yaml, glide.lock, go.mod, go.sum, ... these belong into a separate commit. It allows for easily checking what's beeing bumped and automation verifying the content of vendor folder matches that description.",
			},
		},
	}

	for _, tc := range tt {
		t.Run(tc.name, func(t *testing.T) {
			gotErrs := ValidateBumpCommit(tc.commit)
			if !reflect.DeepEqual(tc.expectedErrs, gotErrs) {
				t.Errorf("expected %#v, got %#v", tc.expectedErrs, gotErrs)
			}
		})
	}
}

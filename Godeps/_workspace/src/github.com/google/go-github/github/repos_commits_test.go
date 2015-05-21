// Copyright 2013 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"fmt"
	"net/http"
	"reflect"
	"testing"
	"time"
)

func TestRepositoriesService_ListCommits(t *testing.T) {
	setup()
	defer teardown()

	// given
	mux.HandleFunc("/repos/o/r/commits", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		testFormValues(t, r,
			values{
				"sha":    "s",
				"path":   "p",
				"author": "a",
				"since":  "2013-08-01T00:00:00Z",
				"until":  "2013-09-03T00:00:00Z",
			})
		fmt.Fprintf(w, `[{"sha": "s"}]`)
	})

	opt := &CommitsListOptions{
		SHA:    "s",
		Path:   "p",
		Author: "a",
		Since:  time.Date(2013, time.August, 1, 0, 0, 0, 0, time.UTC),
		Until:  time.Date(2013, time.September, 3, 0, 0, 0, 0, time.UTC),
	}
	commits, _, err := client.Repositories.ListCommits("o", "r", opt)
	if err != nil {
		t.Errorf("Repositories.ListCommits returned error: %v", err)
	}

	want := []RepositoryCommit{{SHA: String("s")}}
	if !reflect.DeepEqual(commits, want) {
		t.Errorf("Repositories.ListCommits returned %+v, want %+v", commits, want)
	}
}

func TestRepositoriesService_GetCommit(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/commits/s", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprintf(w, `{
		  "sha": "s",
		  "commit": { "message": "m" },
		  "author": { "login": "l" },
		  "committer": { "login": "l" },
		  "parents": [ { "sha": "s" } ],
		  "stats": { "additions": 104, "deletions": 4, "total": 108 },
		  "files": [
		    {
		      "filename": "f",
		      "additions": 10,
		      "deletions": 2,
		      "changes": 12,
		      "status": "s",
		      "raw_url": "r",
		      "blob_url": "b",
		      "patch": "p"
		    }
		  ]
		}`)
	})

	commit, _, err := client.Repositories.GetCommit("o", "r", "s")
	if err != nil {
		t.Errorf("Repositories.GetCommit returned error: %v", err)
	}

	want := &RepositoryCommit{
		SHA: String("s"),
		Commit: &Commit{
			Message: String("m"),
		},
		Author: &User{
			Login: String("l"),
		},
		Committer: &User{
			Login: String("l"),
		},
		Parents: []Commit{
			{
				SHA: String("s"),
			},
		},
		Stats: &CommitStats{
			Additions: Int(104),
			Deletions: Int(4),
			Total:     Int(108),
		},
		Files: []CommitFile{
			{
				Filename:  String("f"),
				Additions: Int(10),
				Deletions: Int(2),
				Changes:   Int(12),
				Status:    String("s"),
				Patch:     String("p"),
			},
		},
	}
	if !reflect.DeepEqual(commit, want) {
		t.Errorf("Repositories.GetCommit returned \n%+v, want \n%+v", commit, want)
	}
}

func TestRepositoriesService_CompareCommits(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/repos/o/r/compare/b...h", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprintf(w, `{
		  "base_commit": {
		    "sha": "s",
		    "commit": {
		      "author": { "name": "n" },
		      "committer": { "name": "n" },
		      "message": "m",
		      "tree": { "sha": "t" }
		    },
		    "author": { "login": "n" },
		    "committer": { "login": "l" },
		    "parents": [ { "sha": "s" } ]
		  },
		  "status": "s",
		  "ahead_by": 1,
		  "behind_by": 2,
		  "total_commits": 1,
		  "commits": [
		    {
		      "sha": "s",
		      "commit": { "author": { "name": "n" } },
		      "author": { "login": "l" },
		      "committer": { "login": "l" },
		      "parents": [ { "sha": "s" } ]
		    }
		  ],
		  "files": [ { "filename": "f" } ]
		}`)
	})

	got, _, err := client.Repositories.CompareCommits("o", "r", "b", "h")
	if err != nil {
		t.Errorf("Repositories.CompareCommits returned error: %v", err)
	}

	want := &CommitsComparison{
		Status:       String("s"),
		AheadBy:      Int(1),
		BehindBy:     Int(2),
		TotalCommits: Int(1),
		BaseCommit: &RepositoryCommit{
			Commit: &Commit{
				Author: &CommitAuthor{Name: String("n")},
			},
			Author:    &User{Login: String("l")},
			Committer: &User{Login: String("l")},
			Message:   String("m"),
		},
		Commits: []RepositoryCommit{
			{
				SHA: String("s"),
			},
		},
		Files: []CommitFile{
			{
				Filename: String("f"),
			},
		},
	}

	if reflect.DeepEqual(got, want) {
		t.Errorf("Repositories.CompareCommits returned \n%+v, want \n%+v", got, want)
	}
}

// Copyright 2014 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
	"time"
)

// ContributorStats represents a contributor to a repository and their
// weekly contributions to a given repo.
type ContributorStats struct {
	Author *Contributor   `json:"author,omitempty"`
	Total  *int           `json:"total,omitempty"`
	Weeks  []*WeeklyStats `json:"weeks,omitempty"`
}

func (c ContributorStats) String() string {
	return Stringify(c)
}

// WeeklyStats represents the number of additions, deletions and commits
// a Contributor made in a given week.
type WeeklyStats struct {
	Week      *Timestamp `json:"w,omitempty"`
	Additions *int       `json:"a,omitempty"`
	Deletions *int       `json:"d,omitempty"`
	Commits   *int       `json:"c,omitempty"`
}

func (w WeeklyStats) String() string {
	return Stringify(w)
}

// ListContributorsStats gets a repo's contributor list with additions,
// deletions and commit counts.
//
// If this is the first time these statistics are requested for the given
// repository, this method will return an *AcceptedError and a status code of
// 202. This is because this is the status that GitHub returns to signify that
// it is now computing the requested statistics. A follow up request, after a
// delay of a second or so, should result in a successful request.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#get-all-contributor-commit-activity
func (s *RepositoriesService) ListContributorsStats(ctx context.Context, owner, repo string) ([]*ContributorStats, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/stats/contributors", owner, repo)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var contributorStats []*ContributorStats
	resp, err := s.client.Do(ctx, req, &contributorStats)
	if err != nil {
		return nil, resp, err
	}

	return contributorStats, resp, nil
}

// WeeklyCommitActivity represents the weekly commit activity for a repository.
// The days array is a group of commits per day, starting on Sunday.
type WeeklyCommitActivity struct {
	Days  []int      `json:"days,omitempty"`
	Total *int       `json:"total,omitempty"`
	Week  *Timestamp `json:"week,omitempty"`
}

func (w WeeklyCommitActivity) String() string {
	return Stringify(w)
}

// ListCommitActivity returns the last year of commit activity
// grouped by week. The days array is a group of commits per day,
// starting on Sunday.
//
// If this is the first time these statistics are requested for the given
// repository, this method will return an *AcceptedError and a status code of
// 202. This is because this is the status that GitHub returns to signify that
// it is now computing the requested statistics. A follow up request, after a
// delay of a second or so, should result in a successful request.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#get-the-last-year-of-commit-activity
func (s *RepositoriesService) ListCommitActivity(ctx context.Context, owner, repo string) ([]*WeeklyCommitActivity, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/stats/commit_activity", owner, repo)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var weeklyCommitActivity []*WeeklyCommitActivity
	resp, err := s.client.Do(ctx, req, &weeklyCommitActivity)
	if err != nil {
		return nil, resp, err
	}

	return weeklyCommitActivity, resp, nil
}

// ListCodeFrequency returns a weekly aggregate of the number of additions and
// deletions pushed to a repository. Returned WeeklyStats will contain
// additions and deletions, but not total commits.
//
// If this is the first time these statistics are requested for the given
// repository, this method will return an *AcceptedError and a status code of
// 202. This is because this is the status that GitHub returns to signify that
// it is now computing the requested statistics. A follow up request, after a
// delay of a second or so, should result in a successful request.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#get-the-weekly-commit-activity
func (s *RepositoriesService) ListCodeFrequency(ctx context.Context, owner, repo string) ([]*WeeklyStats, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/stats/code_frequency", owner, repo)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var weeks [][]int
	resp, err := s.client.Do(ctx, req, &weeks)

	// convert int slices into WeeklyStats
	var stats []*WeeklyStats
	for _, week := range weeks {
		if len(week) != 3 {
			continue
		}
		stat := &WeeklyStats{
			Week:      &Timestamp{time.Unix(int64(week[0]), 0)},
			Additions: Int(week[1]),
			Deletions: Int(week[2]),
		}
		stats = append(stats, stat)
	}

	return stats, resp, err
}

// RepositoryParticipation is the number of commits by everyone
// who has contributed to the repository (including the owner)
// as well as the number of commits by the owner themself.
type RepositoryParticipation struct {
	All   []int `json:"all,omitempty"`
	Owner []int `json:"owner,omitempty"`
}

func (r RepositoryParticipation) String() string {
	return Stringify(r)
}

// ListParticipation returns the total commit counts for the 'owner'
// and total commit counts in 'all'. 'all' is everyone combined,
// including the 'owner' in the last 52 weeks. If you’d like to get
// the commit counts for non-owners, you can subtract 'all' from 'owner'.
//
// The array order is oldest week (index 0) to most recent week.
//
// If this is the first time these statistics are requested for the given
// repository, this method will return an *AcceptedError and a status code of
// 202. This is because this is the status that GitHub returns to signify that
// it is now computing the requested statistics. A follow up request, after a
// delay of a second or so, should result in a successful request.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#get-the-weekly-commit-count
func (s *RepositoriesService) ListParticipation(ctx context.Context, owner, repo string) (*RepositoryParticipation, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/stats/participation", owner, repo)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	participation := new(RepositoryParticipation)
	resp, err := s.client.Do(ctx, req, participation)
	if err != nil {
		return nil, resp, err
	}

	return participation, resp, nil
}

// PunchCard represents the number of commits made during a given hour of a
// day of the week.
type PunchCard struct {
	Day     *int // Day of the week (0-6: =Sunday - Saturday).
	Hour    *int // Hour of day (0-23).
	Commits *int // Number of commits.
}

// ListPunchCard returns the number of commits per hour in each day.
//
// If this is the first time these statistics are requested for the given
// repository, this method will return an *AcceptedError and a status code of
// 202. This is because this is the status that GitHub returns to signify that
// it is now computing the requested statistics. A follow up request, after a
// delay of a second or so, should result in a successful request.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/repos/#get-the-hourly-commit-count-for-each-day
func (s *RepositoriesService) ListPunchCard(ctx context.Context, owner, repo string) ([]*PunchCard, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/stats/punch_card", owner, repo)
	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var results [][]int
	resp, err := s.client.Do(ctx, req, &results)

	// convert int slices into Punchcards
	var cards []*PunchCard
	for _, result := range results {
		if len(result) != 3 {
			continue
		}
		card := &PunchCard{
			Day:     Int(result[0]),
			Hour:    Int(result[1]),
			Commits: Int(result[2]),
		}
		cards = append(cards, card)
	}

	return cards, resp, err
}

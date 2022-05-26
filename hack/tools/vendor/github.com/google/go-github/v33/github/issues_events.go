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

// IssueEvent represents an event that occurred around an Issue or Pull Request.
type IssueEvent struct {
	ID  *int64  `json:"id,omitempty"`
	URL *string `json:"url,omitempty"`

	// The User that generated this event.
	Actor *User `json:"actor,omitempty"`

	// Event identifies the actual type of Event that occurred. Possible
	// values are:
	//
	//     closed
	//       The Actor closed the issue.
	//       If the issue was closed by commit message, CommitID holds the SHA1 hash of the commit.
	//
	//     merged
	//       The Actor merged into master a branch containing a commit mentioning the issue.
	//       CommitID holds the SHA1 of the merge commit.
	//
	//     referenced
	//       The Actor committed to master a commit mentioning the issue in its commit message.
	//       CommitID holds the SHA1 of the commit.
	//
	//     reopened, unlocked
	//       The Actor did that to the issue.
	//
	//     locked
	//       The Actor locked the issue.
	//       LockReason holds the reason of locking the issue (if provided while locking).
	//
	//     renamed
	//       The Actor changed the issue title from Rename.From to Rename.To.
	//
	//     mentioned
	//       Someone unspecified @mentioned the Actor [sic] in an issue comment body.
	//
	//     assigned, unassigned
	//       The Assigner assigned the issue to or removed the assignment from the Assignee.
	//
	//     labeled, unlabeled
	//       The Actor added or removed the Label from the issue.
	//
	//     milestoned, demilestoned
	//       The Actor added or removed the issue from the Milestone.
	//
	//     subscribed, unsubscribed
	//       The Actor subscribed to or unsubscribed from notifications for an issue.
	//
	//     head_ref_deleted, head_ref_restored
	//       The pull request’s branch was deleted or restored.
	//
	//    review_dismissed
	//       The review was dismissed and `DismissedReview` will be populated below.
	//
	//    review_requested, review_request_removed
	//       The Actor requested or removed the request for a review.
	//       RequestedReviewer and ReviewRequester will be populated below.
	//
	Event *string `json:"event,omitempty"`

	CreatedAt *time.Time `json:"created_at,omitempty"`
	Issue     *Issue     `json:"issue,omitempty"`

	// Only present on certain events; see above.
	Assignee          *User            `json:"assignee,omitempty"`
	Assigner          *User            `json:"assigner,omitempty"`
	CommitID          *string          `json:"commit_id,omitempty"`
	Milestone         *Milestone       `json:"milestone,omitempty"`
	Label             *Label           `json:"label,omitempty"`
	Rename            *Rename          `json:"rename,omitempty"`
	LockReason        *string          `json:"lock_reason,omitempty"`
	ProjectCard       *ProjectCard     `json:"project_card,omitempty"`
	DismissedReview   *DismissedReview `json:"dismissed_review,omitempty"`
	RequestedReviewer *User            `json:"requested_reviewer,omitempty"`
	ReviewRequester   *User            `json:"review_requester,omitempty"`
}

// DismissedReview represents details for 'dismissed_review' events.
type DismissedReview struct {
	// State represents the state of the dismissed review.
	// Possible values are: "commented", "approved", and "changes_requested".
	State             *string `json:"state,omitempty"`
	ReviewID          *int64  `json:"review_id,omitempty"`
	DismissalMessage  *string `json:"dismissal_message,omitempty"`
	DismissalCommitID *string `json:"dismissal_commit_id,omitempty"`
}

// ListIssueEvents lists events for the specified issue.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/issues/#list-issue-events
func (s *IssuesService) ListIssueEvents(ctx context.Context, owner, repo string, number int, opts *ListOptions) ([]*IssueEvent, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/issues/%v/events", owner, repo, number)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	req.Header.Set("Accept", mediaTypeProjectCardDetailsPreview)

	var events []*IssueEvent
	resp, err := s.client.Do(ctx, req, &events)
	if err != nil {
		return nil, resp, err
	}

	return events, resp, nil
}

// ListRepositoryEvents lists events for the specified repository.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/issues/#list-issue-events-for-a-repository
func (s *IssuesService) ListRepositoryEvents(ctx context.Context, owner, repo string, opts *ListOptions) ([]*IssueEvent, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/issues/events", owner, repo)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var events []*IssueEvent
	resp, err := s.client.Do(ctx, req, &events)
	if err != nil {
		return nil, resp, err
	}

	return events, resp, nil
}

// GetEvent returns the specified issue event.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/issues/#get-an-issue-event
func (s *IssuesService) GetEvent(ctx context.Context, owner, repo string, id int64) (*IssueEvent, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/issues/events/%v", owner, repo, id)

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	event := new(IssueEvent)
	resp, err := s.client.Do(ctx, req, event)
	if err != nil {
		return nil, resp, err
	}

	return event, resp, nil
}

// Rename contains details for 'renamed' events.
type Rename struct {
	From *string `json:"from,omitempty"`
	To   *string `json:"to,omitempty"`
}

func (r Rename) String() string {
	return Stringify(r)
}

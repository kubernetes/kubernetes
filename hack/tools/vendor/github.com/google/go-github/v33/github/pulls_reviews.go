// Copyright 2016 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"errors"
	"fmt"
	"time"
)

var ErrMixedCommentStyles = errors.New("cannot use both position and side/line form comments")

// PullRequestReview represents a review of a pull request.
type PullRequestReview struct {
	ID             *int64     `json:"id,omitempty"`
	NodeID         *string    `json:"node_id,omitempty"`
	User           *User      `json:"user,omitempty"`
	Body           *string    `json:"body,omitempty"`
	SubmittedAt    *time.Time `json:"submitted_at,omitempty"`
	CommitID       *string    `json:"commit_id,omitempty"`
	HTMLURL        *string    `json:"html_url,omitempty"`
	PullRequestURL *string    `json:"pull_request_url,omitempty"`
	State          *string    `json:"state,omitempty"`
	// AuthorAssociation is the comment author's relationship to the issue's repository.
	// Possible values are "COLLABORATOR", "CONTRIBUTOR", "FIRST_TIMER", "FIRST_TIME_CONTRIBUTOR", "MEMBER", "OWNER", or "NONE".
	AuthorAssociation *string `json:"author_association,omitempty"`
}

func (p PullRequestReview) String() string {
	return Stringify(p)
}

// DraftReviewComment represents a comment part of the review.
type DraftReviewComment struct {
	Path     *string `json:"path,omitempty"`
	Position *int    `json:"position,omitempty"`
	Body     *string `json:"body,omitempty"`

	// The new comfort-fade-preview fields
	StartSide *string `json:"start_side,omitempty"`
	Side      *string `json:"side,omitempty"`
	StartLine *int    `json:"start_line,omitempty"`
	Line      *int    `json:"line,omitempty"`
}

func (c DraftReviewComment) String() string {
	return Stringify(c)
}

// PullRequestReviewRequest represents a request to create a review.
type PullRequestReviewRequest struct {
	NodeID   *string               `json:"node_id,omitempty"`
	CommitID *string               `json:"commit_id,omitempty"`
	Body     *string               `json:"body,omitempty"`
	Event    *string               `json:"event,omitempty"`
	Comments []*DraftReviewComment `json:"comments,omitempty"`
}

func (r PullRequestReviewRequest) String() string {
	return Stringify(r)
}

func (r *PullRequestReviewRequest) isComfortFadePreview() (bool, error) {
	var isCF *bool
	for _, comment := range r.Comments {
		if comment == nil {
			continue
		}
		hasPos := comment.Position != nil
		hasComfortFade := (comment.StartSide != nil) || (comment.Side != nil) ||
			(comment.StartLine != nil) || (comment.Line != nil)

		switch {
		case hasPos && hasComfortFade:
			return false, ErrMixedCommentStyles
		case hasPos && isCF != nil && *isCF:
			return false, ErrMixedCommentStyles
		case hasComfortFade && isCF != nil && !*isCF:
			return false, ErrMixedCommentStyles
		}
		isCF = &hasComfortFade
	}
	if isCF != nil {
		return *isCF, nil
	}
	return false, nil
}

// PullRequestReviewDismissalRequest represents a request to dismiss a review.
type PullRequestReviewDismissalRequest struct {
	Message *string `json:"message,omitempty"`
}

func (r PullRequestReviewDismissalRequest) String() string {
	return Stringify(r)
}

// ListReviews lists all reviews on the specified pull request.
//
// TODO: Follow up with GitHub support about an issue with this method's
// returned error format and remove this comment once it's fixed.
// Read more about it here - https://github.com/google/go-github/issues/540
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/pulls/#list-reviews-for-a-pull-request
func (s *PullRequestsService) ListReviews(ctx context.Context, owner, repo string, number int, opts *ListOptions) ([]*PullRequestReview, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/pulls/%d/reviews", owner, repo, number)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var reviews []*PullRequestReview
	resp, err := s.client.Do(ctx, req, &reviews)
	if err != nil {
		return nil, resp, err
	}

	return reviews, resp, nil
}

// GetReview fetches the specified pull request review.
//
// TODO: Follow up with GitHub support about an issue with this method's
// returned error format and remove this comment once it's fixed.
// Read more about it here - https://github.com/google/go-github/issues/540
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/pulls/#get-a-review-for-a-pull-request
func (s *PullRequestsService) GetReview(ctx context.Context, owner, repo string, number int, reviewID int64) (*PullRequestReview, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/pulls/%d/reviews/%d", owner, repo, number, reviewID)

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	review := new(PullRequestReview)
	resp, err := s.client.Do(ctx, req, review)
	if err != nil {
		return nil, resp, err
	}

	return review, resp, nil
}

// DeletePendingReview deletes the specified pull request pending review.
//
// TODO: Follow up with GitHub support about an issue with this method's
// returned error format and remove this comment once it's fixed.
// Read more about it here - https://github.com/google/go-github/issues/540
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/pulls/#delete-a-pending-review-for-a-pull-request
func (s *PullRequestsService) DeletePendingReview(ctx context.Context, owner, repo string, number int, reviewID int64) (*PullRequestReview, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/pulls/%d/reviews/%d", owner, repo, number, reviewID)

	req, err := s.client.NewRequest("DELETE", u, nil)
	if err != nil {
		return nil, nil, err
	}

	review := new(PullRequestReview)
	resp, err := s.client.Do(ctx, req, review)
	if err != nil {
		return nil, resp, err
	}

	return review, resp, nil
}

// ListReviewComments lists all the comments for the specified review.
//
// TODO: Follow up with GitHub support about an issue with this method's
// returned error format and remove this comment once it's fixed.
// Read more about it here - https://github.com/google/go-github/issues/540
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/pulls/#list-comments-for-a-pull-request-review
func (s *PullRequestsService) ListReviewComments(ctx context.Context, owner, repo string, number int, reviewID int64, opts *ListOptions) ([]*PullRequestComment, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/pulls/%d/reviews/%d/comments", owner, repo, number, reviewID)
	u, err := addOptions(u, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var comments []*PullRequestComment
	resp, err := s.client.Do(ctx, req, &comments)
	if err != nil {
		return nil, resp, err
	}

	return comments, resp, nil
}

// CreateReview creates a new review on the specified pull request.
//
// TODO: Follow up with GitHub support about an issue with this method's
// returned error format and remove this comment once it's fixed.
// Read more about it here - https://github.com/google/go-github/issues/540
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/pulls/#create-a-review-for-a-pull-request
//
// In order to use multi-line comments, you must use the "comfort fade" preview.
// This replaces the use of the "Position" field in comments with 4 new fields:
//   [Start]Side, and [Start]Line.
// These new fields must be used for ALL comments (including single-line),
// with the following restrictions (empirically observed, so subject to change).
//
// For single-line "comfort fade" comments, you must use:
//
//    Path:  &path,  // as before
//    Body:  &body,  // as before
//    Side:  &"RIGHT" (or "LEFT")
//    Line:  &123,  // NOT THE SAME AS POSITION, this is an actual line number.
//
// If StartSide or StartLine is used with single-line comments, a 422 is returned.
//
// For multi-line "comfort fade" comments, you must use:
//
//    Path:      &path,  // as before
//    Body:      &body,  // as before
//    StartSide: &"RIGHT" (or "LEFT")
//    Side:      &"RIGHT" (or "LEFT")
//    StartLine: &120,
//    Line:      &125,
//
// Suggested edits are made by commenting on the lines to replace, and including the
// suggested edit in a block like this (it may be surrounded in non-suggestion markdown):
//
//    ```suggestion
//    Use this instead.
//    It is waaaaaay better.
//    ```
func (s *PullRequestsService) CreateReview(ctx context.Context, owner, repo string, number int, review *PullRequestReviewRequest) (*PullRequestReview, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/pulls/%d/reviews", owner, repo, number)

	req, err := s.client.NewRequest("POST", u, review)
	if err != nil {
		return nil, nil, err
	}

	// Detect which style of review comment is being used.
	if isCF, err := review.isComfortFadePreview(); err != nil {
		return nil, nil, err
	} else if isCF {
		// If the review comments are using the comfort fade preview fields,
		// then pass the comfort fade header.
		req.Header.Set("Accept", mediaTypeMultiLineCommentsPreview)
	}

	r := new(PullRequestReview)
	resp, err := s.client.Do(ctx, req, r)
	if err != nil {
		return nil, resp, err
	}

	return r, resp, nil
}

// UpdateReview updates the review summary on the specified pull request.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/pulls/#update-a-review-for-a-pull-request
func (s *PullRequestsService) UpdateReview(ctx context.Context, owner, repo string, number int, reviewID int64, body string) (*PullRequestReview, *Response, error) {
	opts := &struct {
		Body string `json:"body"`
	}{Body: body}
	u := fmt.Sprintf("repos/%v/%v/pulls/%d/reviews/%d", owner, repo, number, reviewID)

	req, err := s.client.NewRequest("PUT", u, opts)
	if err != nil {
		return nil, nil, err
	}

	review := &PullRequestReview{}
	resp, err := s.client.Do(ctx, req, review)
	if err != nil {
		return nil, resp, err
	}

	return review, resp, nil
}

// SubmitReview submits a specified review on the specified pull request.
//
// TODO: Follow up with GitHub support about an issue with this method's
// returned error format and remove this comment once it's fixed.
// Read more about it here - https://github.com/google/go-github/issues/540
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/pulls/#submit-a-review-for-a-pull-request
func (s *PullRequestsService) SubmitReview(ctx context.Context, owner, repo string, number int, reviewID int64, review *PullRequestReviewRequest) (*PullRequestReview, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/pulls/%d/reviews/%d/events", owner, repo, number, reviewID)

	req, err := s.client.NewRequest("POST", u, review)
	if err != nil {
		return nil, nil, err
	}

	r := new(PullRequestReview)
	resp, err := s.client.Do(ctx, req, r)
	if err != nil {
		return nil, resp, err
	}

	return r, resp, nil
}

// DismissReview dismisses a specified review on the specified pull request.
//
// TODO: Follow up with GitHub support about an issue with this method's
// returned error format and remove this comment once it's fixed.
// Read more about it here - https://github.com/google/go-github/issues/540
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/pulls/#dismiss-a-review-for-a-pull-request
func (s *PullRequestsService) DismissReview(ctx context.Context, owner, repo string, number int, reviewID int64, review *PullRequestReviewDismissalRequest) (*PullRequestReview, *Response, error) {
	u := fmt.Sprintf("repos/%v/%v/pulls/%d/reviews/%d/dismissals", owner, repo, number, reviewID)

	req, err := s.client.NewRequest("PUT", u, review)
	if err != nil {
		return nil, nil, err
	}

	r := new(PullRequestReview)
	resp, err := s.client.Do(ctx, req, r)
	if err != nil {
		return nil, resp, err
	}

	return r, resp, nil
}

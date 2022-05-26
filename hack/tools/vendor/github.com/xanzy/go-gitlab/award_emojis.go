//
// Copyright 2021, Arkbriar
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

package gitlab

import (
	"fmt"
	"time"
)

// AwardEmojiService handles communication with the emoji awards related methods
// of the GitLab API.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/award_emoji.html
type AwardEmojiService struct {
	client *Client
}

// AwardEmoji represents a GitLab Award Emoji.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/award_emoji.html
type AwardEmoji struct {
	ID   int    `json:"id"`
	Name string `json:"name"`
	User struct {
		Name      string `json:"name"`
		Username  string `json:"username"`
		ID        int    `json:"id"`
		State     string `json:"state"`
		AvatarURL string `json:"avatar_url"`
		WebURL    string `json:"web_url"`
	} `json:"user"`
	CreatedAt     *time.Time `json:"created_at"`
	UpdatedAt     *time.Time `json:"updated_at"`
	AwardableID   int        `json:"awardable_id"`
	AwardableType string     `json:"awardable_type"`
}

const (
	awardMergeRequest = "merge_requests"
	awardIssue        = "issues"
	awardSnippets     = "snippets"
)

// ListAwardEmojiOptions represents the available options for listing emoji
// for each resources
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/award_emoji.html
type ListAwardEmojiOptions ListOptions

// ListMergeRequestAwardEmoji gets a list of all award emoji on the merge request.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/award_emoji.html#list-an-awardable-39-s-award-emoji
func (s *AwardEmojiService) ListMergeRequestAwardEmoji(pid interface{}, mergeRequestIID int, opt *ListAwardEmojiOptions, options ...RequestOptionFunc) ([]*AwardEmoji, *Response, error) {
	return s.listAwardEmoji(pid, awardMergeRequest, mergeRequestIID, opt, options...)
}

// ListIssueAwardEmoji gets a list of all award emoji on the issue.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/award_emoji.html#list-an-awardable-39-s-award-emoji
func (s *AwardEmojiService) ListIssueAwardEmoji(pid interface{}, issueIID int, opt *ListAwardEmojiOptions, options ...RequestOptionFunc) ([]*AwardEmoji, *Response, error) {
	return s.listAwardEmoji(pid, awardIssue, issueIID, opt, options...)
}

// ListSnippetAwardEmoji gets a list of all award emoji on the snippet.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/award_emoji.html#list-an-awardable-39-s-award-emoji
func (s *AwardEmojiService) ListSnippetAwardEmoji(pid interface{}, snippetID int, opt *ListAwardEmojiOptions, options ...RequestOptionFunc) ([]*AwardEmoji, *Response, error) {
	return s.listAwardEmoji(pid, awardSnippets, snippetID, opt, options...)
}

func (s *AwardEmojiService) listAwardEmoji(pid interface{}, resource string, resourceID int, opt *ListAwardEmojiOptions, options ...RequestOptionFunc) ([]*AwardEmoji, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/%s/%d/award_emoji",
		pathEscape(project),
		resource,
		resourceID,
	)

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var as []*AwardEmoji
	resp, err := s.client.Do(req, &as)
	if err != nil {
		return nil, resp, err
	}

	return as, resp, err
}

// GetMergeRequestAwardEmoji get an award emoji from merge request.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/award_emoji.html#list-an-awardable-39-s-award-emoji
func (s *AwardEmojiService) GetMergeRequestAwardEmoji(pid interface{}, mergeRequestIID, awardID int, options ...RequestOptionFunc) (*AwardEmoji, *Response, error) {
	return s.getAwardEmoji(pid, awardMergeRequest, mergeRequestIID, awardID, options...)
}

// GetIssueAwardEmoji get an award emoji from issue.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/award_emoji.html#list-an-awardable-39-s-award-emoji
func (s *AwardEmojiService) GetIssueAwardEmoji(pid interface{}, issueIID, awardID int, options ...RequestOptionFunc) (*AwardEmoji, *Response, error) {
	return s.getAwardEmoji(pid, awardIssue, issueIID, awardID, options...)
}

// GetSnippetAwardEmoji get an award emoji from snippet.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/award_emoji.html#list-an-awardable-39-s-award-emoji
func (s *AwardEmojiService) GetSnippetAwardEmoji(pid interface{}, snippetID, awardID int, options ...RequestOptionFunc) (*AwardEmoji, *Response, error) {
	return s.getAwardEmoji(pid, awardSnippets, snippetID, awardID, options...)
}

func (s *AwardEmojiService) getAwardEmoji(pid interface{}, resource string, resourceID, awardID int, options ...RequestOptionFunc) (*AwardEmoji, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/%s/%d/award_emoji/%d",
		pathEscape(project),
		resource,
		resourceID,
		awardID,
	)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	a := new(AwardEmoji)
	resp, err := s.client.Do(req, &a)
	if err != nil {
		return nil, resp, err
	}

	return a, resp, err
}

// CreateAwardEmojiOptions represents the available options for awarding emoji
// for a resource
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/award_emoji.html#award-a-new-emoji
type CreateAwardEmojiOptions struct {
	Name string `json:"name"`
}

// CreateMergeRequestAwardEmoji get an award emoji from merge request.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/award_emoji.html#award-a-new-emoji
func (s *AwardEmojiService) CreateMergeRequestAwardEmoji(pid interface{}, mergeRequestIID int, opt *CreateAwardEmojiOptions, options ...RequestOptionFunc) (*AwardEmoji, *Response, error) {
	return s.createAwardEmoji(pid, awardMergeRequest, mergeRequestIID, opt, options...)
}

// CreateIssueAwardEmoji get an award emoji from issue.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/award_emoji.html#award-a-new-emoji
func (s *AwardEmojiService) CreateIssueAwardEmoji(pid interface{}, issueIID int, opt *CreateAwardEmojiOptions, options ...RequestOptionFunc) (*AwardEmoji, *Response, error) {
	return s.createAwardEmoji(pid, awardIssue, issueIID, opt, options...)
}

// CreateSnippetAwardEmoji get an award emoji from snippet.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/award_emoji.html#award-a-new-emoji
func (s *AwardEmojiService) CreateSnippetAwardEmoji(pid interface{}, snippetID int, opt *CreateAwardEmojiOptions, options ...RequestOptionFunc) (*AwardEmoji, *Response, error) {
	return s.createAwardEmoji(pid, awardSnippets, snippetID, opt, options...)
}

func (s *AwardEmojiService) createAwardEmoji(pid interface{}, resource string, resourceID int, opt *CreateAwardEmojiOptions, options ...RequestOptionFunc) (*AwardEmoji, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/%s/%d/award_emoji",
		pathEscape(project),
		resource,
		resourceID,
	)

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	a := new(AwardEmoji)
	resp, err := s.client.Do(req, &a)
	if err != nil {
		return nil, resp, err
	}

	return a, resp, err
}

// DeleteIssueAwardEmoji delete award emoji on an issue.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/award_emoji.html#award-a-new-emoji-on-a-note
func (s *AwardEmojiService) DeleteIssueAwardEmoji(pid interface{}, issueIID, awardID int, options ...RequestOptionFunc) (*Response, error) {
	return s.deleteAwardEmoji(pid, awardMergeRequest, issueIID, awardID, options...)
}

// DeleteMergeRequestAwardEmoji delete award emoji on a merge request.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/award_emoji.html#award-a-new-emoji-on-a-note
func (s *AwardEmojiService) DeleteMergeRequestAwardEmoji(pid interface{}, mergeRequestIID, awardID int, options ...RequestOptionFunc) (*Response, error) {
	return s.deleteAwardEmoji(pid, awardMergeRequest, mergeRequestIID, awardID, options...)
}

// DeleteSnippetAwardEmoji delete award emoji on a snippet.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/award_emoji.html#award-a-new-emoji-on-a-note
func (s *AwardEmojiService) DeleteSnippetAwardEmoji(pid interface{}, snippetID, awardID int, options ...RequestOptionFunc) (*Response, error) {
	return s.deleteAwardEmoji(pid, awardMergeRequest, snippetID, awardID, options...)
}

// DeleteAwardEmoji Delete an award emoji on the specified resource.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/award_emoji.html#delete-an-award-emoji
func (s *AwardEmojiService) deleteAwardEmoji(pid interface{}, resource string, resourceID, awardID int, options ...RequestOptionFunc) (*Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("projects/%s/%s/%d/award_emoji/%d", pathEscape(project), resource,
		resourceID, awardID)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}
	return s.client.Do(req, nil)
}

// ListIssuesAwardEmojiOnNote gets a list of all award emoji on a note from the
// issue.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/award_emoji.html#award-emoji-on-notes
func (s *AwardEmojiService) ListIssuesAwardEmojiOnNote(pid interface{}, issueID, noteID int, opt *ListAwardEmojiOptions, options ...RequestOptionFunc) ([]*AwardEmoji, *Response, error) {
	return s.listAwardEmojiOnNote(pid, awardIssue, issueID, noteID, opt, options...)
}

// ListMergeRequestAwardEmojiOnNote gets a list of all award emoji on a note
// from the merge request.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/award_emoji.html#award-emoji-on-notes
func (s *AwardEmojiService) ListMergeRequestAwardEmojiOnNote(pid interface{}, mergeRequestIID, noteID int, opt *ListAwardEmojiOptions, options ...RequestOptionFunc) ([]*AwardEmoji, *Response, error) {
	return s.listAwardEmojiOnNote(pid, awardMergeRequest, mergeRequestIID, noteID, opt, options...)
}

// ListSnippetAwardEmojiOnNote gets a list of all award emoji on a note from the
// snippet.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/award_emoji.html#award-emoji-on-notes
func (s *AwardEmojiService) ListSnippetAwardEmojiOnNote(pid interface{}, snippetIID, noteID int, opt *ListAwardEmojiOptions, options ...RequestOptionFunc) ([]*AwardEmoji, *Response, error) {
	return s.listAwardEmojiOnNote(pid, awardSnippets, snippetIID, noteID, opt, options...)
}

func (s *AwardEmojiService) listAwardEmojiOnNote(pid interface{}, resources string, ressourceID, noteID int, opt *ListAwardEmojiOptions, options ...RequestOptionFunc) ([]*AwardEmoji, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/%s/%d/notes/%d/award_emoji", pathEscape(project), resources,
		ressourceID, noteID)

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var as []*AwardEmoji
	resp, err := s.client.Do(req, &as)
	if err != nil {
		return nil, resp, err
	}

	return as, resp, err
}

// GetIssuesAwardEmojiOnNote gets an award emoji on a note from an issue.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/award_emoji.html#award-emoji-on-notes
func (s *AwardEmojiService) GetIssuesAwardEmojiOnNote(pid interface{}, issueID, noteID, awardID int, options ...RequestOptionFunc) (*AwardEmoji, *Response, error) {
	return s.getSingleNoteAwardEmoji(pid, awardIssue, issueID, noteID, awardID, options...)
}

// GetMergeRequestAwardEmojiOnNote gets an award emoji on a note from a
// merge request.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/award_emoji.html#award-emoji-on-notes
func (s *AwardEmojiService) GetMergeRequestAwardEmojiOnNote(pid interface{}, mergeRequestIID, noteID, awardID int, options ...RequestOptionFunc) (*AwardEmoji, *Response, error) {
	return s.getSingleNoteAwardEmoji(pid, awardMergeRequest, mergeRequestIID, noteID, awardID,
		options...)
}

// GetSnippetAwardEmojiOnNote gets an award emoji on a note from a snippet.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/award_emoji.html#award-emoji-on-notes
func (s *AwardEmojiService) GetSnippetAwardEmojiOnNote(pid interface{}, snippetIID, noteID, awardID int, options ...RequestOptionFunc) (*AwardEmoji, *Response, error) {
	return s.getSingleNoteAwardEmoji(pid, awardSnippets, snippetIID, noteID, awardID, options...)
}

func (s *AwardEmojiService) getSingleNoteAwardEmoji(pid interface{}, ressource string, resourceID, noteID, awardID int, options ...RequestOptionFunc) (*AwardEmoji, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/%s/%d/notes/%d/award_emoji/%d",
		pathEscape(project),
		ressource,
		resourceID,
		noteID,
		awardID,
	)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	a := new(AwardEmoji)
	resp, err := s.client.Do(req, &a)
	if err != nil {
		return nil, resp, err
	}

	return a, resp, err
}

// CreateIssuesAwardEmojiOnNote gets an award emoji on a note from an issue.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/award_emoji.html#award-emoji-on-notes
func (s *AwardEmojiService) CreateIssuesAwardEmojiOnNote(pid interface{}, issueID, noteID int, opt *CreateAwardEmojiOptions, options ...RequestOptionFunc) (*AwardEmoji, *Response, error) {
	return s.createAwardEmojiOnNote(pid, awardIssue, issueID, noteID, opt, options...)
}

// CreateMergeRequestAwardEmojiOnNote gets an award emoji on a note from a
// merge request.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/award_emoji.html#award-emoji-on-notes
func (s *AwardEmojiService) CreateMergeRequestAwardEmojiOnNote(pid interface{}, mergeRequestIID, noteID int, opt *CreateAwardEmojiOptions, options ...RequestOptionFunc) (*AwardEmoji, *Response, error) {
	return s.createAwardEmojiOnNote(pid, awardMergeRequest, mergeRequestIID, noteID, opt, options...)
}

// CreateSnippetAwardEmojiOnNote gets an award emoji on a note from a snippet.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/award_emoji.html#award-emoji-on-notes
func (s *AwardEmojiService) CreateSnippetAwardEmojiOnNote(pid interface{}, snippetIID, noteID int, opt *CreateAwardEmojiOptions, options ...RequestOptionFunc) (*AwardEmoji, *Response, error) {
	return s.createAwardEmojiOnNote(pid, awardSnippets, snippetIID, noteID, opt, options...)
}

// CreateAwardEmojiOnNote award emoji on a note.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/award_emoji.html#award-a-new-emoji-on-a-note
func (s *AwardEmojiService) createAwardEmojiOnNote(pid interface{}, resource string, resourceID, noteID int, opt *CreateAwardEmojiOptions, options ...RequestOptionFunc) (*AwardEmoji, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/%s/%d/notes/%d/award_emoji",
		pathEscape(project),
		resource,
		resourceID,
		noteID,
	)

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	a := new(AwardEmoji)
	resp, err := s.client.Do(req, &a)
	if err != nil {
		return nil, resp, err
	}

	return a, resp, err
}

// DeleteIssuesAwardEmojiOnNote deletes an award emoji on a note from an issue.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/award_emoji.html#award-emoji-on-notes
func (s *AwardEmojiService) DeleteIssuesAwardEmojiOnNote(pid interface{}, issueID, noteID, awardID int, options ...RequestOptionFunc) (*Response, error) {
	return s.deleteAwardEmojiOnNote(pid, awardIssue, issueID, noteID, awardID, options...)
}

// DeleteMergeRequestAwardEmojiOnNote deletes an award emoji on a note from a
// merge request.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/award_emoji.html#award-emoji-on-notes
func (s *AwardEmojiService) DeleteMergeRequestAwardEmojiOnNote(pid interface{}, mergeRequestIID, noteID, awardID int, options ...RequestOptionFunc) (*Response, error) {
	return s.deleteAwardEmojiOnNote(pid, awardMergeRequest, mergeRequestIID, noteID, awardID,
		options...)
}

// DeleteSnippetAwardEmojiOnNote deletes an award emoji on a note from a snippet.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/award_emoji.html#award-emoji-on-notes
func (s *AwardEmojiService) DeleteSnippetAwardEmojiOnNote(pid interface{}, snippetIID, noteID, awardID int, options ...RequestOptionFunc) (*Response, error) {
	return s.deleteAwardEmojiOnNote(pid, awardSnippets, snippetIID, noteID, awardID, options...)
}

func (s *AwardEmojiService) deleteAwardEmojiOnNote(pid interface{}, resource string, resourceID, noteID, awardID int, options ...RequestOptionFunc) (*Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("projects/%s/%s/%d/notes/%d/award_emoji/%d",
		pathEscape(project),
		resource,
		resourceID,
		noteID,
		awardID,
	)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

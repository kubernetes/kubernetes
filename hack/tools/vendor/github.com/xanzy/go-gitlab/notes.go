//
// Copyright 2021, Sander van Harmelen
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

// NotesService handles communication with the notes related methods
// of the GitLab API.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/notes.html
type NotesService struct {
	client *Client
}

// Note represents a GitLab note.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/notes.html
type Note struct {
	ID         int    `json:"id"`
	Body       string `json:"body"`
	Attachment string `json:"attachment"`
	Title      string `json:"title"`
	FileName   string `json:"file_name"`
	Author     struct {
		ID        int    `json:"id"`
		Username  string `json:"username"`
		Email     string `json:"email"`
		Name      string `json:"name"`
		State     string `json:"state"`
		AvatarURL string `json:"avatar_url"`
		WebURL    string `json:"web_url"`
	} `json:"author"`
	System       bool          `json:"system"`
	ExpiresAt    *time.Time    `json:"expires_at"`
	UpdatedAt    *time.Time    `json:"updated_at"`
	CreatedAt    *time.Time    `json:"created_at"`
	NoteableID   int           `json:"noteable_id"`
	NoteableType string        `json:"noteable_type"`
	Position     *NotePosition `json:"position"`
	Resolvable   bool          `json:"resolvable"`
	Resolved     bool          `json:"resolved"`
	ResolvedBy   struct {
		ID        int    `json:"id"`
		Username  string `json:"username"`
		Email     string `json:"email"`
		Name      string `json:"name"`
		State     string `json:"state"`
		AvatarURL string `json:"avatar_url"`
		WebURL    string `json:"web_url"`
	} `json:"resolved_by"`
	NoteableIID int `json:"noteable_iid"`
}

// NotePosition represents the position attributes of a note.
type NotePosition struct {
	BaseSHA      string     `json:"base_sha"`
	StartSHA     string     `json:"start_sha"`
	HeadSHA      string     `json:"head_sha"`
	PositionType string     `json:"position_type"`
	NewPath      string     `json:"new_path,omitempty"`
	NewLine      int        `json:"new_line,omitempty"`
	OldPath      string     `json:"old_path,omitempty"`
	OldLine      int        `json:"old_line,omitempty"`
	LineRange    *LineRange `json:"line_range"`
}

// LineRange represents the range of a note.
type LineRange struct {
	StartRange *LinePosition `json:"start"`
	EndRange   *LinePosition `json:"end"`
}

// LinePosition represents a position in a line range.
type LinePosition struct {
	LineCode string `json:"line_code"`
	Type     string `json:"type"`
	OldLine  int    `json:"old_line"`
	NewLine  int    `json:"new_line"`
}

func (n Note) String() string {
	return Stringify(n)
}

// ListIssueNotesOptions represents the available ListIssueNotes() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/notes.html#list-project-issue-notes
type ListIssueNotesOptions struct {
	ListOptions
	OrderBy *string `url:"order_by,omitempty" json:"order_by,omitempty"`
	Sort    *string `url:"sort,omitempty" json:"sort,omitempty"`
}

// ListIssueNotes gets a list of all notes for a single issue.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/notes.html#list-project-issue-notes
func (s *NotesService) ListIssueNotes(pid interface{}, issue int, opt *ListIssueNotesOptions, options ...RequestOptionFunc) ([]*Note, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/issues/%d/notes", pathEscape(project), issue)

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var n []*Note
	resp, err := s.client.Do(req, &n)
	if err != nil {
		return nil, resp, err
	}

	return n, resp, err
}

// GetIssueNote returns a single note for a specific project issue.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/notes.html#get-single-issue-note
func (s *NotesService) GetIssueNote(pid interface{}, issue, note int, options ...RequestOptionFunc) (*Note, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/issues/%d/notes/%d", pathEscape(project), issue, note)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	n := new(Note)
	resp, err := s.client.Do(req, n)
	if err != nil {
		return nil, resp, err
	}

	return n, resp, err
}

// CreateIssueNoteOptions represents the available CreateIssueNote()
// options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/notes.html#create-new-issue-note
type CreateIssueNoteOptions struct {
	Body      *string    `url:"body,omitempty" json:"body,omitempty"`
	CreatedAt *time.Time `url:"created_at,omitempty" json:"created_at,omitempty"`
}

// CreateIssueNote creates a new note to a single project issue.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/notes.html#create-new-issue-note
func (s *NotesService) CreateIssueNote(pid interface{}, issue int, opt *CreateIssueNoteOptions, options ...RequestOptionFunc) (*Note, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/issues/%d/notes", pathEscape(project), issue)

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	n := new(Note)
	resp, err := s.client.Do(req, n)
	if err != nil {
		return nil, resp, err
	}

	return n, resp, err
}

// UpdateIssueNoteOptions represents the available UpdateIssueNote()
// options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/notes.html#modify-existing-issue-note
type UpdateIssueNoteOptions struct {
	Body *string `url:"body,omitempty" json:"body,omitempty"`
}

// UpdateIssueNote modifies existing note of an issue.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/notes.html#modify-existing-issue-note
func (s *NotesService) UpdateIssueNote(pid interface{}, issue, note int, opt *UpdateIssueNoteOptions, options ...RequestOptionFunc) (*Note, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/issues/%d/notes/%d", pathEscape(project), issue, note)

	req, err := s.client.NewRequest("PUT", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	n := new(Note)
	resp, err := s.client.Do(req, n)
	if err != nil {
		return nil, resp, err
	}

	return n, resp, err
}

// DeleteIssueNote deletes an existing note of an issue.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/notes.html#delete-an-issue-note
func (s *NotesService) DeleteIssueNote(pid interface{}, issue, note int, options ...RequestOptionFunc) (*Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("projects/%s/issues/%d/notes/%d", pathEscape(project), issue, note)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

// ListSnippetNotesOptions represents the available ListSnippetNotes() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/notes.html#list-all-snippet-notes
type ListSnippetNotesOptions struct {
	ListOptions
	OrderBy *string `url:"order_by,omitempty" json:"order_by,omitempty"`
	Sort    *string `url:"sort,omitempty" json:"sort,omitempty"`
}

// ListSnippetNotes gets a list of all notes for a single snippet. Snippet
// notes are comments users can post to a snippet.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/notes.html#list-all-snippet-notes
func (s *NotesService) ListSnippetNotes(pid interface{}, snippet int, opt *ListSnippetNotesOptions, options ...RequestOptionFunc) ([]*Note, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/snippets/%d/notes", pathEscape(project), snippet)

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var n []*Note
	resp, err := s.client.Do(req, &n)
	if err != nil {
		return nil, resp, err
	}

	return n, resp, err
}

// GetSnippetNote returns a single note for a given snippet.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/notes.html#get-single-snippet-note
func (s *NotesService) GetSnippetNote(pid interface{}, snippet, note int, options ...RequestOptionFunc) (*Note, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/snippets/%d/notes/%d", pathEscape(project), snippet, note)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	n := new(Note)
	resp, err := s.client.Do(req, n)
	if err != nil {
		return nil, resp, err
	}

	return n, resp, err
}

// CreateSnippetNoteOptions represents the available CreateSnippetNote()
// options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/notes.html#create-new-snippet-note
type CreateSnippetNoteOptions struct {
	Body *string `url:"body,omitempty" json:"body,omitempty"`
}

// CreateSnippetNote creates a new note for a single snippet. Snippet notes are
// comments users can post to a snippet.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/notes.html#create-new-snippet-note
func (s *NotesService) CreateSnippetNote(pid interface{}, snippet int, opt *CreateSnippetNoteOptions, options ...RequestOptionFunc) (*Note, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/snippets/%d/notes", pathEscape(project), snippet)

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	n := new(Note)
	resp, err := s.client.Do(req, n)
	if err != nil {
		return nil, resp, err
	}

	return n, resp, err
}

// UpdateSnippetNoteOptions represents the available UpdateSnippetNote()
// options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/notes.html#modify-existing-snippet-note
type UpdateSnippetNoteOptions struct {
	Body *string `url:"body,omitempty" json:"body,omitempty"`
}

// UpdateSnippetNote modifies existing note of a snippet.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/notes.html#modify-existing-snippet-note
func (s *NotesService) UpdateSnippetNote(pid interface{}, snippet, note int, opt *UpdateSnippetNoteOptions, options ...RequestOptionFunc) (*Note, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/snippets/%d/notes/%d", pathEscape(project), snippet, note)

	req, err := s.client.NewRequest("PUT", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	n := new(Note)
	resp, err := s.client.Do(req, n)
	if err != nil {
		return nil, resp, err
	}

	return n, resp, err
}

// DeleteSnippetNote deletes an existing note of a snippet.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/notes.html#delete-a-snippet-note
func (s *NotesService) DeleteSnippetNote(pid interface{}, snippet, note int, options ...RequestOptionFunc) (*Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("projects/%s/snippets/%d/notes/%d", pathEscape(project), snippet, note)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

// ListMergeRequestNotesOptions represents the available ListMergeRequestNotes()
// options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/notes.html#list-all-merge-request-notes
type ListMergeRequestNotesOptions struct {
	ListOptions
	OrderBy *string `url:"order_by,omitempty" json:"order_by,omitempty"`
	Sort    *string `url:"sort,omitempty" json:"sort,omitempty"`
}

// ListMergeRequestNotes gets a list of all notes for a single merge request.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/notes.html#list-all-merge-request-notes
func (s *NotesService) ListMergeRequestNotes(pid interface{}, mergeRequest int, opt *ListMergeRequestNotesOptions, options ...RequestOptionFunc) ([]*Note, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/merge_requests/%d/notes", pathEscape(project), mergeRequest)

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var n []*Note
	resp, err := s.client.Do(req, &n)
	if err != nil {
		return nil, resp, err
	}

	return n, resp, err
}

// GetMergeRequestNote returns a single note for a given merge request.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/notes.html#get-single-merge-request-note
func (s *NotesService) GetMergeRequestNote(pid interface{}, mergeRequest, note int, options ...RequestOptionFunc) (*Note, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/merge_requests/%d/notes/%d", pathEscape(project), mergeRequest, note)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	n := new(Note)
	resp, err := s.client.Do(req, n)
	if err != nil {
		return nil, resp, err
	}

	return n, resp, err
}

// CreateMergeRequestNoteOptions represents the available
// CreateMergeRequestNote() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/notes.html#create-new-merge-request-note
type CreateMergeRequestNoteOptions struct {
	Body *string `url:"body,omitempty" json:"body,omitempty"`
}

// CreateMergeRequestNote creates a new note for a single merge request.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/notes.html#create-new-merge-request-note
func (s *NotesService) CreateMergeRequestNote(pid interface{}, mergeRequest int, opt *CreateMergeRequestNoteOptions, options ...RequestOptionFunc) (*Note, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/merge_requests/%d/notes", pathEscape(project), mergeRequest)

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	n := new(Note)
	resp, err := s.client.Do(req, n)
	if err != nil {
		return nil, resp, err
	}

	return n, resp, err
}

// UpdateMergeRequestNoteOptions represents the available
// UpdateMergeRequestNote() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/notes.html#modify-existing-merge-request-note
type UpdateMergeRequestNoteOptions struct {
	Body *string `url:"body,omitempty" json:"body,omitempty"`
}

// UpdateMergeRequestNote modifies existing note of a merge request.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/notes.html#modify-existing-merge-request-note
func (s *NotesService) UpdateMergeRequestNote(pid interface{}, mergeRequest, note int, opt *UpdateMergeRequestNoteOptions, options ...RequestOptionFunc) (*Note, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf(
		"projects/%s/merge_requests/%d/notes/%d", pathEscape(project), mergeRequest, note)
	req, err := s.client.NewRequest("PUT", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	n := new(Note)
	resp, err := s.client.Do(req, n)
	if err != nil {
		return nil, resp, err
	}

	return n, resp, err
}

// DeleteMergeRequestNote deletes an existing note of a merge request.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/notes.html#delete-a-merge-request-note
func (s *NotesService) DeleteMergeRequestNote(pid interface{}, mergeRequest, note int, options ...RequestOptionFunc) (*Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf(
		"projects/%s/merge_requests/%d/notes/%d", pathEscape(project), mergeRequest, note)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

// ListEpicNotesOptions represents the available ListEpicNotes() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/notes.html#list-all-epic-notes
type ListEpicNotesOptions struct {
	ListOptions
	OrderBy *string `url:"order_by,omitempty" json:"order_by,omitempty"`
	Sort    *string `url:"sort,omitempty" json:"sort,omitempty"`
}

// ListEpicNotes gets a list of all notes for a single epic.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/notes.html#list-all-epic-notes
func (s *NotesService) ListEpicNotes(gid interface{}, epic int, opt *ListEpicNotesOptions, options ...RequestOptionFunc) ([]*Note, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/epics/%d/notes", pathEscape(group), epic)

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var n []*Note
	resp, err := s.client.Do(req, &n)
	if err != nil {
		return nil, resp, err
	}

	return n, resp, err
}

// GetEpicNote returns a single note for an epic.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/notes.html#get-single-epic-note
func (s *NotesService) GetEpicNote(gid interface{}, epic, note int, options ...RequestOptionFunc) (*Note, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/epics/%d/notes/%d", pathEscape(group), epic, note)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	n := new(Note)
	resp, err := s.client.Do(req, n)
	if err != nil {
		return nil, resp, err
	}

	return n, resp, err
}

// CreateEpicNoteOptions represents the available CreateEpicNote() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/notes.html#create-new-epic-note
type CreateEpicNoteOptions struct {
	Body *string `url:"body,omitempty" json:"body,omitempty"`
}

// CreateEpicNote creates a new note for a single merge request.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/notes.html#create-new-epic-note
func (s *NotesService) CreateEpicNote(gid interface{}, epic int, opt *CreateEpicNoteOptions, options ...RequestOptionFunc) (*Note, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/epics/%d/notes", pathEscape(group), epic)

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	n := new(Note)
	resp, err := s.client.Do(req, n)
	if err != nil {
		return nil, resp, err
	}

	return n, resp, err
}

// UpdateEpicNoteOptions represents the available UpdateEpicNote() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/notes.html#modify-existing-epic-note
type UpdateEpicNoteOptions struct {
	Body *string `url:"body,omitempty" json:"body,omitempty"`
}

// UpdateEpicNote modifies existing note of an epic.
//
// https://docs.gitlab.com/ee/api/notes.html#modify-existing-epic-note
func (s *NotesService) UpdateEpicNote(gid interface{}, epic, note int, opt *UpdateEpicNoteOptions, options ...RequestOptionFunc) (*Note, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/epics/%d/notes/%d", pathEscape(group), epic, note)

	req, err := s.client.NewRequest("PUT", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	n := new(Note)
	resp, err := s.client.Do(req, n)
	if err != nil {
		return nil, resp, err
	}

	return n, resp, err
}

// DeleteEpicNote deletes an existing note of a merge request.
//
// https://docs.gitlab.com/ee/api/notes.html#delete-an-epic-note
func (s *NotesService) DeleteEpicNote(gid interface{}, epic, note int, options ...RequestOptionFunc) (*Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("groups/%s/epics/%d/notes/%d", pathEscape(group), epic, note)

	req, err := s.client.NewRequest("DELETE", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

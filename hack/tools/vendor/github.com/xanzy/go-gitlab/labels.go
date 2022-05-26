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
	"encoding/json"
	"fmt"
)

// LabelsService handles communication with the label related methods of the
// GitLab API.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/labels.html
type LabelsService struct {
	client *Client
}

// Label represents a GitLab label.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/labels.html
type Label struct {
	ID                     int    `json:"id"`
	Name                   string `json:"name"`
	Color                  string `json:"color"`
	TextColor              string `json:"text_color"`
	Description            string `json:"description"`
	OpenIssuesCount        int    `json:"open_issues_count"`
	ClosedIssuesCount      int    `json:"closed_issues_count"`
	OpenMergeRequestsCount int    `json:"open_merge_requests_count"`
	Subscribed             bool   `json:"subscribed"`
	Priority               int    `json:"priority"`
	IsProjectLabel         bool   `json:"is_project_label"`
}

// UnmarshalJSON implements the json.Unmarshaler interface.
func (l *Label) UnmarshalJSON(data []byte) error {
	type alias Label
	if err := json.Unmarshal(data, (*alias)(l)); err != nil {
		return err
	}

	if l.Name == "" {
		var raw map[string]interface{}
		if err := json.Unmarshal(data, &raw); err != nil {
			return err
		}
		if title, ok := raw["title"].(string); ok {
			l.Name = title
		}
	}

	return nil
}

func (l Label) String() string {
	return Stringify(l)
}

// ListLabelsOptions represents the available ListLabels() options.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/labels.html#list-labels
type ListLabelsOptions struct {
	ListOptions
	WithCounts            *bool `url:"with_counts,omitempty" json:"with_counts,omitempty"`
	IncludeAncestorGroups *bool `url:"include_ancestor_groups,omitempty" json:"include_ancestor_groups,omitempty"`
}

// ListLabels gets all labels for given project.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/labels.html#list-labels
func (s *LabelsService) ListLabels(pid interface{}, opt *ListLabelsOptions, options ...RequestOptionFunc) ([]*Label, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/labels", pathEscape(project))

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var l []*Label
	resp, err := s.client.Do(req, &l)
	if err != nil {
		return nil, resp, err
	}

	return l, resp, err
}

// GetLabel get a single label for a given project.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/labels.html#get-a-single-project-label
func (s *LabelsService) GetLabel(pid interface{}, labelID interface{}, options ...RequestOptionFunc) (*Label, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	label, err := parseID(labelID)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/labels/%s", pathEscape(project), label)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	var l *Label
	resp, err := s.client.Do(req, &l)
	if err != nil {
		return nil, resp, err
	}

	return l, resp, err
}

// CreateLabelOptions represents the available CreateLabel() options.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/labels.html#create-a-new-label
type CreateLabelOptions struct {
	Name        *string `url:"name,omitempty" json:"name,omitempty"`
	Color       *string `url:"color,omitempty" json:"color,omitempty"`
	Description *string `url:"description,omitempty" json:"description,omitempty"`
}

// CreateLabel creates a new label for given repository with given name and
// color.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/labels.html#create-a-new-label
func (s *LabelsService) CreateLabel(pid interface{}, opt *CreateLabelOptions, options ...RequestOptionFunc) (*Label, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/labels", pathEscape(project))

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	l := new(Label)
	resp, err := s.client.Do(req, l)
	if err != nil {
		return nil, resp, err
	}

	return l, resp, err
}

// DeleteLabelOptions represents the available DeleteLabel() options.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/labels.html#delete-a-label
type DeleteLabelOptions struct {
	Name *string `url:"name,omitempty" json:"name,omitempty"`
}

// DeleteLabel deletes a label given by its name.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/labels.html#delete-a-label
func (s *LabelsService) DeleteLabel(pid interface{}, opt *DeleteLabelOptions, options ...RequestOptionFunc) (*Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("projects/%s/labels", pathEscape(project))

	req, err := s.client.NewRequest("DELETE", u, opt, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

// UpdateLabelOptions represents the available UpdateLabel() options.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/labels.html#delete-a-label
type UpdateLabelOptions struct {
	Name        *string `url:"name,omitempty" json:"name,omitempty"`
	NewName     *string `url:"new_name,omitempty" json:"new_name,omitempty"`
	Color       *string `url:"color,omitempty" json:"color,omitempty"`
	Description *string `url:"description,omitempty" json:"description,omitempty"`
}

// UpdateLabel updates an existing label with new name or now color. At least
// one parameter is required, to update the label.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/labels.html#edit-an-existing-label
func (s *LabelsService) UpdateLabel(pid interface{}, opt *UpdateLabelOptions, options ...RequestOptionFunc) (*Label, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/labels", pathEscape(project))

	req, err := s.client.NewRequest("PUT", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	l := new(Label)
	resp, err := s.client.Do(req, l)
	if err != nil {
		return nil, resp, err
	}

	return l, resp, err
}

// SubscribeToLabel subscribes the authenticated user to a label to receive
// notifications. If the user is already subscribed to the label, the status
// code 304 is returned.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/labels.html#subscribe-to-a-label
func (s *LabelsService) SubscribeToLabel(pid interface{}, labelID interface{}, options ...RequestOptionFunc) (*Label, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	label, err := parseID(labelID)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/labels/%s/subscribe", pathEscape(project), label)

	req, err := s.client.NewRequest("POST", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	l := new(Label)
	resp, err := s.client.Do(req, l)
	if err != nil {
		return nil, resp, err
	}

	return l, resp, err
}

// UnsubscribeFromLabel unsubscribes the authenticated user from a label to not
// receive notifications from it. If the user is not subscribed to the label, the
// status code 304 is returned.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/labels.html#unsubscribe-from-a-label
func (s *LabelsService) UnsubscribeFromLabel(pid interface{}, labelID interface{}, options ...RequestOptionFunc) (*Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, err
	}
	label, err := parseID(labelID)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("projects/%s/labels/%s/unsubscribe", pathEscape(project), label)

	req, err := s.client.NewRequest("POST", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

// PromoteLabel Promotes a project label to a group label.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/labels.html#promote-a-project-label-to-a-group-label
func (s *LabelsService) PromoteLabel(pid interface{}, labelID interface{}, options ...RequestOptionFunc) (*Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, err
	}
	label, err := parseID(labelID)
	if err != nil {
		return nil, err
	}
	u := fmt.Sprintf("projects/%s/labels/%s/promote", pathEscape(project), label)

	req, err := s.client.NewRequest("PUT", u, nil, options)
	if err != nil {
		return nil, err
	}

	return s.client.Do(req, nil)
}

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
)

// timeStatsService handles communication with the time tracking related
// methods of the GitLab API.
//
// GitLab docs: https://docs.gitlab.com/ce/workflow/time_tracking.html
type timeStatsService struct {
	client *Client
}

// TimeStats represents the time estimates and time spent for an issue.
//
// GitLab docs: https://docs.gitlab.com/ce/workflow/time_tracking.html
type TimeStats struct {
	HumanTimeEstimate   string `json:"human_time_estimate"`
	HumanTotalTimeSpent string `json:"human_total_time_spent"`
	TimeEstimate        int    `json:"time_estimate"`
	TotalTimeSpent      int    `json:"total_time_spent"`
}

func (t TimeStats) String() string {
	return Stringify(t)
}

// SetTimeEstimateOptions represents the available SetTimeEstimate()
// options.
//
// GitLab docs: https://docs.gitlab.com/ce/workflow/time_tracking.html
type SetTimeEstimateOptions struct {
	Duration *string `url:"duration,omitempty" json:"duration,omitempty"`
}

// setTimeEstimate sets the time estimate for a single project issue.
//
// GitLab docs: https://docs.gitlab.com/ce/workflow/time_tracking.html
func (s *timeStatsService) setTimeEstimate(pid interface{}, entity string, issue int, opt *SetTimeEstimateOptions, options ...RequestOptionFunc) (*TimeStats, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/%s/%d/time_estimate", pathEscape(project), entity, issue)

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	t := new(TimeStats)
	resp, err := s.client.Do(req, t)
	if err != nil {
		return nil, resp, err
	}

	return t, resp, err
}

// resetTimeEstimate resets the time estimate for a single project issue.
//
// GitLab docs: https://docs.gitlab.com/ce/workflow/time_tracking.html
func (s *timeStatsService) resetTimeEstimate(pid interface{}, entity string, issue int, options ...RequestOptionFunc) (*TimeStats, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/%s/%d/reset_time_estimate", pathEscape(project), entity, issue)

	req, err := s.client.NewRequest("POST", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	t := new(TimeStats)
	resp, err := s.client.Do(req, t)
	if err != nil {
		return nil, resp, err
	}

	return t, resp, err
}

// AddSpentTimeOptions represents the available AddSpentTime() options.
//
// GitLab docs: https://docs.gitlab.com/ce/workflow/time_tracking.html
type AddSpentTimeOptions struct {
	Duration *string `url:"duration,omitempty" json:"duration,omitempty"`
}

// addSpentTime adds spent time for a single project issue.
//
// GitLab docs: https://docs.gitlab.com/ce/workflow/time_tracking.html
func (s *timeStatsService) addSpentTime(pid interface{}, entity string, issue int, opt *AddSpentTimeOptions, options ...RequestOptionFunc) (*TimeStats, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/%s/%d/add_spent_time", pathEscape(project), entity, issue)

	req, err := s.client.NewRequest("POST", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	t := new(TimeStats)
	resp, err := s.client.Do(req, t)
	if err != nil {
		return nil, resp, err
	}

	return t, resp, err
}

// resetSpentTime resets the spent time for a single project issue.
//
// GitLab docs: https://docs.gitlab.com/ce/workflow/time_tracking.html
func (s *timeStatsService) resetSpentTime(pid interface{}, entity string, issue int, options ...RequestOptionFunc) (*TimeStats, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/%s/%d/reset_spent_time", pathEscape(project), entity, issue)

	req, err := s.client.NewRequest("POST", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	t := new(TimeStats)
	resp, err := s.client.Do(req, t)
	if err != nil {
		return nil, resp, err
	}

	return t, resp, err
}

// getTimeSpent gets the spent time for a single project issue.
//
// GitLab docs: https://docs.gitlab.com/ce/workflow/time_tracking.html
func (s *timeStatsService) getTimeSpent(pid interface{}, entity string, issue int, options ...RequestOptionFunc) (*TimeStats, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/%s/%d/time_stats", pathEscape(project), entity, issue)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	t := new(TimeStats)
	resp, err := s.client.Do(req, t)
	if err != nil {
		return nil, resp, err
	}

	return t, resp, err
}

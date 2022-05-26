// Copyright 2018 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"encoding/json"
	"time"
)

// Event represents a GitHub event.
type Event struct {
	Type       *string          `json:"type,omitempty"`
	Public     *bool            `json:"public,omitempty"`
	RawPayload *json.RawMessage `json:"payload,omitempty"`
	Repo       *Repository      `json:"repo,omitempty"`
	Actor      *User            `json:"actor,omitempty"`
	Org        *Organization    `json:"org,omitempty"`
	CreatedAt  *time.Time       `json:"created_at,omitempty"`
	ID         *string          `json:"id,omitempty"`
}

func (e Event) String() string {
	return Stringify(e)
}

// ParsePayload parses the event payload. For recognized event types,
// a value of the corresponding struct type will be returned.
func (e *Event) ParsePayload() (payload interface{}, err error) {
	switch *e.Type {
	case "CheckRunEvent":
		payload = &CheckRunEvent{}
	case "CheckSuiteEvent":
		payload = &CheckSuiteEvent{}
	case "CommitCommentEvent":
		payload = &CommitCommentEvent{}
	case "ContentReferenceEvent":
		payload = &ContentReferenceEvent{}
	case "CreateEvent":
		payload = &CreateEvent{}
	case "DeleteEvent":
		payload = &DeleteEvent{}
	case "DeployKeyEvent":
		payload = &DeployKeyEvent{}
	case "DeploymentEvent":
		payload = &DeploymentEvent{}
	case "DeploymentStatusEvent":
		payload = &DeploymentStatusEvent{}
	case "ForkEvent":
		payload = &ForkEvent{}
	case "GitHubAppAuthorizationEvent":
		payload = &GitHubAppAuthorizationEvent{}
	case "GollumEvent":
		payload = &GollumEvent{}
	case "InstallationEvent":
		payload = &InstallationEvent{}
	case "InstallationRepositoriesEvent":
		payload = &InstallationRepositoriesEvent{}
	case "IssueCommentEvent":
		payload = &IssueCommentEvent{}
	case "IssuesEvent":
		payload = &IssuesEvent{}
	case "LabelEvent":
		payload = &LabelEvent{}
	case "MarketplacePurchaseEvent":
		payload = &MarketplacePurchaseEvent{}
	case "MemberEvent":
		payload = &MemberEvent{}
	case "MembershipEvent":
		payload = &MembershipEvent{}
	case "MetaEvent":
		payload = &MetaEvent{}
	case "MilestoneEvent":
		payload = &MilestoneEvent{}
	case "OrganizationEvent":
		payload = &OrganizationEvent{}
	case "OrgBlockEvent":
		payload = &OrgBlockEvent{}
	case "PackageEvent":
		payload = &PackageEvent{}
	case "PageBuildEvent":
		payload = &PageBuildEvent{}
	case "PingEvent":
		payload = &PingEvent{}
	case "ProjectEvent":
		payload = &ProjectEvent{}
	case "ProjectCardEvent":
		payload = &ProjectCardEvent{}
	case "ProjectColumnEvent":
		payload = &ProjectColumnEvent{}
	case "PublicEvent":
		payload = &PublicEvent{}
	case "PullRequestEvent":
		payload = &PullRequestEvent{}
	case "PullRequestReviewEvent":
		payload = &PullRequestReviewEvent{}
	case "PullRequestReviewCommentEvent":
		payload = &PullRequestReviewCommentEvent{}
	case "PushEvent":
		payload = &PushEvent{}
	case "ReleaseEvent":
		payload = &ReleaseEvent{}
	case "RepositoryEvent":
		payload = &RepositoryEvent{}
	case "RepositoryDispatchEvent":
		payload = &RepositoryDispatchEvent{}
	case "RepositoryVulnerabilityAlertEvent":
		payload = &RepositoryVulnerabilityAlertEvent{}
	case "StarEvent":
		payload = &StarEvent{}
	case "StatusEvent":
		payload = &StatusEvent{}
	case "TeamEvent":
		payload = &TeamEvent{}
	case "TeamAddEvent":
		payload = &TeamAddEvent{}
	case "UserEvent":
		payload = &UserEvent{}
	case "WatchEvent":
		payload = &WatchEvent{}
	case "WorkflowDispatchEvent":
		payload = &WorkflowDispatchEvent{}
	case "WorkflowRunEvent":
		payload = &WorkflowRunEvent{}
	}
	err = json.Unmarshal(*e.RawPayload, &payload)
	return payload, err
}

// Payload returns the parsed event payload. For recognized event types,
// a value of the corresponding struct type will be returned.
//
// Deprecated: Use ParsePayload instead, which returns an error
// rather than panics if JSON unmarshaling raw payload fails.
func (e *Event) Payload() (payload interface{}) {
	var err error
	payload, err = e.ParsePayload()
	if err != nil {
		panic(err)
	}
	return payload
}

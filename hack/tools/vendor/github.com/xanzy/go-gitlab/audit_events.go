package gitlab

import (
	"fmt"
	"time"
)

// AuditEvent represents an audit event for a group or project.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/audit_events.html
type AuditEvent struct {
	ID         int               `json:"id"`
	AuthorID   int               `json:"author_id"`
	EntityID   int               `json:"entity_id"`
	EntityType string            `json:"entity_type"`
	Details    AuditEventDetails `json:"details"`
	CreatedAt  *time.Time        `json:"created_at"`
}

// AuditEventDetails represents the details portion of an audit event for
// a group or project. The exact fields that are returned for an audit event
// depend on the action being recorded.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/audit_events.html
type AuditEventDetails struct {
	With          string      `json:"with"`
	Add           string      `json:"add"`
	As            string      `json:"as"`
	Change        string      `json:"change"`
	From          string      `json:"from"`
	To            string      `json:"to"`
	Remove        string      `json:"remove"`
	CustomMessage string      `json:"custom_message"`
	AuthorName    string      `json:"author_name"`
	TargetID      interface{} `json:"target_id"`
	TargetType    string      `json:"target_type"`
	TargetDetails string      `json:"target_details"`
	IPAddress     string      `json:"ip_address"`
	EntityPath    string      `json:"entity_path"`
}

// AuditEventsService handles communication with the project/group audit
// event related methods of the GitLab API.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/audit_events.html
type AuditEventsService struct {
	client *Client
}

// ListAuditEventsOptions represents the available ListProjectAuditEvents()
// or ListGroupAuditEvents() options.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/audit_events.html
type ListAuditEventsOptions struct {
	ListOptions
	CreatedAfter  *time.Time `url:"created_after,omitempty" json:"created_after,omitempty"`
	CreatedBefore *time.Time `url:"created_before,omitempty" json:"created_before,omitempty"`
}

// ListGroupAuditEvents gets a list of audit events for the specified group
// viewable by the authenticated user.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/audit_events.html
func (s *AuditEventsService) ListGroupAuditEvents(gid interface{}, opt *ListAuditEventsOptions, options ...RequestOptionFunc) ([]*AuditEvent, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/audit_events", pathEscape(group))

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var aes []*AuditEvent
	resp, err := s.client.Do(req, &aes)
	if err != nil {
		return nil, resp, err
	}

	return aes, resp, err
}

// GetGroupAuditEvent gets a specific group audit event.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/audit_events.html
func (s *AuditEventsService) GetGroupAuditEvent(gid interface{}, event int, options ...RequestOptionFunc) (*AuditEvent, *Response, error) {
	group, err := parseID(gid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("groups/%s/audit_events/%d", pathEscape(group), event)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	ae := new(AuditEvent)
	resp, err := s.client.Do(req, ae)
	if err != nil {
		return nil, resp, err
	}

	return ae, resp, err
}

// ListProjectAuditEvents gets a list of audit events for the specified project
// viewable by the authenticated user.
//
// GitLab API docs: https://docs.gitlab.com/ee/api/audit_events.html
func (s *AuditEventsService) ListProjectAuditEvents(pid interface{}, opt *ListAuditEventsOptions, options ...RequestOptionFunc) ([]*AuditEvent, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/audit_events", pathEscape(project))

	req, err := s.client.NewRequest("GET", u, opt, options)
	if err != nil {
		return nil, nil, err
	}

	var aes []*AuditEvent
	resp, err := s.client.Do(req, &aes)
	if err != nil {
		return nil, resp, err
	}

	return aes, resp, err
}

// GetProjectAuditEvent gets a specific project audit event.
//
// GitLab API docs:
// https://docs.gitlab.com/ee/api/audit_events.html
func (s *AuditEventsService) GetProjectAuditEvent(pid interface{}, event int, options ...RequestOptionFunc) (*AuditEvent, *Response, error) {
	project, err := parseID(pid)
	if err != nil {
		return nil, nil, err
	}
	u := fmt.Sprintf("projects/%s/audit_events/%d", pathEscape(project), event)

	req, err := s.client.NewRequest("GET", u, nil, options)
	if err != nil {
		return nil, nil, err
	}

	ae := new(AuditEvent)
	resp, err := s.client.Do(req, ae)
	if err != nil {
		return nil, resp, err
	}

	return ae, resp, err
}

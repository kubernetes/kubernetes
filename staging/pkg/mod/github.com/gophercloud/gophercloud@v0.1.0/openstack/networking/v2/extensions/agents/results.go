package agents

import (
	"encoding/json"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

type commonResult struct {
	gophercloud.Result
}

// Extract is a function that accepts a result and extracts an agent resource.
func (r commonResult) Extract() (*Agent, error) {
	var s struct {
		Agent *Agent `json:"agent"`
	}
	err := r.ExtractInto(&s)
	return s.Agent, err
}

// GetResult represents the result of a get operation. Call its Extract
// method to interpret it as an Agent.
type GetResult struct {
	commonResult
}

// Agent represents a Neutron agent.
type Agent struct {
	// ID is the id of the agent.
	ID string `json:"id"`

	// AdminStateUp is an administrative state of the agent.
	AdminStateUp bool `json:"admin_state_up"`

	// AgentType is a type of the agent.
	AgentType string `json:"agent_type"`

	// Alive indicates whether agent is alive or not.
	Alive bool `json:"alive"`

	// AvailabilityZone is a zone of the agent.
	AvailabilityZone string `json:"availability_zone"`

	// Binary is an executable binary of the agent.
	Binary string `json:"binary"`

	// Configurations is a configuration specific key/value pairs that are
	// determined by the agent binary and type.
	Configurations map[string]interface{} `json:"configurations"`

	// CreatedAt is a creation timestamp.
	CreatedAt time.Time `json:"-"`

	// StartedAt is a starting timestamp.
	StartedAt time.Time `json:"-"`

	// HeartbeatTimestamp is a last heartbeat timestamp.
	HeartbeatTimestamp time.Time `json:"-"`

	// Description contains agent description.
	Description string `json:"description"`

	// Host is a hostname of the agent system.
	Host string `json:"host"`

	// Topic contains name of AMQP topic.
	Topic string `json:"topic"`
}

// UnmarshalJSON helps to convert the timestamps into the time.Time type.
func (r *Agent) UnmarshalJSON(b []byte) error {
	type tmp Agent
	var s struct {
		tmp
		CreatedAt          gophercloud.JSONRFC3339ZNoTNoZ `json:"created_at"`
		StartedAt          gophercloud.JSONRFC3339ZNoTNoZ `json:"started_at"`
		HeartbeatTimestamp gophercloud.JSONRFC3339ZNoTNoZ `json:"heartbeat_timestamp"`
	}
	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}
	*r = Agent(s.tmp)

	r.CreatedAt = time.Time(s.CreatedAt)
	r.StartedAt = time.Time(s.StartedAt)
	r.HeartbeatTimestamp = time.Time(s.HeartbeatTimestamp)

	return nil
}

// AgentPage stores a single page of Agents from a List() API call.
type AgentPage struct {
	pagination.LinkedPageBase
}

// NextPageURL is invoked when a paginated collection of agent has
// reached the end of a page and the pager seeks to traverse over a new one.
// In order to do this, it needs to construct the next page's URL.
func (r AgentPage) NextPageURL() (string, error) {
	var s struct {
		Links []gophercloud.Link `json:"agents_links"`
	}
	err := r.ExtractInto(&s)
	if err != nil {
		return "", err
	}
	return gophercloud.ExtractNextURL(s.Links)
}

// IsEmpty determines whether or not a AgentPage is empty.
func (r AgentPage) IsEmpty() (bool, error) {
	agents, err := ExtractAgents(r)
	return len(agents) == 0, err
}

// ExtractAgents interprets the results of a single page from a List()
// API call, producing a slice of Agents structs.
func ExtractAgents(r pagination.Page) ([]Agent, error) {
	var s struct {
		Agents []Agent `json:"agents"`
	}
	err := (r.(AgentPage)).ExtractInto(&s)
	return s.Agents, err
}

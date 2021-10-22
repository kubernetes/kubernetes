package remoteconsoles

import "github.com/gophercloud/gophercloud"

type commonResult struct {
	gophercloud.Result
}

// CreateResult represents the result of a create operation. Call its Extract
// method to interpret it as a RemoteConsole.
type CreateResult struct {
	commonResult
}

// RemoteConsole represents the Compute service remote console object.
type RemoteConsole struct {
	// Protocol contains remote console protocol.
	// You can use the RemoteConsoleProtocol custom type to unmarshal raw JSON
	// response into the pre-defined valid console protocol.
	Protocol string `json:"protocol"`

	// Type contains remote console type.
	// You can use the RemoteConsoleType custom type to unmarshal raw JSON
	// response into the pre-defined valid console type.
	Type string `json:"type"`

	// URL can be used to connect to the remote console.
	URL string `json:"url"`
}

// Extract interprets any commonResult as a RemoteConsole.
func (r commonResult) Extract() (*RemoteConsole, error) {
	var s struct {
		RemoteConsole *RemoteConsole `json:"remote_console"`
	}
	err := r.ExtractInto(&s)
	return s.RemoteConsole, err
}

package packp

import (
	"errors"
	"io"

	"gopkg.in/src-d/go-git.v4/plumbing"
	"gopkg.in/src-d/go-git.v4/plumbing/protocol/packp/capability"
	"gopkg.in/src-d/go-git.v4/plumbing/protocol/packp/sideband"
)

var (
	ErrEmptyCommands    = errors.New("commands cannot be empty")
	ErrMalformedCommand = errors.New("malformed command")
)

// ReferenceUpdateRequest values represent reference upload requests.
// Values from this type are not zero-value safe, use the New function instead.
type ReferenceUpdateRequest struct {
	Capabilities *capability.List
	Commands     []*Command
	Shallow      *plumbing.Hash
	// Packfile contains an optional packfile reader.
	Packfile io.ReadCloser

	// Progress receives sideband progress messages from the server
	Progress sideband.Progress
}

// New returns a pointer to a new ReferenceUpdateRequest value.
func NewReferenceUpdateRequest() *ReferenceUpdateRequest {
	return &ReferenceUpdateRequest{
		// TODO: Add support for push-cert
		Capabilities: capability.NewList(),
		Commands:     nil,
	}
}

// NewReferenceUpdateRequestFromCapabilities returns a pointer to a new
// ReferenceUpdateRequest value, the request capabilities are filled with the
// most optimal ones, based on the adv value (advertised capabilities), the
// ReferenceUpdateRequest contains no commands
//
// It does set the following capabilities:
//   - agent
//   - report-status
//   - ofs-delta
//   - ref-delta
//   - delete-refs
// It leaves up to the user to add the following capabilities later:
//   - atomic
//   - ofs-delta
//   - side-band
//   - side-band-64k
//   - quiet
//   - push-cert
func NewReferenceUpdateRequestFromCapabilities(adv *capability.List) *ReferenceUpdateRequest {
	r := NewReferenceUpdateRequest()

	if adv.Supports(capability.Agent) {
		r.Capabilities.Set(capability.Agent, capability.DefaultAgent)
	}

	if adv.Supports(capability.ReportStatus) {
		r.Capabilities.Set(capability.ReportStatus)
	}

	return r
}

func (r *ReferenceUpdateRequest) validate() error {
	if len(r.Commands) == 0 {
		return ErrEmptyCommands
	}

	for _, c := range r.Commands {
		if err := c.validate(); err != nil {
			return err
		}
	}

	return nil
}

type Action string

const (
	Create  Action = "create"
	Update         = "update"
	Delete         = "delete"
	Invalid        = "invalid"
)

type Command struct {
	Name plumbing.ReferenceName
	Old  plumbing.Hash
	New  plumbing.Hash
}

func (c *Command) Action() Action {
	if c.Old == plumbing.ZeroHash && c.New == plumbing.ZeroHash {
		return Invalid
	}

	if c.Old == plumbing.ZeroHash {
		return Create
	}

	if c.New == plumbing.ZeroHash {
		return Delete
	}

	return Update
}

func (c *Command) validate() error {
	if c.Action() == Invalid {
		return ErrMalformedCommand
	}

	return nil
}

package packp

import (
	"fmt"
	"time"

	"github.com/go-git/go-git/v5/plumbing"
	"github.com/go-git/go-git/v5/plumbing/protocol/packp/capability"
)

// UploadRequest values represent the information transmitted on a
// upload-request message.  Values from this type are not zero-value
// safe, use the New function instead.
// This is a low level type, use UploadPackRequest instead.
type UploadRequest struct {
	Capabilities *capability.List
	Wants        []plumbing.Hash
	Shallows     []plumbing.Hash
	Depth        Depth
}

// Depth values stores the desired depth of the requested packfile: see
// DepthCommit, DepthSince and DepthReference.
type Depth interface {
	isDepth()
	IsZero() bool
}

// DepthCommits values stores the maximum number of requested commits in
// the packfile.  Zero means infinite.  A negative value will have
// undefined consequences.
type DepthCommits int

func (d DepthCommits) isDepth() {}

func (d DepthCommits) IsZero() bool {
	return d == 0
}

// DepthSince values requests only commits newer than the specified time.
type DepthSince time.Time

func (d DepthSince) isDepth() {}

func (d DepthSince) IsZero() bool {
	return time.Time(d).IsZero()
}

// DepthReference requests only commits not to found in the specified reference.
type DepthReference string

func (d DepthReference) isDepth() {}

func (d DepthReference) IsZero() bool {
	return string(d) == ""
}

// NewUploadRequest returns a pointer to a new UploadRequest value, ready to be
// used. It has no capabilities, wants or shallows and an infinite depth. Please
// note that to encode an upload-request it has to have at least one wanted hash.
func NewUploadRequest() *UploadRequest {
	return &UploadRequest{
		Capabilities: capability.NewList(),
		Wants:        []plumbing.Hash{},
		Shallows:     []plumbing.Hash{},
		Depth:        DepthCommits(0),
	}
}

// NewUploadRequestFromCapabilities returns a pointer to a new UploadRequest
// value, the request capabilities are filled with the most optimal ones, based
// on the adv value (advertised capabilities), the UploadRequest generated it
// has no wants or shallows and an infinite depth.
func NewUploadRequestFromCapabilities(adv *capability.List) *UploadRequest {
	r := NewUploadRequest()

	if adv.Supports(capability.MultiACKDetailed) {
		r.Capabilities.Set(capability.MultiACKDetailed)
	} else if adv.Supports(capability.MultiACK) {
		r.Capabilities.Set(capability.MultiACK)
	}

	if adv.Supports(capability.Sideband64k) {
		r.Capabilities.Set(capability.Sideband64k)
	} else if adv.Supports(capability.Sideband) {
		r.Capabilities.Set(capability.Sideband)
	}

	if adv.Supports(capability.ThinPack) {
		r.Capabilities.Set(capability.ThinPack)
	}

	if adv.Supports(capability.OFSDelta) {
		r.Capabilities.Set(capability.OFSDelta)
	}

	if adv.Supports(capability.Agent) {
		r.Capabilities.Set(capability.Agent, capability.DefaultAgent)
	}

	return r
}

// Validate validates the content of UploadRequest, following the next rules:
//   - Wants MUST have at least one reference
//   - capability.Shallow MUST be present if Shallows is not empty
//   - is a non-zero DepthCommits is given capability.Shallow MUST be present
//   - is a DepthSince is given capability.Shallow MUST be present
//   - is a DepthReference is given capability.DeepenNot MUST be present
//   - MUST contain only maximum of one of capability.Sideband and capability.Sideband64k
//   - MUST contain only maximum of one of capability.MultiACK and capability.MultiACKDetailed
func (req *UploadRequest) Validate() error {
	if len(req.Wants) == 0 {
		return fmt.Errorf("want can't be empty")
	}

	if err := req.validateRequiredCapabilities(); err != nil {
		return err
	}

	if err := req.validateConflictCapabilities(); err != nil {
		return err
	}

	return nil
}

func (req *UploadRequest) validateRequiredCapabilities() error {
	msg := "missing capability %s"

	if len(req.Shallows) != 0 && !req.Capabilities.Supports(capability.Shallow) {
		return fmt.Errorf(msg, capability.Shallow)
	}

	switch req.Depth.(type) {
	case DepthCommits:
		if req.Depth != DepthCommits(0) {
			if !req.Capabilities.Supports(capability.Shallow) {
				return fmt.Errorf(msg, capability.Shallow)
			}
		}
	case DepthSince:
		if !req.Capabilities.Supports(capability.DeepenSince) {
			return fmt.Errorf(msg, capability.DeepenSince)
		}
	case DepthReference:
		if !req.Capabilities.Supports(capability.DeepenNot) {
			return fmt.Errorf(msg, capability.DeepenNot)
		}
	}

	return nil
}

func (req *UploadRequest) validateConflictCapabilities() error {
	msg := "capabilities %s and %s are mutually exclusive"
	if req.Capabilities.Supports(capability.Sideband) &&
		req.Capabilities.Supports(capability.Sideband64k) {
		return fmt.Errorf(msg, capability.Sideband, capability.Sideband64k)
	}

	if req.Capabilities.Supports(capability.MultiACK) &&
		req.Capabilities.Supports(capability.MultiACKDetailed) {
		return fmt.Errorf(msg, capability.MultiACK, capability.MultiACKDetailed)
	}

	return nil
}

package agent

import (
	"fmt"
	"regexp"

	"github.com/hashicorp/consul/consul/structs"
	"github.com/hashicorp/go-uuid"
)

const (
	// userEventMaxVersion is the maximum protocol version we understand
	userEventMaxVersion = 1

	// remoteExecName is the event name for a remote exec command
	remoteExecName = "_rexec"
)

// UserEventParam is used to parameterize a user event
type UserEvent struct {
	// ID of the user event. Automatically generated.
	ID string

	// Name of the event
	Name string `codec:"n"`

	// Optional payload
	Payload []byte `codec:"p,omitempty"`

	// NodeFilter is a regular expression to filter on nodes
	NodeFilter string `codec:"nf,omitempty"`

	// ServiceFilter is a regular expression to filter on services
	ServiceFilter string `codec:"sf,omitempty"`

	// TagFilter is a regular expression to filter on tags of a service,
	// must be provided with ServiceFilter
	TagFilter string `codec:"tf,omitempty"`

	// Version of the user event. Automatically generated.
	Version int `codec:"v"`

	// LTime is the lamport time. Automatically generated.
	LTime uint64 `codec:"-"`
}

// validateUserEventParams is used to sanity check the inputs
func validateUserEventParams(params *UserEvent) error {
	// Validate the inputs
	if params.Name == "" {
		return fmt.Errorf("User event missing name")
	}
	if params.TagFilter != "" && params.ServiceFilter == "" {
		return fmt.Errorf("Cannot provide tag filter without service filter")
	}
	if params.NodeFilter != "" {
		if _, err := regexp.Compile(params.NodeFilter); err != nil {
			return fmt.Errorf("Invalid node filter: %v", err)
		}
	}
	if params.ServiceFilter != "" {
		if _, err := regexp.Compile(params.ServiceFilter); err != nil {
			return fmt.Errorf("Invalid service filter: %v", err)
		}
	}
	if params.TagFilter != "" {
		if _, err := regexp.Compile(params.TagFilter); err != nil {
			return fmt.Errorf("Invalid tag filter: %v", err)
		}
	}
	return nil
}

// UserEvent is used to fire an event via the Serf layer on the LAN
func (a *Agent) UserEvent(dc, token string, params *UserEvent) error {
	// Validate the params
	if err := validateUserEventParams(params); err != nil {
		return err
	}

	// Format message
	var err error
	if params.ID, err = uuid.GenerateUUID(); err != nil {
		return fmt.Errorf("UUID generation failed: %v", err)
	}
	params.Version = userEventMaxVersion
	payload, err := encodeMsgPack(&params)
	if err != nil {
		return fmt.Errorf("UserEvent encoding failed: %v", err)
	}

	// Service the event fire over RPC. This ensures that we authorize
	// the request against the token first.
	args := structs.EventFireRequest{
		Datacenter:   dc,
		Name:         params.Name,
		Payload:      payload,
		QueryOptions: structs.QueryOptions{Token: token},
	}

	// Any server can process in the remote DC, since the
	// gossip will take over anyways
	args.AllowStale = true
	var out structs.EventFireResponse
	return a.RPC("Internal.EventFire", &args, &out)
}

// handleEvents is used to process incoming user events
func (a *Agent) handleEvents() {
	for {
		select {
		case e := <-a.eventCh:
			// Decode the event
			msg := new(UserEvent)
			if err := decodeMsgPack(e.Payload, msg); err != nil {
				a.logger.Printf("[ERR] agent: Failed to decode event: %v", err)
				continue
			}
			msg.LTime = uint64(e.LTime)

			// Skip if we don't pass filtering
			if !a.shouldProcessUserEvent(msg) {
				continue
			}

			// Ingest the event
			a.ingestUserEvent(msg)

		case <-a.shutdownCh:
			return
		}
	}
}

// shouldProcessUserEvent checks if an event makes it through our filters
func (a *Agent) shouldProcessUserEvent(msg *UserEvent) bool {
	// Check the version
	if msg.Version > userEventMaxVersion {
		a.logger.Printf("[WARN] agent: Event version %d may have unsupported features (%s)",
			msg.Version, msg.Name)
	}

	// Apply the filters
	if msg.NodeFilter != "" {
		re, err := regexp.Compile(msg.NodeFilter)
		if err != nil {
			a.logger.Printf("[ERR] agent: Failed to parse node filter '%s' for event '%s': %v",
				msg.NodeFilter, msg.Name, err)
			return false
		}
		if !re.MatchString(a.config.NodeName) {
			return false
		}
	}

	if msg.ServiceFilter != "" {
		re, err := regexp.Compile(msg.ServiceFilter)
		if err != nil {
			a.logger.Printf("[ERR] agent: Failed to parse service filter '%s' for event '%s': %v",
				msg.ServiceFilter, msg.Name, err)
			return false
		}

		var tagRe *regexp.Regexp
		if msg.TagFilter != "" {
			re, err := regexp.Compile(msg.TagFilter)
			if err != nil {
				a.logger.Printf("[ERR] agent: Failed to parse tag filter '%s' for event '%s': %v",
					msg.TagFilter, msg.Name, err)
				return false
			}
			tagRe = re
		}

		// Scan for a match
		services := a.state.Services()
		found := false
	OUTER:
		for name, info := range services {
			// Check the service name
			if !re.MatchString(name) {
				continue
			}
			if tagRe == nil {
				found = true
				break
			}

			// Look for a matching tag
			for _, tag := range info.Tags {
				if !tagRe.MatchString(tag) {
					continue
				}
				found = true
				break OUTER
			}
		}

		// No matching services
		if !found {
			return false
		}
	}
	return true
}

// ingestUserEvent is used to process an event that passes filtering
func (a *Agent) ingestUserEvent(msg *UserEvent) {
	// Special handling for internal events
	switch msg.Name {
	case remoteExecName:
		if a.config.DisableRemoteExec {
			a.logger.Printf("[INFO] agent: ignoring remote exec event (%s), disabled.", msg.ID)
		} else {
			go a.handleRemoteExec(msg)
		}
		return
	default:
		a.logger.Printf("[DEBUG] agent: new event: %s (%s)", msg.Name, msg.ID)
	}

	a.eventLock.Lock()
	defer func() {
		a.eventLock.Unlock()
		a.eventNotify.Notify()
	}()

	idx := a.eventIndex
	a.eventBuf[idx] = msg
	a.eventIndex = (idx + 1) % len(a.eventBuf)
}

// UserEvents is used to return a slice of the most recent
// user events.
func (a *Agent) UserEvents() []*UserEvent {
	n := len(a.eventBuf)
	out := make([]*UserEvent, n)
	a.eventLock.RLock()
	defer a.eventLock.RUnlock()

	// Check if the buffer is full
	if a.eventBuf[a.eventIndex] != nil {
		if a.eventIndex == 0 {
			copy(out, a.eventBuf)
		} else {
			copy(out, a.eventBuf[a.eventIndex:])
			copy(out[n-a.eventIndex:], a.eventBuf[:a.eventIndex])
		}
	} else {
		// We haven't filled the buffer yet
		copy(out, a.eventBuf[:a.eventIndex])
		out = out[:a.eventIndex]
	}
	return out
}

// LastUserEvent is used to return the lastest user event.
// This will return nil if there is no recent event.
func (a *Agent) LastUserEvent() *UserEvent {
	a.eventLock.RLock()
	defer a.eventLock.RUnlock()
	n := len(a.eventBuf)
	idx := (((a.eventIndex - 1) % n) + n) % n
	return a.eventBuf[idx]
}

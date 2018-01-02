package plugin

import (
	"fmt"
	"reflect"

	"github.com/docker/docker/api/types"
)

// Event is emitted for actions performed on the plugin manager
type Event interface {
	matches(Event) bool
}

// EventCreate is an event which is emitted when a plugin is created
// This is either by pull or create from context.
//
// Use the `Interfaces` field to match only plugins that implement a specific
// interface.
// These are matched against using "or" logic.
// If no interfaces are listed, all are matched.
type EventCreate struct {
	Interfaces map[string]bool
	Plugin     types.Plugin
}

func (e EventCreate) matches(observed Event) bool {
	oe, ok := observed.(EventCreate)
	if !ok {
		return false
	}
	if len(e.Interfaces) == 0 {
		return true
	}

	var ifaceMatch bool
	for _, in := range oe.Plugin.Config.Interface.Types {
		if e.Interfaces[in.Capability] {
			ifaceMatch = true
			break
		}
	}
	return ifaceMatch
}

// EventRemove is an event which is emitted when a plugin is removed
// It maches on the passed in plugin's ID only.
type EventRemove struct {
	Plugin types.Plugin
}

func (e EventRemove) matches(observed Event) bool {
	oe, ok := observed.(EventRemove)
	if !ok {
		return false
	}
	return e.Plugin.ID == oe.Plugin.ID
}

// EventDisable is an event that is emitted when a plugin is disabled
// It maches on the passed in plugin's ID only.
type EventDisable struct {
	Plugin types.Plugin
}

func (e EventDisable) matches(observed Event) bool {
	oe, ok := observed.(EventDisable)
	if !ok {
		return false
	}
	return e.Plugin.ID == oe.Plugin.ID
}

// EventEnable is an event that is emitted when a plugin is disabled
// It maches on the passed in plugin's ID only.
type EventEnable struct {
	Plugin types.Plugin
}

func (e EventEnable) matches(observed Event) bool {
	oe, ok := observed.(EventEnable)
	if !ok {
		return false
	}
	return e.Plugin.ID == oe.Plugin.ID
}

// SubscribeEvents provides an event channel to listen for structured events from
// the plugin manager actions, CRUD operations.
// The caller must call the returned `cancel()` function once done with the channel
// or this will leak resources.
func (pm *Manager) SubscribeEvents(buffer int, watchEvents ...Event) (eventCh <-chan interface{}, cancel func()) {
	topic := func(i interface{}) bool {
		observed, ok := i.(Event)
		if !ok {
			panic(fmt.Sprintf("unexpected type passed to event channel: %v", reflect.TypeOf(i)))
		}
		for _, e := range watchEvents {
			if e.matches(observed) {
				return true
			}
		}
		// If no specific events are specified always assume a matched event
		// If some events were specified and none matched above, then the event
		// doesn't match
		return watchEvents == nil
	}
	ch := pm.publisher.SubscribeTopicWithBuffer(topic, buffer)
	cancelFunc := func() { pm.publisher.Evict(ch) }
	return ch, cancelFunc
}

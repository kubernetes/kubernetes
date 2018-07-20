package events

import (
	"github.com/docker/distribution/reference"
	"github.com/docker/docker/api/types/events"
	"github.com/docker/docker/api/types/filters"
)

// Filter can filter out docker events from a stream
type Filter struct {
	filter filters.Args
}

// NewFilter creates a new Filter
func NewFilter(filter filters.Args) *Filter {
	return &Filter{filter: filter}
}

// Include returns true when the event ev is included by the filters
func (ef *Filter) Include(ev events.Message) bool {
	return ef.matchEvent(ev) &&
		ef.filter.ExactMatch("type", ev.Type) &&
		ef.matchScope(ev.Scope) &&
		ef.matchDaemon(ev) &&
		ef.matchContainer(ev) &&
		ef.matchPlugin(ev) &&
		ef.matchVolume(ev) &&
		ef.matchNetwork(ev) &&
		ef.matchImage(ev) &&
		ef.matchNode(ev) &&
		ef.matchService(ev) &&
		ef.matchSecret(ev) &&
		ef.matchConfig(ev) &&
		ef.matchLabels(ev.Actor.Attributes)
}

func (ef *Filter) matchEvent(ev events.Message) bool {
	// #25798 if an event filter contains either health_status, exec_create or exec_start without a colon
	// Let's to a FuzzyMatch instead of an ExactMatch.
	if ef.filterContains("event", map[string]struct{}{"health_status": {}, "exec_create": {}, "exec_start": {}}) {
		return ef.filter.FuzzyMatch("event", ev.Action)
	}
	return ef.filter.ExactMatch("event", ev.Action)
}

func (ef *Filter) filterContains(field string, values map[string]struct{}) bool {
	for _, v := range ef.filter.Get(field) {
		if _, ok := values[v]; ok {
			return true
		}
	}
	return false
}

func (ef *Filter) matchScope(scope string) bool {
	if !ef.filter.Contains("scope") {
		return true
	}
	return ef.filter.ExactMatch("scope", scope)
}

func (ef *Filter) matchLabels(attributes map[string]string) bool {
	if !ef.filter.Contains("label") {
		return true
	}
	return ef.filter.MatchKVList("label", attributes)
}

func (ef *Filter) matchDaemon(ev events.Message) bool {
	return ef.fuzzyMatchName(ev, events.DaemonEventType)
}

func (ef *Filter) matchContainer(ev events.Message) bool {
	return ef.fuzzyMatchName(ev, events.ContainerEventType)
}

func (ef *Filter) matchPlugin(ev events.Message) bool {
	return ef.fuzzyMatchName(ev, events.PluginEventType)
}

func (ef *Filter) matchVolume(ev events.Message) bool {
	return ef.fuzzyMatchName(ev, events.VolumeEventType)
}

func (ef *Filter) matchNetwork(ev events.Message) bool {
	return ef.fuzzyMatchName(ev, events.NetworkEventType)
}

func (ef *Filter) matchService(ev events.Message) bool {
	return ef.fuzzyMatchName(ev, events.ServiceEventType)
}

func (ef *Filter) matchNode(ev events.Message) bool {
	return ef.fuzzyMatchName(ev, events.NodeEventType)
}

func (ef *Filter) matchSecret(ev events.Message) bool {
	return ef.fuzzyMatchName(ev, events.SecretEventType)
}

func (ef *Filter) matchConfig(ev events.Message) bool {
	return ef.fuzzyMatchName(ev, events.ConfigEventType)
}

func (ef *Filter) fuzzyMatchName(ev events.Message, eventType string) bool {
	return ef.filter.FuzzyMatch(eventType, ev.Actor.ID) ||
		ef.filter.FuzzyMatch(eventType, ev.Actor.Attributes["name"])
}

// matchImage matches against both event.Actor.ID (for image events)
// and event.Actor.Attributes["image"] (for container events), so that any container that was created
// from an image will be included in the image events. Also compare both
// against the stripped repo name without any tags.
func (ef *Filter) matchImage(ev events.Message) bool {
	id := ev.Actor.ID
	nameAttr := "image"
	var imageName string

	if ev.Type == events.ImageEventType {
		nameAttr = "name"
	}

	if n, ok := ev.Actor.Attributes[nameAttr]; ok {
		imageName = n
	}
	return ef.filter.ExactMatch("image", id) ||
		ef.filter.ExactMatch("image", imageName) ||
		ef.filter.ExactMatch("image", stripTag(id)) ||
		ef.filter.ExactMatch("image", stripTag(imageName))
}

func stripTag(image string) string {
	ref, err := reference.ParseNormalizedNamed(image)
	if err != nil {
		return image
	}
	return reference.FamiliarName(ref)
}

package serf

type coalesceEvent struct {
	Type   EventType
	Member *Member
}

type memberEventCoalescer struct {
	lastEvents   map[string]EventType
	latestEvents map[string]coalesceEvent
}

func (c *memberEventCoalescer) Handle(e Event) bool {
	switch e.EventType() {
	case EventMemberJoin:
		return true
	case EventMemberLeave:
		return true
	case EventMemberFailed:
		return true
	case EventMemberUpdate:
		return true
	case EventMemberReap:
		return true
	default:
		return false
	}
}

func (c *memberEventCoalescer) Coalesce(raw Event) {
	e := raw.(MemberEvent)
	for _, m := range e.Members {
		c.latestEvents[m.Name] = coalesceEvent{
			Type:   e.Type,
			Member: &m,
		}
	}
}

func (c *memberEventCoalescer) Flush(outCh chan<- Event) {
	// Coalesce the various events we got into a single set of events.
	events := make(map[EventType]*MemberEvent)
	for name, cevent := range c.latestEvents {
		previous, ok := c.lastEvents[name]

		// If we sent the same event before, then ignore
		// unless it is a MemberUpdate
		if ok && previous == cevent.Type && cevent.Type != EventMemberUpdate {
			continue
		}

		// Update our last event
		c.lastEvents[name] = cevent.Type

		// Add it to our event
		newEvent, ok := events[cevent.Type]
		if !ok {
			newEvent = &MemberEvent{Type: cevent.Type}
			events[cevent.Type] = newEvent
		}
		newEvent.Members = append(newEvent.Members, *cevent.Member)
	}

	// Send out those events
	for _, event := range events {
		outCh <- *event
	}
}

package serf

type latestUserEvents struct {
	LTime  LamportTime
	Events []Event
}

type userEventCoalescer struct {
	// Maps an event name into the latest versions
	events map[string]*latestUserEvents
}

func (c *userEventCoalescer) Handle(e Event) bool {
	// Only handle EventUser messages
	if e.EventType() != EventUser {
		return false
	}

	// Check if coalescing is enabled
	user := e.(UserEvent)
	return user.Coalesce
}

func (c *userEventCoalescer) Coalesce(e Event) {
	user := e.(UserEvent)
	latest, ok := c.events[user.Name]

	// Create a new entry if there are none, or
	// if this message has the newest LTime
	if !ok || latest.LTime < user.LTime {
		latest = &latestUserEvents{
			LTime:  user.LTime,
			Events: []Event{e},
		}
		c.events[user.Name] = latest
		return
	}

	// If the the same age, save it
	if latest.LTime == user.LTime {
		latest.Events = append(latest.Events, e)
	}
}

func (c *userEventCoalescer) Flush(outChan chan<- Event) {
	for _, latest := range c.events {
		for _, e := range latest.Events {
			outChan <- e
		}
	}
	c.events = make(map[string]*latestUserEvents)
}

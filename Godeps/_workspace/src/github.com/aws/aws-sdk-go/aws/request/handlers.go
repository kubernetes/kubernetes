package request

// A Handlers provides a collection of request handlers for various
// stages of handling requests.
type Handlers struct {
	Validate         HandlerList
	Build            HandlerList
	Sign             HandlerList
	Send             HandlerList
	ValidateResponse HandlerList
	Unmarshal        HandlerList
	UnmarshalMeta    HandlerList
	UnmarshalError   HandlerList
	Retry            HandlerList
	AfterRetry       HandlerList
}

// Copy returns of this handler's lists.
func (h *Handlers) Copy() Handlers {
	return Handlers{
		Validate:         h.Validate.copy(),
		Build:            h.Build.copy(),
		Sign:             h.Sign.copy(),
		Send:             h.Send.copy(),
		ValidateResponse: h.ValidateResponse.copy(),
		Unmarshal:        h.Unmarshal.copy(),
		UnmarshalError:   h.UnmarshalError.copy(),
		UnmarshalMeta:    h.UnmarshalMeta.copy(),
		Retry:            h.Retry.copy(),
		AfterRetry:       h.AfterRetry.copy(),
	}
}

// Clear removes callback functions for all handlers
func (h *Handlers) Clear() {
	h.Validate.Clear()
	h.Build.Clear()
	h.Send.Clear()
	h.Sign.Clear()
	h.Unmarshal.Clear()
	h.UnmarshalMeta.Clear()
	h.UnmarshalError.Clear()
	h.ValidateResponse.Clear()
	h.Retry.Clear()
	h.AfterRetry.Clear()
}

// A HandlerList manages zero or more handlers in a list.
type HandlerList struct {
	list []NamedHandler
}

// A NamedHandler is a struct that contains a name and function callback.
type NamedHandler struct {
	Name string
	Fn   func(*Request)
}

// copy creates a copy of the handler list.
func (l *HandlerList) copy() HandlerList {
	var n HandlerList
	n.list = append([]NamedHandler{}, l.list...)
	return n
}

// Clear clears the handler list.
func (l *HandlerList) Clear() {
	l.list = []NamedHandler{}
}

// Len returns the number of handlers in the list.
func (l *HandlerList) Len() int {
	return len(l.list)
}

// PushBack pushes handler f to the back of the handler list.
func (l *HandlerList) PushBack(f func(*Request)) {
	l.list = append(l.list, NamedHandler{"__anonymous", f})
}

// PushFront pushes handler f to the front of the handler list.
func (l *HandlerList) PushFront(f func(*Request)) {
	l.list = append([]NamedHandler{{"__anonymous", f}}, l.list...)
}

// PushBackNamed pushes named handler f to the back of the handler list.
func (l *HandlerList) PushBackNamed(n NamedHandler) {
	l.list = append(l.list, n)
}

// PushFrontNamed pushes named handler f to the front of the handler list.
func (l *HandlerList) PushFrontNamed(n NamedHandler) {
	l.list = append([]NamedHandler{n}, l.list...)
}

// Remove removes a NamedHandler n
func (l *HandlerList) Remove(n NamedHandler) {
	newlist := []NamedHandler{}
	for _, m := range l.list {
		if m.Name != n.Name {
			newlist = append(newlist, m)
		}
	}
	l.list = newlist
}

// Run executes all handlers in the list with a given request object.
func (l *HandlerList) Run(r *Request) {
	for _, f := range l.list {
		f.Fn(r)
	}
}

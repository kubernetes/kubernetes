package aws

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

// copy returns of this handler's lists.
func (h *Handlers) copy() Handlers {
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
	list []func(*Request)
}

// copy creates a copy of the handler list.
func (l *HandlerList) copy() HandlerList {
	var n HandlerList
	n.list = append([]func(*Request){}, l.list...)
	return n
}

// Clear clears the handler list.
func (l *HandlerList) Clear() {
	l.list = []func(*Request){}
}

// Len returns the number of handlers in the list.
func (l *HandlerList) Len() int {
	return len(l.list)
}

// PushBack pushes handlers f to the back of the handler list.
func (l *HandlerList) PushBack(f ...func(*Request)) {
	l.list = append(l.list, f...)
}

// PushFront pushes handlers f to the front of the handler list.
func (l *HandlerList) PushFront(f ...func(*Request)) {
	l.list = append(f, l.list...)
}

// Run executes all handlers in the list with a given request object.
func (l *HandlerList) Run(r *Request) {
	for _, f := range l.list {
		f(r)
	}
}

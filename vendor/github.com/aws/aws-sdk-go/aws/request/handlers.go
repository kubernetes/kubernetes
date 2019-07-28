package request

import (
	"fmt"
	"strings"
)

// A Handlers provides a collection of request handlers for various
// stages of handling requests.
type Handlers struct {
	Validate         HandlerList
	Build            HandlerList
	Sign             HandlerList
	Send             HandlerList
	ValidateResponse HandlerList
	Unmarshal        HandlerList
	UnmarshalStream  HandlerList
	UnmarshalMeta    HandlerList
	UnmarshalError   HandlerList
	Retry            HandlerList
	AfterRetry       HandlerList
	CompleteAttempt  HandlerList
	Complete         HandlerList
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
		UnmarshalStream:  h.UnmarshalStream.copy(),
		UnmarshalError:   h.UnmarshalError.copy(),
		UnmarshalMeta:    h.UnmarshalMeta.copy(),
		Retry:            h.Retry.copy(),
		AfterRetry:       h.AfterRetry.copy(),
		CompleteAttempt:  h.CompleteAttempt.copy(),
		Complete:         h.Complete.copy(),
	}
}

// Clear removes callback functions for all handlers
func (h *Handlers) Clear() {
	h.Validate.Clear()
	h.Build.Clear()
	h.Send.Clear()
	h.Sign.Clear()
	h.Unmarshal.Clear()
	h.UnmarshalStream.Clear()
	h.UnmarshalMeta.Clear()
	h.UnmarshalError.Clear()
	h.ValidateResponse.Clear()
	h.Retry.Clear()
	h.AfterRetry.Clear()
	h.CompleteAttempt.Clear()
	h.Complete.Clear()
}

// A HandlerListRunItem represents an entry in the HandlerList which
// is being run.
type HandlerListRunItem struct {
	Index   int
	Handler NamedHandler
	Request *Request
}

// A HandlerList manages zero or more handlers in a list.
type HandlerList struct {
	list []NamedHandler

	// Called after each request handler in the list is called. If set
	// and the func returns true the HandlerList will continue to iterate
	// over the request handlers. If false is returned the HandlerList
	// will stop iterating.
	//
	// Should be used if extra logic to be performed between each handler
	// in the list. This can be used to terminate a list's iteration
	// based on a condition such as error like, HandlerListStopOnError.
	// Or for logging like HandlerListLogItem.
	AfterEachFn func(item HandlerListRunItem) bool
}

// A NamedHandler is a struct that contains a name and function callback.
type NamedHandler struct {
	Name string
	Fn   func(*Request)
}

// copy creates a copy of the handler list.
func (l *HandlerList) copy() HandlerList {
	n := HandlerList{
		AfterEachFn: l.AfterEachFn,
	}
	if len(l.list) == 0 {
		return n
	}

	n.list = append(make([]NamedHandler, 0, len(l.list)), l.list...)
	return n
}

// Clear clears the handler list.
func (l *HandlerList) Clear() {
	l.list = l.list[0:0]
}

// Len returns the number of handlers in the list.
func (l *HandlerList) Len() int {
	return len(l.list)
}

// PushBack pushes handler f to the back of the handler list.
func (l *HandlerList) PushBack(f func(*Request)) {
	l.PushBackNamed(NamedHandler{"__anonymous", f})
}

// PushBackNamed pushes named handler f to the back of the handler list.
func (l *HandlerList) PushBackNamed(n NamedHandler) {
	if cap(l.list) == 0 {
		l.list = make([]NamedHandler, 0, 5)
	}
	l.list = append(l.list, n)
}

// PushFront pushes handler f to the front of the handler list.
func (l *HandlerList) PushFront(f func(*Request)) {
	l.PushFrontNamed(NamedHandler{"__anonymous", f})
}

// PushFrontNamed pushes named handler f to the front of the handler list.
func (l *HandlerList) PushFrontNamed(n NamedHandler) {
	if cap(l.list) == len(l.list) {
		// Allocating new list required
		l.list = append([]NamedHandler{n}, l.list...)
	} else {
		// Enough room to prepend into list.
		l.list = append(l.list, NamedHandler{})
		copy(l.list[1:], l.list)
		l.list[0] = n
	}
}

// Remove removes a NamedHandler n
func (l *HandlerList) Remove(n NamedHandler) {
	l.RemoveByName(n.Name)
}

// RemoveByName removes a NamedHandler by name.
func (l *HandlerList) RemoveByName(name string) {
	for i := 0; i < len(l.list); i++ {
		m := l.list[i]
		if m.Name == name {
			// Shift array preventing creating new arrays
			copy(l.list[i:], l.list[i+1:])
			l.list[len(l.list)-1] = NamedHandler{}
			l.list = l.list[:len(l.list)-1]

			// decrement list so next check to length is correct
			i--
		}
	}
}

// SwapNamed will swap out any existing handlers with the same name as the
// passed in NamedHandler returning true if handlers were swapped. False is
// returned otherwise.
func (l *HandlerList) SwapNamed(n NamedHandler) (swapped bool) {
	for i := 0; i < len(l.list); i++ {
		if l.list[i].Name == n.Name {
			l.list[i].Fn = n.Fn
			swapped = true
		}
	}

	return swapped
}

// Swap will swap out all handlers matching the name passed in. The matched
// handlers will be swapped in. True is returned if the handlers were swapped.
func (l *HandlerList) Swap(name string, replace NamedHandler) bool {
	var swapped bool

	for i := 0; i < len(l.list); i++ {
		if l.list[i].Name == name {
			l.list[i] = replace
			swapped = true
		}
	}

	return swapped
}

// SetBackNamed will replace the named handler if it exists in the handler list.
// If the handler does not exist the handler will be added to the end of the list.
func (l *HandlerList) SetBackNamed(n NamedHandler) {
	if !l.SwapNamed(n) {
		l.PushBackNamed(n)
	}
}

// SetFrontNamed will replace the named handler if it exists in the handler list.
// If the handler does not exist the handler will be added to the beginning of
// the list.
func (l *HandlerList) SetFrontNamed(n NamedHandler) {
	if !l.SwapNamed(n) {
		l.PushFrontNamed(n)
	}
}

// Run executes all handlers in the list with a given request object.
func (l *HandlerList) Run(r *Request) {
	for i, h := range l.list {
		h.Fn(r)
		item := HandlerListRunItem{
			Index: i, Handler: h, Request: r,
		}
		if l.AfterEachFn != nil && !l.AfterEachFn(item) {
			return
		}
	}
}

// HandlerListLogItem logs the request handler and the state of the
// request's Error value. Always returns true to continue iterating
// request handlers in a HandlerList.
func HandlerListLogItem(item HandlerListRunItem) bool {
	if item.Request.Config.Logger == nil {
		return true
	}
	item.Request.Config.Logger.Log("DEBUG: RequestHandler",
		item.Index, item.Handler.Name, item.Request.Error)

	return true
}

// HandlerListStopOnError returns false to stop the HandlerList iterating
// over request handlers if Request.Error is not nil. True otherwise
// to continue iterating.
func HandlerListStopOnError(item HandlerListRunItem) bool {
	return item.Request.Error == nil
}

// WithAppendUserAgent will add a string to the user agent prefixed with a
// single white space.
func WithAppendUserAgent(s string) Option {
	return func(r *Request) {
		r.Handlers.Build.PushBack(func(r2 *Request) {
			AddToUserAgent(r, s)
		})
	}
}

// MakeAddToUserAgentHandler will add the name/version pair to the User-Agent request
// header. If the extra parameters are provided they will be added as metadata to the
// name/version pair resulting in the following format.
// "name/version (extra0; extra1; ...)"
// The user agent part will be concatenated with this current request's user agent string.
func MakeAddToUserAgentHandler(name, version string, extra ...string) func(*Request) {
	ua := fmt.Sprintf("%s/%s", name, version)
	if len(extra) > 0 {
		ua += fmt.Sprintf(" (%s)", strings.Join(extra, "; "))
	}
	return func(r *Request) {
		AddToUserAgent(r, ua)
	}
}

// MakeAddToUserAgentFreeFormHandler adds the input to the User-Agent request header.
// The input string will be concatenated with the current request's user agent string.
func MakeAddToUserAgentFreeFormHandler(s string) func(*Request) {
	return func(r *Request) {
		AddToUserAgent(r, s)
	}
}

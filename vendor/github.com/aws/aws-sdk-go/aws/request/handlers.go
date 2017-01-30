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

package memberlist

// EventDelegate is a simpler delegate that is used only to receive
// notifications about members joining and leaving. The methods in this
// delegate may be called by multiple goroutines, but never concurrently.
// This allows you to reason about ordering.
type EventDelegate interface {
	// NotifyJoin is invoked when a node is detected to have joined.
	// The Node argument must not be modified.
	NotifyJoin(*Node)

	// NotifyLeave is invoked when a node is detected to have left.
	// The Node argument must not be modified.
	NotifyLeave(*Node)

	// NotifyUpdate is invoked when a node is detected to have
	// updated, usually involving the meta data. The Node argument
	// must not be modified.
	NotifyUpdate(*Node)
}

// ChannelEventDelegate is used to enable an application to receive
// events about joins and leaves over a channel instead of a direct
// function call.
//
// Care must be taken that events are processed in a timely manner from
// the channel, since this delegate will block until an event can be sent.
type ChannelEventDelegate struct {
	Ch chan<- NodeEvent
}

// NodeEventType are the types of events that can be sent from the
// ChannelEventDelegate.
type NodeEventType int

const (
	NodeJoin NodeEventType = iota
	NodeLeave
	NodeUpdate
)

// NodeEvent is a single event related to node activity in the memberlist.
// The Node member of this struct must not be directly modified. It is passed
// as a pointer to avoid unnecessary copies. If you wish to modify the node,
// make a copy first.
type NodeEvent struct {
	Event NodeEventType
	Node  *Node
}

func (c *ChannelEventDelegate) NotifyJoin(n *Node) {
	c.Ch <- NodeEvent{NodeJoin, n}
}

func (c *ChannelEventDelegate) NotifyLeave(n *Node) {
	c.Ch <- NodeEvent{NodeLeave, n}
}

func (c *ChannelEventDelegate) NotifyUpdate(n *Node) {
	c.Ch <- NodeEvent{NodeUpdate, n}
}

package memberlist

// Delegate is the interface that clients must implement if they want to hook
// into the gossip layer of Memberlist. All the methods must be thread-safe,
// as they can and generally will be called concurrently.
type Delegate interface {
	// NodeMeta is used to retrieve meta-data about the current node
	// when broadcasting an alive message. It's length is limited to
	// the given byte size. This metadata is available in the Node structure.
	NodeMeta(limit int) []byte

	// NotifyMsg is called when a user-data message is received.
	// Care should be taken that this method does not block, since doing
	// so would block the entire UDP packet receive loop. Additionally, the byte
	// slice may be modified after the call returns, so it should be copied if needed.
	NotifyMsg([]byte)

	// GetBroadcasts is called when user data messages can be broadcast.
	// It can return a list of buffers to send. Each buffer should assume an
	// overhead as provided with a limit on the total byte size allowed.
	// The total byte size of the resulting data to send must not exceed
	// the limit. Care should be taken that this method does not block,
	// since doing so would block the entire UDP packet receive loop.
	GetBroadcasts(overhead, limit int) [][]byte

	// LocalState is used for a TCP Push/Pull. This is sent to
	// the remote side in addition to the membership information. Any
	// data can be sent here. See MergeRemoteState as well. The `join`
	// boolean indicates this is for a join instead of a push/pull.
	LocalState(join bool) []byte

	// MergeRemoteState is invoked after a TCP Push/Pull. This is the
	// state received from the remote side and is the result of the
	// remote side's LocalState call. The 'join'
	// boolean indicates this is for a join instead of a push/pull.
	MergeRemoteState(buf []byte, join bool)
}

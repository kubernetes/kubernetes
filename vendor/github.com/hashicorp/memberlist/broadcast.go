package memberlist

/*
The broadcast mechanism works by maintaining a sorted list of messages to be
sent out. When a message is to be broadcast, the retransmit count
is set to zero and appended to the queue. The retransmit count serves
as the "priority", ensuring that newer messages get sent first. Once
a message hits the retransmit limit, it is removed from the queue.

Additionally, older entries can be invalidated by new messages that
are contradictory. For example, if we send "{suspect M1 inc: 1},
then a following {alive M1 inc: 2} will invalidate that message
*/

type memberlistBroadcast struct {
	node   string
	msg    []byte
	notify chan struct{}
}

func (b *memberlistBroadcast) Invalidates(other Broadcast) bool {
	// Check if that broadcast is a memberlist type
	mb, ok := other.(*memberlistBroadcast)
	if !ok {
		return false
	}

	// Invalidates any message about the same node
	return b.node == mb.node
}

func (b *memberlistBroadcast) Message() []byte {
	return b.msg
}

func (b *memberlistBroadcast) Finished() {
	select {
	case b.notify <- struct{}{}:
	default:
	}
}

// encodeAndBroadcast encodes a message and enqueues it for broadcast. Fails
// silently if there is an encoding error.
func (m *Memberlist) encodeAndBroadcast(node string, msgType messageType, msg interface{}) {
	m.encodeBroadcastNotify(node, msgType, msg, nil)
}

// encodeBroadcastNotify encodes a message and enqueues it for broadcast
// and notifies the given channel when transmission is finished. Fails
// silently if there is an encoding error.
func (m *Memberlist) encodeBroadcastNotify(node string, msgType messageType, msg interface{}, notify chan struct{}) {
	buf, err := encode(msgType, msg)
	if err != nil {
		m.logger.Printf("[ERR] memberlist: Failed to encode message for broadcast: %s", err)
	} else {
		m.queueBroadcast(node, buf.Bytes(), notify)
	}
}

// queueBroadcast is used to start dissemination of a message. It will be
// sent up to a configured number of times. The message could potentially
// be invalidated by a future message about the same node
func (m *Memberlist) queueBroadcast(node string, msg []byte, notify chan struct{}) {
	b := &memberlistBroadcast{node, msg, notify}
	m.broadcasts.QueueBroadcast(b)
}

// getBroadcasts is used to return a slice of broadcasts to send up to
// a maximum byte size, while imposing a per-broadcast overhead. This is used
// to fill a UDP packet with piggybacked data
func (m *Memberlist) getBroadcasts(overhead, limit int) [][]byte {
	// Get memberlist messages first
	toSend := m.broadcasts.GetBroadcasts(overhead, limit)

	// Check if the user has anything to broadcast
	d := m.config.Delegate
	if d != nil {
		// Determine the bytes used already
		bytesUsed := 0
		for _, msg := range toSend {
			bytesUsed += len(msg) + overhead
		}

		// Check space remaining for user messages
		avail := limit - bytesUsed
		if avail > overhead+userMsgOverhead {
			userMsgs := d.GetBroadcasts(overhead+userMsgOverhead, avail)

			// Frame each user message
			for _, msg := range userMsgs {
				buf := make([]byte, 1, len(msg)+1)
				buf[0] = byte(userMsg)
				buf = append(buf, msg...)
				toSend = append(toSend, buf)
			}
		}
	}
	return toSend
}

package memberlist

import "time"

// PingDelegate is used to notify an observer how long it took for a ping message to
// complete a round trip.  It can also be used for writing arbitrary byte slices
// into ack messages. Note that in order to be meaningful for RTT estimates, this
// delegate does not apply to indirect pings, nor fallback pings sent over TCP.
type PingDelegate interface {
	// AckPayload is invoked when an ack is being sent; the returned bytes will be appended to the ack
	AckPayload() []byte
	// NotifyPing is invoked when an ack for a ping is received
	NotifyPingComplete(other *Node, rtt time.Duration, payload []byte)
}

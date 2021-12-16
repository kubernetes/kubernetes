package congestion

import (
	"math"
	"time"

	"github.com/lucas-clemente/quic-go/internal/protocol"
	"github.com/lucas-clemente/quic-go/internal/utils"
)

// This cubic implementation is based on the one found in Chromiums's QUIC
// implementation, in the files net/quic/congestion_control/cubic.{hh,cc}.

// Constants based on TCP defaults.
// The following constants are in 2^10 fractions of a second instead of ms to
// allow a 10 shift right to divide.

// 1024*1024^3 (first 1024 is from 0.100^3)
// where 0.100 is 100 ms which is the scaling round trip time.
const (
	cubeScale                                    = 40
	cubeCongestionWindowScale                    = 410
	cubeFactor                protocol.ByteCount = 1 << cubeScale / cubeCongestionWindowScale / maxDatagramSize
	// TODO: when re-enabling cubic, make sure to use the actual packet size here
	maxDatagramSize = protocol.ByteCount(protocol.InitialPacketSizeIPv4)
)

const defaultNumConnections = 1

// Default Cubic backoff factor
const beta float32 = 0.7

// Additional backoff factor when loss occurs in the concave part of the Cubic
// curve. This additional backoff factor is expected to give up bandwidth to
// new concurrent flows and speed up convergence.
const betaLastMax float32 = 0.85

// Cubic implements the cubic algorithm from TCP
type Cubic struct {
	clock Clock

	// Number of connections to simulate.
	numConnections int

	// Time when this cycle started, after last loss event.
	epoch time.Time

	// Max congestion window used just before last loss event.
	// Note: to improve fairness to other streams an additional back off is
	// applied to this value if the new value is below our latest value.
	lastMaxCongestionWindow protocol.ByteCount

	// Number of acked bytes since the cycle started (epoch).
	ackedBytesCount protocol.ByteCount

	// TCP Reno equivalent congestion window in packets.
	estimatedTCPcongestionWindow protocol.ByteCount

	// Origin point of cubic function.
	originPointCongestionWindow protocol.ByteCount

	// Time to origin point of cubic function in 2^10 fractions of a second.
	timeToOriginPoint uint32

	// Last congestion window in packets computed by cubic function.
	lastTargetCongestionWindow protocol.ByteCount
}

// NewCubic returns a new Cubic instance
func NewCubic(clock Clock) *Cubic {
	c := &Cubic{
		clock:          clock,
		numConnections: defaultNumConnections,
	}
	c.Reset()
	return c
}

// Reset is called after a timeout to reset the cubic state
func (c *Cubic) Reset() {
	c.epoch = time.Time{}
	c.lastMaxCongestionWindow = 0
	c.ackedBytesCount = 0
	c.estimatedTCPcongestionWindow = 0
	c.originPointCongestionWindow = 0
	c.timeToOriginPoint = 0
	c.lastTargetCongestionWindow = 0
}

func (c *Cubic) alpha() float32 {
	// TCPFriendly alpha is described in Section 3.3 of the CUBIC paper. Note that
	// beta here is a cwnd multiplier, and is equal to 1-beta from the paper.
	// We derive the equivalent alpha for an N-connection emulation as:
	b := c.beta()
	return 3 * float32(c.numConnections) * float32(c.numConnections) * (1 - b) / (1 + b)
}

func (c *Cubic) beta() float32 {
	// kNConnectionBeta is the backoff factor after loss for our N-connection
	// emulation, which emulates the effective backoff of an ensemble of N
	// TCP-Reno connections on a single loss event. The effective multiplier is
	// computed as:
	return (float32(c.numConnections) - 1 + beta) / float32(c.numConnections)
}

func (c *Cubic) betaLastMax() float32 {
	// betaLastMax is the additional backoff factor after loss for our
	// N-connection emulation, which emulates the additional backoff of
	// an ensemble of N TCP-Reno connections on a single loss event. The
	// effective multiplier is computed as:
	return (float32(c.numConnections) - 1 + betaLastMax) / float32(c.numConnections)
}

// OnApplicationLimited is called on ack arrival when sender is unable to use
// the available congestion window. Resets Cubic state during quiescence.
func (c *Cubic) OnApplicationLimited() {
	// When sender is not using the available congestion window, the window does
	// not grow. But to be RTT-independent, Cubic assumes that the sender has been
	// using the entire window during the time since the beginning of the current
	// "epoch" (the end of the last loss recovery period). Since
	// application-limited periods break this assumption, we reset the epoch when
	// in such a period. This reset effectively freezes congestion window growth
	// through application-limited periods and allows Cubic growth to continue
	// when the entire window is being used.
	c.epoch = time.Time{}
}

// CongestionWindowAfterPacketLoss computes a new congestion window to use after
// a loss event. Returns the new congestion window in packets. The new
// congestion window is a multiplicative decrease of our current window.
func (c *Cubic) CongestionWindowAfterPacketLoss(currentCongestionWindow protocol.ByteCount) protocol.ByteCount {
	if currentCongestionWindow+maxDatagramSize < c.lastMaxCongestionWindow {
		// We never reached the old max, so assume we are competing with another
		// flow. Use our extra back off factor to allow the other flow to go up.
		c.lastMaxCongestionWindow = protocol.ByteCount(c.betaLastMax() * float32(currentCongestionWindow))
	} else {
		c.lastMaxCongestionWindow = currentCongestionWindow
	}
	c.epoch = time.Time{} // Reset time.
	return protocol.ByteCount(float32(currentCongestionWindow) * c.beta())
}

// CongestionWindowAfterAck computes a new congestion window to use after a received ACK.
// Returns the new congestion window in packets. The new congestion window
// follows a cubic function that depends on the time passed since last
// packet loss.
func (c *Cubic) CongestionWindowAfterAck(
	ackedBytes protocol.ByteCount,
	currentCongestionWindow protocol.ByteCount,
	delayMin time.Duration,
	eventTime time.Time,
) protocol.ByteCount {
	c.ackedBytesCount += ackedBytes

	if c.epoch.IsZero() {
		// First ACK after a loss event.
		c.epoch = eventTime            // Start of epoch.
		c.ackedBytesCount = ackedBytes // Reset count.
		// Reset estimated_tcp_congestion_window_ to be in sync with cubic.
		c.estimatedTCPcongestionWindow = currentCongestionWindow
		if c.lastMaxCongestionWindow <= currentCongestionWindow {
			c.timeToOriginPoint = 0
			c.originPointCongestionWindow = currentCongestionWindow
		} else {
			c.timeToOriginPoint = uint32(math.Cbrt(float64(cubeFactor * (c.lastMaxCongestionWindow - currentCongestionWindow))))
			c.originPointCongestionWindow = c.lastMaxCongestionWindow
		}
	}

	// Change the time unit from microseconds to 2^10 fractions per second. Take
	// the round trip time in account. This is done to allow us to use shift as a
	// divide operator.
	elapsedTime := int64(eventTime.Add(delayMin).Sub(c.epoch)/time.Microsecond) << 10 / (1000 * 1000)

	// Right-shifts of negative, signed numbers have implementation-dependent
	// behavior, so force the offset to be positive, as is done in the kernel.
	offset := int64(c.timeToOriginPoint) - elapsedTime
	if offset < 0 {
		offset = -offset
	}

	deltaCongestionWindow := protocol.ByteCount(cubeCongestionWindowScale*offset*offset*offset) * maxDatagramSize >> cubeScale
	var targetCongestionWindow protocol.ByteCount
	if elapsedTime > int64(c.timeToOriginPoint) {
		targetCongestionWindow = c.originPointCongestionWindow + deltaCongestionWindow
	} else {
		targetCongestionWindow = c.originPointCongestionWindow - deltaCongestionWindow
	}
	// Limit the CWND increase to half the acked bytes.
	targetCongestionWindow = utils.MinByteCount(targetCongestionWindow, currentCongestionWindow+c.ackedBytesCount/2)

	// Increase the window by approximately Alpha * 1 MSS of bytes every
	// time we ack an estimated tcp window of bytes.  For small
	// congestion windows (less than 25), the formula below will
	// increase slightly slower than linearly per estimated tcp window
	// of bytes.
	c.estimatedTCPcongestionWindow += protocol.ByteCount(float32(c.ackedBytesCount) * c.alpha() * float32(maxDatagramSize) / float32(c.estimatedTCPcongestionWindow))
	c.ackedBytesCount = 0

	// We have a new cubic congestion window.
	c.lastTargetCongestionWindow = targetCongestionWindow

	// Compute target congestion_window based on cubic target and estimated TCP
	// congestion_window, use highest (fastest).
	if targetCongestionWindow < c.estimatedTCPcongestionWindow {
		targetCongestionWindow = c.estimatedTCPcongestionWindow
	}
	return targetCongestionWindow
}

// SetNumConnections sets the number of emulated connections
func (c *Cubic) SetNumConnections(n int) {
	c.numConnections = n
}

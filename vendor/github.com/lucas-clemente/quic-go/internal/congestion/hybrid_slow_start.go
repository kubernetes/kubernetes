package congestion

import (
	"time"

	"github.com/lucas-clemente/quic-go/internal/protocol"
	"github.com/lucas-clemente/quic-go/internal/utils"
)

// Note(pwestin): the magic clamping numbers come from the original code in
// tcp_cubic.c.
const hybridStartLowWindow = protocol.ByteCount(16)

// Number of delay samples for detecting the increase of delay.
const hybridStartMinSamples = uint32(8)

// Exit slow start if the min rtt has increased by more than 1/8th.
const hybridStartDelayFactorExp = 3 // 2^3 = 8
// The original paper specifies 2 and 8ms, but those have changed over time.
const (
	hybridStartDelayMinThresholdUs = int64(4000)
	hybridStartDelayMaxThresholdUs = int64(16000)
)

// HybridSlowStart implements the TCP hybrid slow start algorithm
type HybridSlowStart struct {
	endPacketNumber      protocol.PacketNumber
	lastSentPacketNumber protocol.PacketNumber
	started              bool
	currentMinRTT        time.Duration
	rttSampleCount       uint32
	hystartFound         bool
}

// StartReceiveRound is called for the start of each receive round (burst) in the slow start phase.
func (s *HybridSlowStart) StartReceiveRound(lastSent protocol.PacketNumber) {
	s.endPacketNumber = lastSent
	s.currentMinRTT = 0
	s.rttSampleCount = 0
	s.started = true
}

// IsEndOfRound returns true if this ack is the last packet number of our current slow start round.
func (s *HybridSlowStart) IsEndOfRound(ack protocol.PacketNumber) bool {
	return s.endPacketNumber < ack
}

// ShouldExitSlowStart should be called on every new ack frame, since a new
// RTT measurement can be made then.
// rtt: the RTT for this ack packet.
// minRTT: is the lowest delay (RTT) we have seen during the session.
// congestionWindow: the congestion window in packets.
func (s *HybridSlowStart) ShouldExitSlowStart(latestRTT time.Duration, minRTT time.Duration, congestionWindow protocol.ByteCount) bool {
	if !s.started {
		// Time to start the hybrid slow start.
		s.StartReceiveRound(s.lastSentPacketNumber)
	}
	if s.hystartFound {
		return true
	}
	// Second detection parameter - delay increase detection.
	// Compare the minimum delay (s.currentMinRTT) of the current
	// burst of packets relative to the minimum delay during the session.
	// Note: we only look at the first few(8) packets in each burst, since we
	// only want to compare the lowest RTT of the burst relative to previous
	// bursts.
	s.rttSampleCount++
	if s.rttSampleCount <= hybridStartMinSamples {
		if s.currentMinRTT == 0 || s.currentMinRTT > latestRTT {
			s.currentMinRTT = latestRTT
		}
	}
	// We only need to check this once per round.
	if s.rttSampleCount == hybridStartMinSamples {
		// Divide minRTT by 8 to get a rtt increase threshold for exiting.
		minRTTincreaseThresholdUs := int64(minRTT / time.Microsecond >> hybridStartDelayFactorExp)
		// Ensure the rtt threshold is never less than 2ms or more than 16ms.
		minRTTincreaseThresholdUs = utils.MinInt64(minRTTincreaseThresholdUs, hybridStartDelayMaxThresholdUs)
		minRTTincreaseThreshold := time.Duration(utils.MaxInt64(minRTTincreaseThresholdUs, hybridStartDelayMinThresholdUs)) * time.Microsecond

		if s.currentMinRTT > (minRTT + minRTTincreaseThreshold) {
			s.hystartFound = true
		}
	}
	// Exit from slow start if the cwnd is greater than 16 and
	// increasing delay is found.
	return congestionWindow >= hybridStartLowWindow && s.hystartFound
}

// OnPacketSent is called when a packet was sent
func (s *HybridSlowStart) OnPacketSent(packetNumber protocol.PacketNumber) {
	s.lastSentPacketNumber = packetNumber
}

// OnPacketAcked gets invoked after ShouldExitSlowStart, so it's best to end
// the round when the final packet of the burst is received and start it on
// the next incoming ack.
func (s *HybridSlowStart) OnPacketAcked(ackedPacketNumber protocol.PacketNumber) {
	if s.IsEndOfRound(ackedPacketNumber) {
		s.started = false
	}
}

// Started returns true if started
func (s *HybridSlowStart) Started() bool {
	return s.started
}

// Restart the slow start phase
func (s *HybridSlowStart) Restart() {
	s.started = false
	s.hystartFound = false
}

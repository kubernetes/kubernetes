package ackhandler

import (
	"github.com/lucas-clemente/quic-go/internal/protocol"
	"github.com/lucas-clemente/quic-go/internal/utils"
	"github.com/lucas-clemente/quic-go/internal/wire"
)

// The receivedPacketHistory stores if a packet number has already been received.
// It generates ACK ranges which can be used to assemble an ACK frame.
// It does not store packet contents.
type receivedPacketHistory struct {
	ranges *utils.PacketIntervalList

	deletedBelow protocol.PacketNumber
}

func newReceivedPacketHistory() *receivedPacketHistory {
	return &receivedPacketHistory{
		ranges: utils.NewPacketIntervalList(),
	}
}

// ReceivedPacket registers a packet with PacketNumber p and updates the ranges
func (h *receivedPacketHistory) ReceivedPacket(p protocol.PacketNumber) bool /* is a new packet (and not a duplicate / delayed packet) */ {
	// ignore delayed packets, if we already deleted the range
	if p < h.deletedBelow {
		return false
	}
	isNew := h.addToRanges(p)
	h.maybeDeleteOldRanges()
	return isNew
}

func (h *receivedPacketHistory) addToRanges(p protocol.PacketNumber) bool /* is a new packet (and not a duplicate / delayed packet) */ {
	if h.ranges.Len() == 0 {
		h.ranges.PushBack(utils.PacketInterval{Start: p, End: p})
		return true
	}

	for el := h.ranges.Back(); el != nil; el = el.Prev() {
		// p already included in an existing range. Nothing to do here
		if p >= el.Value.Start && p <= el.Value.End {
			return false
		}

		if el.Value.End == p-1 { // extend a range at the end
			el.Value.End = p
			return true
		}
		if el.Value.Start == p+1 { // extend a range at the beginning
			el.Value.Start = p

			prev := el.Prev()
			if prev != nil && prev.Value.End+1 == el.Value.Start { // merge two ranges
				prev.Value.End = el.Value.End
				h.ranges.Remove(el)
			}
			return true
		}

		// create a new range at the end
		if p > el.Value.End {
			h.ranges.InsertAfter(utils.PacketInterval{Start: p, End: p}, el)
			return true
		}
	}

	// create a new range at the beginning
	h.ranges.InsertBefore(utils.PacketInterval{Start: p, End: p}, h.ranges.Front())
	return true
}

// Delete old ranges, if we're tracking more than 500 of them.
// This is a DoS defense against a peer that sends us too many gaps.
func (h *receivedPacketHistory) maybeDeleteOldRanges() {
	for h.ranges.Len() > protocol.MaxNumAckRanges {
		h.ranges.Remove(h.ranges.Front())
	}
}

// DeleteBelow deletes all entries below (but not including) p
func (h *receivedPacketHistory) DeleteBelow(p protocol.PacketNumber) {
	if p < h.deletedBelow {
		return
	}
	h.deletedBelow = p

	nextEl := h.ranges.Front()
	for el := h.ranges.Front(); nextEl != nil; el = nextEl {
		nextEl = el.Next()

		if el.Value.End < p { // delete a whole range
			h.ranges.Remove(el)
		} else if p > el.Value.Start && p <= el.Value.End {
			el.Value.Start = p
			return
		} else { // no ranges affected. Nothing to do
			return
		}
	}
}

// GetAckRanges gets a slice of all AckRanges that can be used in an AckFrame
func (h *receivedPacketHistory) GetAckRanges() []wire.AckRange {
	if h.ranges.Len() == 0 {
		return nil
	}

	ackRanges := make([]wire.AckRange, h.ranges.Len())
	i := 0
	for el := h.ranges.Back(); el != nil; el = el.Prev() {
		ackRanges[i] = wire.AckRange{Smallest: el.Value.Start, Largest: el.Value.End}
		i++
	}
	return ackRanges
}

func (h *receivedPacketHistory) GetHighestAckRange() wire.AckRange {
	ackRange := wire.AckRange{}
	if h.ranges.Len() > 0 {
		r := h.ranges.Back().Value
		ackRange.Smallest = r.Start
		ackRange.Largest = r.End
	}
	return ackRange
}

func (h *receivedPacketHistory) IsPotentiallyDuplicate(p protocol.PacketNumber) bool {
	if p < h.deletedBelow {
		return true
	}
	for el := h.ranges.Back(); el != nil; el = el.Prev() {
		if p > el.Value.End {
			return false
		}
		if p <= el.Value.End && p >= el.Value.Start {
			return true
		}
	}
	return false
}

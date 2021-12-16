package ackhandler

import (
	"fmt"
	"time"

	"github.com/lucas-clemente/quic-go/internal/protocol"
	"github.com/lucas-clemente/quic-go/internal/utils"
)

type sentPacketHistory struct {
	rttStats    *utils.RTTStats
	packetList  *PacketList
	packetMap   map[protocol.PacketNumber]*PacketElement
	highestSent protocol.PacketNumber
}

func newSentPacketHistory(rttStats *utils.RTTStats) *sentPacketHistory {
	return &sentPacketHistory{
		rttStats:    rttStats,
		packetList:  NewPacketList(),
		packetMap:   make(map[protocol.PacketNumber]*PacketElement),
		highestSent: protocol.InvalidPacketNumber,
	}
}

func (h *sentPacketHistory) SentPacket(p *Packet, isAckEliciting bool) {
	if p.PacketNumber <= h.highestSent {
		panic("non-sequential packet number use")
	}
	// Skipped packet numbers.
	for pn := h.highestSent + 1; pn < p.PacketNumber; pn++ {
		el := h.packetList.PushBack(Packet{
			PacketNumber:    pn,
			EncryptionLevel: p.EncryptionLevel,
			SendTime:        p.SendTime,
			skippedPacket:   true,
		})
		h.packetMap[pn] = el
	}
	h.highestSent = p.PacketNumber

	if isAckEliciting {
		el := h.packetList.PushBack(*p)
		h.packetMap[p.PacketNumber] = el
	}
}

// Iterate iterates through all packets.
func (h *sentPacketHistory) Iterate(cb func(*Packet) (cont bool, err error)) error {
	cont := true
	var next *PacketElement
	for el := h.packetList.Front(); cont && el != nil; el = next {
		var err error
		next = el.Next()
		cont, err = cb(&el.Value)
		if err != nil {
			return err
		}
	}
	return nil
}

// FirstOutStanding returns the first outstanding packet.
func (h *sentPacketHistory) FirstOutstanding() *Packet {
	for el := h.packetList.Front(); el != nil; el = el.Next() {
		p := &el.Value
		if !p.declaredLost && !p.skippedPacket && !p.IsPathMTUProbePacket {
			return p
		}
	}
	return nil
}

func (h *sentPacketHistory) Len() int {
	return len(h.packetMap)
}

func (h *sentPacketHistory) Remove(p protocol.PacketNumber) error {
	el, ok := h.packetMap[p]
	if !ok {
		return fmt.Errorf("packet %d not found in sent packet history", p)
	}
	h.packetList.Remove(el)
	delete(h.packetMap, p)
	return nil
}

func (h *sentPacketHistory) HasOutstandingPackets() bool {
	return h.FirstOutstanding() != nil
}

func (h *sentPacketHistory) DeleteOldPackets(now time.Time) {
	maxAge := 3 * h.rttStats.PTO(false)
	var nextEl *PacketElement
	for el := h.packetList.Front(); el != nil; el = nextEl {
		nextEl = el.Next()
		p := el.Value
		if p.SendTime.After(now.Add(-maxAge)) {
			break
		}
		if !p.skippedPacket && !p.declaredLost { // should only happen in the case of drastic RTT changes
			continue
		}
		delete(h.packetMap, p.PacketNumber)
		h.packetList.Remove(el)
	}
}

package logging

import (
	"context"
	"net"
	"time"
)

type tracerMultiplexer struct {
	tracers []Tracer
}

var _ Tracer = &tracerMultiplexer{}

// NewMultiplexedTracer creates a new tracer that multiplexes events to multiple tracers.
func NewMultiplexedTracer(tracers ...Tracer) Tracer {
	if len(tracers) == 0 {
		return nil
	}
	if len(tracers) == 1 {
		return tracers[0]
	}
	return &tracerMultiplexer{tracers}
}

func (m *tracerMultiplexer) TracerForConnection(ctx context.Context, p Perspective, odcid ConnectionID) ConnectionTracer {
	var connTracers []ConnectionTracer
	for _, t := range m.tracers {
		if ct := t.TracerForConnection(ctx, p, odcid); ct != nil {
			connTracers = append(connTracers, ct)
		}
	}
	return NewMultiplexedConnectionTracer(connTracers...)
}

func (m *tracerMultiplexer) SentPacket(remote net.Addr, hdr *Header, size ByteCount, frames []Frame) {
	for _, t := range m.tracers {
		t.SentPacket(remote, hdr, size, frames)
	}
}

func (m *tracerMultiplexer) DroppedPacket(remote net.Addr, typ PacketType, size ByteCount, reason PacketDropReason) {
	for _, t := range m.tracers {
		t.DroppedPacket(remote, typ, size, reason)
	}
}

type connTracerMultiplexer struct {
	tracers []ConnectionTracer
}

var _ ConnectionTracer = &connTracerMultiplexer{}

// NewMultiplexedConnectionTracer creates a new connection tracer that multiplexes events to multiple tracers.
func NewMultiplexedConnectionTracer(tracers ...ConnectionTracer) ConnectionTracer {
	if len(tracers) == 0 {
		return nil
	}
	if len(tracers) == 1 {
		return tracers[0]
	}
	return &connTracerMultiplexer{tracers: tracers}
}

func (m *connTracerMultiplexer) StartedConnection(local, remote net.Addr, srcConnID, destConnID ConnectionID) {
	for _, t := range m.tracers {
		t.StartedConnection(local, remote, srcConnID, destConnID)
	}
}

func (m *connTracerMultiplexer) NegotiatedVersion(chosen VersionNumber, clientVersions, serverVersions []VersionNumber) {
	for _, t := range m.tracers {
		t.NegotiatedVersion(chosen, clientVersions, serverVersions)
	}
}

func (m *connTracerMultiplexer) ClosedConnection(e error) {
	for _, t := range m.tracers {
		t.ClosedConnection(e)
	}
}

func (m *connTracerMultiplexer) SentTransportParameters(tp *TransportParameters) {
	for _, t := range m.tracers {
		t.SentTransportParameters(tp)
	}
}

func (m *connTracerMultiplexer) ReceivedTransportParameters(tp *TransportParameters) {
	for _, t := range m.tracers {
		t.ReceivedTransportParameters(tp)
	}
}

func (m *connTracerMultiplexer) RestoredTransportParameters(tp *TransportParameters) {
	for _, t := range m.tracers {
		t.RestoredTransportParameters(tp)
	}
}

func (m *connTracerMultiplexer) SentPacket(hdr *ExtendedHeader, size ByteCount, ack *AckFrame, frames []Frame) {
	for _, t := range m.tracers {
		t.SentPacket(hdr, size, ack, frames)
	}
}

func (m *connTracerMultiplexer) ReceivedVersionNegotiationPacket(hdr *Header, versions []VersionNumber) {
	for _, t := range m.tracers {
		t.ReceivedVersionNegotiationPacket(hdr, versions)
	}
}

func (m *connTracerMultiplexer) ReceivedRetry(hdr *Header) {
	for _, t := range m.tracers {
		t.ReceivedRetry(hdr)
	}
}

func (m *connTracerMultiplexer) ReceivedPacket(hdr *ExtendedHeader, size ByteCount, frames []Frame) {
	for _, t := range m.tracers {
		t.ReceivedPacket(hdr, size, frames)
	}
}

func (m *connTracerMultiplexer) BufferedPacket(typ PacketType) {
	for _, t := range m.tracers {
		t.BufferedPacket(typ)
	}
}

func (m *connTracerMultiplexer) DroppedPacket(typ PacketType, size ByteCount, reason PacketDropReason) {
	for _, t := range m.tracers {
		t.DroppedPacket(typ, size, reason)
	}
}

func (m *connTracerMultiplexer) UpdatedCongestionState(state CongestionState) {
	for _, t := range m.tracers {
		t.UpdatedCongestionState(state)
	}
}

func (m *connTracerMultiplexer) UpdatedMetrics(rttStats *RTTStats, cwnd, bytesInFLight ByteCount, packetsInFlight int) {
	for _, t := range m.tracers {
		t.UpdatedMetrics(rttStats, cwnd, bytesInFLight, packetsInFlight)
	}
}

func (m *connTracerMultiplexer) AcknowledgedPacket(encLevel EncryptionLevel, pn PacketNumber) {
	for _, t := range m.tracers {
		t.AcknowledgedPacket(encLevel, pn)
	}
}

func (m *connTracerMultiplexer) LostPacket(encLevel EncryptionLevel, pn PacketNumber, reason PacketLossReason) {
	for _, t := range m.tracers {
		t.LostPacket(encLevel, pn, reason)
	}
}

func (m *connTracerMultiplexer) UpdatedPTOCount(value uint32) {
	for _, t := range m.tracers {
		t.UpdatedPTOCount(value)
	}
}

func (m *connTracerMultiplexer) UpdatedKeyFromTLS(encLevel EncryptionLevel, perspective Perspective) {
	for _, t := range m.tracers {
		t.UpdatedKeyFromTLS(encLevel, perspective)
	}
}

func (m *connTracerMultiplexer) UpdatedKey(generation KeyPhase, remote bool) {
	for _, t := range m.tracers {
		t.UpdatedKey(generation, remote)
	}
}

func (m *connTracerMultiplexer) DroppedEncryptionLevel(encLevel EncryptionLevel) {
	for _, t := range m.tracers {
		t.DroppedEncryptionLevel(encLevel)
	}
}

func (m *connTracerMultiplexer) DroppedKey(generation KeyPhase) {
	for _, t := range m.tracers {
		t.DroppedKey(generation)
	}
}

func (m *connTracerMultiplexer) SetLossTimer(typ TimerType, encLevel EncryptionLevel, exp time.Time) {
	for _, t := range m.tracers {
		t.SetLossTimer(typ, encLevel, exp)
	}
}

func (m *connTracerMultiplexer) LossTimerExpired(typ TimerType, encLevel EncryptionLevel) {
	for _, t := range m.tracers {
		t.LossTimerExpired(typ, encLevel)
	}
}

func (m *connTracerMultiplexer) LossTimerCanceled() {
	for _, t := range m.tracers {
		t.LossTimerCanceled()
	}
}

func (m *connTracerMultiplexer) Debug(name, msg string) {
	for _, t := range m.tracers {
		t.Debug(name, msg)
	}
}

func (m *connTracerMultiplexer) Close() {
	for _, t := range m.tracers {
		t.Close()
	}
}

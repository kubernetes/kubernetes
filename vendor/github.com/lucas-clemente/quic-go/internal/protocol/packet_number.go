package protocol

// A PacketNumber in QUIC
type PacketNumber int64

// InvalidPacketNumber is a packet number that is never sent.
// In QUIC, 0 is a valid packet number.
const InvalidPacketNumber PacketNumber = -1

// PacketNumberLen is the length of the packet number in bytes
type PacketNumberLen uint8

const (
	// PacketNumberLen1 is a packet number length of 1 byte
	PacketNumberLen1 PacketNumberLen = 1
	// PacketNumberLen2 is a packet number length of 2 bytes
	PacketNumberLen2 PacketNumberLen = 2
	// PacketNumberLen3 is a packet number length of 3 bytes
	PacketNumberLen3 PacketNumberLen = 3
	// PacketNumberLen4 is a packet number length of 4 bytes
	PacketNumberLen4 PacketNumberLen = 4
)

// DecodePacketNumber calculates the packet number based on the received packet number, its length and the last seen packet number
func DecodePacketNumber(
	packetNumberLength PacketNumberLen,
	lastPacketNumber PacketNumber,
	wirePacketNumber PacketNumber,
) PacketNumber {
	var epochDelta PacketNumber
	switch packetNumberLength {
	case PacketNumberLen1:
		epochDelta = PacketNumber(1) << 8
	case PacketNumberLen2:
		epochDelta = PacketNumber(1) << 16
	case PacketNumberLen3:
		epochDelta = PacketNumber(1) << 24
	case PacketNumberLen4:
		epochDelta = PacketNumber(1) << 32
	}
	epoch := lastPacketNumber & ^(epochDelta - 1)
	var prevEpochBegin PacketNumber
	if epoch > epochDelta {
		prevEpochBegin = epoch - epochDelta
	}
	nextEpochBegin := epoch + epochDelta
	return closestTo(
		lastPacketNumber+1,
		epoch+wirePacketNumber,
		closestTo(lastPacketNumber+1, prevEpochBegin+wirePacketNumber, nextEpochBegin+wirePacketNumber),
	)
}

func closestTo(target, a, b PacketNumber) PacketNumber {
	if delta(target, a) < delta(target, b) {
		return a
	}
	return b
}

func delta(a, b PacketNumber) PacketNumber {
	if a < b {
		return b - a
	}
	return a - b
}

// GetPacketNumberLengthForHeader gets the length of the packet number for the public header
// it never chooses a PacketNumberLen of 1 byte, since this is too short under certain circumstances
func GetPacketNumberLengthForHeader(packetNumber, leastUnacked PacketNumber) PacketNumberLen {
	diff := uint64(packetNumber - leastUnacked)
	if diff < (1 << (16 - 1)) {
		return PacketNumberLen2
	}
	if diff < (1 << (24 - 1)) {
		return PacketNumberLen3
	}
	return PacketNumberLen4
}

package wire

import (
	"fmt"
	"strings"

	"github.com/lucas-clemente/quic-go/internal/protocol"
	"github.com/lucas-clemente/quic-go/internal/utils"
)

// LogFrame logs a frame, either sent or received
func LogFrame(logger utils.Logger, frame Frame, sent bool) {
	if !logger.Debug() {
		return
	}
	dir := "<-"
	if sent {
		dir = "->"
	}
	switch f := frame.(type) {
	case *CryptoFrame:
		dataLen := protocol.ByteCount(len(f.Data))
		logger.Debugf("\t%s &wire.CryptoFrame{Offset: %d, Data length: %d, Offset + Data length: %d}", dir, f.Offset, dataLen, f.Offset+dataLen)
	case *StreamFrame:
		logger.Debugf("\t%s &wire.StreamFrame{StreamID: %d, Fin: %t, Offset: %d, Data length: %d, Offset + Data length: %d}", dir, f.StreamID, f.Fin, f.Offset, f.DataLen(), f.Offset+f.DataLen())
	case *ResetStreamFrame:
		logger.Debugf("\t%s &wire.ResetStreamFrame{StreamID: %d, ErrorCode: %#x, FinalSize: %d}", dir, f.StreamID, f.ErrorCode, f.FinalSize)
	case *AckFrame:
		hasECN := f.ECT0 > 0 || f.ECT1 > 0 || f.ECNCE > 0
		var ecn string
		if hasECN {
			ecn = fmt.Sprintf(", ECT0: %d, ECT1: %d, CE: %d", f.ECT0, f.ECT1, f.ECNCE)
		}
		if len(f.AckRanges) > 1 {
			ackRanges := make([]string, len(f.AckRanges))
			for i, r := range f.AckRanges {
				ackRanges[i] = fmt.Sprintf("{Largest: %d, Smallest: %d}", r.Largest, r.Smallest)
			}
			logger.Debugf("\t%s &wire.AckFrame{LargestAcked: %d, LowestAcked: %d, AckRanges: {%s}, DelayTime: %s%s}", dir, f.LargestAcked(), f.LowestAcked(), strings.Join(ackRanges, ", "), f.DelayTime.String(), ecn)
		} else {
			logger.Debugf("\t%s &wire.AckFrame{LargestAcked: %d, LowestAcked: %d, DelayTime: %s%s}", dir, f.LargestAcked(), f.LowestAcked(), f.DelayTime.String(), ecn)
		}
	case *MaxDataFrame:
		logger.Debugf("\t%s &wire.MaxDataFrame{MaximumData: %d}", dir, f.MaximumData)
	case *MaxStreamDataFrame:
		logger.Debugf("\t%s &wire.MaxStreamDataFrame{StreamID: %d, MaximumStreamData: %d}", dir, f.StreamID, f.MaximumStreamData)
	case *DataBlockedFrame:
		logger.Debugf("\t%s &wire.DataBlockedFrame{MaximumData: %d}", dir, f.MaximumData)
	case *StreamDataBlockedFrame:
		logger.Debugf("\t%s &wire.StreamDataBlockedFrame{StreamID: %d, MaximumStreamData: %d}", dir, f.StreamID, f.MaximumStreamData)
	case *MaxStreamsFrame:
		switch f.Type {
		case protocol.StreamTypeUni:
			logger.Debugf("\t%s &wire.MaxStreamsFrame{Type: uni, MaxStreamNum: %d}", dir, f.MaxStreamNum)
		case protocol.StreamTypeBidi:
			logger.Debugf("\t%s &wire.MaxStreamsFrame{Type: bidi, MaxStreamNum: %d}", dir, f.MaxStreamNum)
		}
	case *StreamsBlockedFrame:
		switch f.Type {
		case protocol.StreamTypeUni:
			logger.Debugf("\t%s &wire.StreamsBlockedFrame{Type: uni, MaxStreams: %d}", dir, f.StreamLimit)
		case protocol.StreamTypeBidi:
			logger.Debugf("\t%s &wire.StreamsBlockedFrame{Type: bidi, MaxStreams: %d}", dir, f.StreamLimit)
		}
	case *NewConnectionIDFrame:
		logger.Debugf("\t%s &wire.NewConnectionIDFrame{SequenceNumber: %d, ConnectionID: %s, StatelessResetToken: %#x}", dir, f.SequenceNumber, f.ConnectionID, f.StatelessResetToken)
	case *NewTokenFrame:
		logger.Debugf("\t%s &wire.NewTokenFrame{Token: %#x}", dir, f.Token)
	default:
		logger.Debugf("\t%s %#v", dir, frame)
	}
}

package networkdb

import "github.com/gogo/protobuf/proto"

const (
	// Compound message header overhead 1 byte(message type) + 4
	// bytes (num messages)
	compoundHeaderOverhead = 5

	// Overhead for each embedded message in a compound message 4
	// bytes (len of embedded message)
	compoundOverhead = 4
)

func encodeRawMessage(t MessageType, raw []byte) ([]byte, error) {
	gMsg := GossipMessage{
		Type: t,
		Data: raw,
	}

	buf, err := proto.Marshal(&gMsg)
	if err != nil {
		return nil, err
	}

	return buf, nil
}

func encodeMessage(t MessageType, msg interface{}) ([]byte, error) {
	buf, err := proto.Marshal(msg.(proto.Message))
	if err != nil {
		return nil, err
	}

	buf, err = encodeRawMessage(t, buf)
	if err != nil {
		return nil, err
	}

	return buf, nil
}

func decodeMessage(buf []byte) (MessageType, []byte, error) {
	var gMsg GossipMessage

	err := proto.Unmarshal(buf, &gMsg)
	if err != nil {
		return MessageTypeInvalid, nil, err
	}

	return gMsg.Type, gMsg.Data, nil
}

// makeCompoundMessage takes a list of messages and generates
// a single compound message containing all of them
func makeCompoundMessage(msgs [][]byte) []byte {
	cMsg := CompoundMessage{}

	cMsg.Messages = make([]*CompoundMessage_SimpleMessage, 0, len(msgs))
	for _, m := range msgs {
		cMsg.Messages = append(cMsg.Messages, &CompoundMessage_SimpleMessage{
			Payload: m,
		})
	}

	buf, err := proto.Marshal(&cMsg)
	if err != nil {
		return nil
	}

	gMsg := GossipMessage{
		Type: MessageTypeCompound,
		Data: buf,
	}

	buf, err = proto.Marshal(&gMsg)
	if err != nil {
		return nil
	}

	return buf
}

// decodeCompoundMessage splits a compound message and returns
// the slices of individual messages. Returns any potential error.
func decodeCompoundMessage(buf []byte) ([][]byte, error) {
	var cMsg CompoundMessage
	if err := proto.Unmarshal(buf, &cMsg); err != nil {
		return nil, err
	}

	parts := make([][]byte, 0, len(cMsg.Messages))
	for _, m := range cMsg.Messages {
		parts = append(parts, m.Payload)
	}

	return parts, nil
}

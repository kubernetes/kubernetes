package ber

import "io"

func readByte(reader io.Reader) (byte, error) {
	bytes := make([]byte, 1, 1)
	_, err := io.ReadFull(reader, bytes)
	if err != nil {
		if err == io.EOF {
			return 0, io.ErrUnexpectedEOF
		}
		return 0, err
	}
	return bytes[0], nil
}

func isEOCPacket(p *Packet) bool {
	return p != nil &&
		p.Tag == TagEOC &&
		p.ClassType == ClassUniversal &&
		p.TagType == TypePrimitive &&
		len(p.ByteValue) == 0 &&
		len(p.Children) == 0
}

package githttp

import (
	"encoding/hex"
	"errors"
	"fmt"
)

// pktLineParser is a parser for git pkt-line Format,
// as documented in https://github.com/git/git/blob/master/Documentation/technical/protocol-common.txt.
// A zero value of pktLineParser is valid to use as a parser in ready state.
// Output should be read from Lines and Error after Step returns finished true.
// pktLineParser reads until a terminating "0000" flush-pkt. It's good for a single use only.
type pktLineParser struct {
	// Lines contains all pkt-lines.
	Lines []string

	// Error contains the first error encountered while parsing, or nil otherwise.
	Error error

	// Internal state machine.
	state state
	next  int // next is the number of bytes that need to be written to buf before its contents should be processed by the state machine.
	buf   []byte
}

// Feed accumulates and parses data.
// It will return early if it reaches end of pkt-line data (indicated by a flush-pkt "0000"),
// or if it encounters a parsing error.
// It must not be called when state is done.
// When done, all of pkt-lines will be available in Lines, and Error will be set if any error occurred.
func (p *pktLineParser) Feed(data []byte) {
	for {
		// If not enough data to reach next state, append it to buf and return.
		if len(data) < p.next {
			p.buf = append(p.buf, data...)
			p.next -= len(data)
			return
		}

		// There's enough data to reach next state. Take from data only what's needed.
		b := data[:p.next]
		data = data[p.next:]
		p.buf = append(p.buf, b...)
		p.next = 0

		// Take a step to next state.
		err := p.step()
		if err != nil {
			p.state = done
			p.Error = err
			return
		}

		// Break out once reached done state.
		if p.state == done {
			return
		}
	}
}

const (
	// pkt-len = 4*(HEXDIG)
	pktLenSize = 4
)

type state uint8

const (
	ready state = iota
	readingLen
	readingPayload
	done
)

// step moves the state machine to the next state.
// buf must contain all the data ready for consumption for current state.
// It must not be called when state is done.
func (p *pktLineParser) step() error {
	switch p.state {
	case ready:
		p.state = readingLen
		p.next = pktLenSize
		return nil
	case readingLen:
		// len(p.buf) is 4.
		pktLen, err := parsePktLen(p.buf)
		if err != nil {
			return err
		}

		switch {
		case pktLen == 0:
			p.state = done
			p.next = 0
			p.buf = nil
			return nil
		default:
			p.state = readingPayload
			p.next = pktLen - pktLenSize // (pkt-len - 4)*(OCTET)
			p.buf = p.buf[:0]
			return nil
		}
	case readingPayload:
		p.state = readingLen
		p.next = pktLenSize
		p.Lines = append(p.Lines, string(p.buf))
		p.buf = p.buf[:0]
		return nil
	default:
		panic(fmt.Errorf("unreachable: %v", p.state))
	}
}

// parsePktLen parses a pkt-len segment.
// len(b) must be 4.
func parsePktLen(b []byte) (int, error) {
	pktLen, err := parseHex(b)
	switch {
	case err != nil:
		return 0, err
	case 1 <= pktLen && pktLen < pktLenSize:
		return 0, fmt.Errorf("invalid pkt-len: %v", pktLen)
	case pktLen > 65524:
		// The maximum length of a pkt-line is 65524 bytes (65520 bytes of payload + 4 bytes of length data).
		return 0, fmt.Errorf("invalid pkt-len: %v", pktLen)
	}
	return int(pktLen), nil
}

// parseHex parses a 4-byte hex number.
// len(h) must be 4.
func parseHex(h []byte) (uint16, error) {
	var b [2]uint8
	n, err := hex.Decode(b[:], h)
	switch {
	case err != nil:
		return 0, err
	case n != 2:
		return 0, errors.New("short output")
	}
	return uint16(b[0])<<8 | uint16(b[1]), nil
}

package dns

// MsgAcceptFunc is used early in the server code to accept or reject a message with RcodeFormatError.
// It returns a MsgAcceptAction to indicate what should happen with the message.
type MsgAcceptFunc func(dh Header) MsgAcceptAction

// DefaultMsgAcceptFunc checks the request and will reject if:
//
// * isn't a request (don't respond in that case)
//
// * opcode isn't OpcodeQuery or OpcodeNotify
//
// * Zero bit isn't zero
//
// * has more than 1 question in the question section
//
// * has more than 1 RR in the Answer section
//
// * has more than 0 RRs in the Authority section
//
// * has more than 2 RRs in the Additional section
//
var DefaultMsgAcceptFunc MsgAcceptFunc = defaultMsgAcceptFunc

// MsgAcceptAction represents the action to be taken.
type MsgAcceptAction int

const (
	MsgAccept               MsgAcceptAction = iota // Accept the message
	MsgReject                                      // Reject the message with a RcodeFormatError
	MsgIgnore                                      // Ignore the error and send nothing back.
	MsgRejectNotImplemented                        // Reject the message with a RcodeNotImplemented
)

func defaultMsgAcceptFunc(dh Header) MsgAcceptAction {
	if isResponse := dh.Bits&_QR != 0; isResponse {
		return MsgIgnore
	}

	// Don't allow dynamic updates, because then the sections can contain a whole bunch of RRs.
	opcode := int(dh.Bits>>11) & 0xF
	if opcode != OpcodeQuery && opcode != OpcodeNotify {
		return MsgRejectNotImplemented
	}

	if dh.Qdcount != 1 {
		return MsgReject
	}
	// NOTIFY requests can have a SOA in the ANSWER section. See RFC 1996 Section 3.7 and 3.11.
	if dh.Ancount > 1 {
		return MsgReject
	}
	// IXFR request could have one SOA RR in the NS section. See RFC 1995, section 3.
	if dh.Nscount > 1 {
		return MsgReject
	}
	if dh.Arcount > 2 {
		return MsgReject
	}
	return MsgAccept
}

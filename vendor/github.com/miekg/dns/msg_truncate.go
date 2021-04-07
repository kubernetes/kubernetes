package dns

// Truncate ensures the reply message will fit into the requested buffer
// size by removing records that exceed the requested size.
//
// It will first check if the reply fits without compression and then with
// compression. If it won't fit with compression, Truncate then walks the
// record adding as many records as possible without exceeding the
// requested buffer size.
//
// The TC bit will be set if any records were excluded from the message.
// If the TC bit is already set on the message it will be retained.
// TC indicates that the client should retry over TCP.
//
// According to RFC 2181, the TC bit should only be set if not all of the
// "required" RRs can be included in the response. Unfortunately, we have
// no way of knowing which RRs are required so we set the TC bit if any RR
// had to be omitted from the response.
//
// The appropriate buffer size can be retrieved from the requests OPT
// record, if present, and is transport specific otherwise. dns.MinMsgSize
// should be used for UDP requests without an OPT record, and
// dns.MaxMsgSize for TCP requests without an OPT record.
func (dns *Msg) Truncate(size int) {
	if dns.IsTsig() != nil {
		// To simplify this implementation, we don't perform
		// truncation on responses with a TSIG record.
		return
	}

	// RFC 6891 mandates that the payload size in an OPT record
	// less than 512 (MinMsgSize) bytes must be treated as equal to 512 bytes.
	//
	// For ease of use, we impose that restriction here.
	if size < MinMsgSize {
		size = MinMsgSize
	}

	l := msgLenWithCompressionMap(dns, nil) // uncompressed length
	if l <= size {
		// Don't waste effort compressing this message.
		dns.Compress = false
		return
	}

	dns.Compress = true

	edns0 := dns.popEdns0()
	if edns0 != nil {
		// Account for the OPT record that gets added at the end,
		// by subtracting that length from our budget.
		//
		// The EDNS(0) OPT record must have the root domain and
		// it's length is thus unaffected by compression.
		size -= Len(edns0)
	}

	compression := make(map[string]struct{})

	l = headerSize
	for _, r := range dns.Question {
		l += r.len(l, compression)
	}

	var numAnswer int
	if l < size {
		l, numAnswer = truncateLoop(dns.Answer, size, l, compression)
	}

	var numNS int
	if l < size {
		l, numNS = truncateLoop(dns.Ns, size, l, compression)
	}

	var numExtra int
	if l < size {
		_, numExtra = truncateLoop(dns.Extra, size, l, compression)
	}

	// See the function documentation for when we set this.
	dns.Truncated = dns.Truncated || len(dns.Answer) > numAnswer ||
		len(dns.Ns) > numNS || len(dns.Extra) > numExtra

	dns.Answer = dns.Answer[:numAnswer]
	dns.Ns = dns.Ns[:numNS]
	dns.Extra = dns.Extra[:numExtra]

	if edns0 != nil {
		// Add the OPT record back onto the additional section.
		dns.Extra = append(dns.Extra, edns0)
	}
}

func truncateLoop(rrs []RR, size, l int, compression map[string]struct{}) (int, int) {
	for i, r := range rrs {
		if r == nil {
			continue
		}

		l += r.len(l, compression)
		if l > size {
			// Return size, rather than l prior to this record,
			// to prevent any further records being added.
			return size, i
		}
		if l == size {
			return l, i + 1
		}
	}

	return l, len(rrs)
}

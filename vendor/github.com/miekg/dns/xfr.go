package dns

import (
	"fmt"
	"time"
)

// Envelope is used when doing a zone transfer with a remote server.
type Envelope struct {
	RR    []RR  // The set of RRs in the answer section of the xfr reply message.
	Error error // If something went wrong, this contains the error.
}

// A Transfer defines parameters that are used during a zone transfer.
type Transfer struct {
	*Conn
	DialTimeout    time.Duration     // net.DialTimeout, defaults to 2 seconds
	ReadTimeout    time.Duration     // net.Conn.SetReadTimeout value for connections, defaults to 2 seconds
	WriteTimeout   time.Duration     // net.Conn.SetWriteTimeout value for connections, defaults to 2 seconds
	TsigSecret     map[string]string // Secret(s) for Tsig map[<zonename>]<base64 secret>, zonename must be in canonical form (lowercase, fqdn, see RFC 4034 Section 6.2)
	tsigTimersOnly bool
}

// Think we need to away to stop the transfer

// In performs an incoming transfer with the server in a.
// If you would like to set the source IP, or some other attribute
// of a Dialer for a Transfer, you can do so by specifying the attributes
// in the Transfer.Conn:
//
//	d := net.Dialer{LocalAddr: transfer_source}
//	con, err := d.Dial("tcp", master)
//	dnscon := &dns.Conn{Conn:con}
//	transfer = &dns.Transfer{Conn: dnscon}
//	channel, err := transfer.In(message, master)
//
func (t *Transfer) In(q *Msg, a string) (env chan *Envelope, err error) {
	switch q.Question[0].Qtype {
	case TypeAXFR, TypeIXFR:
	default:
		return nil, &Error{"unsupported question type"}
	}

	timeout := dnsTimeout
	if t.DialTimeout != 0 {
		timeout = t.DialTimeout
	}

	if t.Conn == nil {
		t.Conn, err = DialTimeout("tcp", a, timeout)
		if err != nil {
			return nil, err
		}
	}

	if err := t.WriteMsg(q); err != nil {
		return nil, err
	}

	env = make(chan *Envelope)
	switch q.Question[0].Qtype {
	case TypeAXFR:
		go t.inAxfr(q, env)
	case TypeIXFR:
		go t.inIxfr(q, env)
	}

	return env, nil
}

func (t *Transfer) inAxfr(q *Msg, c chan *Envelope) {
	first := true
	defer t.Close()
	defer close(c)
	timeout := dnsTimeout
	if t.ReadTimeout != 0 {
		timeout = t.ReadTimeout
	}
	for {
		t.Conn.SetReadDeadline(time.Now().Add(timeout))
		in, err := t.ReadMsg()
		if err != nil {
			c <- &Envelope{nil, err}
			return
		}
		if q.Id != in.Id {
			c <- &Envelope{in.Answer, ErrId}
			return
		}
		if first {
			if in.Rcode != RcodeSuccess {
				c <- &Envelope{in.Answer, &Error{err: fmt.Sprintf(errXFR, in.Rcode)}}
				return
			}
			if !isSOAFirst(in) {
				c <- &Envelope{in.Answer, ErrSoa}
				return
			}
			first = !first
			// only one answer that is SOA, receive more
			if len(in.Answer) == 1 {
				t.tsigTimersOnly = true
				c <- &Envelope{in.Answer, nil}
				continue
			}
		}

		if !first {
			t.tsigTimersOnly = true // Subsequent envelopes use this.
			if isSOALast(in) {
				c <- &Envelope{in.Answer, nil}
				return
			}
			c <- &Envelope{in.Answer, nil}
		}
	}
}

func (t *Transfer) inIxfr(q *Msg, c chan *Envelope) {
	var serial uint32 // The first serial seen is the current server serial
	axfr := true
	n := 0
	qser := q.Ns[0].(*SOA).Serial
	defer t.Close()
	defer close(c)
	timeout := dnsTimeout
	if t.ReadTimeout != 0 {
		timeout = t.ReadTimeout
	}
	for {
		t.SetReadDeadline(time.Now().Add(timeout))
		in, err := t.ReadMsg()
		if err != nil {
			c <- &Envelope{nil, err}
			return
		}
		if q.Id != in.Id {
			c <- &Envelope{in.Answer, ErrId}
			return
		}
		if in.Rcode != RcodeSuccess {
			c <- &Envelope{in.Answer, &Error{err: fmt.Sprintf(errXFR, in.Rcode)}}
			return
		}
		if n == 0 {
			// Check if the returned answer is ok
			if !isSOAFirst(in) {
				c <- &Envelope{in.Answer, ErrSoa}
				return
			}
			// This serial is important
			serial = in.Answer[0].(*SOA).Serial
			// Check if there are no changes in zone
			if qser >= serial {
				c <- &Envelope{in.Answer, nil}
				return
			}
		}
		// Now we need to check each message for SOA records, to see what we need to do
		t.tsigTimersOnly = true
		for _, rr := range in.Answer {
			if v, ok := rr.(*SOA); ok {
				if v.Serial == serial {
					n++
					// quit if it's a full axfr or the the servers' SOA is repeated the third time
					if axfr && n == 2 || n == 3 {
						c <- &Envelope{in.Answer, nil}
						return
					}
				} else if axfr {
					// it's an ixfr
					axfr = false
				}
			}
		}
		c <- &Envelope{in.Answer, nil}
	}
}

// Out performs an outgoing transfer with the client connecting in w.
// Basic use pattern:
//
//	ch := make(chan *dns.Envelope)
//	tr := new(dns.Transfer)
//	go tr.Out(w, r, ch)
//	ch <- &dns.Envelope{RR: []dns.RR{soa, rr1, rr2, rr3, soa}}
//	close(ch)
//	w.Hijack()
//	// w.Close() // Client closes connection
//
// The server is responsible for sending the correct sequence of RRs through the
// channel ch.
func (t *Transfer) Out(w ResponseWriter, q *Msg, ch chan *Envelope) error {
	for x := range ch {
		r := new(Msg)
		// Compress?
		r.SetReply(q)
		r.Authoritative = true
		// assume it fits TODO(miek): fix
		r.Answer = append(r.Answer, x.RR...)
		if err := w.WriteMsg(r); err != nil {
			return err
		}
	}
	w.TsigTimersOnly(true)
	return nil
}

// ReadMsg reads a message from the transfer connection t.
func (t *Transfer) ReadMsg() (*Msg, error) {
	m := new(Msg)
	p := make([]byte, MaxMsgSize)
	n, err := t.Read(p)
	if err != nil && n == 0 {
		return nil, err
	}
	p = p[:n]
	if err := m.Unpack(p); err != nil {
		return nil, err
	}
	if ts := m.IsTsig(); ts != nil && t.TsigSecret != nil {
		if _, ok := t.TsigSecret[ts.Hdr.Name]; !ok {
			return m, ErrSecret
		}
		// Need to work on the original message p, as that was used to calculate the tsig.
		err = TsigVerify(p, t.TsigSecret[ts.Hdr.Name], t.tsigRequestMAC, t.tsigTimersOnly)
		t.tsigRequestMAC = ts.MAC
	}
	return m, err
}

// WriteMsg writes a message through the transfer connection t.
func (t *Transfer) WriteMsg(m *Msg) (err error) {
	var out []byte
	if ts := m.IsTsig(); ts != nil && t.TsigSecret != nil {
		if _, ok := t.TsigSecret[ts.Hdr.Name]; !ok {
			return ErrSecret
		}
		out, t.tsigRequestMAC, err = TsigGenerate(m, t.TsigSecret[ts.Hdr.Name], t.tsigRequestMAC, t.tsigTimersOnly)
	} else {
		out, err = m.Pack()
	}
	if err != nil {
		return err
	}
	_, err = t.Write(out)
	return err
}

func isSOAFirst(in *Msg) bool {
	return len(in.Answer) > 0 &&
		in.Answer[0].Header().Rrtype == TypeSOA
}

func isSOALast(in *Msg) bool {
	return len(in.Answer) > 0 &&
		in.Answer[len(in.Answer)-1].Header().Rrtype == TypeSOA
}

const errXFR = "bad xfr rcode: %d"

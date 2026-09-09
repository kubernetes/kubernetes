package netlink

import (
	"bytes"
	"errors"
	"io"
)

const (
	tcpBBRInfoLen = 20
	memInfoLen    = 16
)

func checkDeserErr(err error) error {
	if err == io.EOF {
		return nil
	}
	return err
}

func (t *TCPInfo) deserialize(b []byte) error {
	var err error
	rb := bytes.NewBuffer(b)

	t.State, err = rb.ReadByte()
	if err != nil {
		return checkDeserErr(err)
	}

	t.Ca_state, err = rb.ReadByte()
	if err != nil {
		return checkDeserErr(err)
	}

	t.Retransmits, err = rb.ReadByte()
	if err != nil {
		return checkDeserErr(err)
	}

	t.Probes, err = rb.ReadByte()
	if err != nil {
		return checkDeserErr(err)
	}

	t.Backoff, err = rb.ReadByte()
	if err != nil {
		return checkDeserErr(err)
	}
	t.Options, err = rb.ReadByte()
	if err != nil {
		return checkDeserErr(err)
	}

	scales, err := rb.ReadByte()
	if err != nil {
		return checkDeserErr(err)
	}
	t.Snd_wscale = scales >> 4  // first 4 bits
	t.Rcv_wscale = scales & 0xf // last 4 bits

	rateLimAndFastOpen, err := rb.ReadByte()
	if err != nil {
		return checkDeserErr(err)
	}
	t.Delivery_rate_app_limited = rateLimAndFastOpen >> 7 // get first bit
	t.Fastopen_client_fail = rateLimAndFastOpen >> 5 & 3  // get next two bits

	next := rb.Next(4)
	if len(next) == 0 {
		return nil
	}
	t.Rto = native.Uint32(next)

	next = rb.Next(4)
	if len(next) == 0 {
		return nil
	}
	t.Ato = native.Uint32(next)

	next = rb.Next(4)
	if len(next) == 0 {
		return nil
	}
	t.Snd_mss = native.Uint32(next)

	next = rb.Next(4)
	if len(next) == 0 {
		return nil
	}
	t.Rcv_mss = native.Uint32(next)

	next = rb.Next(4)
	if len(next) == 0 {
		return nil
	}
	t.Unacked = native.Uint32(next)

	next = rb.Next(4)
	if len(next) == 0 {
		return nil
	}
	t.Sacked = native.Uint32(next)

	next = rb.Next(4)
	if len(next) == 0 {
		return nil
	}
	t.Lost = native.Uint32(next)

	next = rb.Next(4)
	if len(next) == 0 {
		return nil
	}
	t.Retrans = native.Uint32(next)

	next = rb.Next(4)
	if len(next) == 0 {
		return nil
	}
	t.Fackets = native.Uint32(next)

	next = rb.Next(4)
	if len(next) == 0 {
		return nil
	}
	t.Last_data_sent = native.Uint32(next)

	next = rb.Next(4)
	if len(next) == 0 {
		return nil
	}
	t.Last_ack_sent = native.Uint32(next)

	next = rb.Next(4)
	if len(next) == 0 {
		return nil
	}
	t.Last_data_recv = native.Uint32(next)

	next = rb.Next(4)
	if len(next) == 0 {
		return nil
	}
	t.Last_ack_recv = native.Uint32(next)

	next = rb.Next(4)
	if len(next) == 0 {
		return nil
	}
	t.Pmtu = native.Uint32(next)

	next = rb.Next(4)
	if len(next) == 0 {
		return nil
	}
	t.Rcv_ssthresh = native.Uint32(next)

	next = rb.Next(4)
	if len(next) == 0 {
		return nil
	}
	t.Rtt = native.Uint32(next)

	next = rb.Next(4)
	if len(next) == 0 {
		return nil
	}
	t.Rttvar = native.Uint32(next)

	next = rb.Next(4)
	if len(next) == 0 {
		return nil
	}
	t.Snd_ssthresh = native.Uint32(next)

	next = rb.Next(4)
	if len(next) == 0 {
		return nil
	}
	t.Snd_cwnd = native.Uint32(next)

	next = rb.Next(4)
	if len(next) == 0 {
		return nil
	}
	t.Advmss = native.Uint32(next)

	next = rb.Next(4)
	if len(next) == 0 {
		return nil
	}
	t.Reordering = native.Uint32(next)

	next = rb.Next(4)
	if len(next) == 0 {
		return nil
	}
	t.Rcv_rtt = native.Uint32(next)

	next = rb.Next(4)
	if len(next) == 0 {
		return nil
	}
	t.Rcv_space = native.Uint32(next)

	next = rb.Next(4)
	if len(next) == 0 {
		return nil
	}
	t.Total_retrans = native.Uint32(next)

	next = rb.Next(8)
	if len(next) == 0 {
		return nil
	}
	t.Pacing_rate = native.Uint64(next)

	next = rb.Next(8)
	if len(next) == 0 {
		return nil
	}
	t.Max_pacing_rate = native.Uint64(next)

	next = rb.Next(8)
	if len(next) == 0 {
		return nil
	}
	t.Bytes_acked = native.Uint64(next)

	next = rb.Next(8)
	if len(next) == 0 {
		return nil
	}
	t.Bytes_received = native.Uint64(next)

	next = rb.Next(4)
	if len(next) == 0 {
		return nil
	}
	t.Segs_out = native.Uint32(next)

	next = rb.Next(4)
	if len(next) == 0 {
		return nil
	}
	t.Segs_in = native.Uint32(next)
	next = rb.Next(4)
	if len(next) == 0 {
		return nil
	}
	t.Notsent_bytes = native.Uint32(next)
	next = rb.Next(4)
	if len(next) == 0 {
		return nil
	}
	t.Min_rtt = native.Uint32(next)
	next = rb.Next(4)
	if len(next) == 0 {
		return nil
	}
	t.Data_segs_in = native.Uint32(next)
	next = rb.Next(4)
	if len(next) == 0 {
		return nil
	}
	t.Data_segs_out = native.Uint32(next)

	next = rb.Next(8)
	if len(next) == 0 {
		return nil
	}
	t.Delivery_rate = native.Uint64(next)

	next = rb.Next(8)
	if len(next) == 0 {
		return nil
	}
	t.Busy_time = native.Uint64(next)

	next = rb.Next(8)
	if len(next) == 0 {
		return nil
	}
	t.Rwnd_limited = native.Uint64(next)

	next = rb.Next(8)
	if len(next) == 0 {
		return nil
	}
	t.Sndbuf_limited = native.Uint64(next)

	next = rb.Next(4)
	if len(next) == 0 {
		return nil
	}
	t.Delivered = native.Uint32(next)

	next = rb.Next(4)
	if len(next) == 0 {
		return nil
	}
	t.Delivered_ce = native.Uint32(next)

	next = rb.Next(8)
	if len(next) == 0 {
		return nil
	}
	t.Bytes_sent = native.Uint64(next)

	next = rb.Next(8)
	if len(next) == 0 {
		return nil
	}
	t.Bytes_retrans = native.Uint64(next)

	next = rb.Next(4)
	if len(next) == 0 {
		return nil
	}
	t.Dsack_dups = native.Uint32(next)

	next = rb.Next(4)
	if len(next) == 0 {
		return nil
	}
	t.Reord_seen = native.Uint32(next)

	next = rb.Next(4)
	if len(next) == 0 {
		return nil
	}
	t.Rcv_ooopack = native.Uint32(next)

	next = rb.Next(4)
	if len(next) == 0 {
		return nil
	}
	t.Snd_wnd = native.Uint32(next)
	return nil
}

func (t *TCPBBRInfo) deserialize(b []byte) error {
	if len(b) != tcpBBRInfoLen {
		return errors.New("Invalid length")
	}

	rb := bytes.NewBuffer(b)
	t.BBRBW = native.Uint64(rb.Next(8))
	t.BBRMinRTT = native.Uint32(rb.Next(4))
	t.BBRPacingGain = native.Uint32(rb.Next(4))
	t.BBRCwndGain = native.Uint32(rb.Next(4))

	return nil
}

func (m *MemInfo) deserialize(b []byte) error {
	if len(b) != memInfoLen {
		return errors.New("Invalid length")
	}

	rb := bytes.NewBuffer(b)
	m.RMem = native.Uint32(rb.Next(4))
	m.WMem = native.Uint32(rb.Next(4))
	m.FMem = native.Uint32(rb.Next(4))
	m.TMem = native.Uint32(rb.Next(4))

	return nil
}

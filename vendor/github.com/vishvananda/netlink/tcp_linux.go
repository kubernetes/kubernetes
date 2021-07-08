package netlink

import (
	"bytes"
	"io"
)

type TCPInfo struct {
	State                     uint8
	Ca_state                  uint8
	Retransmits               uint8
	Probes                    uint8
	Backoff                   uint8
	Options                   uint8
	Snd_wscale                uint8 // no uint4
	Rcv_wscale                uint8
	Delivery_rate_app_limited uint8
	Fastopen_client_fail      uint8
	Rto                       uint32
	Ato                       uint32
	Snd_mss                   uint32
	Rcv_mss                   uint32
	Unacked                   uint32
	Sacked                    uint32
	Lost                      uint32
	Retrans                   uint32
	Fackets                   uint32
	Last_data_sent            uint32
	Last_ack_sent             uint32
	Last_data_recv            uint32
	Last_ack_recv             uint32
	Pmtu                      uint32
	Rcv_ssthresh              uint32
	Rtt                       uint32
	Rttvar                    uint32
	Snd_ssthresh              uint32
	Snd_cwnd                  uint32
	Advmss                    uint32
	Reordering                uint32
	Rcv_rtt                   uint32
	Rcv_space                 uint32
	Total_retrans             uint32
	Pacing_rate               uint64
	Max_pacing_rate           uint64
	Bytes_acked               uint64 /* RFC4898 tcpEStatsAppHCThruOctetsAcked */
	Bytes_received            uint64 /* RFC4898 tcpEStatsAppHCThruOctetsReceived */
	Segs_out                  uint32 /* RFC4898 tcpEStatsPerfSegsOut */
	Segs_in                   uint32 /* RFC4898 tcpEStatsPerfSegsIn */
	Notsent_bytes             uint32
	Min_rtt                   uint32
	Data_segs_in              uint32 /* RFC4898 tcpEStatsDataSegsIn */
	Data_segs_out             uint32 /* RFC4898 tcpEStatsDataSegsOut */
	Delivery_rate             uint64
	Busy_time                 uint64 /* Time (usec) busy sending data */
	Rwnd_limited              uint64 /* Time (usec) limited by receive window */
	Sndbuf_limited            uint64 /* Time (usec) limited by send buffer */
	Delivered                 uint32
	Delivered_ce              uint32
	Bytes_sent                uint64 /* RFC4898 tcpEStatsPerfHCDataOctetsOut */
	Bytes_retrans             uint64 /* RFC4898 tcpEStatsPerfOctetsRetrans */
	Dsack_dups                uint32 /* RFC4898 tcpEStatsStackDSACKDups */
	Reord_seen                uint32 /* reordering events seen */
	Rcv_ooopack               uint32 /* Out-of-order packets received */
	Snd_wnd                   uint32 /* peer's advertised receive window after * scaling (bytes) */
}

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

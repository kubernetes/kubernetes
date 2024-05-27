package netlink

// TCP States
const (
	TCP_ESTABLISHED = iota + 0x01
	TCP_SYN_SENT
	TCP_SYN_RECV
	TCP_FIN_WAIT1
	TCP_FIN_WAIT2
	TCP_TIME_WAIT
	TCP_CLOSE
	TCP_CLOSE_WAIT
	TCP_LAST_ACK
	TCP_LISTEN
	TCP_CLOSING
	TCP_NEW_SYN_REC
	TCP_MAX_STATES
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

type TCPBBRInfo struct {
	BBRBW         uint64
	BBRMinRTT     uint32
	BBRPacingGain uint32
	BBRCwndGain   uint32
}

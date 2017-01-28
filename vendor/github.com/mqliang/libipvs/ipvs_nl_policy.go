package libipvs

import (
	"github.com/hkwi/nlgo"
)

var ipvs_stats_policy = nlgo.MapPolicy{
	Prefix: "IPVS_STATS_ATTR",
	Names: map[uint16]string{
		IPVS_STATS_ATTR_CONNS:    "CONNS",
		IPVS_STATS_ATTR_INPKTS:   "INPKTS",
		IPVS_STATS_ATTR_OUTPKTS:  "OUTPKTS",
		IPVS_STATS_ATTR_INBYTES:  "INBYTES",
		IPVS_STATS_ATTR_OUTBYTES: "OUTBYTES",
		IPVS_STATS_ATTR_CPS:      "CPS",
		IPVS_STATS_ATTR_INPPS:    "INPPS",
		IPVS_STATS_ATTR_OUTPPS:   "OUTPPS",
		IPVS_STATS_ATTR_INBPS:    "INBPS",
		IPVS_STATS_ATTR_OUTBPS:   "OUTBPS",
	},
	Rule: map[uint16]nlgo.Policy{
		IPVS_STATS_ATTR_CONNS:    nlgo.U32Policy,
		IPVS_STATS_ATTR_INPKTS:   nlgo.U32Policy,
		IPVS_STATS_ATTR_OUTPKTS:  nlgo.U32Policy,
		IPVS_STATS_ATTR_INBYTES:  nlgo.U64Policy,
		IPVS_STATS_ATTR_OUTBYTES: nlgo.U64Policy,
		IPVS_STATS_ATTR_CPS:      nlgo.U32Policy,
		IPVS_STATS_ATTR_INPPS:    nlgo.U32Policy,
		IPVS_STATS_ATTR_OUTPPS:   nlgo.U32Policy,
		IPVS_STATS_ATTR_INBPS:    nlgo.U32Policy,
		IPVS_STATS_ATTR_OUTBPS:   nlgo.U32Policy,
	},
}

var ipvs_service_policy = nlgo.MapPolicy{
	Prefix: "IPVS_SVC_ATTR",
	Names: map[uint16]string{
		IPVS_SVC_ATTR_AF:         "AF",
		IPVS_SVC_ATTR_PROTOCOL:   "PROTOCOL",
		IPVS_SVC_ATTR_ADDR:       "ADDR",
		IPVS_SVC_ATTR_PORT:       "PORT",
		IPVS_SVC_ATTR_FWMARK:     "FWMARK",
		IPVS_SVC_ATTR_SCHED_NAME: "SCHED_NAME",
		IPVS_SVC_ATTR_FLAGS:      "FLAGS",
		IPVS_SVC_ATTR_TIMEOUT:    "TIMEOUT",
		IPVS_SVC_ATTR_NETMASK:    "NETMASK",
		IPVS_SVC_ATTR_STATS:      "STATS",
		IPVS_SVC_ATTR_PE_NAME:    "PE_NAME",
	},
	Rule: map[uint16]nlgo.Policy{
		IPVS_SVC_ATTR_AF:         nlgo.U16Policy,
		IPVS_SVC_ATTR_PROTOCOL:   nlgo.U16Policy,
		IPVS_SVC_ATTR_ADDR:       nlgo.BinaryPolicy, // struct in6_addr
		IPVS_SVC_ATTR_PORT:       nlgo.U16Policy,
		IPVS_SVC_ATTR_FWMARK:     nlgo.U32Policy,
		IPVS_SVC_ATTR_SCHED_NAME: nlgo.NulStringPolicy, // IP_VS_SCHEDNAME_MAXLEN
		IPVS_SVC_ATTR_FLAGS:      nlgo.BinaryPolicy,    // struct ip_vs_flags
		IPVS_SVC_ATTR_TIMEOUT:    nlgo.U32Policy,
		IPVS_SVC_ATTR_NETMASK:    nlgo.U32Policy,
		IPVS_SVC_ATTR_STATS:      ipvs_stats_policy,
	},
}

var ipvs_dest_policy = nlgo.MapPolicy{
	Prefix: "IPVS_DEST_ATTR",
	Names: map[uint16]string{
		IPVS_DEST_ATTR_ADDR:          "ADDR",
		IPVS_DEST_ATTR_PORT:          "PORT",
		IPVS_DEST_ATTR_FWD_METHOD:    "FWD_METHOD",
		IPVS_DEST_ATTR_WEIGHT:        "WEIGHT",
		IPVS_DEST_ATTR_U_THRESH:      "U_THRESH",
		IPVS_DEST_ATTR_L_THRESH:      "L_THRESH",
		IPVS_DEST_ATTR_ACTIVE_CONNS:  "ACTIVE_CONNS",
		IPVS_DEST_ATTR_INACT_CONNS:   "INACT_CONNS",
		IPVS_DEST_ATTR_PERSIST_CONNS: "PERSIST_CONNS",
		IPVS_DEST_ATTR_STATS:         "STATS",
	},
	Rule: map[uint16]nlgo.Policy{
		IPVS_DEST_ATTR_ADDR:          nlgo.BinaryPolicy, // struct in6_addr
		IPVS_DEST_ATTR_PORT:          nlgo.U16Policy,
		IPVS_DEST_ATTR_FWD_METHOD:    nlgo.U32Policy,
		IPVS_DEST_ATTR_WEIGHT:        nlgo.U32Policy,
		IPVS_DEST_ATTR_U_THRESH:      nlgo.U32Policy,
		IPVS_DEST_ATTR_L_THRESH:      nlgo.U32Policy,
		IPVS_DEST_ATTR_ACTIVE_CONNS:  nlgo.U32Policy,
		IPVS_DEST_ATTR_INACT_CONNS:   nlgo.U32Policy,
		IPVS_DEST_ATTR_PERSIST_CONNS: nlgo.U32Policy,
		IPVS_DEST_ATTR_STATS:         ipvs_stats_policy,
	},
}

var ipvs_daemon_policy = nlgo.MapPolicy{
	Prefix: "IPVS_DAEMON_ATTR",
	Names: map[uint16]string{
		IPVS_DAEMON_ATTR_STATE:     "STATE",
		IPVS_DAEMON_ATTR_MCAST_IFN: "MCAST_IFN",
		IPVS_DAEMON_ATTR_SYNC_ID:   "SYNC_ID",
	},
	Rule: map[uint16]nlgo.Policy{
		IPVS_DAEMON_ATTR_STATE:     nlgo.U32Policy,
		IPVS_DAEMON_ATTR_MCAST_IFN: nlgo.StringPolicy, // maxlen = IP_VS_IFNAME_MAXLEN
		IPVS_DAEMON_ATTR_SYNC_ID:   nlgo.U32Policy,
	},
}

var ipvs_cmd_policy = nlgo.MapPolicy{
	Prefix: "IPVS_CMD_ATTR",
	Names: map[uint16]string{
		IPVS_CMD_ATTR_SERVICE:         "SERVICE",
		IPVS_CMD_ATTR_DEST:            "DEST",
		IPVS_CMD_ATTR_DAEMON:          "DAEMON",
		IPVS_CMD_ATTR_TIMEOUT_TCP:     "TIMEOUT_TCP",
		IPVS_CMD_ATTR_TIMEOUT_TCP_FIN: "TIMEOUT_TCP_FIN",
		IPVS_CMD_ATTR_TIMEOUT_UDP:     "TIMEOUT_UDP",
	},
	Rule: map[uint16]nlgo.Policy{
		IPVS_CMD_ATTR_SERVICE:         ipvs_service_policy,
		IPVS_CMD_ATTR_DEST:            ipvs_dest_policy,
		IPVS_CMD_ATTR_DAEMON:          ipvs_daemon_policy,
		IPVS_CMD_ATTR_TIMEOUT_TCP:     nlgo.U32Policy,
		IPVS_CMD_ATTR_TIMEOUT_TCP_FIN: nlgo.U32Policy,
		IPVS_CMD_ATTR_TIMEOUT_UDP:     nlgo.U32Policy,
	},
}

var ipvs_info_policy = nlgo.MapPolicy{
	Prefix: "IPVS_INFO_ATTR",
	Names: map[uint16]string{
		IPVS_INFO_ATTR_VERSION:       "VERSION",
		IPVS_INFO_ATTR_CONN_TAB_SIZE: "CONN_TAB_SIZE",
	},
	Rule: map[uint16]nlgo.Policy{
		IPVS_INFO_ATTR_VERSION:       nlgo.U32Policy,
		IPVS_INFO_ATTR_CONN_TAB_SIZE: nlgo.U32Policy,
	},
}

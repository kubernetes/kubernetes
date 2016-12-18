// +build linux

package libipvs

const (
	GENL_CTRL_ID = 0x10
)

/* Generic Netlink family info */
const (
	IPVS_GENL_NAME    = "IPVS"
	IPVS_GENL_VERSION = 0x1
)

// GENL control commands
const (
	GENL_CTRL_CMD_UNSPEC uint8 = iota
	GENL_CTRL_CMD_NEW_FAMILY
	GENL_CTRL_CMD_DEL_FAMILY
	GENL_CTRL_CMD_GET_FAMILY
)

// GENL family attributes
const (
	GEN_CTRL_ATTR_UNSPEC int = iota
	GENL_CTRL_ATTR_FAMILY_ID
	GENL_CTRL_ATTR_FAMILY_NAME
)

// Generic Netlink command attributes
const (
	IPVS_CMD_UNSPEC uint8 = iota

	IPVS_CMD_NEW_SERVICE /* add service */
	IPVS_CMD_SET_SERVICE /* modify service */
	IPVS_CMD_DEL_SERVICE /* delete service */
	IPVS_CMD_GET_SERVICE /* get info about specific service */

	IPVS_CMD_NEW_DEST /* add destination */
	IPVS_CMD_SET_DEST /* modify destination */
	IPVS_CMD_DEL_DEST /* delete destination */
	IPVS_CMD_GET_DEST /* get list of all service dests */

	IPVS_CMD_NEW_DAEMON /* start sync daemon */
	IPVS_CMD_DEL_DAEMON /* stop sync daemon */
	IPVS_CMD_GET_DAEMON /* get sync daemon status */

	IPVS_CMD_SET_TIMEOUT /* set TCP and UDP timeouts */
	IPVS_CMD_GET_TIMEOUT /* get TCP and UDP timeouts */

	IPVS_CMD_SET_INFO /* only used in GET_INFO reply */
	IPVS_CMD_GET_INFO /* get general IPVS info */

	IPVS_CMD_ZERO  /* zero all counters and stats */
	IPVS_CMD_FLUSH /* flush services and dests */
)

// Attributes used in the first level of commands
const (
	IPVS_CMD_ATTR_UNSPEC          = iota
	IPVS_CMD_ATTR_SERVICE         /* nested service attribute */
	IPVS_CMD_ATTR_DEST            /* nested destination attribute */
	IPVS_CMD_ATTR_DAEMON          /* nested sync daemon attribute */
	IPVS_CMD_ATTR_TIMEOUT_TCP     /* TCP connection timeout */
	IPVS_CMD_ATTR_TIMEOUT_TCP_FIN /* TCP FIN wait timeout */
	IPVS_CMD_ATTR_TIMEOUT_UDP     /* UDP timeout */
)

// Attributes used to describe a service
// Used inside nested attribute IPVS_CMD_ATTR_SERVICE
const (
	IPVS_SVC_ATTR_UNSPEC   uint16 = iota
	IPVS_SVC_ATTR_AF           /* address family */
	IPVS_SVC_ATTR_PROTOCOL     /* virtual service protocol */
	IPVS_SVC_ATTR_ADDR         /* virtual service address */
	IPVS_SVC_ATTR_PORT         /* virtual service port */
	IPVS_SVC_ATTR_FWMARK       /* firewall mark of service */

	IPVS_SVC_ATTR_SCHED_NAME /* name of scheduler */
	IPVS_SVC_ATTR_FLAGS      /* virtual service flags */
	IPVS_SVC_ATTR_TIMEOUT    /* persistent timeout */
	IPVS_SVC_ATTR_NETMASK    /* persistent netmask */

	IPVS_SVC_ATTR_STATS /* nested attribute for service stats */

	IPVS_SVC_ATTR_PE_NAME /* name of scheduler */
)

// Attributes used to describe a destination (real server)
// Used inside nested attribute IPVS_CMD_ATTR_DEST
const (
	IPVS_DEST_ATTR_UNSPEC uint16 = iota
	IPVS_DEST_ATTR_ADDR       /* real server address */
	IPVS_DEST_ATTR_PORT       /* real server port */

	IPVS_DEST_ATTR_FWD_METHOD /* forwarding method */
	IPVS_DEST_ATTR_WEIGHT     /* destination weight */

	IPVS_DEST_ATTR_U_THRESH /* upper threshold */
	IPVS_DEST_ATTR_L_THRESH /* lower threshold */

	IPVS_DEST_ATTR_ACTIVE_CONNS  /* active connections */
	IPVS_DEST_ATTR_INACT_CONNS   /* inactive connections */
	IPVS_DEST_ATTR_PERSIST_CONNS /* persistent connections */

	IPVS_DEST_ATTR_STATS /* nested attribute for dest stats */

	IPVS_DEST_ATTR_ADDR_FAMILY /* Address family of address */
)

// Attributes describing a sync daemon
// Used inside nested attribute IPVS_CMD_ATTR_DAEMON
const (
	IPVS_DAEMON_ATTR_UNSPEC    uint16 = iota
	IPVS_DAEMON_ATTR_STATE         /* sync daemon state (master/backup) */
	IPVS_DAEMON_ATTR_MCAST_IFN     /* multicast interface name */
	IPVS_DAEMON_ATTR_SYNC_ID       /* SyncID we belong to */
)

// Attributes used to describe service or destination entry statistics
// Used inside nested attributes IPVS_SVC_ATTR_STATS and IPVS_DEST_ATTR_STATS
const (
	IPVS_STATS_ATTR_UNSPEC   uint16 = iota
	IPVS_STATS_ATTR_CONNS        /* connections scheduled */
	IPVS_STATS_ATTR_INPKTS       /* incoming packets */
	IPVS_STATS_ATTR_OUTPKTS      /* outgoing packets */
	IPVS_STATS_ATTR_INBYTES      /* incoming bytes */
	IPVS_STATS_ATTR_OUTBYTES     /* outgoing bytes */

	IPVS_STATS_ATTR_CPS    /* current connection rate */
	IPVS_STATS_ATTR_INPPS  /* current in packet rate */
	IPVS_STATS_ATTR_OUTPPS /* current out packet rate */
	IPVS_STATS_ATTR_INBPS  /* current in byte rate */
	IPVS_STATS_ATTR_OUTBPS /* current out byte rate */
)

/* Attributes used in response to IPVS_CMD_GET_INFO command */
const (
	IPVS_INFO_ATTR_UNSPEC        uint16 = iota
	IPVS_INFO_ATTR_VERSION           /* IPVS version number */
	IPVS_INFO_ATTR_CONN_TAB_SIZE     /* size of connection hash table */
)

//  IPVS sync daemon states
const (
	IP_VS_STATE_NONE   = 0x0000 /* daemon is stopped */
	IP_VS_STATE_MASTER = 0x0001 /* started as master */
	IP_VS_STATE_BACKUP = 0x0002 /* started as backup */
)

// IPVS socket options
const (
	IP_VS_BASE_CTL = (64 + 1024 + 64) /* base */

	IP_VS_SO_SET_NONE        = IP_VS_BASE_CTL /* just peek */
	IP_VS_SO_SET_INSERT      = (IP_VS_BASE_CTL + 1)
	IP_VS_SO_SET_ADD         = (IP_VS_BASE_CTL + 2)
	IP_VS_SO_SET_EDIT        = (IP_VS_BASE_CTL + 3)
	IP_VS_SO_SET_DEL         = (IP_VS_BASE_CTL + 4)
	IP_VS_SO_SET_FLUSH       = (IP_VS_BASE_CTL + 5)
	IP_VS_SO_SET_LIST        = (IP_VS_BASE_CTL + 6)
	IP_VS_SO_SET_ADDDEST     = (IP_VS_BASE_CTL + 7)
	IP_VS_SO_SET_DELDEST     = (IP_VS_BASE_CTL + 8)
	IP_VS_SO_SET_EDITDEST    = (IP_VS_BASE_CTL + 9)
	IP_VS_SO_SET_TIMEOUT     = (IP_VS_BASE_CTL + 10)
	IP_VS_SO_SET_STARTDAEMON = (IP_VS_BASE_CTL + 11)
	IP_VS_SO_SET_STOPDAEMON  = (IP_VS_BASE_CTL + 12)
	IP_VS_SO_SET_RESTORE     = (IP_VS_BASE_CTL + 13)
	IP_VS_SO_SET_SAVE        = (IP_VS_BASE_CTL + 14)
	IP_VS_SO_SET_ZERO        = (IP_VS_BASE_CTL + 15)
	IP_VS_SO_SET_MAX         = IP_VS_SO_SET_ZERO

	IP_VS_SO_GET_VERSION  = IP_VS_BASE_CTL
	IP_VS_SO_GET_INFO     = (IP_VS_BASE_CTL + 1)
	IP_VS_SO_GET_SERVICES = (IP_VS_BASE_CTL + 2)
	IP_VS_SO_GET_SERVICE  = (IP_VS_BASE_CTL + 3)
	IP_VS_SO_GET_DESTS    = (IP_VS_BASE_CTL + 4)
	IP_VS_SO_GET_DEST     = (IP_VS_BASE_CTL + 5) /* not used now */
	IP_VS_SO_GET_TIMEOUT  = (IP_VS_BASE_CTL + 6)
	IP_VS_SO_GET_DAEMON   = (IP_VS_BASE_CTL + 7)
	IP_VS_SO_GET_MAX      = IP_VS_SO_GET_DAEMON
)

// Virtual Service Flags
const (
	IP_VS_SVC_F_PERSISTENT = 0x0001 /* persistent port */
	IP_VS_SVC_F_HASHED     = 0x0002 /* hashed entry */
	IP_VS_SVC_F_ONEPACKET  = 0x0004 /* one-packet scheduling */
	IP_VS_SVC_F_SCHED1     = 0x0008 /* scheduler flag 1 */
	IP_VS_SVC_F_SCHED2     = 0x0010 /* scheduler flag 2 */
	IP_VS_SVC_F_SCHED3     = 0x0020 /* scheduler flag 3 */

	IP_VS_SVC_F_SCHED_SH_FALLBACK = IP_VS_SVC_F_SCHED1 /* SH fallback */
	IP_VS_SVC_F_SCHED_SH_PORT     = IP_VS_SVC_F_SCHED2 /* SH use port */
)

//  IPVS Connection Flags
const (
	IP_VS_CONN_F_FWD_MASK   = 0x0007 /* mask for the fwd methods */
	IP_VS_CONN_F_MASQ       = 0x0000 /* masquerading/NAT */
	IP_VS_CONN_F_LOCALNODE  = 0x0001 /* local node */
	IP_VS_CONN_F_TUNNEL     = 0x0002 /* tunneling */
	IP_VS_CONN_F_DROUTE     = 0x0003 /* direct routing */
	IP_VS_CONN_F_BYPASS     = 0x0004 /* cache bypass */
	IP_VS_CONN_F_SYNC       = 0x0020 /* entry created by sync */
	IP_VS_CONN_F_HASHED     = 0x0040 /* hashed entry */
	IP_VS_CONN_F_NOOUTPUT   = 0x0080 /* no output packets */
	IP_VS_CONN_F_INACTIVE   = 0x0100 /* not established */
	IP_VS_CONN_F_OUT_SEQ    = 0x0200 /* must do output seq adjust */
	IP_VS_CONN_F_IN_SEQ     = 0x0400 /* must do input seq adjust */
	IP_VS_CONN_F_SEQ_MASK   = 0x0600 /* in/out sequence mask */
	IP_VS_CONN_F_NO_CPORT   = 0x0800 /* no client port set yet */
	IP_VS_CONN_F_TEMPLATE   = 0x1000 /* template, not connection */
	IP_VS_CONN_F_ONE_PACKET = 0x2000 /* forward only one packet */
)

const (
	// RoundRobin distributes jobs equally amongst the available
	// real servers.
	RoundRobin = "rr"

	// LeastConnection assigns more jobs to real servers with
	// fewer active jobs.
	LeastConnection = "lc"

	// DestinationHashing assigns jobs to servers through looking
	// up a statically assigned hash table by their destination IP
	// addresses.
	DestinationHashing = "dh"

	// SourceHashing assigns jobs to servers through looking up
	// a statically assigned hash table by their source IP
	// addresses.
	SourceHashing = "sh"
)

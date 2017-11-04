package nl

// syscall package lack of rule atributes type.
// Thus there are defined below
const (
	FRA_UNSPEC  = iota
	FRA_DST     /* destination address */
	FRA_SRC     /* source address */
	FRA_IIFNAME /* interface name */
	FRA_GOTO    /* target to jump to (FR_ACT_GOTO) */
	FRA_UNUSED2
	FRA_PRIORITY /* priority/preference */
	FRA_UNUSED3
	FRA_UNUSED4
	FRA_UNUSED5
	FRA_FWMARK /* mark */
	FRA_FLOW   /* flow/class id */
	FRA_TUN_ID
	FRA_SUPPRESS_IFGROUP
	FRA_SUPPRESS_PREFIXLEN
	FRA_TABLE  /* Extended table id */
	FRA_FWMASK /* mask for netfilter mark */
	FRA_OIFNAME
)

// ip rule netlink request types
const (
	FR_ACT_UNSPEC = iota
	FR_ACT_TO_TBL /* Pass to fixed table */
	FR_ACT_GOTO   /* Jump to another rule */
	FR_ACT_NOP    /* No operation */
	FR_ACT_RES3
	FR_ACT_RES4
	FR_ACT_BLACKHOLE   /* Drop without notification */
	FR_ACT_UNREACHABLE /* Drop with ENETUNREACH */
	FR_ACT_PROHIBIT    /* Drop with EACCES */
)

// socket diags related
const (
	SOCK_DIAG_BY_FAMILY = 20         /* linux.sock_diag.h */
	TCPDIAG_NOCOOKIE    = 0xFFFFFFFF /* TCPDIAG_NOCOOKIE in net/ipv4/tcp_diag.h*/
)

const (
	AF_MPLS = 28
)

const (
	RTA_NEWDST     = 0x13
	RTA_ENCAP_TYPE = 0x15
	RTA_ENCAP      = 0x16
)

// RTA_ENCAP subtype
const (
	MPLS_IPTUNNEL_UNSPEC = iota
	MPLS_IPTUNNEL_DST
)

// light weight tunnel encap types
const (
	LWTUNNEL_ENCAP_NONE = iota
	LWTUNNEL_ENCAP_MPLS
	LWTUNNEL_ENCAP_IP
	LWTUNNEL_ENCAP_ILA
	LWTUNNEL_ENCAP_IP6
)

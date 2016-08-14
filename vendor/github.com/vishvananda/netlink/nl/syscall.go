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

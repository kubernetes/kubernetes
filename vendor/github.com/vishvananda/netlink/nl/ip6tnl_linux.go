package nl

// id's of route attribute from https://elixir.bootlin.com/linux/v5.17.3/source/include/uapi/linux/lwtunnel.h#L38
// the value's size are specified in https://elixir.bootlin.com/linux/v5.17.3/source/net/ipv4/ip_tunnel_core.c#L928

const (
	LWTUNNEL_IP6_UNSPEC = iota
	LWTUNNEL_IP6_ID
	LWTUNNEL_IP6_DST
	LWTUNNEL_IP6_SRC
	LWTUNNEL_IP6_HOPLIMIT
	LWTUNNEL_IP6_TC
	LWTUNNEL_IP6_FLAGS
	LWTUNNEL_IP6_PAD // not implemented
	LWTUNNEL_IP6_OPTS // not implemented
	__LWTUNNEL_IP6_MAX
)





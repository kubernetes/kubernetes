// +build ignore

package nlgo

/*
#include <linux/netlink.h>
#include <linux/rtnetlink.h>
#include <linux/genetlink.h>
*/
import "C"

const (
	NLA_HDRLEN  = C.NLA_HDRLEN
	GENL_HDRLEN = C.GENL_HDRLEN
)

type GenlMsghdr C.struct_genlmsghdr

type Ndmsg C.struct_ndmsg

type Tcmsg C.struct_tcmsg

const (
	SizeofGenlMsghdr = C.sizeof_struct_genlmsghdr
	SizeofNdmsg      = C.sizeof_struct_ndmsg
	SizeofTcmsg      = C.sizeof_struct_tcmsg
)

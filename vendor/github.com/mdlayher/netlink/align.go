package netlink

import "unsafe"

// Functions and values used to properly align netlink messages, headers,
// and attributes.  Definitions taken from Linux kernel source.

// #define NLMSG_ALIGNTO   4U
const nlmsgAlignTo = 4

// #define NLMSG_ALIGN(len) ( ((len)+NLMSG_ALIGNTO-1) & ~(NLMSG_ALIGNTO-1) )
func nlmsgAlign(len int) int {
	return ((len) + nlmsgAlignTo - 1) & ^(nlmsgAlignTo - 1)
}

// #define NLMSG_LENGTH(len) ((len) + NLMSG_HDRLEN)
func nlmsgLength(len int) int {
	return len + nlmsgHeaderLen
}

// #define NLMSG_HDRLEN     ((int) NLMSG_ALIGN(sizeof(struct nlmsghdr)))
var nlmsgHeaderLen = nlmsgAlign(int(unsafe.Sizeof(Header{})))

// #define NLA_ALIGNTO             4
const nlaAlignTo = 4

// #define NLA_ALIGN(len)          (((len) + NLA_ALIGNTO - 1) & ~(NLA_ALIGNTO - 1))
func nlaAlign(len int) int {
	return ((len) + nlaAlignTo - 1) & ^(nlaAlignTo - 1)
}

// Because this package's Attribute type contains a byte slice, unsafe.Sizeof
// can't be used to determine the correct length.
const sizeofAttribute = 4

// #define NLA_HDRLEN              ((int) NLA_ALIGN(sizeof(struct nlattr)))
var nlaHeaderLen = nlaAlign(sizeofAttribute)

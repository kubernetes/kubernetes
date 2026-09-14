package netlink

import "unsafe"

// Functions and values used to properly align netlink messages, headers,
// and attributes.  Definitions taken from Linux kernel source.

// #define NLMSG_ALIGNTO   4U
const nlmsgAlignTo = 4

// #define NLMSG_ALIGN(length) ( ((length)+NLMSG_ALIGNTO-1) & ~(NLMSG_ALIGNTO-1) )
func nlmsgAlign(length int) int {
	return ((length) + nlmsgAlignTo - 1) & ^(nlmsgAlignTo - 1)
}

// #define NLMSG_LENGTH(length) ((length) + NLMSG_HDRLEN)
func nlmsgLength(length int) int {
	return length + nlmsgHeaderLen
}

// #define NLMSG_HDRLEN     ((int) NLMSG_ALIGN(sizeof(struct nlmsghdr)))
var nlmsgHeaderLen = nlmsgAlign(int(unsafe.Sizeof(Header{})))

// #define NLA_ALIGNTO             4
const nlaAlignTo = 4

// #define NLA_ALIGN(length)          (((length) + NLA_ALIGNTO - 1) & ~(NLA_ALIGNTO - 1))
func nlaAlign(length int) int {
	return ((length) + nlaAlignTo - 1) & ^(nlaAlignTo - 1)
}

// Because this package's Attribute type contains a byte slice, unsafe.Sizeof
// can't be used to determine the correct length.
const sizeofAttribute = 4

// #define NLA_HDRLEN              ((int) NLA_ALIGN(sizeof(struct nlattr)))
var nlaHeaderLen = nlaAlign(sizeofAttribute)

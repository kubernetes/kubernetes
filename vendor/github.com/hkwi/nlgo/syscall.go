// +build linux

//go:generate sh syscall.sh $GOFILE $GOOS $GOARCH

package nlgo

const SOL_NETLINK = 0x10e // 270

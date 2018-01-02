// +build linux,cgo

package loopback

/*
#include <linux/loop.h> // FIXME: present only for defines, maybe we can remove it?

#ifndef LOOP_CTL_GET_FREE
  #define LOOP_CTL_GET_FREE 0x4C82
#endif

#ifndef LO_FLAGS_PARTSCAN
  #define LO_FLAGS_PARTSCAN 8
#endif

*/
import "C"

type loopInfo64 struct {
	loDevice         uint64 /* ioctl r/o */
	loInode          uint64 /* ioctl r/o */
	loRdevice        uint64 /* ioctl r/o */
	loOffset         uint64
	loSizelimit      uint64 /* bytes, 0 == max available */
	loNumber         uint32 /* ioctl r/o */
	loEncryptType    uint32
	loEncryptKeySize uint32 /* ioctl w/o */
	loFlags          uint32 /* ioctl r/o */
	loFileName       [LoNameSize]uint8
	loCryptName      [LoNameSize]uint8
	loEncryptKey     [LoKeySize]uint8 /* ioctl w/o */
	loInit           [2]uint64
}

// IOCTL consts
const (
	LoopSetFd       = C.LOOP_SET_FD
	LoopCtlGetFree  = C.LOOP_CTL_GET_FREE
	LoopGetStatus64 = C.LOOP_GET_STATUS64
	LoopSetStatus64 = C.LOOP_SET_STATUS64
	LoopClrFd       = C.LOOP_CLR_FD
	LoopSetCapacity = C.LOOP_SET_CAPACITY
)

// LOOP consts.
const (
	LoFlagsAutoClear = C.LO_FLAGS_AUTOCLEAR
	LoFlagsReadOnly  = C.LO_FLAGS_READ_ONLY
	LoFlagsPartScan  = C.LO_FLAGS_PARTSCAN
	LoKeySize        = C.LO_KEY_SIZE
	LoNameSize       = C.LO_NAME_SIZE
)

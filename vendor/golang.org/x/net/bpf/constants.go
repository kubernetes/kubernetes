// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bpf

// A Register is a register of the BPF virtual machine.
type Register uint16

const (
	// RegA is the accumulator register. RegA is always the
	// destination register of ALU operations.
	RegA Register = iota
	// RegX is the indirection register, used by LoadIndirect
	// operations.
	RegX
)

// An ALUOp is an arithmetic or logic operation.
type ALUOp uint16

// ALU binary operation types.
const (
	ALUOpAdd ALUOp = iota << 4
	ALUOpSub
	ALUOpMul
	ALUOpDiv
	ALUOpOr
	ALUOpAnd
	ALUOpShiftLeft
	ALUOpShiftRight
	aluOpNeg // Not exported because it's the only unary ALU operation, and gets its own instruction type.
	ALUOpMod
	ALUOpXor
)

// A JumpTest is a comparison operator used in conditional jumps.
type JumpTest uint16

// Supported operators for conditional jumps.
const (
	// K == A
	JumpEqual JumpTest = iota
	// K != A
	JumpNotEqual
	// K > A
	JumpGreaterThan
	// K < A
	JumpLessThan
	// K >= A
	JumpGreaterOrEqual
	// K <= A
	JumpLessOrEqual
	// K & A != 0
	JumpBitsSet
	// K & A == 0
	JumpBitsNotSet
)

// An Extension is a function call provided by the kernel that
// performs advanced operations that are expensive or impossible
// within the BPF virtual machine.
//
// Extensions are only implemented by the Linux kernel.
//
// TODO: should we prune this list? Some of these extensions seem
// either broken or near-impossible to use correctly, whereas other
// (len, random, ifindex) are quite useful.
type Extension int

// Extension functions available in the Linux kernel.
const (
	// extOffset is the negative maximum number of instructions used
	// to load instructions by overloading the K argument.
	extOffset = -0x1000
	// ExtLen returns the length of the packet.
	ExtLen Extension = 1
	// ExtProto returns the packet's L3 protocol type.
	ExtProto Extension = 0
	// ExtType returns the packet's type (skb->pkt_type in the kernel)
	//
	// TODO: better documentation. How nice an API do we want to
	// provide for these esoteric extensions?
	ExtType Extension = 4
	// ExtPayloadOffset returns the offset of the packet payload, or
	// the first protocol header that the kernel does not know how to
	// parse.
	ExtPayloadOffset Extension = 52
	// ExtInterfaceIndex returns the index of the interface on which
	// the packet was received.
	ExtInterfaceIndex Extension = 8
	// ExtNetlinkAttr returns the netlink attribute of type X at
	// offset A.
	ExtNetlinkAttr Extension = 12
	// ExtNetlinkAttrNested returns the nested netlink attribute of
	// type X at offset A.
	ExtNetlinkAttrNested Extension = 16
	// ExtMark returns the packet's mark value.
	ExtMark Extension = 20
	// ExtQueue returns the packet's assigned hardware queue.
	ExtQueue Extension = 24
	// ExtLinkLayerType returns the packet's hardware address type
	// (e.g. Ethernet, Infiniband).
	ExtLinkLayerType Extension = 28
	// ExtRXHash returns the packets receive hash.
	//
	// TODO: figure out what this rxhash actually is.
	ExtRXHash Extension = 32
	// ExtCPUID returns the ID of the CPU processing the current
	// packet.
	ExtCPUID Extension = 36
	// ExtVLANTag returns the packet's VLAN tag.
	ExtVLANTag Extension = 44
	// ExtVLANTagPresent returns non-zero if the packet has a VLAN
	// tag.
	//
	// TODO: I think this might be a lie: it reads bit 0x1000 of the
	// VLAN header, which changed meaning in recent revisions of the
	// spec - this extension may now return meaningless information.
	ExtVLANTagPresent Extension = 48
	// ExtVLANProto returns 0x8100 if the frame has a VLAN header,
	// 0x88a8 if the frame has a "Q-in-Q" double VLAN header, or some
	// other value if no VLAN information is present.
	ExtVLANProto Extension = 60
	// ExtRand returns a uniformly random uint32.
	ExtRand Extension = 56
)

// The following gives names to various bit patterns used in opcode construction.

const (
	opMaskCls uint16 = 0x7
	// opClsLoad masks
	opMaskLoadDest  = 0x01
	opMaskLoadWidth = 0x18
	opMaskLoadMode  = 0xe0
	// opClsALU
	opMaskOperandSrc = 0x08
	opMaskOperator   = 0xf0
	// opClsJump
	opMaskJumpConst = 0x0f
	opMaskJumpCond  = 0xf0
)

const (
	// +---------------+-----------------+---+---+---+
	// | AddrMode (3b) | LoadWidth (2b)  | 0 | 0 | 0 |
	// +---------------+-----------------+---+---+---+
	opClsLoadA uint16 = iota
	// +---------------+-----------------+---+---+---+
	// | AddrMode (3b) | LoadWidth (2b)  | 0 | 0 | 1 |
	// +---------------+-----------------+---+---+---+
	opClsLoadX
	// +---+---+---+---+---+---+---+---+
	// | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
	// +---+---+---+---+---+---+---+---+
	opClsStoreA
	// +---+---+---+---+---+---+---+---+
	// | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
	// +---+---+---+---+---+---+---+---+
	opClsStoreX
	// +---------------+-----------------+---+---+---+
	// | Operator (4b) | OperandSrc (1b) | 1 | 0 | 0 |
	// +---------------+-----------------+---+---+---+
	opClsALU
	// +-----------------------------+---+---+---+---+
	// |      TestOperator (4b)      | 0 | 1 | 0 | 1 |
	// +-----------------------------+---+---+---+---+
	opClsJump
	// +---+-------------------------+---+---+---+---+
	// | 0 | 0 | 0 |   RetSrc (1b)   | 0 | 1 | 1 | 0 |
	// +---+-------------------------+---+---+---+---+
	opClsReturn
	// +---+-------------------------+---+---+---+---+
	// | 0 | 0 | 0 |  TXAorTAX (1b)  | 0 | 1 | 1 | 1 |
	// +---+-------------------------+---+---+---+---+
	opClsMisc
)

const (
	opAddrModeImmediate uint16 = iota << 5
	opAddrModeAbsolute
	opAddrModeIndirect
	opAddrModeScratch
	opAddrModePacketLen // actually an extension, not an addressing mode.
	opAddrModeMemShift
)

const (
	opLoadWidth4 uint16 = iota << 3
	opLoadWidth2
	opLoadWidth1
)

// Operator defined by ALUOp*

const (
	opALUSrcConstant uint16 = iota << 3
	opALUSrcX
)

const (
	opJumpAlways = iota << 4
	opJumpEqual
	opJumpGT
	opJumpGE
	opJumpSet
)

const (
	opRetSrcConstant uint16 = iota << 4
	opRetSrcA
)

const (
	opMiscTAX = 0x00
	opMiscTXA = 0x80
)

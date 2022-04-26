// Code generated "gen_opcodes.go"; DO NOT EDIT.

package quasigo

//go:generate stringer -type=opcode -trimprefix=op
type opcode byte

const (
	opInvalid opcode = 0

	// Encoding: 0x01 (width=1)
	// Stack effect: (value) -> ()
	opPop opcode = 1

	// Encoding: 0x02 (width=1)
	// Stack effect: (x) -> (x x)
	opDup opcode = 2

	// Encoding: 0x03 index:u8 (width=2)
	// Stack effect: () -> (value)
	opPushParam opcode = 3

	// Encoding: 0x04 index:u8 (width=2)
	// Stack effect: () -> (value:int)
	opPushIntParam opcode = 4

	// Encoding: 0x05 index:u8 (width=2)
	// Stack effect: () -> (value)
	opPushLocal opcode = 5

	// Encoding: 0x06 index:u8 (width=2)
	// Stack effect: () -> (value:int)
	opPushIntLocal opcode = 6

	// Encoding: 0x07 (width=1)
	// Stack effect: () -> (false)
	opPushFalse opcode = 7

	// Encoding: 0x08 (width=1)
	// Stack effect: () -> (true)
	opPushTrue opcode = 8

	// Encoding: 0x09 constid:u8 (width=2)
	// Stack effect: () -> (const)
	opPushConst opcode = 9

	// Encoding: 0x0a constid:u8 (width=2)
	// Stack effect: () -> (const:int)
	opPushIntConst opcode = 10

	// Encoding: 0x0b (width=1)
	// Stack effect: (value:int) -> (value)
	opConvIntToIface opcode = 11

	// Encoding: 0x0c index:u8 (width=2)
	// Stack effect: (value) -> ()
	opSetLocal opcode = 12

	// Encoding: 0x0d index:u8 (width=2)
	// Stack effect: (value:int) -> ()
	opSetIntLocal opcode = 13

	// Encoding: 0x0e index:u8 (width=2)
	// Stack effect: unchanged
	opIncLocal opcode = 14

	// Encoding: 0x0f index:u8 (width=2)
	// Stack effect: unchanged
	opDecLocal opcode = 15

	// Encoding: 0x10 (width=1)
	// Stack effect: (value) -> (value)
	opReturnTop opcode = 16

	// Encoding: 0x11 (width=1)
	// Stack effect: (value) -> (value)
	opReturnIntTop opcode = 17

	// Encoding: 0x12 (width=1)
	// Stack effect: unchanged
	opReturnFalse opcode = 18

	// Encoding: 0x13 (width=1)
	// Stack effect: unchanged
	opReturnTrue opcode = 19

	// Encoding: 0x14 (width=1)
	// Stack effect: unchanged
	opReturn opcode = 20

	// Encoding: 0x15 offset:i16 (width=3)
	// Stack effect: unchanged
	opJump opcode = 21

	// Encoding: 0x16 offset:i16 (width=3)
	// Stack effect: (cond:bool) -> ()
	opJumpFalse opcode = 22

	// Encoding: 0x17 offset:i16 (width=3)
	// Stack effect: (cond:bool) -> ()
	opJumpTrue opcode = 23

	// Encoding: 0x18 len:u8 (width=2)
	// Stack effect: unchanged
	opSetVariadicLen opcode = 24

	// Encoding: 0x19 funcid:u16 (width=3)
	// Stack effect: (args...) -> (results...)
	opCallNative opcode = 25

	// Encoding: 0x1a (width=1)
	// Stack effect: (value) -> (result:bool)
	opIsNil opcode = 26

	// Encoding: 0x1b (width=1)
	// Stack effect: (value) -> (result:bool)
	opIsNotNil opcode = 27

	// Encoding: 0x1c (width=1)
	// Stack effect: (value:bool) -> (result:bool)
	opNot opcode = 28

	// Encoding: 0x1d (width=1)
	// Stack effect: (x:int y:int) -> (result:bool)
	opEqInt opcode = 29

	// Encoding: 0x1e (width=1)
	// Stack effect: (x:int y:int) -> (result:bool)
	opNotEqInt opcode = 30

	// Encoding: 0x1f (width=1)
	// Stack effect: (x:int y:int) -> (result:bool)
	opGtInt opcode = 31

	// Encoding: 0x20 (width=1)
	// Stack effect: (x:int y:int) -> (result:bool)
	opGtEqInt opcode = 32

	// Encoding: 0x21 (width=1)
	// Stack effect: (x:int y:int) -> (result:bool)
	opLtInt opcode = 33

	// Encoding: 0x22 (width=1)
	// Stack effect: (x:int y:int) -> (result:bool)
	opLtEqInt opcode = 34

	// Encoding: 0x23 (width=1)
	// Stack effect: (x:string y:string) -> (result:bool)
	opEqString opcode = 35

	// Encoding: 0x24 (width=1)
	// Stack effect: (x:string y:string) -> (result:bool)
	opNotEqString opcode = 36

	// Encoding: 0x25 (width=1)
	// Stack effect: (x:string y:string) -> (result:string)
	opConcat opcode = 37

	// Encoding: 0x26 (width=1)
	// Stack effect: (x:int y:int) -> (result:int)
	opAdd opcode = 38

	// Encoding: 0x27 (width=1)
	// Stack effect: (x:int y:int) -> (result:int)
	opSub opcode = 39

	// Encoding: 0x28 (width=1)
	// Stack effect: (s:string from:int to:int) -> (result:string)
	opStringSlice opcode = 40

	// Encoding: 0x29 (width=1)
	// Stack effect: (s:string from:int) -> (result:string)
	opStringSliceFrom opcode = 41

	// Encoding: 0x2a (width=1)
	// Stack effect: (s:string to:int) -> (result:string)
	opStringSliceTo opcode = 42

	// Encoding: 0x2b (width=1)
	// Stack effect: (s:string) -> (result:int)
	opStringLen opcode = 43
)

type opcodeInfo struct {
	width int
}

var opcodeInfoTable = [256]opcodeInfo{
	opInvalid: {width: 1},

	opPop:             {width: 1},
	opDup:             {width: 1},
	opPushParam:       {width: 2},
	opPushIntParam:    {width: 2},
	opPushLocal:       {width: 2},
	opPushIntLocal:    {width: 2},
	opPushFalse:       {width: 1},
	opPushTrue:        {width: 1},
	opPushConst:       {width: 2},
	opPushIntConst:    {width: 2},
	opConvIntToIface:  {width: 1},
	opSetLocal:        {width: 2},
	opSetIntLocal:     {width: 2},
	opIncLocal:        {width: 2},
	opDecLocal:        {width: 2},
	opReturnTop:       {width: 1},
	opReturnIntTop:    {width: 1},
	opReturnFalse:     {width: 1},
	opReturnTrue:      {width: 1},
	opReturn:          {width: 1},
	opJump:            {width: 3},
	opJumpFalse:       {width: 3},
	opJumpTrue:        {width: 3},
	opSetVariadicLen:  {width: 2},
	opCallNative:      {width: 3},
	opIsNil:           {width: 1},
	opIsNotNil:        {width: 1},
	opNot:             {width: 1},
	opEqInt:           {width: 1},
	opNotEqInt:        {width: 1},
	opGtInt:           {width: 1},
	opGtEqInt:         {width: 1},
	opLtInt:           {width: 1},
	opLtEqInt:         {width: 1},
	opEqString:        {width: 1},
	opNotEqString:     {width: 1},
	opConcat:          {width: 1},
	opAdd:             {width: 1},
	opSub:             {width: 1},
	opStringSlice:     {width: 1},
	opStringSliceFrom: {width: 1},
	opStringSliceTo:   {width: 1},
	opStringLen:       {width: 1},
}

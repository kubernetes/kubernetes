//+build !amd64

package flate

const (
	// Masks for shifts with register sizes of the shift value.
	// This can be used to work around the x86 design of shifting by mod register size.
	// It can be used when a variable shift is always smaller than the register size.

	// reg8SizeMaskX - shift value is 8 bits, shifted is X
	reg8SizeMask8  = 0xff
	reg8SizeMask16 = 0xff
	reg8SizeMask32 = 0xff
	reg8SizeMask64 = 0xff

	// reg16SizeMaskX - shift value is 16 bits, shifted is X
	reg16SizeMask8  = 0xffff
	reg16SizeMask16 = 0xffff
	reg16SizeMask32 = 0xffff
	reg16SizeMask64 = 0xffff

	// reg32SizeMaskX - shift value is 32 bits, shifted is X
	reg32SizeMask8  = 0xffffffff
	reg32SizeMask16 = 0xffffffff
	reg32SizeMask32 = 0xffffffff
	reg32SizeMask64 = 0xffffffff

	// reg64SizeMaskX - shift value is 64 bits, shifted is X
	reg64SizeMask8  = 0xffffffffffffffff
	reg64SizeMask16 = 0xffffffffffffffff
	reg64SizeMask32 = 0xffffffffffffffff
	reg64SizeMask64 = 0xffffffffffffffff

	// regSizeMaskUintX - shift value is uint, shifted is X
	regSizeMaskUint8  = ^uint(0)
	regSizeMaskUint16 = ^uint(0)
	regSizeMaskUint32 = ^uint(0)
	regSizeMaskUint64 = ^uint(0)
)

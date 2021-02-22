package flate

const (
	// Masks for shifts with register sizes of the shift value.
	// This can be used to work around the x86 design of shifting by mod register size.
	// It can be used when a variable shift is always smaller than the register size.

	// reg8SizeMaskX - shift value is 8 bits, shifted is X
	reg8SizeMask8  = 7
	reg8SizeMask16 = 15
	reg8SizeMask32 = 31
	reg8SizeMask64 = 63

	// reg16SizeMaskX - shift value is 16 bits, shifted is X
	reg16SizeMask8  = reg8SizeMask8
	reg16SizeMask16 = reg8SizeMask16
	reg16SizeMask32 = reg8SizeMask32
	reg16SizeMask64 = reg8SizeMask64

	// reg32SizeMaskX - shift value is 32 bits, shifted is X
	reg32SizeMask8  = reg8SizeMask8
	reg32SizeMask16 = reg8SizeMask16
	reg32SizeMask32 = reg8SizeMask32
	reg32SizeMask64 = reg8SizeMask64

	// reg64SizeMaskX - shift value is 64 bits, shifted is X
	reg64SizeMask8  = reg8SizeMask8
	reg64SizeMask16 = reg8SizeMask16
	reg64SizeMask32 = reg8SizeMask32
	reg64SizeMask64 = reg8SizeMask64

	// regSizeMaskUintX - shift value is uint, shifted is X
	regSizeMaskUint8  = reg8SizeMask8
	regSizeMaskUint16 = reg8SizeMask16
	regSizeMaskUint32 = reg8SizeMask32
	regSizeMaskUint64 = reg8SizeMask64
)

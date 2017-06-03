package msgp

// size of every object on the wire,
// plus type information. gives us
// constant-time type information
// for traversing composite objects.
//
var sizes = [256]bytespec{
	mnil:      {size: 1, extra: constsize, typ: NilType},
	mfalse:    {size: 1, extra: constsize, typ: BoolType},
	mtrue:     {size: 1, extra: constsize, typ: BoolType},
	mbin8:     {size: 2, extra: extra8, typ: BinType},
	mbin16:    {size: 3, extra: extra16, typ: BinType},
	mbin32:    {size: 5, extra: extra32, typ: BinType},
	mext8:     {size: 3, extra: extra8, typ: ExtensionType},
	mext16:    {size: 4, extra: extra16, typ: ExtensionType},
	mext32:    {size: 6, extra: extra32, typ: ExtensionType},
	mfloat32:  {size: 5, extra: constsize, typ: Float32Type},
	mfloat64:  {size: 9, extra: constsize, typ: Float64Type},
	muint8:    {size: 2, extra: constsize, typ: UintType},
	muint16:   {size: 3, extra: constsize, typ: UintType},
	muint32:   {size: 5, extra: constsize, typ: UintType},
	muint64:   {size: 9, extra: constsize, typ: UintType},
	mint8:     {size: 2, extra: constsize, typ: IntType},
	mint16:    {size: 3, extra: constsize, typ: IntType},
	mint32:    {size: 5, extra: constsize, typ: IntType},
	mint64:    {size: 9, extra: constsize, typ: IntType},
	mfixext1:  {size: 3, extra: constsize, typ: ExtensionType},
	mfixext2:  {size: 4, extra: constsize, typ: ExtensionType},
	mfixext4:  {size: 6, extra: constsize, typ: ExtensionType},
	mfixext8:  {size: 10, extra: constsize, typ: ExtensionType},
	mfixext16: {size: 18, extra: constsize, typ: ExtensionType},
	mstr8:     {size: 2, extra: extra8, typ: StrType},
	mstr16:    {size: 3, extra: extra16, typ: StrType},
	mstr32:    {size: 5, extra: extra32, typ: StrType},
	marray16:  {size: 3, extra: array16v, typ: ArrayType},
	marray32:  {size: 5, extra: array32v, typ: ArrayType},
	mmap16:    {size: 3, extra: map16v, typ: MapType},
	mmap32:    {size: 5, extra: map32v, typ: MapType},
}

func init() {
	// set up fixed fields

	// fixint
	for i := mfixint; i < 0x80; i++ {
		sizes[i] = bytespec{size: 1, extra: constsize, typ: IntType}
	}

	// nfixint
	for i := uint16(mnfixint); i < 0x100; i++ {
		sizes[uint8(i)] = bytespec{size: 1, extra: constsize, typ: IntType}
	}

	// fixstr gets constsize,
	// since the prefix yields the size
	for i := mfixstr; i < 0xc0; i++ {
		sizes[i] = bytespec{size: 1 + rfixstr(i), extra: constsize, typ: StrType}
	}

	// fixmap
	for i := mfixmap; i < 0x90; i++ {
		sizes[i] = bytespec{size: 1, extra: varmode(2 * rfixmap(i)), typ: MapType}
	}

	// fixarray
	for i := mfixarray; i < 0xa0; i++ {
		sizes[i] = bytespec{size: 1, extra: varmode(rfixarray(i)), typ: ArrayType}
	}
}

// a valid bytespsec has
// non-zero 'size' and
// non-zero 'typ'
type bytespec struct {
	size  uint8   // prefix size information
	extra varmode // extra size information
	typ   Type    // type
	_     byte    // makes bytespec 4 bytes (yes, this matters)
}

// size mode
// if positive, # elements for composites
type varmode int8

const (
	constsize varmode = 0  // constant size (size bytes + uint8(varmode) objects)
	extra8            = -1 // has uint8(p[1]) extra bytes
	extra16           = -2 // has be16(p[1:]) extra bytes
	extra32           = -3 // has be32(p[1:]) extra bytes
	map16v            = -4 // use map16
	map32v            = -5 // use map32
	array16v          = -6 // use array16
	array32v          = -7 // use array32
)

func getType(v byte) Type {
	return sizes[v].typ
}

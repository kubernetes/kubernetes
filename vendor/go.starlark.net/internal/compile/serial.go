package compile

// This file defines functions to read and write a compile.Program to a file.
//
// It is the client's responsibility to avoid version skew between the
// compiler used to produce a file and the interpreter that consumes it.
// The version number is provided as a constant.
// Incompatible protocol changes should also increment the version number.
//
// Encoding
//
// Program:
//	"sky!"		[4]byte		# magic number
//	str		uint32le	# offset of <strings> section
//	version		varint		# must match Version
//	filename	string
//	numloads	varint
//	loads		[]Ident
//	numnames	varint
//	names		[]string
//	numconsts	varint
//	consts		[]Constant
//	numglobals	varint
//	globals		[]Ident
//	toplevel	Funcode
//	numfuncs	varint
//	funcs		[]Funcode
//	recursion	varint (0 or 1)
//	<strings>	[]byte		# concatenation of all referenced strings
//	EOF
//
// Funcode:
//	id		Ident
//	code		[]byte
//	pclinetablen	varint
//	pclinetab	[]varint
//	numlocals	varint
//	locals		[]Ident
//	numcells	varint
//	cells		[]int
//	numfreevars	varint
//	freevar		[]Ident
//	maxstack	varint
//	numparams	varint
//	numkwonlyparams	varint
//	hasvarargs	varint (0 or 1)
//	haskwargs	varint (0 or 1)
//
// Ident:
//	filename	string
//	line, col	varint
//
// Constant:                            # type      data
//      type            varint          # 0=string  string
//      data            ...             # 1=bytes   string
//                                      # 2=int     varint
//                                      # 3=float   varint (bits as uint64)
//                                      # 4=bigint  string (decimal ASCII text)
//
// The encoding starts with a four-byte magic number.
// The next four bytes are a little-endian uint32
// that provides the offset of the string section
// at the end of the file, which contains the ordered
// concatenation of all strings referenced by the
// program. This design permits the decoder to read
// the first and second parts of the file into different
// memory allocations: the first (the encoded program)
// is transient, but the second (the strings) persists
// for the life of the Program.
//
// Within the encoded program, all strings are referred
// to by their length. As the encoder and decoder process
// the entire file sequentially, they are in lock step,
// so the start offset of each string is implicit.
//
// Program.Code is represented as a []byte slice to permit
// modification when breakpoints are set. All other strings
// are represented as strings. They all (unsafely) share the
// same backing byte slice.
//
// Aside from the str field, all integers are encoded as varints.

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/big"
	debugpkg "runtime/debug"
	"unsafe"

	"go.starlark.net/syntax"
)

const magic = "!sky"

// Encode encodes a compiled Starlark program.
func (prog *Program) Encode() []byte {
	var e encoder
	e.p = append(e.p, magic...)
	e.p = append(e.p, "????"...) // string data offset; filled in later
	e.int(Version)
	e.string(prog.Toplevel.Pos.Filename())
	e.bindings(prog.Loads)
	e.int(len(prog.Names))
	for _, name := range prog.Names {
		e.string(name)
	}
	e.int(len(prog.Constants))
	for _, c := range prog.Constants {
		switch c := c.(type) {
		case string:
			e.int(0)
			e.string(c)
		case Bytes:
			e.int(1)
			e.string(string(c))
		case int64:
			e.int(2)
			e.int64(c)
		case float64:
			e.int(3)
			e.uint64(math.Float64bits(c))
		case *big.Int:
			e.int(4)
			e.string(c.Text(10))
		}
	}
	e.bindings(prog.Globals)
	e.function(prog.Toplevel)
	e.int(len(prog.Functions))
	for _, fn := range prog.Functions {
		e.function(fn)
	}
	e.int(b2i(prog.Recursion))

	// Patch in the offset of the string data section.
	binary.LittleEndian.PutUint32(e.p[4:8], uint32(len(e.p)))

	return append(e.p, e.s...)
}

type encoder struct {
	p   []byte // encoded program
	s   []byte // strings
	tmp [binary.MaxVarintLen64]byte
}

func (e *encoder) int(x int) {
	e.int64(int64(x))
}

func (e *encoder) int64(x int64) {
	n := binary.PutVarint(e.tmp[:], x)
	e.p = append(e.p, e.tmp[:n]...)
}

func (e *encoder) uint64(x uint64) {
	n := binary.PutUvarint(e.tmp[:], x)
	e.p = append(e.p, e.tmp[:n]...)
}

func (e *encoder) string(s string) {
	e.int(len(s))
	e.s = append(e.s, s...)
}

func (e *encoder) bytes(b []byte) {
	e.int(len(b))
	e.s = append(e.s, b...)
}

func (e *encoder) binding(bind Binding) {
	e.string(bind.Name)
	e.int(int(bind.Pos.Line))
	e.int(int(bind.Pos.Col))
}

func (e *encoder) bindings(binds []Binding) {
	e.int(len(binds))
	for _, bind := range binds {
		e.binding(bind)
	}
}

func (e *encoder) function(fn *Funcode) {
	e.binding(Binding{fn.Name, fn.Pos})
	e.string(fn.Doc)
	e.bytes(fn.Code)
	e.int(len(fn.pclinetab))
	for _, x := range fn.pclinetab {
		e.int64(int64(x))
	}
	e.bindings(fn.Locals)
	e.int(len(fn.Cells))
	for _, index := range fn.Cells {
		e.int(index)
	}
	e.bindings(fn.FreeVars)
	e.int(fn.MaxStack)
	e.int(fn.NumParams)
	e.int(fn.NumKwonlyParams)
	e.int(b2i(fn.HasVarargs))
	e.int(b2i(fn.HasKwargs))
}

func b2i(b bool) int {
	if b {
		return 1
	} else {
		return 0
	}
}

// DecodeProgram decodes a compiled Starlark program from data.
func DecodeProgram(data []byte) (_ *Program, err error) {
	if len(data) < len(magic) {
		return nil, fmt.Errorf("not a compiled module: no magic number")
	}
	if got := string(data[:4]); got != magic {
		return nil, fmt.Errorf("not a compiled module: got magic number %q, want %q",
			got, magic)
	}
	defer func() {
		if x := recover(); x != nil {
			debugpkg.PrintStack()
			err = fmt.Errorf("internal error while decoding program: %v", x)
		}
	}()

	offset := binary.LittleEndian.Uint32(data[4:8])
	d := decoder{
		p: data[8:offset],
		s: append([]byte(nil), data[offset:]...), // allocate a copy, which will persist
	}

	if v := d.int(); v != Version {
		return nil, fmt.Errorf("version mismatch: read %d, want %d", v, Version)
	}

	filename := d.string()
	d.filename = &filename

	loads := d.bindings()

	names := make([]string, d.int())
	for i := range names {
		names[i] = d.string()
	}

	// constants
	constants := make([]interface{}, d.int())
	for i := range constants {
		var c interface{}
		switch d.int() {
		case 0:
			c = d.string()
		case 1:
			c = Bytes(d.string())
		case 2:
			c = d.int64()
		case 3:
			c = math.Float64frombits(d.uint64())
		case 4:
			c, _ = new(big.Int).SetString(d.string(), 10)
		}
		constants[i] = c
	}

	globals := d.bindings()
	toplevel := d.function()
	funcs := make([]*Funcode, d.int())
	for i := range funcs {
		funcs[i] = d.function()
	}
	recursion := d.int() != 0

	prog := &Program{
		Loads:     loads,
		Names:     names,
		Constants: constants,
		Globals:   globals,
		Functions: funcs,
		Toplevel:  toplevel,
		Recursion: recursion,
	}
	toplevel.Prog = prog
	for _, f := range funcs {
		f.Prog = prog
	}

	if len(d.p)+len(d.s) > 0 {
		return nil, fmt.Errorf("internal error: unconsumed data during decoding")
	}

	return prog, nil
}

type decoder struct {
	p        []byte  // encoded program
	s        []byte  // strings
	filename *string // (indirect to avoid keeping decoder live)
}

func (d *decoder) int() int {
	return int(d.int64())
}

func (d *decoder) int64() int64 {
	x, len := binary.Varint(d.p[:])
	d.p = d.p[len:]
	return x
}

func (d *decoder) uint64() uint64 {
	x, len := binary.Uvarint(d.p[:])
	d.p = d.p[len:]
	return x
}

func (d *decoder) string() (s string) {
	if slice := d.bytes(); len(slice) > 0 {
		// Avoid a memory allocation for each string
		// by unsafely aliasing slice.
		type string struct {
			data *byte
			len  int
		}
		ptr := (*string)(unsafe.Pointer(&s))
		ptr.data = &slice[0]
		ptr.len = len(slice)
	}
	return s
}

func (d *decoder) bytes() []byte {
	len := d.int()
	r := d.s[:len:len]
	d.s = d.s[len:]
	return r
}

func (d *decoder) binding() Binding {
	name := d.string()
	line := int32(d.int())
	col := int32(d.int())
	return Binding{Name: name, Pos: syntax.MakePosition(d.filename, line, col)}
}

func (d *decoder) bindings() []Binding {
	bindings := make([]Binding, d.int())
	for i := range bindings {
		bindings[i] = d.binding()
	}
	return bindings
}

func (d *decoder) ints() []int {
	ints := make([]int, d.int())
	for i := range ints {
		ints[i] = d.int()
	}
	return ints
}

func (d *decoder) bool() bool { return d.int() != 0 }

func (d *decoder) function() *Funcode {
	id := d.binding()
	doc := d.string()
	code := d.bytes()
	pclinetab := make([]uint16, d.int())
	for i := range pclinetab {
		pclinetab[i] = uint16(d.int())
	}
	locals := d.bindings()
	cells := d.ints()
	freevars := d.bindings()
	maxStack := d.int()
	numParams := d.int()
	numKwonlyParams := d.int()
	hasVarargs := d.int() != 0
	hasKwargs := d.int() != 0
	return &Funcode{
		// Prog is filled in later.
		Pos:             id.Pos,
		Name:            id.Name,
		Doc:             doc,
		Code:            code,
		pclinetab:       pclinetab,
		Locals:          locals,
		Cells:           cells,
		FreeVars:        freevars,
		MaxStack:        maxStack,
		NumParams:       numParams,
		NumKwonlyParams: numKwonlyParams,
		HasVarargs:      hasVarargs,
		HasKwargs:       hasKwargs,
	}
}

package btf

import (
	"bytes"
	"debug/elf"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"reflect"
	"sync"
	"unsafe"

	"github.com/cilium/ebpf/internal"
	"github.com/cilium/ebpf/internal/unix"
)

const btfMagic = 0xeB9F

// Errors returned by BTF functions.
var (
	ErrNotSupported   = internal.ErrNotSupported
	ErrNotFound       = errors.New("not found")
	ErrNoExtendedInfo = errors.New("no extended info")
)

// ID represents the unique ID of a BTF object.
type ID uint32

// Spec represents decoded BTF.
type Spec struct {
	rawTypes   []rawType
	strings    stringTable
	types      []Type
	namedTypes map[string][]NamedType
	funcInfos  map[string]extInfo
	lineInfos  map[string]extInfo
	coreRelos  map[string]coreRelos
	byteOrder  binary.ByteOrder
}

type btfHeader struct {
	Magic   uint16
	Version uint8
	Flags   uint8
	HdrLen  uint32

	TypeOff   uint32
	TypeLen   uint32
	StringOff uint32
	StringLen uint32
}

// LoadSpecFromReader reads BTF sections from an ELF.
//
// Returns ErrNotFound if the reader contains no BTF.
func LoadSpecFromReader(rd io.ReaderAt) (*Spec, error) {
	file, err := internal.NewSafeELFFile(rd)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	symbols, err := file.Symbols()
	if err != nil {
		return nil, fmt.Errorf("can't read symbols: %v", err)
	}

	variableOffsets := make(map[variable]uint32)
	for _, symbol := range symbols {
		if idx := symbol.Section; idx >= elf.SHN_LORESERVE && idx <= elf.SHN_HIRESERVE {
			// Ignore things like SHN_ABS
			continue
		}

		if int(symbol.Section) >= len(file.Sections) {
			return nil, fmt.Errorf("symbol %s: invalid section %d", symbol.Name, symbol.Section)
		}

		secName := file.Sections[symbol.Section].Name
		if symbol.Value > math.MaxUint32 {
			return nil, fmt.Errorf("section %s: symbol %s: size exceeds maximum", secName, symbol.Name)
		}

		variableOffsets[variable{secName, symbol.Name}] = uint32(symbol.Value)
	}

	return loadSpecFromELF(file, variableOffsets)
}

func loadSpecFromELF(file *internal.SafeELFFile, variableOffsets map[variable]uint32) (*Spec, error) {
	var (
		btfSection    *elf.Section
		btfExtSection *elf.Section
		sectionSizes  = make(map[string]uint32)
	)

	for _, sec := range file.Sections {
		switch sec.Name {
		case ".BTF":
			btfSection = sec
		case ".BTF.ext":
			btfExtSection = sec
		default:
			if sec.Type != elf.SHT_PROGBITS && sec.Type != elf.SHT_NOBITS {
				break
			}

			if sec.Size > math.MaxUint32 {
				return nil, fmt.Errorf("section %s exceeds maximum size", sec.Name)
			}

			sectionSizes[sec.Name] = uint32(sec.Size)
		}
	}

	if btfSection == nil {
		return nil, fmt.Errorf("btf: %w", ErrNotFound)
	}

	spec, err := loadRawSpec(btfSection.Open(), file.ByteOrder, sectionSizes, variableOffsets)
	if err != nil {
		return nil, err
	}

	if btfExtSection == nil {
		return spec, nil
	}

	spec.funcInfos, spec.lineInfos, spec.coreRelos, err = parseExtInfos(btfExtSection.Open(), file.ByteOrder, spec.strings)
	if err != nil {
		return nil, fmt.Errorf("can't read ext info: %w", err)
	}

	return spec, nil
}

// LoadRawSpec reads a blob of BTF data that isn't wrapped in an ELF file.
//
// Prefer using LoadSpecFromReader, since this function only supports a subset
// of BTF.
func LoadRawSpec(btf io.Reader, bo binary.ByteOrder) (*Spec, error) {
	// This will return an error if we encounter a Datasec, since we can't fix
	// it up.
	return loadRawSpec(btf, bo, nil, nil)
}

func loadRawSpec(btf io.Reader, bo binary.ByteOrder, sectionSizes map[string]uint32, variableOffsets map[variable]uint32) (*Spec, error) {
	rawTypes, rawStrings, err := parseBTF(btf, bo)
	if err != nil {
		return nil, err
	}

	err = fixupDatasec(rawTypes, rawStrings, sectionSizes, variableOffsets)
	if err != nil {
		return nil, err
	}

	types, typesByName, err := inflateRawTypes(rawTypes, rawStrings)
	if err != nil {
		return nil, err
	}

	return &Spec{
		rawTypes:   rawTypes,
		namedTypes: typesByName,
		types:      types,
		strings:    rawStrings,
		byteOrder:  bo,
	}, nil
}

var kernelBTF struct {
	sync.Mutex
	*Spec
}

// LoadKernelSpec returns the current kernel's BTF information.
//
// Requires a >= 5.5 kernel with CONFIG_DEBUG_INFO_BTF enabled. Returns
// ErrNotSupported if BTF is not enabled.
func LoadKernelSpec() (*Spec, error) {
	kernelBTF.Lock()
	defer kernelBTF.Unlock()

	if kernelBTF.Spec != nil {
		return kernelBTF.Spec, nil
	}

	var err error
	kernelBTF.Spec, err = loadKernelSpec()
	return kernelBTF.Spec, err
}

func loadKernelSpec() (*Spec, error) {
	release, err := unix.KernelRelease()
	if err != nil {
		return nil, fmt.Errorf("can't read kernel release number: %w", err)
	}

	fh, err := os.Open("/sys/kernel/btf/vmlinux")
	if err == nil {
		defer fh.Close()

		return LoadRawSpec(fh, internal.NativeEndian)
	}

	// use same list of locations as libbpf
	// https://github.com/libbpf/libbpf/blob/9a3a42608dbe3731256a5682a125ac1e23bced8f/src/btf.c#L3114-L3122
	locations := []string{
		"/boot/vmlinux-%s",
		"/lib/modules/%s/vmlinux-%[1]s",
		"/lib/modules/%s/build/vmlinux",
		"/usr/lib/modules/%s/kernel/vmlinux",
		"/usr/lib/debug/boot/vmlinux-%s",
		"/usr/lib/debug/boot/vmlinux-%s.debug",
		"/usr/lib/debug/lib/modules/%s/vmlinux",
	}

	for _, loc := range locations {
		path := fmt.Sprintf(loc, release)

		fh, err := os.Open(path)
		if err != nil {
			continue
		}
		defer fh.Close()

		file, err := internal.NewSafeELFFile(fh)
		if err != nil {
			return nil, err
		}
		defer file.Close()

		return loadSpecFromELF(file, nil)
	}

	return nil, fmt.Errorf("no BTF for kernel version %s: %w", release, internal.ErrNotSupported)
}

func parseBTF(btf io.Reader, bo binary.ByteOrder) ([]rawType, stringTable, error) {
	rawBTF, err := io.ReadAll(btf)
	if err != nil {
		return nil, nil, fmt.Errorf("can't read BTF: %v", err)
	}

	rd := bytes.NewReader(rawBTF)

	var header btfHeader
	if err := binary.Read(rd, bo, &header); err != nil {
		return nil, nil, fmt.Errorf("can't read header: %v", err)
	}

	if header.Magic != btfMagic {
		return nil, nil, fmt.Errorf("incorrect magic value %v", header.Magic)
	}

	if header.Version != 1 {
		return nil, nil, fmt.Errorf("unexpected version %v", header.Version)
	}

	if header.Flags != 0 {
		return nil, nil, fmt.Errorf("unsupported flags %v", header.Flags)
	}

	remainder := int64(header.HdrLen) - int64(binary.Size(&header))
	if remainder < 0 {
		return nil, nil, errors.New("header is too short")
	}

	if _, err := io.CopyN(internal.DiscardZeroes{}, rd, remainder); err != nil {
		return nil, nil, fmt.Errorf("header padding: %v", err)
	}

	if _, err := rd.Seek(int64(header.HdrLen+header.StringOff), io.SeekStart); err != nil {
		return nil, nil, fmt.Errorf("can't seek to start of string section: %v", err)
	}

	rawStrings, err := readStringTable(io.LimitReader(rd, int64(header.StringLen)))
	if err != nil {
		return nil, nil, fmt.Errorf("can't read type names: %w", err)
	}

	if _, err := rd.Seek(int64(header.HdrLen+header.TypeOff), io.SeekStart); err != nil {
		return nil, nil, fmt.Errorf("can't seek to start of type section: %v", err)
	}

	rawTypes, err := readTypes(io.LimitReader(rd, int64(header.TypeLen)), bo)
	if err != nil {
		return nil, nil, fmt.Errorf("can't read types: %w", err)
	}

	return rawTypes, rawStrings, nil
}

type variable struct {
	section string
	name    string
}

func fixupDatasec(rawTypes []rawType, rawStrings stringTable, sectionSizes map[string]uint32, variableOffsets map[variable]uint32) error {
	for i, rawType := range rawTypes {
		if rawType.Kind() != kindDatasec {
			continue
		}

		name, err := rawStrings.Lookup(rawType.NameOff)
		if err != nil {
			return err
		}

		if name == ".kconfig" || name == ".ksyms" {
			return fmt.Errorf("reference to %s: %w", name, ErrNotSupported)
		}

		if rawTypes[i].SizeType != 0 {
			continue
		}

		size, ok := sectionSizes[name]
		if !ok {
			return fmt.Errorf("data section %s: missing size", name)
		}

		rawTypes[i].SizeType = size

		secinfos := rawType.data.([]btfVarSecinfo)
		for j, secInfo := range secinfos {
			id := int(secInfo.Type - 1)
			if id >= len(rawTypes) {
				return fmt.Errorf("data section %s: invalid type id %d for variable %d", name, id, j)
			}

			varName, err := rawStrings.Lookup(rawTypes[id].NameOff)
			if err != nil {
				return fmt.Errorf("data section %s: can't get name for type %d: %w", name, id, err)
			}

			offset, ok := variableOffsets[variable{name, varName}]
			if !ok {
				return fmt.Errorf("data section %s: missing offset for variable %s", name, varName)
			}

			secinfos[j].Offset = offset
		}
	}

	return nil
}

// Copy creates a copy of Spec.
func (s *Spec) Copy() *Spec {
	types, _ := copyTypes(s.types, nil)
	namedTypes := make(map[string][]NamedType)
	for _, typ := range types {
		if named, ok := typ.(NamedType); ok {
			name := essentialName(named.TypeName())
			namedTypes[name] = append(namedTypes[name], named)
		}
	}

	// NB: Other parts of spec are not copied since they are immutable.
	return &Spec{
		s.rawTypes,
		s.strings,
		types,
		namedTypes,
		s.funcInfos,
		s.lineInfos,
		s.coreRelos,
		s.byteOrder,
	}
}

type marshalOpts struct {
	ByteOrder        binary.ByteOrder
	StripFuncLinkage bool
}

func (s *Spec) marshal(opts marshalOpts) ([]byte, error) {
	var (
		buf       bytes.Buffer
		header    = new(btfHeader)
		headerLen = binary.Size(header)
	)

	// Reserve space for the header. We have to write it last since
	// we don't know the size of the type section yet.
	_, _ = buf.Write(make([]byte, headerLen))

	// Write type section, just after the header.
	for _, raw := range s.rawTypes {
		switch {
		case opts.StripFuncLinkage && raw.Kind() == kindFunc:
			raw.SetLinkage(StaticFunc)
		}

		if err := raw.Marshal(&buf, opts.ByteOrder); err != nil {
			return nil, fmt.Errorf("can't marshal BTF: %w", err)
		}
	}

	typeLen := uint32(buf.Len() - headerLen)

	// Write string section after type section.
	_, _ = buf.Write(s.strings)

	// Fill out the header, and write it out.
	header = &btfHeader{
		Magic:     btfMagic,
		Version:   1,
		Flags:     0,
		HdrLen:    uint32(headerLen),
		TypeOff:   0,
		TypeLen:   typeLen,
		StringOff: typeLen,
		StringLen: uint32(len(s.strings)),
	}

	raw := buf.Bytes()
	err := binary.Write(sliceWriter(raw[:headerLen]), opts.ByteOrder, header)
	if err != nil {
		return nil, fmt.Errorf("can't write header: %v", err)
	}

	return raw, nil
}

type sliceWriter []byte

func (sw sliceWriter) Write(p []byte) (int, error) {
	if len(p) != len(sw) {
		return 0, errors.New("size doesn't match")
	}

	return copy(sw, p), nil
}

// Program finds the BTF for a specific section.
//
// Length is the number of bytes in the raw BPF instruction stream.
//
// Returns an error which may wrap ErrNoExtendedInfo if the Spec doesn't
// contain extended BTF info.
func (s *Spec) Program(name string, length uint64) (*Program, error) {
	if length == 0 {
		return nil, errors.New("length musn't be zero")
	}

	if s.funcInfos == nil && s.lineInfos == nil && s.coreRelos == nil {
		return nil, fmt.Errorf("BTF for section %s: %w", name, ErrNoExtendedInfo)
	}

	funcInfos, funcOK := s.funcInfos[name]
	lineInfos, lineOK := s.lineInfos[name]
	relos, coreOK := s.coreRelos[name]

	if !funcOK && !lineOK && !coreOK {
		return nil, fmt.Errorf("no extended BTF info for section %s", name)
	}

	return &Program{s, length, funcInfos, lineInfos, relos}, nil
}

// FindType searches for a type with a specific name.
//
// Called T a type that satisfies Type, typ must be a non-nil **T.
// On success, the address of the found type will be copied in typ.
//
// Returns an error wrapping ErrNotFound if no matching
// type exists in spec.
func (s *Spec) FindType(name string, typ interface{}) error {
	typValue := reflect.ValueOf(typ)
	if typValue.Kind() != reflect.Ptr {
		return fmt.Errorf("%T is not a pointer", typ)
	}

	typPtr := typValue.Elem()
	if !typPtr.CanSet() {
		return fmt.Errorf("%T cannot be set", typ)
	}

	wanted := typPtr.Type()
	if !wanted.AssignableTo(reflect.TypeOf((*Type)(nil)).Elem()) {
		return fmt.Errorf("%T does not satisfy Type interface", typ)
	}

	var candidate Type
	for _, typ := range s.namedTypes[essentialName(name)] {
		if reflect.TypeOf(typ) != wanted {
			continue
		}

		// Match against the full name, not just the essential one.
		if typ.TypeName() != name {
			continue
		}

		if candidate != nil {
			return fmt.Errorf("type %s: multiple candidates for %T", name, typ)
		}

		candidate = typ
	}

	if candidate == nil {
		return fmt.Errorf("type %s: %w", name, ErrNotFound)
	}

	typPtr.Set(reflect.ValueOf(candidate))

	return nil
}

// Handle is a reference to BTF loaded into the kernel.
type Handle struct {
	spec *Spec
	fd   *internal.FD
}

// NewHandle loads BTF into the kernel.
//
// Returns ErrNotSupported if BTF is not supported.
func NewHandle(spec *Spec) (*Handle, error) {
	if err := haveBTF(); err != nil {
		return nil, err
	}

	if spec.byteOrder != internal.NativeEndian {
		return nil, fmt.Errorf("can't load %s BTF on %s", spec.byteOrder, internal.NativeEndian)
	}

	btf, err := spec.marshal(marshalOpts{
		ByteOrder:        internal.NativeEndian,
		StripFuncLinkage: haveFuncLinkage() != nil,
	})
	if err != nil {
		return nil, fmt.Errorf("can't marshal BTF: %w", err)
	}

	if uint64(len(btf)) > math.MaxUint32 {
		return nil, errors.New("BTF exceeds the maximum size")
	}

	attr := &bpfLoadBTFAttr{
		btf:     internal.NewSlicePointer(btf),
		btfSize: uint32(len(btf)),
	}

	fd, err := bpfLoadBTF(attr)
	if err != nil {
		logBuf := make([]byte, 64*1024)
		attr.logBuf = internal.NewSlicePointer(logBuf)
		attr.btfLogSize = uint32(len(logBuf))
		attr.btfLogLevel = 1
		_, logErr := bpfLoadBTF(attr)
		return nil, internal.ErrorWithLog(err, logBuf, logErr)
	}

	return &Handle{spec.Copy(), fd}, nil
}

// NewHandleFromID returns the BTF handle for a given id.
//
// Returns ErrNotExist, if there is no BTF with the given id.
//
// Requires CAP_SYS_ADMIN.
func NewHandleFromID(id ID) (*Handle, error) {
	fd, err := internal.BPFObjGetFDByID(internal.BPF_BTF_GET_FD_BY_ID, uint32(id))
	if err != nil {
		return nil, fmt.Errorf("get BTF by id: %w", err)
	}

	info, err := newInfoFromFd(fd)
	if err != nil {
		_ = fd.Close()
		return nil, fmt.Errorf("get BTF spec for handle: %w", err)
	}

	return &Handle{info.BTF, fd}, nil
}

// Spec returns the Spec that defined the BTF loaded into the kernel.
func (h *Handle) Spec() *Spec {
	return h.spec
}

// Close destroys the handle.
//
// Subsequent calls to FD will return an invalid value.
func (h *Handle) Close() error {
	return h.fd.Close()
}

// FD returns the file descriptor for the handle.
func (h *Handle) FD() int {
	value, err := h.fd.Value()
	if err != nil {
		return -1
	}

	return int(value)
}

// Map is the BTF for a map.
type Map struct {
	Spec       *Spec
	Key, Value Type
}

// Program is the BTF information for a stream of instructions.
type Program struct {
	spec                 *Spec
	length               uint64
	funcInfos, lineInfos extInfo
	coreRelos            coreRelos
}

// Spec returns the BTF spec of this program.
func (p *Program) Spec() *Spec {
	return p.spec
}

// Append the information from other to the Program.
func (p *Program) Append(other *Program) error {
	if other.spec != p.spec {
		return fmt.Errorf("can't append program with different BTF specs")
	}

	funcInfos, err := p.funcInfos.append(other.funcInfos, p.length)
	if err != nil {
		return fmt.Errorf("func infos: %w", err)
	}

	lineInfos, err := p.lineInfos.append(other.lineInfos, p.length)
	if err != nil {
		return fmt.Errorf("line infos: %w", err)
	}

	p.funcInfos = funcInfos
	p.lineInfos = lineInfos
	p.coreRelos = p.coreRelos.append(other.coreRelos, p.length)
	p.length += other.length
	return nil
}

// FuncInfos returns the binary form of BTF function infos.
func (p *Program) FuncInfos() (recordSize uint32, bytes []byte, err error) {
	bytes, err = p.funcInfos.MarshalBinary()
	if err != nil {
		return 0, nil, fmt.Errorf("func infos: %w", err)
	}

	return p.funcInfos.recordSize, bytes, nil
}

// LineInfos returns the binary form of BTF line infos.
func (p *Program) LineInfos() (recordSize uint32, bytes []byte, err error) {
	bytes, err = p.lineInfos.MarshalBinary()
	if err != nil {
		return 0, nil, fmt.Errorf("line infos: %w", err)
	}

	return p.lineInfos.recordSize, bytes, nil
}

// Fixups returns the changes required to adjust the program to the target.
//
// Passing a nil target will relocate against the running kernel.
func (p *Program) Fixups(target *Spec) (COREFixups, error) {
	if len(p.coreRelos) == 0 {
		return nil, nil
	}

	if target == nil {
		var err error
		target, err = LoadKernelSpec()
		if err != nil {
			return nil, err
		}
	}

	return coreRelocate(p.spec, target, p.coreRelos)
}

type bpfLoadBTFAttr struct {
	btf         internal.Pointer
	logBuf      internal.Pointer
	btfSize     uint32
	btfLogSize  uint32
	btfLogLevel uint32
}

func bpfLoadBTF(attr *bpfLoadBTFAttr) (*internal.FD, error) {
	fd, err := internal.BPF(internal.BPF_BTF_LOAD, unsafe.Pointer(attr), unsafe.Sizeof(*attr))
	if err != nil {
		return nil, err
	}

	return internal.NewFD(uint32(fd)), nil
}

func marshalBTF(types interface{}, strings []byte, bo binary.ByteOrder) []byte {
	const minHeaderLength = 24

	typesLen := uint32(binary.Size(types))
	header := btfHeader{
		Magic:     btfMagic,
		Version:   1,
		HdrLen:    minHeaderLength,
		TypeOff:   0,
		TypeLen:   typesLen,
		StringOff: typesLen,
		StringLen: uint32(len(strings)),
	}

	buf := new(bytes.Buffer)
	_ = binary.Write(buf, bo, &header)
	_ = binary.Write(buf, bo, types)
	buf.Write(strings)

	return buf.Bytes()
}

var haveBTF = internal.FeatureTest("BTF", "5.1", func() error {
	var (
		types struct {
			Integer btfType
			Var     btfType
			btfVar  struct{ Linkage uint32 }
		}
		strings = []byte{0, 'a', 0}
	)

	// We use a BTF_KIND_VAR here, to make sure that
	// the kernel understands BTF at least as well as we
	// do. BTF_KIND_VAR was introduced ~5.1.
	types.Integer.SetKind(kindPointer)
	types.Var.NameOff = 1
	types.Var.SetKind(kindVar)
	types.Var.SizeType = 1

	btf := marshalBTF(&types, strings, internal.NativeEndian)

	fd, err := bpfLoadBTF(&bpfLoadBTFAttr{
		btf:     internal.NewSlicePointer(btf),
		btfSize: uint32(len(btf)),
	})
	if errors.Is(err, unix.EINVAL) || errors.Is(err, unix.EPERM) {
		// Treat both EINVAL and EPERM as not supported: loading the program
		// might still succeed without BTF.
		return internal.ErrNotSupported
	}
	if err != nil {
		return err
	}

	fd.Close()
	return nil
})

var haveFuncLinkage = internal.FeatureTest("BTF func linkage", "5.6", func() error {
	if err := haveBTF(); err != nil {
		return err
	}

	var (
		types struct {
			FuncProto btfType
			Func      btfType
		}
		strings = []byte{0, 'a', 0}
	)

	types.FuncProto.SetKind(kindFuncProto)
	types.Func.SetKind(kindFunc)
	types.Func.SizeType = 1 // aka FuncProto
	types.Func.NameOff = 1
	types.Func.SetLinkage(GlobalFunc)

	btf := marshalBTF(&types, strings, internal.NativeEndian)

	fd, err := bpfLoadBTF(&bpfLoadBTFAttr{
		btf:     internal.NewSlicePointer(btf),
		btfSize: uint32(len(btf)),
	})
	if errors.Is(err, unix.EINVAL) {
		return internal.ErrNotSupported
	}
	if err != nil {
		return err
	}

	fd.Close()
	return nil
})

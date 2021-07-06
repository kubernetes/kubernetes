package btf

import (
	"bytes"
	"debug/elf"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
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

// Spec represents decoded BTF.
type Spec struct {
	rawTypes   []rawType
	strings    stringTable
	types      []Type
	namedTypes map[string][]namedType
	funcInfos  map[string]extInfo
	lineInfos  map[string]extInfo
	coreRelos  map[string]bpfCoreRelos
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
// Returns a nil Spec and no error if no BTF was present.
func LoadSpecFromReader(rd io.ReaderAt) (*Spec, error) {
	file, err := internal.NewSafeELFFile(rd)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	btfSection, btfExtSection, sectionSizes, err := findBtfSections(file)
	if err != nil {
		return nil, err
	}

	if btfSection == nil {
		return nil, nil
	}

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
		if _, ok := sectionSizes[secName]; !ok {
			continue
		}

		if symbol.Value > math.MaxUint32 {
			return nil, fmt.Errorf("section %s: symbol %s: size exceeds maximum", secName, symbol.Name)
		}

		variableOffsets[variable{secName, symbol.Name}] = uint32(symbol.Value)
	}

	spec, err := loadNakedSpec(btfSection.Open(), file.ByteOrder, sectionSizes, variableOffsets)
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

func findBtfSections(file *internal.SafeELFFile) (*elf.Section, *elf.Section, map[string]uint32, error) {
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
				return nil, nil, nil, fmt.Errorf("section %s exceeds maximum size", sec.Name)
			}

			sectionSizes[sec.Name] = uint32(sec.Size)
		}
	}
	return btfSection, btfExtSection, sectionSizes, nil
}

func loadSpecFromVmlinux(rd io.ReaderAt) (*Spec, error) {
	file, err := internal.NewSafeELFFile(rd)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	btfSection, _, _, err := findBtfSections(file)
	if err != nil {
		return nil, fmt.Errorf(".BTF ELF section: %s", err)
	}
	if btfSection == nil {
		return nil, fmt.Errorf("unable to find .BTF ELF section")
	}
	return loadNakedSpec(btfSection.Open(), file.ByteOrder, nil, nil)
}

func loadNakedSpec(btf io.ReadSeeker, bo binary.ByteOrder, sectionSizes map[string]uint32, variableOffsets map[variable]uint32) (*Spec, error) {
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

		return loadNakedSpec(fh, internal.NativeEndian, nil, nil)
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

		return loadSpecFromVmlinux(fh)
	}

	return nil, fmt.Errorf("no BTF for kernel version %s: %w", release, internal.ErrNotSupported)
}

func parseBTF(btf io.ReadSeeker, bo binary.ByteOrder) ([]rawType, stringTable, error) {
	rawBTF, err := ioutil.ReadAll(btf)
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
			raw.SetLinkage(linkageStatic)
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
	coreRelos, coreOK := s.coreRelos[name]

	if !funcOK && !lineOK && !coreOK {
		return nil, fmt.Errorf("no extended BTF info for section %s", name)
	}

	return &Program{s, length, funcInfos, lineInfos, coreRelos}, nil
}

// Datasec returns the BTF required to create maps which represent data sections.
func (s *Spec) Datasec(name string) (*Map, error) {
	var datasec Datasec
	if err := s.FindType(name, &datasec); err != nil {
		return nil, fmt.Errorf("data section %s: can't get BTF: %w", name, err)
	}

	m := NewMap(s, &Void{}, &datasec)
	return &m, nil
}

// FindType searches for a type with a specific name.
//
// hint determines the type of the returned Type.
//
// Returns an error wrapping ErrNotFound if no matching
// type exists in spec.
func (s *Spec) FindType(name string, typ Type) error {
	var (
		wanted    = reflect.TypeOf(typ)
		candidate Type
	)

	for _, typ := range s.namedTypes[essentialName(name)] {
		if reflect.TypeOf(typ) != wanted {
			continue
		}

		// Match against the full name, not just the essential one.
		if typ.name() != name {
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

	value := reflect.Indirect(reflect.ValueOf(copyType(candidate)))
	reflect.Indirect(reflect.ValueOf(typ)).Set(value)
	return nil
}

// Handle is a reference to BTF loaded into the kernel.
type Handle struct {
	fd *internal.FD
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

	return &Handle{fd}, nil
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
	spec       *Spec
	key, value Type
}

// NewMap returns a new Map containing the given values.
// The key and value arguments are initialized to Void if nil values are given.
func NewMap(spec *Spec, key Type, value Type) Map {
	if key == nil {
		key = &Void{}
	}
	if value == nil {
		value = &Void{}
	}

	return Map{
		spec:  spec,
		key:   key,
		value: value,
	}
}

// MapSpec should be a method on Map, but is a free function
// to hide it from users of the ebpf package.
func MapSpec(m *Map) *Spec {
	return m.spec
}

// MapKey should be a method on Map, but is a free function
// to hide it from users of the ebpf package.
func MapKey(m *Map) Type {
	return m.key
}

// MapValue should be a method on Map, but is a free function
// to hide it from users of the ebpf package.
func MapValue(m *Map) Type {
	return m.value
}

// Program is the BTF information for a stream of instructions.
type Program struct {
	spec                 *Spec
	length               uint64
	funcInfos, lineInfos extInfo
	coreRelos            bpfCoreRelos
}

// ProgramSpec returns the Spec needed for loading function and line infos into the kernel.
//
// This is a free function instead of a method to hide it from users
// of package ebpf.
func ProgramSpec(s *Program) *Spec {
	return s.spec
}

// ProgramAppend the information from other to the Program.
//
// This is a free function instead of a method to hide it from users
// of package ebpf.
func ProgramAppend(s, other *Program) error {
	funcInfos, err := s.funcInfos.append(other.funcInfos, s.length)
	if err != nil {
		return fmt.Errorf("func infos: %w", err)
	}

	lineInfos, err := s.lineInfos.append(other.lineInfos, s.length)
	if err != nil {
		return fmt.Errorf("line infos: %w", err)
	}

	s.funcInfos = funcInfos
	s.lineInfos = lineInfos
	s.coreRelos = s.coreRelos.append(other.coreRelos, s.length)
	s.length += other.length
	return nil
}

// ProgramFuncInfos returns the binary form of BTF function infos.
//
// This is a free function instead of a method to hide it from users
// of package ebpf.
func ProgramFuncInfos(s *Program) (recordSize uint32, bytes []byte, err error) {
	bytes, err = s.funcInfos.MarshalBinary()
	if err != nil {
		return 0, nil, err
	}

	return s.funcInfos.recordSize, bytes, nil
}

// ProgramLineInfos returns the binary form of BTF line infos.
//
// This is a free function instead of a method to hide it from users
// of package ebpf.
func ProgramLineInfos(s *Program) (recordSize uint32, bytes []byte, err error) {
	bytes, err = s.lineInfos.MarshalBinary()
	if err != nil {
		return 0, nil, err
	}

	return s.lineInfos.recordSize, bytes, nil
}

// ProgramRelocations returns the CO-RE relocations required to adjust the
// program to the target.
//
// This is a free function instead of a method to hide it from users
// of package ebpf.
func ProgramRelocations(s *Program, target *Spec) (map[uint64]Relocation, error) {
	if len(s.coreRelos) == 0 {
		return nil, nil
	}

	return coreRelocate(s.spec, target, s.coreRelos)
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
	types.Func.SetLinkage(linkageGlobal)

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

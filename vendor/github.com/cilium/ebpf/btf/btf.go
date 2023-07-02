package btf

import (
	"bufio"
	"bytes"
	"debug/elf"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"reflect"

	"github.com/cilium/ebpf/internal"
	"github.com/cilium/ebpf/internal/sys"
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
type ID = sys.BTFID

// Spec represents decoded BTF.
type Spec struct {
	// Data from .BTF.
	rawTypes []rawType
	strings  *stringTable

	// All types contained by the spec. For the base type, the position of
	// a type in the slice is its ID.
	types types

	// Type IDs indexed by type.
	typeIDs map[Type]TypeID

	// Types indexed by essential name.
	// Includes all struct flavors and types with the same name.
	namedTypes map[essentialName][]Type

	byteOrder binary.ByteOrder
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

// typeStart returns the offset from the beginning of the .BTF section
// to the start of its type entries.
func (h *btfHeader) typeStart() int64 {
	return int64(h.HdrLen + h.TypeOff)
}

// stringStart returns the offset from the beginning of the .BTF section
// to the start of its string table.
func (h *btfHeader) stringStart() int64 {
	return int64(h.HdrLen + h.StringOff)
}

// LoadSpec opens file and calls LoadSpecFromReader on it.
func LoadSpec(file string) (*Spec, error) {
	fh, err := os.Open(file)
	if err != nil {
		return nil, err
	}
	defer fh.Close()

	return LoadSpecFromReader(fh)
}

// LoadSpecFromReader reads from an ELF or a raw BTF blob.
//
// Returns ErrNotFound if reading from an ELF which contains no BTF. ExtInfos
// may be nil.
func LoadSpecFromReader(rd io.ReaderAt) (*Spec, error) {
	file, err := internal.NewSafeELFFile(rd)
	if err != nil {
		if bo := guessRawBTFByteOrder(rd); bo != nil {
			// Try to parse a naked BTF blob. This will return an error if
			// we encounter a Datasec, since we can't fix it up.
			spec, err := loadRawSpec(io.NewSectionReader(rd, 0, math.MaxInt64), bo, nil, nil)
			return spec, err
		}

		return nil, err
	}

	return loadSpecFromELF(file)
}

// LoadSpecAndExtInfosFromReader reads from an ELF.
//
// ExtInfos may be nil if the ELF doesn't contain section metadta.
// Returns ErrNotFound if the ELF contains no BTF.
func LoadSpecAndExtInfosFromReader(rd io.ReaderAt) (*Spec, *ExtInfos, error) {
	file, err := internal.NewSafeELFFile(rd)
	if err != nil {
		return nil, nil, err
	}

	spec, err := loadSpecFromELF(file)
	if err != nil {
		return nil, nil, err
	}

	extInfos, err := loadExtInfosFromELF(file, spec.types, spec.strings)
	if err != nil && !errors.Is(err, ErrNotFound) {
		return nil, nil, err
	}

	return spec, extInfos, nil
}

// variableOffsets extracts all symbols offsets from an ELF and indexes them by
// section and variable name.
//
// References to variables in BTF data sections carry unsigned 32-bit offsets.
// Some ELF symbols (e.g. in vmlinux) may point to virtual memory that is well
// beyond this range. Since these symbols cannot be described by BTF info,
// ignore them here.
func variableOffsets(file *internal.SafeELFFile) (map[variable]uint32, error) {
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

		if symbol.Value > math.MaxUint32 {
			// VarSecinfo offset is u32, cannot reference symbols in higher regions.
			continue
		}

		if int(symbol.Section) >= len(file.Sections) {
			return nil, fmt.Errorf("symbol %s: invalid section %d", symbol.Name, symbol.Section)
		}

		secName := file.Sections[symbol.Section].Name
		variableOffsets[variable{secName, symbol.Name}] = uint32(symbol.Value)
	}

	return variableOffsets, nil
}

func loadSpecFromELF(file *internal.SafeELFFile) (*Spec, error) {
	var (
		btfSection   *elf.Section
		sectionSizes = make(map[string]uint32)
	)

	for _, sec := range file.Sections {
		switch sec.Name {
		case ".BTF":
			btfSection = sec
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

	vars, err := variableOffsets(file)
	if err != nil {
		return nil, err
	}

	if btfSection.ReaderAt == nil {
		return nil, fmt.Errorf("compressed BTF is not supported")
	}

	rawTypes, rawStrings, err := parseBTF(btfSection.ReaderAt, file.ByteOrder, nil)
	if err != nil {
		return nil, err
	}

	err = fixupDatasec(rawTypes, rawStrings, sectionSizes, vars)
	if err != nil {
		return nil, err
	}

	return inflateSpec(rawTypes, rawStrings, file.ByteOrder, nil)
}

func loadRawSpec(btf io.ReaderAt, bo binary.ByteOrder,
	baseTypes types, baseStrings *stringTable) (*Spec, error) {

	rawTypes, rawStrings, err := parseBTF(btf, bo, baseStrings)
	if err != nil {
		return nil, err
	}

	return inflateSpec(rawTypes, rawStrings, bo, baseTypes)
}

func inflateSpec(rawTypes []rawType, rawStrings *stringTable, bo binary.ByteOrder,
	baseTypes types) (*Spec, error) {

	types, err := inflateRawTypes(rawTypes, baseTypes, rawStrings)
	if err != nil {
		return nil, err
	}

	typeIDs, typesByName := indexTypes(types, TypeID(len(baseTypes)))

	return &Spec{
		rawTypes:   rawTypes,
		namedTypes: typesByName,
		typeIDs:    typeIDs,
		types:      types,
		strings:    rawStrings,
		byteOrder:  bo,
	}, nil
}

func indexTypes(types []Type, typeIDOffset TypeID) (map[Type]TypeID, map[essentialName][]Type) {
	namedTypes := 0
	for _, typ := range types {
		if typ.TypeName() != "" {
			// Do a pre-pass to figure out how big types by name has to be.
			// Most types have unique names, so it's OK to ignore essentialName
			// here.
			namedTypes++
		}
	}

	typeIDs := make(map[Type]TypeID, len(types))
	typesByName := make(map[essentialName][]Type, namedTypes)

	for i, typ := range types {
		if name := newEssentialName(typ.TypeName()); name != "" {
			typesByName[name] = append(typesByName[name], typ)
		}
		typeIDs[typ] = TypeID(i) + typeIDOffset
	}

	return typeIDs, typesByName
}

// LoadKernelSpec returns the current kernel's BTF information.
//
// Defaults to /sys/kernel/btf/vmlinux and falls back to scanning the file system
// for vmlinux ELFs. Returns an error wrapping ErrNotSupported if BTF is not enabled.
func LoadKernelSpec() (*Spec, error) {
	fh, err := os.Open("/sys/kernel/btf/vmlinux")
	if err == nil {
		defer fh.Close()

		return loadRawSpec(fh, internal.NativeEndian, nil, nil)
	}

	file, err := findVMLinux()
	if err != nil {
		return nil, err
	}
	defer file.Close()

	return loadSpecFromELF(file)
}

// findVMLinux scans multiple well-known paths for vmlinux kernel images.
func findVMLinux() (*internal.SafeELFFile, error) {
	release, err := internal.KernelRelease()
	if err != nil {
		return nil, err
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
		file, err := internal.OpenSafeELFFile(fmt.Sprintf(loc, release))
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		return file, err
	}

	return nil, fmt.Errorf("no BTF found for kernel version %s: %w", release, internal.ErrNotSupported)
}

// parseBTFHeader parses the header of the .BTF section.
func parseBTFHeader(r io.Reader, bo binary.ByteOrder) (*btfHeader, error) {
	var header btfHeader
	if err := binary.Read(r, bo, &header); err != nil {
		return nil, fmt.Errorf("can't read header: %v", err)
	}

	if header.Magic != btfMagic {
		return nil, fmt.Errorf("incorrect magic value %v", header.Magic)
	}

	if header.Version != 1 {
		return nil, fmt.Errorf("unexpected version %v", header.Version)
	}

	if header.Flags != 0 {
		return nil, fmt.Errorf("unsupported flags %v", header.Flags)
	}

	remainder := int64(header.HdrLen) - int64(binary.Size(&header))
	if remainder < 0 {
		return nil, errors.New("header length shorter than btfHeader size")
	}

	if _, err := io.CopyN(internal.DiscardZeroes{}, r, remainder); err != nil {
		return nil, fmt.Errorf("header padding: %v", err)
	}

	return &header, nil
}

func guessRawBTFByteOrder(r io.ReaderAt) binary.ByteOrder {
	buf := new(bufio.Reader)
	for _, bo := range []binary.ByteOrder{
		binary.LittleEndian,
		binary.BigEndian,
	} {
		buf.Reset(io.NewSectionReader(r, 0, math.MaxInt64))
		if _, err := parseBTFHeader(buf, bo); err == nil {
			return bo
		}
	}

	return nil
}

// parseBTF reads a .BTF section into memory and parses it into a list of
// raw types and a string table.
func parseBTF(btf io.ReaderAt, bo binary.ByteOrder, baseStrings *stringTable) ([]rawType, *stringTable, error) {
	buf := internal.NewBufferedSectionReader(btf, 0, math.MaxInt64)
	header, err := parseBTFHeader(buf, bo)
	if err != nil {
		return nil, nil, fmt.Errorf("parsing .BTF header: %v", err)
	}

	rawStrings, err := readStringTable(io.NewSectionReader(btf, header.stringStart(), int64(header.StringLen)),
		baseStrings)
	if err != nil {
		return nil, nil, fmt.Errorf("can't read type names: %w", err)
	}

	buf.Reset(io.NewSectionReader(btf, header.typeStart(), int64(header.TypeLen)))
	rawTypes, err := readTypes(buf, bo, header.TypeLen)
	if err != nil {
		return nil, nil, fmt.Errorf("can't read types: %w", err)
	}

	return rawTypes, rawStrings, nil
}

type variable struct {
	section string
	name    string
}

func fixupDatasec(rawTypes []rawType, rawStrings *stringTable, sectionSizes map[string]uint32, variableOffsets map[variable]uint32) error {
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
	types := copyTypes(s.types, nil)

	typeIDOffset := TypeID(0)
	if len(s.types) != 0 {
		typeIDOffset = s.typeIDs[s.types[0]]
	}
	typeIDs, typesByName := indexTypes(types, typeIDOffset)

	// NB: Other parts of spec are not copied since they are immutable.
	return &Spec{
		s.rawTypes,
		s.strings,
		types,
		typeIDs,
		typesByName,
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
	stringsLen := s.strings.Length()
	buf.Grow(stringsLen)
	if err := s.strings.Marshal(&buf); err != nil {
		return nil, err
	}

	// Fill out the header, and write it out.
	header = &btfHeader{
		Magic:     btfMagic,
		Version:   1,
		Flags:     0,
		HdrLen:    uint32(headerLen),
		TypeOff:   0,
		TypeLen:   typeLen,
		StringOff: typeLen,
		StringLen: uint32(stringsLen),
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

// TypeByID returns the BTF Type with the given type ID.
//
// Returns an error wrapping ErrNotFound if a Type with the given ID
// does not exist in the Spec.
func (s *Spec) TypeByID(id TypeID) (Type, error) {
	return s.types.ByID(id)
}

// TypeID returns the ID for a given Type.
//
// Returns an error wrapping ErrNoFound if the type isn't part of the Spec.
func (s *Spec) TypeID(typ Type) (TypeID, error) {
	if _, ok := typ.(*Void); ok {
		// Equality is weird for void, since it is a zero sized type.
		return 0, nil
	}

	id, ok := s.typeIDs[typ]
	if !ok {
		return 0, fmt.Errorf("no ID for type %s: %w", typ, ErrNotFound)
	}

	return id, nil
}

// AnyTypesByName returns a list of BTF Types with the given name.
//
// If the BTF blob describes multiple compilation units like vmlinux, multiple
// Types with the same name and kind can exist, but might not describe the same
// data structure.
//
// Returns an error wrapping ErrNotFound if no matching Type exists in the Spec.
func (s *Spec) AnyTypesByName(name string) ([]Type, error) {
	types := s.namedTypes[newEssentialName(name)]
	if len(types) == 0 {
		return nil, fmt.Errorf("type name %s: %w", name, ErrNotFound)
	}

	// Return a copy to prevent changes to namedTypes.
	result := make([]Type, 0, len(types))
	for _, t := range types {
		// Match against the full name, not just the essential one
		// in case the type being looked up is a struct flavor.
		if t.TypeName() == name {
			result = append(result, t)
		}
	}
	return result, nil
}

// AnyTypeByName returns a Type with the given name.
//
// Returns an error if multiple types of that name exist.
func (s *Spec) AnyTypeByName(name string) (Type, error) {
	types, err := s.AnyTypesByName(name)
	if err != nil {
		return nil, err
	}

	if len(types) > 1 {
		return nil, fmt.Errorf("found multiple types: %v", types)
	}

	return types[0], nil
}

// TypeByName searches for a Type with a specific name. Since multiple
// Types with the same name can exist, the parameter typ is taken to
// narrow down the search in case of a clash.
//
// typ must be a non-nil pointer to an implementation of a Type.
// On success, the address of the found Type will be copied to typ.
//
// Returns an error wrapping ErrNotFound if no matching
// Type exists in the Spec. If multiple candidates are found,
// an error is returned.
func (s *Spec) TypeByName(name string, typ interface{}) error {
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

	types, err := s.AnyTypesByName(name)
	if err != nil {
		return err
	}

	var candidate Type
	for _, typ := range types {
		if reflect.TypeOf(typ) != wanted {
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

// LoadSplitSpecFromReader loads split BTF from a reader.
//
// Types from base are used to resolve references in the split BTF.
// The returned Spec only contains types from the split BTF, not from the base.
func LoadSplitSpecFromReader(r io.ReaderAt, base *Spec) (*Spec, error) {
	return loadRawSpec(r, internal.NativeEndian, base.types, base.strings)
}

// TypesIterator iterates over types of a given spec.
type TypesIterator struct {
	spec  *Spec
	index int
	// The last visited type in the spec.
	Type Type
}

// Iterate returns the types iterator.
func (s *Spec) Iterate() *TypesIterator {
	return &TypesIterator{spec: s, index: 0}
}

// Next returns true as long as there are any remaining types.
func (iter *TypesIterator) Next() bool {
	if len(iter.spec.types) <= iter.index {
		return false
	}

	iter.Type = iter.spec.types[iter.index]
	iter.index++
	return true
}

// Handle is a reference to BTF loaded into the kernel.
type Handle struct {
	fd *sys.FD

	// Size of the raw BTF in bytes.
	size uint32
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

	attr := &sys.BtfLoadAttr{
		Btf:     sys.NewSlicePointer(btf),
		BtfSize: uint32(len(btf)),
	}

	fd, err := sys.BtfLoad(attr)
	if err != nil {
		logBuf := make([]byte, 64*1024)
		attr.BtfLogBuf = sys.NewSlicePointer(logBuf)
		attr.BtfLogSize = uint32(len(logBuf))
		attr.BtfLogLevel = 1
		// NB: The syscall will never return ENOSPC as of 5.18-rc4.
		_, _ = sys.BtfLoad(attr)
		return nil, internal.ErrorWithLog(err, logBuf)
	}

	return &Handle{fd, attr.BtfSize}, nil
}

// NewHandleFromID returns the BTF handle for a given id.
//
// Prefer calling [ebpf.Program.Handle] or [ebpf.Map.Handle] if possible.
//
// Returns ErrNotExist, if there is no BTF with the given id.
//
// Requires CAP_SYS_ADMIN.
func NewHandleFromID(id ID) (*Handle, error) {
	fd, err := sys.BtfGetFdById(&sys.BtfGetFdByIdAttr{
		Id: uint32(id),
	})
	if err != nil {
		return nil, fmt.Errorf("get FD for ID %d: %w", id, err)
	}

	info, err := newHandleInfoFromFD(fd)
	if err != nil {
		_ = fd.Close()
		return nil, err
	}

	return &Handle{fd, info.size}, nil
}

// Spec parses the kernel BTF into Go types.
//
// base is used to decode split BTF and may be nil.
func (h *Handle) Spec(base *Spec) (*Spec, error) {
	var btfInfo sys.BtfInfo
	btfBuffer := make([]byte, h.size)
	btfInfo.Btf, btfInfo.BtfSize = sys.NewSlicePointerLen(btfBuffer)

	if err := sys.ObjInfo(h.fd, &btfInfo); err != nil {
		return nil, err
	}

	var baseTypes types
	var baseStrings *stringTable
	if base != nil {
		baseTypes = base.types
		baseStrings = base.strings
	}

	return loadRawSpec(bytes.NewReader(btfBuffer), internal.NativeEndian, baseTypes, baseStrings)
}

// Close destroys the handle.
//
// Subsequent calls to FD will return an invalid value.
func (h *Handle) Close() error {
	if h == nil {
		return nil
	}

	return h.fd.Close()
}

// FD returns the file descriptor for the handle.
func (h *Handle) FD() int {
	return h.fd.Int()
}

// Info returns metadata about the handle.
func (h *Handle) Info() (*HandleInfo, error) {
	return newHandleInfoFromFD(h.fd)
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

	fd, err := sys.BtfLoad(&sys.BtfLoadAttr{
		Btf:     sys.NewSlicePointer(btf),
		BtfSize: uint32(len(btf)),
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

	fd, err := sys.BtfLoad(&sys.BtfLoadAttr{
		Btf:     sys.NewSlicePointer(btf),
		BtfSize: uint32(len(btf)),
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

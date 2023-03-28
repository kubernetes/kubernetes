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
	"sync"

	"github.com/cilium/ebpf/internal"
	"github.com/cilium/ebpf/internal/sys"
	"github.com/cilium/ebpf/internal/unix"
)

const btfMagic = 0xeB9F

// Errors returned by BTF functions.
var (
	ErrNotSupported    = internal.ErrNotSupported
	ErrNotFound        = errors.New("not found")
	ErrNoExtendedInfo  = errors.New("no extended info")
	ErrMultipleMatches = errors.New("multiple matching types")
)

// ID represents the unique ID of a BTF object.
type ID = sys.BTFID

// Spec represents decoded BTF.
type Spec struct {
	// Data from .BTF.
	strings *stringTable

	// All types contained by the spec, not including types from the base in
	// case the spec was parsed from split BTF.
	types []Type

	// Type IDs indexed by type.
	typeIDs map[Type]TypeID

	// Types indexed by essential name.
	// Includes all struct flavors and types with the same name.
	namedTypes map[essentialName][]Type

	byteOrder binary.ByteOrder
}

var btfHeaderLen = binary.Size(&btfHeader{})

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
			return loadRawSpec(io.NewSectionReader(rd, 0, math.MaxInt64), bo, nil, nil)
		}

		return nil, err
	}

	return loadSpecFromELF(file)
}

// LoadSpecAndExtInfosFromReader reads from an ELF.
//
// ExtInfos may be nil if the ELF doesn't contain section metadata.
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

// symbolOffsets extracts all symbols offsets from an ELF and indexes them by
// section and variable name.
//
// References to variables in BTF data sections carry unsigned 32-bit offsets.
// Some ELF symbols (e.g. in vmlinux) may point to virtual memory that is well
// beyond this range. Since these symbols cannot be described by BTF info,
// ignore them here.
func symbolOffsets(file *internal.SafeELFFile) (map[symbol]uint32, error) {
	symbols, err := file.Symbols()
	if err != nil {
		return nil, fmt.Errorf("can't read symbols: %v", err)
	}

	offsets := make(map[symbol]uint32)
	for _, sym := range symbols {
		if idx := sym.Section; idx >= elf.SHN_LORESERVE && idx <= elf.SHN_HIRESERVE {
			// Ignore things like SHN_ABS
			continue
		}

		if sym.Value > math.MaxUint32 {
			// VarSecinfo offset is u32, cannot reference symbols in higher regions.
			continue
		}

		if int(sym.Section) >= len(file.Sections) {
			return nil, fmt.Errorf("symbol %s: invalid section %d", sym.Name, sym.Section)
		}

		secName := file.Sections[sym.Section].Name
		offsets[symbol{secName, sym.Name}] = uint32(sym.Value)
	}

	return offsets, nil
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

	offsets, err := symbolOffsets(file)
	if err != nil {
		return nil, err
	}

	if btfSection.ReaderAt == nil {
		return nil, fmt.Errorf("compressed BTF is not supported")
	}

	spec, err := loadRawSpec(btfSection.ReaderAt, file.ByteOrder, nil, nil)
	if err != nil {
		return nil, err
	}

	err = fixupDatasec(spec.types, sectionSizes, offsets)
	if err != nil {
		return nil, err
	}

	return spec, nil
}

func loadRawSpec(btf io.ReaderAt, bo binary.ByteOrder,
	baseTypes types, baseStrings *stringTable) (*Spec, error) {

	rawTypes, rawStrings, err := parseBTF(btf, bo, baseStrings)
	if err != nil {
		return nil, err
	}

	types, err := inflateRawTypes(rawTypes, baseTypes, rawStrings)
	if err != nil {
		return nil, err
	}

	typeIDs, typesByName := indexTypes(types, TypeID(len(baseTypes)))

	return &Spec{
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
	spec, _, err := kernelSpec()
	return spec, err
}

var kernelBTF struct {
	sync.RWMutex
	spec *Spec
	// True if the spec was read from an ELF instead of raw BTF in /sys.
	fallback bool
}

// FlushKernelSpec removes any cached kernel type information.
func FlushKernelSpec() {
	kernelBTF.Lock()
	defer kernelBTF.Unlock()

	kernelBTF.spec, kernelBTF.fallback = nil, false
}

func kernelSpec() (*Spec, bool, error) {
	kernelBTF.RLock()
	spec, fallback := kernelBTF.spec, kernelBTF.fallback
	kernelBTF.RUnlock()

	if spec == nil {
		kernelBTF.Lock()
		defer kernelBTF.Unlock()

		spec, fallback = kernelBTF.spec, kernelBTF.fallback
	}

	if spec != nil {
		return spec.Copy(), fallback, nil
	}

	spec, fallback, err := loadKernelSpec()
	if err != nil {
		return nil, false, err
	}

	kernelBTF.spec, kernelBTF.fallback = spec, fallback
	return spec.Copy(), fallback, nil
}

func loadKernelSpec() (_ *Spec, fallback bool, _ error) {
	fh, err := os.Open("/sys/kernel/btf/vmlinux")
	if err == nil {
		defer fh.Close()

		spec, err := loadRawSpec(fh, internal.NativeEndian, nil, nil)
		return spec, false, err
	}

	file, err := findVMLinux()
	if err != nil {
		return nil, false, err
	}
	defer file.Close()

	spec, err := loadSpecFromELF(file)
	return spec, true, err
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

type symbol struct {
	section string
	name    string
}

func fixupDatasec(types []Type, sectionSizes map[string]uint32, offsets map[symbol]uint32) error {
	for _, typ := range types {
		ds, ok := typ.(*Datasec)
		if !ok {
			continue
		}

		name := ds.Name
		if name == ".kconfig" || name == ".ksyms" {
			return fmt.Errorf("reference to %s: %w", name, ErrNotSupported)
		}

		if ds.Size != 0 {
			continue
		}

		ds.Size, ok = sectionSizes[name]
		if !ok {
			return fmt.Errorf("data section %s: missing size", name)
		}

		for i := range ds.Vars {
			symName := ds.Vars[i].Type.TypeName()
			ds.Vars[i].Offset, ok = offsets[symbol{name, symName}]
			if !ok {
				return fmt.Errorf("data section %s: missing offset for symbol %s", name, symName)
			}
		}
	}

	return nil
}

// Copy creates a copy of Spec.
func (s *Spec) Copy() *Spec {
	types := copyTypes(s.types, nil)

	typeIDs, typesByName := indexTypes(types, s.firstTypeID())

	// NB: Other parts of spec are not copied since they are immutable.
	return &Spec{
		s.strings,
		types,
		typeIDs,
		typesByName,
		s.byteOrder,
	}
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
	firstID := s.firstTypeID()
	lastID := firstID + TypeID(len(s.types))

	if id < firstID || id >= lastID {
		return nil, fmt.Errorf("expected type ID between %d and %d, got %d: %w", firstID, lastID, id, ErrNotFound)
	}

	return s.types[id-firstID], nil
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

// TypeByName searches for a Type with a specific name. Since multiple Types
// with the same name can exist, the parameter typ is taken to narrow down the
// search in case of a clash.
//
// typ must be a non-nil pointer to an implementation of a Type. On success, the
// address of the found Type will be copied to typ.
//
// Returns an error wrapping ErrNotFound if no matching Type exists in the Spec.
// Returns an error wrapping ErrMultipleTypes if multiple candidates are found.
func (s *Spec) TypeByName(name string, typ interface{}) error {
	typeInterface := reflect.TypeOf((*Type)(nil)).Elem()

	// typ may be **T or *Type
	typValue := reflect.ValueOf(typ)
	if typValue.Kind() != reflect.Ptr {
		return fmt.Errorf("%T is not a pointer", typ)
	}

	typPtr := typValue.Elem()
	if !typPtr.CanSet() {
		return fmt.Errorf("%T cannot be set", typ)
	}

	wanted := typPtr.Type()
	if wanted == typeInterface {
		// This is *Type. Unwrap the value's type.
		wanted = typPtr.Elem().Type()
	}

	if !wanted.AssignableTo(typeInterface) {
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
			return fmt.Errorf("type %s(%T): %w", name, typ, ErrMultipleMatches)
		}

		candidate = typ
	}

	if candidate == nil {
		return fmt.Errorf("%s %s: %w", wanted, name, ErrNotFound)
	}

	typPtr.Set(reflect.ValueOf(candidate))

	return nil
}

// firstTypeID returns the first type ID or zero.
func (s *Spec) firstTypeID() TypeID {
	if len(s.types) > 0 {
		return s.typeIDs[s.types[0]]
	}
	return 0
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

// haveBTF attempts to load a BTF blob containing an Int. It should pass on any
// kernel that supports BPF_BTF_LOAD.
var haveBTF = internal.NewFeatureTest("BTF", "4.18", func() error {
	var (
		types struct {
			Integer btfType
			btfInt
		}
		strings = []byte{0}
	)
	types.Integer.SetKind(kindInt) // 0-length anonymous integer

	btf := marshalBTF(&types, strings, internal.NativeEndian)

	fd, err := sys.BtfLoad(&sys.BtfLoadAttr{
		Btf:     sys.NewSlicePointer(btf),
		BtfSize: uint32(len(btf)),
	})
	if errors.Is(err, unix.EINVAL) || errors.Is(err, unix.EPERM) {
		return internal.ErrNotSupported
	}
	if err != nil {
		return err
	}

	fd.Close()
	return nil
})

// haveMapBTF attempts to load a minimal BTF blob containing a Var. It is
// used as a proxy for .bss, .data and .rodata map support, which generally
// come with a Var and Datasec. These were introduced in Linux 5.2.
var haveMapBTF = internal.NewFeatureTest("Map BTF (Var/Datasec)", "5.2", func() error {
	if err := haveBTF(); err != nil {
		return err
	}

	var (
		types struct {
			Integer btfType
			Var     btfType
			btfVariable
		}
		strings = []byte{0, 'a', 0}
	)

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
		// Treat both EINVAL and EPERM as not supported: creating the map may still
		// succeed without Btf* attrs.
		return internal.ErrNotSupported
	}
	if err != nil {
		return err
	}

	fd.Close()
	return nil
})

// haveProgBTF attempts to load a BTF blob containing a Func and FuncProto. It
// is used as a proxy for ext_info (func_info) support, which depends on
// Func(Proto) by definition.
var haveProgBTF = internal.NewFeatureTest("Program BTF (func/line_info)", "5.0", func() error {
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

	btf := marshalBTF(&types, strings, internal.NativeEndian)

	fd, err := sys.BtfLoad(&sys.BtfLoadAttr{
		Btf:     sys.NewSlicePointer(btf),
		BtfSize: uint32(len(btf)),
	})
	if errors.Is(err, unix.EINVAL) || errors.Is(err, unix.EPERM) {
		return internal.ErrNotSupported
	}
	if err != nil {
		return err
	}

	fd.Close()
	return nil
})

var haveFuncLinkage = internal.NewFeatureTest("BTF func linkage", "5.6", func() error {
	if err := haveProgBTF(); err != nil {
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

package btf

import (
	"bufio"
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

// Spec allows querying a set of Types and loading the set into the
// kernel.
type Spec struct {
	// All types contained by the spec, not including types from the base in
	// case the spec was parsed from split BTF.
	types []Type

	// Type IDs indexed by type.
	typeIDs map[Type]TypeID

	// The ID of the first type in types.
	firstTypeID TypeID

	// Types indexed by essential name.
	// Includes all struct flavors and types with the same name.
	namedTypes map[essentialName][]Type

	// String table from ELF, may be nil.
	strings *stringTable

	// Byte order of the ELF we decoded the spec from, may be nil.
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

// newSpec creates a Spec containing only Void.
func newSpec() *Spec {
	return &Spec{
		[]Type{(*Void)(nil)},
		map[Type]TypeID{(*Void)(nil): 0},
		0,
		make(map[essentialName][]Type),
		nil,
		nil,
	}
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
			return loadRawSpec(io.NewSectionReader(rd, 0, math.MaxInt64), bo, nil)
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

	extInfos, err := loadExtInfosFromELF(file, spec)
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

	spec, err := loadRawSpec(btfSection.ReaderAt, file.ByteOrder, nil)
	if err != nil {
		return nil, err
	}

	err = fixupDatasec(spec.types, sectionSizes, offsets)
	if err != nil {
		return nil, err
	}

	return spec, nil
}

func loadRawSpec(btf io.ReaderAt, bo binary.ByteOrder, base *Spec) (*Spec, error) {
	var (
		baseStrings *stringTable
		firstTypeID TypeID
		err         error
	)

	if base != nil {
		if base.firstTypeID != 0 {
			return nil, fmt.Errorf("can't use split BTF as base")
		}

		if base.strings == nil {
			return nil, fmt.Errorf("parse split BTF: base must be loaded from an ELF")
		}

		baseStrings = base.strings

		firstTypeID, err = base.nextTypeID()
		if err != nil {
			return nil, err
		}
	}

	rawTypes, rawStrings, err := parseBTF(btf, bo, baseStrings)
	if err != nil {
		return nil, err
	}

	types, err := inflateRawTypes(rawTypes, rawStrings, base)
	if err != nil {
		return nil, err
	}

	typeIDs, typesByName := indexTypes(types, firstTypeID)

	return &Spec{
		namedTypes:  typesByName,
		typeIDs:     typeIDs,
		types:       types,
		firstTypeID: firstTypeID,
		strings:     rawStrings,
		byteOrder:   bo,
	}, nil
}

func indexTypes(types []Type, firstTypeID TypeID) (map[Type]TypeID, map[essentialName][]Type) {
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
		typeIDs[typ] = firstTypeID + TypeID(i)
	}

	return typeIDs, typesByName
}

// LoadKernelSpec returns the current kernel's BTF information.
//
// Defaults to /sys/kernel/btf/vmlinux and falls back to scanning the file system
// for vmlinux ELFs. Returns an error wrapping ErrNotSupported if BTF is not enabled.
func LoadKernelSpec() (*Spec, error) {
	spec, _, err := kernelSpec()
	if err != nil {
		return nil, err
	}
	return spec.Copy(), nil
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
		return spec, fallback, nil
	}

	spec, fallback, err := loadKernelSpec()
	if err != nil {
		return nil, false, err
	}

	kernelBTF.spec, kernelBTF.fallback = spec, fallback
	return spec, fallback, nil
}

func loadKernelSpec() (_ *Spec, fallback bool, _ error) {
	fh, err := os.Open("/sys/kernel/btf/vmlinux")
	if err == nil {
		defer fh.Close()

		spec, err := loadRawSpec(fh, internal.NativeEndian, nil)
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

// fixupDatasec attempts to patch up missing info in Datasecs and its members by
// supplementing them with information from the ELF headers and symbol table.
func fixupDatasec(types []Type, sectionSizes map[string]uint32, offsets map[symbol]uint32) error {
	for _, typ := range types {
		ds, ok := typ.(*Datasec)
		if !ok {
			continue
		}

		name := ds.Name

		// Some Datasecs are virtual and don't have corresponding ELF sections.
		switch name {
		case ".ksyms":
			// .ksyms describes forward declarations of kfunc signatures.
			// Nothing to fix up, all sizes and offsets are 0.
			for _, vsi := range ds.Vars {
				_, ok := vsi.Type.(*Func)
				if !ok {
					// Only Funcs are supported in the .ksyms Datasec.
					return fmt.Errorf("data section %s: expected *btf.Func, not %T: %w", name, vsi.Type, ErrNotSupported)
				}
			}

			continue
		case ".kconfig":
			// .kconfig has a size of 0 and has all members' offsets set to 0.
			// Fix up all offsets and set the Datasec's size.
			if err := fixupDatasecLayout(ds); err != nil {
				return err
			}

			// Fix up extern to global linkage to avoid a BTF verifier error.
			for _, vsi := range ds.Vars {
				vsi.Type.(*Var).Linkage = GlobalVar
			}

			continue
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

// fixupDatasecLayout populates ds.Vars[].Offset according to var sizes and
// alignment. Calculate and set ds.Size.
func fixupDatasecLayout(ds *Datasec) error {
	var off uint32

	for i, vsi := range ds.Vars {
		v, ok := vsi.Type.(*Var)
		if !ok {
			return fmt.Errorf("member %d: unsupported type %T", i, vsi.Type)
		}

		size, err := Sizeof(v.Type)
		if err != nil {
			return fmt.Errorf("variable %s: getting size: %w", v.Name, err)
		}
		align, err := alignof(v.Type)
		if err != nil {
			return fmt.Errorf("variable %s: getting alignment: %w", v.Name, err)
		}

		// Align the current member based on the offset of the end of the previous
		// member and the alignment of the current member.
		off = internal.Align(off, uint32(align))

		ds.Vars[i].Offset = off

		off += uint32(size)
	}

	ds.Size = off

	return nil
}

// Copy creates a copy of Spec.
func (s *Spec) Copy() *Spec {
	types := copyTypes(s.types, nil)
	typeIDs, typesByName := indexTypes(types, s.firstTypeID)

	// NB: Other parts of spec are not copied since they are immutable.
	return &Spec{
		types,
		typeIDs,
		s.firstTypeID,
		typesByName,
		s.strings,
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

// nextTypeID returns the next unallocated type ID or an error if there are no
// more type IDs.
func (s *Spec) nextTypeID() (TypeID, error) {
	id := s.firstTypeID + TypeID(len(s.types))
	if id < s.firstTypeID {
		return 0, fmt.Errorf("no more type IDs")
	}
	return id, nil
}

// TypeByID returns the BTF Type with the given type ID.
//
// Returns an error wrapping ErrNotFound if a Type with the given ID
// does not exist in the Spec.
func (s *Spec) TypeByID(id TypeID) (Type, error) {
	if id < s.firstTypeID {
		return nil, fmt.Errorf("look up type with ID %d (first ID is %d): %w", id, s.firstTypeID, ErrNotFound)
	}

	index := int(id - s.firstTypeID)
	if index >= len(s.types) {
		return nil, fmt.Errorf("look up type with ID %d: %w", id, ErrNotFound)
	}

	return s.types[index], nil
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

// LoadSplitSpecFromReader loads split BTF from a reader.
//
// Types from base are used to resolve references in the split BTF.
// The returned Spec only contains types from the split BTF, not from the base.
func LoadSplitSpecFromReader(r io.ReaderAt, base *Spec) (*Spec, error) {
	return loadRawSpec(r, internal.NativeEndian, base)
}

// TypesIterator iterates over types of a given spec.
type TypesIterator struct {
	types []Type
	index int
	// The last visited type in the spec.
	Type Type
}

// Iterate returns the types iterator.
func (s *Spec) Iterate() *TypesIterator {
	// We share the backing array of types with the Spec. This is safe since
	// we don't allow deletion or shuffling of types.
	return &TypesIterator{types: s.types, index: 0}
}

// Next returns true as long as there are any remaining types.
func (iter *TypesIterator) Next() bool {
	if len(iter.types) <= iter.index {
		return false
	}

	iter.Type = iter.types[iter.index]
	iter.index++
	return true
}

// haveBTF attempts to load a BTF blob containing an Int. It should pass on any
// kernel that supports BPF_BTF_LOAD.
var haveBTF = internal.NewFeatureTest("BTF", "4.18", func() error {
	// 0-length anonymous integer
	err := probeBTF(&Int{})
	if errors.Is(err, unix.EINVAL) || errors.Is(err, unix.EPERM) {
		return internal.ErrNotSupported
	}
	return err
})

// haveMapBTF attempts to load a minimal BTF blob containing a Var. It is
// used as a proxy for .bss, .data and .rodata map support, which generally
// come with a Var and Datasec. These were introduced in Linux 5.2.
var haveMapBTF = internal.NewFeatureTest("Map BTF (Var/Datasec)", "5.2", func() error {
	if err := haveBTF(); err != nil {
		return err
	}

	v := &Var{
		Name: "a",
		Type: &Pointer{(*Void)(nil)},
	}

	err := probeBTF(v)
	if errors.Is(err, unix.EINVAL) || errors.Is(err, unix.EPERM) {
		// Treat both EINVAL and EPERM as not supported: creating the map may still
		// succeed without Btf* attrs.
		return internal.ErrNotSupported
	}
	return err
})

// haveProgBTF attempts to load a BTF blob containing a Func and FuncProto. It
// is used as a proxy for ext_info (func_info) support, which depends on
// Func(Proto) by definition.
var haveProgBTF = internal.NewFeatureTest("Program BTF (func/line_info)", "5.0", func() error {
	if err := haveBTF(); err != nil {
		return err
	}

	fn := &Func{
		Name: "a",
		Type: &FuncProto{Return: (*Void)(nil)},
	}

	err := probeBTF(fn)
	if errors.Is(err, unix.EINVAL) || errors.Is(err, unix.EPERM) {
		return internal.ErrNotSupported
	}
	return err
})

var haveFuncLinkage = internal.NewFeatureTest("BTF func linkage", "5.6", func() error {
	if err := haveProgBTF(); err != nil {
		return err
	}

	fn := &Func{
		Name:    "a",
		Type:    &FuncProto{Return: (*Void)(nil)},
		Linkage: GlobalFunc,
	}

	err := probeBTF(fn)
	if errors.Is(err, unix.EINVAL) {
		return internal.ErrNotSupported
	}
	return err
})

func probeBTF(typ Type) error {
	b, err := NewBuilder([]Type{typ})
	if err != nil {
		return err
	}

	buf, err := b.Marshal(nil, nil)
	if err != nil {
		return err
	}

	fd, err := sys.BtfLoad(&sys.BtfLoadAttr{
		Btf:     sys.NewSlicePointer(buf),
		BtfSize: uint32(len(buf)),
	})

	if err == nil {
		fd.Close()
	}

	return err
}

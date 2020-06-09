package btf

import (
	"bytes"
	"debug/elf"
	"encoding/binary"
	"io"
	"io/ioutil"
	"math"
	"reflect"
	"unsafe"

	"github.com/cilium/ebpf/internal"
	"github.com/cilium/ebpf/internal/unix"

	"golang.org/x/xerrors"
)

const btfMagic = 0xeB9F

// Errors returned by BTF functions.
var (
	ErrNotSupported = internal.ErrNotSupported
)

// Spec represents decoded BTF.
type Spec struct {
	rawTypes  []rawType
	strings   stringTable
	types     map[string][]Type
	funcInfos map[string]extInfo
	lineInfos map[string]extInfo
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

// LoadSpecFromReader reads BTF sections from an ELF.
//
// Returns a nil Spec and no error if no BTF was present.
func LoadSpecFromReader(rd io.ReaderAt) (*Spec, error) {
	file, err := elf.NewFile(rd)
	if err != nil {
		return nil, err
	}
	defer file.Close()

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
				return nil, xerrors.Errorf("section %s exceeds maximum size", sec.Name)
			}

			sectionSizes[sec.Name] = uint32(sec.Size)
		}
	}

	if btfSection == nil {
		return nil, nil
	}

	symbols, err := file.Symbols()
	if err != nil {
		return nil, xerrors.Errorf("can't read symbols: %v", err)
	}

	variableOffsets := make(map[variable]uint32)
	for _, symbol := range symbols {
		if idx := symbol.Section; idx >= elf.SHN_LORESERVE && idx <= elf.SHN_HIRESERVE {
			// Ignore things like SHN_ABS
			continue
		}

		secName := file.Sections[symbol.Section].Name
		if _, ok := sectionSizes[secName]; !ok {
			continue
		}

		if symbol.Value > math.MaxUint32 {
			return nil, xerrors.Errorf("section %s: symbol %s: size exceeds maximum", secName, symbol.Name)
		}

		variableOffsets[variable{secName, symbol.Name}] = uint32(symbol.Value)
	}

	rawTypes, rawStrings, err := parseBTF(btfSection.Open(), file.ByteOrder)
	if err != nil {
		return nil, err
	}

	err = fixupDatasec(rawTypes, rawStrings, sectionSizes, variableOffsets)
	if err != nil {
		return nil, err
	}

	types, err := inflateRawTypes(rawTypes, rawStrings)
	if err != nil {
		return nil, err
	}

	var (
		funcInfos = make(map[string]extInfo)
		lineInfos = make(map[string]extInfo)
	)
	if btfExtSection != nil {
		funcInfos, lineInfos, err = parseExtInfos(btfExtSection.Open(), file.ByteOrder, rawStrings)
		if err != nil {
			return nil, xerrors.Errorf("can't read ext info: %w", err)
		}
	}

	return &Spec{
		rawTypes:  rawTypes,
		types:     types,
		strings:   rawStrings,
		funcInfos: funcInfos,
		lineInfos: lineInfos,
		byteOrder: file.ByteOrder,
	}, nil
}

func parseBTF(btf io.ReadSeeker, bo binary.ByteOrder) ([]rawType, stringTable, error) {
	rawBTF, err := ioutil.ReadAll(btf)
	if err != nil {
		return nil, nil, xerrors.Errorf("can't read BTF: %v", err)
	}

	rd := bytes.NewReader(rawBTF)

	var header btfHeader
	if err := binary.Read(rd, bo, &header); err != nil {
		return nil, nil, xerrors.Errorf("can't read header: %v", err)
	}

	if header.Magic != btfMagic {
		return nil, nil, xerrors.Errorf("incorrect magic value %v", header.Magic)
	}

	if header.Version != 1 {
		return nil, nil, xerrors.Errorf("unexpected version %v", header.Version)
	}

	if header.Flags != 0 {
		return nil, nil, xerrors.Errorf("unsupported flags %v", header.Flags)
	}

	remainder := int64(header.HdrLen) - int64(binary.Size(&header))
	if remainder < 0 {
		return nil, nil, xerrors.New("header is too short")
	}

	if _, err := io.CopyN(internal.DiscardZeroes{}, rd, remainder); err != nil {
		return nil, nil, xerrors.Errorf("header padding: %v", err)
	}

	if _, err := rd.Seek(int64(header.HdrLen+header.StringOff), io.SeekStart); err != nil {
		return nil, nil, xerrors.Errorf("can't seek to start of string section: %v", err)
	}

	rawStrings, err := readStringTable(io.LimitReader(rd, int64(header.StringLen)))
	if err != nil {
		return nil, nil, xerrors.Errorf("can't read type names: %w", err)
	}

	if _, err := rd.Seek(int64(header.HdrLen+header.TypeOff), io.SeekStart); err != nil {
		return nil, nil, xerrors.Errorf("can't seek to start of type section: %v", err)
	}

	rawTypes, err := readTypes(io.LimitReader(rd, int64(header.TypeLen)), bo)
	if err != nil {
		return nil, nil, xerrors.Errorf("can't read types: %w", err)
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

		size, ok := sectionSizes[name]
		if !ok {
			return xerrors.Errorf("data section %s: missing size", name)
		}

		rawTypes[i].SizeType = size

		secinfos := rawType.data.([]btfVarSecinfo)
		for j, secInfo := range secinfos {
			id := int(secInfo.Type - 1)
			if id >= len(rawTypes) {
				return xerrors.Errorf("data section %s: invalid type id %d for variable %d", name, id, j)
			}

			varName, err := rawStrings.Lookup(rawTypes[id].NameOff)
			if err != nil {
				return xerrors.Errorf("data section %s: can't get name for type %d: %w", name, id, err)
			}

			offset, ok := variableOffsets[variable{name, varName}]
			if !ok {
				return xerrors.Errorf("data section %s: missing offset for variable %s", name, varName)
			}

			secinfos[j].Offset = offset
		}
	}

	return nil
}

func (s *Spec) marshal(bo binary.ByteOrder) ([]byte, error) {
	var (
		buf       bytes.Buffer
		header    = new(btfHeader)
		headerLen = binary.Size(header)
	)

	// Reserve space for the header. We have to write it last since
	// we don't know the size of the type section yet.
	_, _ = buf.Write(make([]byte, headerLen))

	// Write type section, just after the header.
	for _, typ := range s.rawTypes {
		if err := typ.Marshal(&buf, bo); err != nil {
			return nil, xerrors.Errorf("can't marshal BTF: %w", err)
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
	err := binary.Write(sliceWriter(raw[:headerLen]), bo, header)
	if err != nil {
		return nil, xerrors.Errorf("can't write header: %v", err)
	}

	return raw, nil
}

type sliceWriter []byte

func (sw sliceWriter) Write(p []byte) (int, error) {
	if len(p) != len(sw) {
		return 0, xerrors.New("size doesn't match")
	}

	return copy(sw, p), nil
}

// Program finds the BTF for a specific section.
//
// Length is the number of bytes in the raw BPF instruction stream.
//
// Returns an error if there is no BTF.
func (s *Spec) Program(name string, length uint64) (*Program, error) {
	if length == 0 {
		return nil, xerrors.New("length musn't be zero")
	}

	funcInfos, funcOK := s.funcInfos[name]
	lineInfos, lineOK := s.lineInfos[name]

	if !funcOK && !lineOK {
		return nil, xerrors.Errorf("no BTF for program %s", name)
	}

	return &Program{s, length, funcInfos, lineInfos}, nil
}

// Map finds the BTF for a map.
//
// Returns an error if there is no BTF for the given name.
func (s *Spec) Map(name string) (*Map, []Member, error) {
	var mapVar Var
	if err := s.FindType(name, &mapVar); err != nil {
		return nil, nil, err
	}

	mapStruct, ok := mapVar.Type.(*Struct)
	if !ok {
		return nil, nil, xerrors.Errorf("expected struct, have %s", mapVar.Type)
	}

	var key, value Type
	for _, member := range mapStruct.Members {
		switch member.Name {
		case "key":
			key = member.Type

		case "value":
			value = member.Type
		}
	}

	if key == nil {
		return nil, nil, xerrors.Errorf("map %s: missing 'key' in type", name)
	}

	if value == nil {
		return nil, nil, xerrors.Errorf("map %s: missing 'value' in type", name)
	}

	return &Map{s, key, value}, mapStruct.Members, nil
}

// Datasec returns the BTF required to create maps which represent data sections.
func (s *Spec) Datasec(name string) (*Map, error) {
	var datasec Datasec
	if err := s.FindType(name, &datasec); err != nil {
		return nil, xerrors.Errorf("data section %s: can't get BTF: %w", name, err)
	}

	return &Map{s, &Void{}, &datasec}, nil
}

var errNotFound = xerrors.New("not found")

// FindType searches for a type with a specific name.
//
// hint determines the type of the returned Type.
//
// Returns an error if there is no or multiple matches.
func (s *Spec) FindType(name string, typ Type) error {
	var (
		wanted    = reflect.TypeOf(typ)
		candidate Type
	)

	for _, typ := range s.types[name] {
		if reflect.TypeOf(typ) != wanted {
			continue
		}

		if candidate != nil {
			return xerrors.Errorf("type %s: multiple candidates for %T", name, typ)
		}

		candidate = typ
	}

	if candidate == nil {
		return xerrors.Errorf("type %s: %w", name, errNotFound)
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
		return nil, xerrors.Errorf("can't load %s BTF on %s", spec.byteOrder, internal.NativeEndian)
	}

	btf, err := spec.marshal(internal.NativeEndian)
	if err != nil {
		return nil, xerrors.Errorf("can't marshal BTF: %w", err)
	}

	if uint64(len(btf)) > math.MaxUint32 {
		return nil, xerrors.New("BTF exceeds the maximum size")
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
		return xerrors.Errorf("func infos: %w", err)
	}

	lineInfos, err := s.lineInfos.append(other.lineInfos, s.length)
	if err != nil {
		return xerrors.Errorf("line infos: %w", err)
	}

	s.length += other.length
	s.funcInfos = funcInfos
	s.lineInfos = lineInfos
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

type bpfLoadBTFAttr struct {
	btf         internal.Pointer
	logBuf      internal.Pointer
	btfSize     uint32
	btfLogSize  uint32
	btfLogLevel uint32
}

func bpfLoadBTF(attr *bpfLoadBTFAttr) (*internal.FD, error) {
	const _BTFLoad = 18

	fd, err := internal.BPF(_BTFLoad, unsafe.Pointer(attr), unsafe.Sizeof(*attr))
	if err != nil {
		return nil, err
	}

	return internal.NewFD(uint32(fd)), nil
}

func minimalBTF(bo binary.ByteOrder) []byte {
	const minHeaderLength = 24

	var (
		types struct {
			Integer btfType
			Var     btfType
			btfVar  struct{ Linkage uint32 }
		}
		typLen  = uint32(binary.Size(&types))
		strings = []byte{0, 'a', 0}
		header  = btfHeader{
			Magic:     btfMagic,
			Version:   1,
			HdrLen:    minHeaderLength,
			TypeOff:   0,
			TypeLen:   typLen,
			StringOff: typLen,
			StringLen: uint32(len(strings)),
		}
	)

	// We use a BTF_KIND_VAR here, to make sure that
	// the kernel understands BTF at least as well as we
	// do. BTF_KIND_VAR was introduced ~5.1.
	types.Integer.SetKind(kindPointer)
	types.Var.NameOff = 1
	types.Var.SetKind(kindVar)
	types.Var.SizeType = 1

	buf := new(bytes.Buffer)
	_ = binary.Write(buf, bo, &header)
	_ = binary.Write(buf, bo, &types)
	buf.Write(strings)

	return buf.Bytes()
}

var haveBTF = internal.FeatureTest("BTF", "5.1", func() bool {
	btf := minimalBTF(internal.NativeEndian)
	fd, err := bpfLoadBTF(&bpfLoadBTFAttr{
		btf:     internal.NewSlicePointer(btf),
		btfSize: uint32(len(btf)),
	})
	if err == nil {
		fd.Close()
	}
	// Check for EINVAL specifically, rather than err != nil since we
	// otherwise misdetect due to insufficient permissions.
	return !xerrors.Is(err, unix.EINVAL)
})

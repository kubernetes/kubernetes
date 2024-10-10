package btf

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"sort"

	"github.com/cilium/ebpf/asm"
	"github.com/cilium/ebpf/internal"
)

// ExtInfos contains ELF section metadata.
type ExtInfos struct {
	// The slices are sorted by offset in ascending order.
	funcInfos       map[string]FuncInfos
	lineInfos       map[string]LineInfos
	relocationInfos map[string]CORERelocationInfos
}

// loadExtInfosFromELF parses ext infos from the .BTF.ext section in an ELF.
//
// Returns an error wrapping ErrNotFound if no ext infos are present.
func loadExtInfosFromELF(file *internal.SafeELFFile, spec *Spec) (*ExtInfos, error) {
	section := file.Section(".BTF.ext")
	if section == nil {
		return nil, fmt.Errorf("btf ext infos: %w", ErrNotFound)
	}

	if section.ReaderAt == nil {
		return nil, fmt.Errorf("compressed ext_info is not supported")
	}

	return loadExtInfos(section.ReaderAt, file.ByteOrder, spec)
}

// loadExtInfos parses bare ext infos.
func loadExtInfos(r io.ReaderAt, bo binary.ByteOrder, spec *Spec) (*ExtInfos, error) {
	// Open unbuffered section reader. binary.Read() calls io.ReadFull on
	// the header structs, resulting in one syscall per header.
	headerRd := io.NewSectionReader(r, 0, math.MaxInt64)
	extHeader, err := parseBTFExtHeader(headerRd, bo)
	if err != nil {
		return nil, fmt.Errorf("parsing BTF extension header: %w", err)
	}

	coreHeader, err := parseBTFExtCOREHeader(headerRd, bo, extHeader)
	if err != nil {
		return nil, fmt.Errorf("parsing BTF CO-RE header: %w", err)
	}

	buf := internal.NewBufferedSectionReader(r, extHeader.funcInfoStart(), int64(extHeader.FuncInfoLen))
	btfFuncInfos, err := parseFuncInfos(buf, bo, spec.strings)
	if err != nil {
		return nil, fmt.Errorf("parsing BTF function info: %w", err)
	}

	funcInfos := make(map[string]FuncInfos, len(btfFuncInfos))
	for section, bfis := range btfFuncInfos {
		funcInfos[section], err = newFuncInfos(bfis, spec)
		if err != nil {
			return nil, fmt.Errorf("section %s: func infos: %w", section, err)
		}
	}

	buf = internal.NewBufferedSectionReader(r, extHeader.lineInfoStart(), int64(extHeader.LineInfoLen))
	btfLineInfos, err := parseLineInfos(buf, bo, spec.strings)
	if err != nil {
		return nil, fmt.Errorf("parsing BTF line info: %w", err)
	}

	lineInfos := make(map[string]LineInfos, len(btfLineInfos))
	for section, blis := range btfLineInfos {
		lineInfos[section], err = newLineInfos(blis, spec.strings)
		if err != nil {
			return nil, fmt.Errorf("section %s: line infos: %w", section, err)
		}
	}

	if coreHeader == nil || coreHeader.COREReloLen == 0 {
		return &ExtInfos{funcInfos, lineInfos, nil}, nil
	}

	var btfCORERelos map[string][]bpfCORERelo
	buf = internal.NewBufferedSectionReader(r, extHeader.coreReloStart(coreHeader), int64(coreHeader.COREReloLen))
	btfCORERelos, err = parseCORERelos(buf, bo, spec.strings)
	if err != nil {
		return nil, fmt.Errorf("parsing CO-RE relocation info: %w", err)
	}

	coreRelos := make(map[string]CORERelocationInfos, len(btfCORERelos))
	for section, brs := range btfCORERelos {
		coreRelos[section], err = newRelocationInfos(brs, spec, spec.strings)
		if err != nil {
			return nil, fmt.Errorf("section %s: CO-RE relocations: %w", section, err)
		}
	}

	return &ExtInfos{funcInfos, lineInfos, coreRelos}, nil
}

type funcInfoMeta struct{}
type coreRelocationMeta struct{}

// Assign per-section metadata from BTF to a section's instructions.
func (ei *ExtInfos) Assign(insns asm.Instructions, section string) {
	funcInfos := ei.funcInfos[section]
	lineInfos := ei.lineInfos[section]
	reloInfos := ei.relocationInfos[section]

	AssignMetadataToInstructions(insns, funcInfos, lineInfos, reloInfos)
}

// Assign per-instruction metadata to the instructions in insns.
func AssignMetadataToInstructions(
	insns asm.Instructions,
	funcInfos FuncInfos,
	lineInfos LineInfos,
	reloInfos CORERelocationInfos,
) {
	iter := insns.Iterate()
	for iter.Next() {
		if len(funcInfos.infos) > 0 && funcInfos.infos[0].offset == iter.Offset {
			*iter.Ins = WithFuncMetadata(*iter.Ins, funcInfos.infos[0].fn)
			funcInfos.infos = funcInfos.infos[1:]
		}

		if len(lineInfos.infos) > 0 && lineInfos.infos[0].offset == iter.Offset {
			*iter.Ins = iter.Ins.WithSource(lineInfos.infos[0].line)
			lineInfos.infos = lineInfos.infos[1:]
		}

		if len(reloInfos.infos) > 0 && reloInfos.infos[0].offset == iter.Offset {
			iter.Ins.Metadata.Set(coreRelocationMeta{}, reloInfos.infos[0].relo)
			reloInfos.infos = reloInfos.infos[1:]
		}
	}
}

// MarshalExtInfos encodes function and line info embedded in insns into kernel
// wire format.
//
// If an instruction has an [asm.Comment], it will be synthesized into a mostly
// empty line info.
func MarshalExtInfos(insns asm.Instructions, b *Builder) (funcInfos, lineInfos []byte, _ error) {
	iter := insns.Iterate()
	for iter.Next() {
		if iter.Ins.Source() != nil || FuncMetadata(iter.Ins) != nil {
			goto marshal
		}
	}

	return nil, nil, nil

marshal:
	var fiBuf, liBuf bytes.Buffer
	for {
		if fn := FuncMetadata(iter.Ins); fn != nil {
			fi := &funcInfo{
				fn:     fn,
				offset: iter.Offset,
			}
			if err := fi.marshal(&fiBuf, b); err != nil {
				return nil, nil, fmt.Errorf("write func info: %w", err)
			}
		}

		if source := iter.Ins.Source(); source != nil {
			var line *Line
			if l, ok := source.(*Line); ok {
				line = l
			} else {
				line = &Line{
					line: source.String(),
				}
			}

			li := &lineInfo{
				line:   line,
				offset: iter.Offset,
			}
			if err := li.marshal(&liBuf, b); err != nil {
				return nil, nil, fmt.Errorf("write line info: %w", err)
			}
		}

		if !iter.Next() {
			break
		}
	}

	return fiBuf.Bytes(), liBuf.Bytes(), nil
}

// btfExtHeader is found at the start of the .BTF.ext section.
type btfExtHeader struct {
	Magic   uint16
	Version uint8
	Flags   uint8

	// HdrLen is larger than the size of struct btfExtHeader when it is
	// immediately followed by a btfExtCOREHeader.
	HdrLen uint32

	FuncInfoOff uint32
	FuncInfoLen uint32
	LineInfoOff uint32
	LineInfoLen uint32
}

// parseBTFExtHeader parses the header of the .BTF.ext section.
func parseBTFExtHeader(r io.Reader, bo binary.ByteOrder) (*btfExtHeader, error) {
	var header btfExtHeader
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

	if int64(header.HdrLen) < int64(binary.Size(&header)) {
		return nil, fmt.Errorf("header length shorter than btfExtHeader size")
	}

	return &header, nil
}

// funcInfoStart returns the offset from the beginning of the .BTF.ext section
// to the start of its func_info entries.
func (h *btfExtHeader) funcInfoStart() int64 {
	return int64(h.HdrLen + h.FuncInfoOff)
}

// lineInfoStart returns the offset from the beginning of the .BTF.ext section
// to the start of its line_info entries.
func (h *btfExtHeader) lineInfoStart() int64 {
	return int64(h.HdrLen + h.LineInfoOff)
}

// coreReloStart returns the offset from the beginning of the .BTF.ext section
// to the start of its CO-RE relocation entries.
func (h *btfExtHeader) coreReloStart(ch *btfExtCOREHeader) int64 {
	return int64(h.HdrLen + ch.COREReloOff)
}

// btfExtCOREHeader is found right after the btfExtHeader when its HdrLen
// field is larger than its size.
type btfExtCOREHeader struct {
	COREReloOff uint32
	COREReloLen uint32
}

// parseBTFExtCOREHeader parses the tail of the .BTF.ext header. If additional
// header bytes are present, extHeader.HdrLen will be larger than the struct,
// indicating the presence of a CO-RE extension header.
func parseBTFExtCOREHeader(r io.Reader, bo binary.ByteOrder, extHeader *btfExtHeader) (*btfExtCOREHeader, error) {
	extHdrSize := int64(binary.Size(&extHeader))
	remainder := int64(extHeader.HdrLen) - extHdrSize

	if remainder == 0 {
		return nil, nil
	}

	var coreHeader btfExtCOREHeader
	if err := binary.Read(r, bo, &coreHeader); err != nil {
		return nil, fmt.Errorf("can't read header: %v", err)
	}

	return &coreHeader, nil
}

type btfExtInfoSec struct {
	SecNameOff uint32
	NumInfo    uint32
}

// parseExtInfoSec parses a btf_ext_info_sec header within .BTF.ext,
// appearing within func_info and line_info sub-sections.
// These headers appear once for each program section in the ELF and are
// followed by one or more func/line_info records for the section.
func parseExtInfoSec(r io.Reader, bo binary.ByteOrder, strings *stringTable) (string, *btfExtInfoSec, error) {
	var infoHeader btfExtInfoSec
	if err := binary.Read(r, bo, &infoHeader); err != nil {
		return "", nil, fmt.Errorf("read ext info header: %w", err)
	}

	secName, err := strings.Lookup(infoHeader.SecNameOff)
	if err != nil {
		return "", nil, fmt.Errorf("get section name: %w", err)
	}
	if secName == "" {
		return "", nil, fmt.Errorf("extinfo header refers to empty section name")
	}

	if infoHeader.NumInfo == 0 {
		return "", nil, fmt.Errorf("section %s has zero records", secName)
	}

	return secName, &infoHeader, nil
}

// parseExtInfoRecordSize parses the uint32 at the beginning of a func_infos
// or line_infos segment that describes the length of all extInfoRecords in
// that segment.
func parseExtInfoRecordSize(r io.Reader, bo binary.ByteOrder) (uint32, error) {
	const maxRecordSize = 256

	var recordSize uint32
	if err := binary.Read(r, bo, &recordSize); err != nil {
		return 0, fmt.Errorf("can't read record size: %v", err)
	}

	if recordSize < 4 {
		// Need at least InsnOff worth of bytes per record.
		return 0, errors.New("record size too short")
	}
	if recordSize > maxRecordSize {
		return 0, fmt.Errorf("record size %v exceeds %v", recordSize, maxRecordSize)
	}

	return recordSize, nil
}

// FuncInfos contains a sorted list of func infos.
type FuncInfos struct {
	infos []funcInfo
}

// The size of a FuncInfo in BTF wire format.
var FuncInfoSize = uint32(binary.Size(bpfFuncInfo{}))

type funcInfo struct {
	fn     *Func
	offset asm.RawInstructionOffset
}

type bpfFuncInfo struct {
	// Instruction offset of the function within an ELF section.
	InsnOff uint32
	TypeID  TypeID
}

func newFuncInfo(fi bpfFuncInfo, spec *Spec) (*funcInfo, error) {
	typ, err := spec.TypeByID(fi.TypeID)
	if err != nil {
		return nil, err
	}

	fn, ok := typ.(*Func)
	if !ok {
		return nil, fmt.Errorf("type ID %d is a %T, but expected a Func", fi.TypeID, typ)
	}

	// C doesn't have anonymous functions, but check just in case.
	if fn.Name == "" {
		return nil, fmt.Errorf("func with type ID %d doesn't have a name", fi.TypeID)
	}

	return &funcInfo{
		fn,
		asm.RawInstructionOffset(fi.InsnOff),
	}, nil
}

func newFuncInfos(bfis []bpfFuncInfo, spec *Spec) (FuncInfos, error) {
	fis := FuncInfos{
		infos: make([]funcInfo, 0, len(bfis)),
	}
	for _, bfi := range bfis {
		fi, err := newFuncInfo(bfi, spec)
		if err != nil {
			return FuncInfos{}, fmt.Errorf("offset %d: %w", bfi.InsnOff, err)
		}
		fis.infos = append(fis.infos, *fi)
	}
	sort.Slice(fis.infos, func(i, j int) bool {
		return fis.infos[i].offset <= fis.infos[j].offset
	})
	return fis, nil
}

// LoadFuncInfos parses BTF func info in kernel wire format.
func LoadFuncInfos(reader io.Reader, bo binary.ByteOrder, recordNum uint32, spec *Spec) (FuncInfos, error) {
	fis, err := parseFuncInfoRecords(
		reader,
		bo,
		FuncInfoSize,
		recordNum,
		false,
	)
	if err != nil {
		return FuncInfos{}, fmt.Errorf("parsing BTF func info: %w", err)
	}

	return newFuncInfos(fis, spec)
}

// marshal into the BTF wire format.
func (fi *funcInfo) marshal(w *bytes.Buffer, b *Builder) error {
	id, err := b.Add(fi.fn)
	if err != nil {
		return err
	}
	bfi := bpfFuncInfo{
		InsnOff: uint32(fi.offset),
		TypeID:  id,
	}
	buf := make([]byte, FuncInfoSize)
	internal.NativeEndian.PutUint32(buf, bfi.InsnOff)
	internal.NativeEndian.PutUint32(buf[4:], uint32(bfi.TypeID))
	_, err = w.Write(buf)
	return err
}

// parseFuncInfos parses a func_info sub-section within .BTF.ext ito a map of
// func infos indexed by section name.
func parseFuncInfos(r io.Reader, bo binary.ByteOrder, strings *stringTable) (map[string][]bpfFuncInfo, error) {
	recordSize, err := parseExtInfoRecordSize(r, bo)
	if err != nil {
		return nil, err
	}

	result := make(map[string][]bpfFuncInfo)
	for {
		secName, infoHeader, err := parseExtInfoSec(r, bo, strings)
		if errors.Is(err, io.EOF) {
			return result, nil
		}
		if err != nil {
			return nil, err
		}

		records, err := parseFuncInfoRecords(r, bo, recordSize, infoHeader.NumInfo, true)
		if err != nil {
			return nil, fmt.Errorf("section %v: %w", secName, err)
		}

		result[secName] = records
	}
}

// parseFuncInfoRecords parses a stream of func_infos into a funcInfos.
// These records appear after a btf_ext_info_sec header in the func_info
// sub-section of .BTF.ext.
func parseFuncInfoRecords(r io.Reader, bo binary.ByteOrder, recordSize uint32, recordNum uint32, offsetInBytes bool) ([]bpfFuncInfo, error) {
	var out []bpfFuncInfo
	var fi bpfFuncInfo

	if exp, got := FuncInfoSize, recordSize; exp != got {
		// BTF blob's record size is longer than we know how to parse.
		return nil, fmt.Errorf("expected FuncInfo record size %d, but BTF blob contains %d", exp, got)
	}

	for i := uint32(0); i < recordNum; i++ {
		if err := binary.Read(r, bo, &fi); err != nil {
			return nil, fmt.Errorf("can't read function info: %v", err)
		}

		if offsetInBytes {
			if fi.InsnOff%asm.InstructionSize != 0 {
				return nil, fmt.Errorf("offset %v is not aligned with instruction size", fi.InsnOff)
			}

			// ELF tracks offset in bytes, the kernel expects raw BPF instructions.
			// Convert as early as possible.
			fi.InsnOff /= asm.InstructionSize
		}

		out = append(out, fi)
	}

	return out, nil
}

var LineInfoSize = uint32(binary.Size(bpfLineInfo{}))

// Line represents the location and contents of a single line of source
// code a BPF ELF was compiled from.
type Line struct {
	fileName   string
	line       string
	lineNumber uint32
	lineColumn uint32
}

func (li *Line) FileName() string {
	return li.fileName
}

func (li *Line) Line() string {
	return li.line
}

func (li *Line) LineNumber() uint32 {
	return li.lineNumber
}

func (li *Line) LineColumn() uint32 {
	return li.lineColumn
}

func (li *Line) String() string {
	return li.line
}

// LineInfos contains a sorted list of line infos.
type LineInfos struct {
	infos []lineInfo
}

type lineInfo struct {
	line   *Line
	offset asm.RawInstructionOffset
}

// Constants for the format of bpfLineInfo.LineCol.
const (
	bpfLineShift = 10
	bpfLineMax   = (1 << (32 - bpfLineShift)) - 1
	bpfColumnMax = (1 << bpfLineShift) - 1
)

type bpfLineInfo struct {
	// Instruction offset of the line within the whole instruction stream, in instructions.
	InsnOff     uint32
	FileNameOff uint32
	LineOff     uint32
	LineCol     uint32
}

// LoadLineInfos parses BTF line info in kernel wire format.
func LoadLineInfos(reader io.Reader, bo binary.ByteOrder, recordNum uint32, spec *Spec) (LineInfos, error) {
	lis, err := parseLineInfoRecords(
		reader,
		bo,
		LineInfoSize,
		recordNum,
		false,
	)
	if err != nil {
		return LineInfos{}, fmt.Errorf("parsing BTF line info: %w", err)
	}

	return newLineInfos(lis, spec.strings)
}

func newLineInfo(li bpfLineInfo, strings *stringTable) (lineInfo, error) {
	line, err := strings.Lookup(li.LineOff)
	if err != nil {
		return lineInfo{}, fmt.Errorf("lookup of line: %w", err)
	}

	fileName, err := strings.Lookup(li.FileNameOff)
	if err != nil {
		return lineInfo{}, fmt.Errorf("lookup of filename: %w", err)
	}

	lineNumber := li.LineCol >> bpfLineShift
	lineColumn := li.LineCol & bpfColumnMax

	return lineInfo{
		&Line{
			fileName,
			line,
			lineNumber,
			lineColumn,
		},
		asm.RawInstructionOffset(li.InsnOff),
	}, nil
}

func newLineInfos(blis []bpfLineInfo, strings *stringTable) (LineInfos, error) {
	lis := LineInfos{
		infos: make([]lineInfo, 0, len(blis)),
	}
	for _, bli := range blis {
		li, err := newLineInfo(bli, strings)
		if err != nil {
			return LineInfos{}, fmt.Errorf("offset %d: %w", bli.InsnOff, err)
		}
		lis.infos = append(lis.infos, li)
	}
	sort.Slice(lis.infos, func(i, j int) bool {
		return lis.infos[i].offset <= lis.infos[j].offset
	})
	return lis, nil
}

// marshal writes the binary representation of the LineInfo to w.
func (li *lineInfo) marshal(w *bytes.Buffer, b *Builder) error {
	line := li.line
	if line.lineNumber > bpfLineMax {
		return fmt.Errorf("line %d exceeds %d", line.lineNumber, bpfLineMax)
	}

	if line.lineColumn > bpfColumnMax {
		return fmt.Errorf("column %d exceeds %d", line.lineColumn, bpfColumnMax)
	}

	fileNameOff, err := b.addString(line.fileName)
	if err != nil {
		return fmt.Errorf("file name %q: %w", line.fileName, err)
	}

	lineOff, err := b.addString(line.line)
	if err != nil {
		return fmt.Errorf("line %q: %w", line.line, err)
	}

	bli := bpfLineInfo{
		uint32(li.offset),
		fileNameOff,
		lineOff,
		(line.lineNumber << bpfLineShift) | line.lineColumn,
	}

	buf := make([]byte, LineInfoSize)
	internal.NativeEndian.PutUint32(buf, bli.InsnOff)
	internal.NativeEndian.PutUint32(buf[4:], bli.FileNameOff)
	internal.NativeEndian.PutUint32(buf[8:], bli.LineOff)
	internal.NativeEndian.PutUint32(buf[12:], bli.LineCol)
	_, err = w.Write(buf)
	return err
}

// parseLineInfos parses a line_info sub-section within .BTF.ext ito a map of
// line infos indexed by section name.
func parseLineInfos(r io.Reader, bo binary.ByteOrder, strings *stringTable) (map[string][]bpfLineInfo, error) {
	recordSize, err := parseExtInfoRecordSize(r, bo)
	if err != nil {
		return nil, err
	}

	result := make(map[string][]bpfLineInfo)
	for {
		secName, infoHeader, err := parseExtInfoSec(r, bo, strings)
		if errors.Is(err, io.EOF) {
			return result, nil
		}
		if err != nil {
			return nil, err
		}

		records, err := parseLineInfoRecords(r, bo, recordSize, infoHeader.NumInfo, true)
		if err != nil {
			return nil, fmt.Errorf("section %v: %w", secName, err)
		}

		result[secName] = records
	}
}

// parseLineInfoRecords parses a stream of line_infos into a lineInfos.
// These records appear after a btf_ext_info_sec header in the line_info
// sub-section of .BTF.ext.
func parseLineInfoRecords(r io.Reader, bo binary.ByteOrder, recordSize uint32, recordNum uint32, offsetInBytes bool) ([]bpfLineInfo, error) {
	var li bpfLineInfo

	if exp, got := uint32(binary.Size(li)), recordSize; exp != got {
		// BTF blob's record size is longer than we know how to parse.
		return nil, fmt.Errorf("expected LineInfo record size %d, but BTF blob contains %d", exp, got)
	}

	out := make([]bpfLineInfo, 0, recordNum)
	for i := uint32(0); i < recordNum; i++ {
		if err := binary.Read(r, bo, &li); err != nil {
			return nil, fmt.Errorf("can't read line info: %v", err)
		}

		if offsetInBytes {
			if li.InsnOff%asm.InstructionSize != 0 {
				return nil, fmt.Errorf("offset %v is not aligned with instruction size", li.InsnOff)
			}

			// ELF tracks offset in bytes, the kernel expects raw BPF instructions.
			// Convert as early as possible.
			li.InsnOff /= asm.InstructionSize
		}

		out = append(out, li)
	}

	return out, nil
}

// bpfCORERelo matches the kernel's struct bpf_core_relo.
type bpfCORERelo struct {
	InsnOff      uint32
	TypeID       TypeID
	AccessStrOff uint32
	Kind         coreKind
}

type CORERelocation struct {
	// The local type of the relocation, stripped of typedefs and qualifiers.
	typ      Type
	accessor coreAccessor
	kind     coreKind
	// The ID of the local type in the source BTF.
	id TypeID
}

func (cr *CORERelocation) String() string {
	return fmt.Sprintf("CORERelocation(%s, %s[%s], local_id=%d)", cr.kind, cr.typ, cr.accessor, cr.id)
}

func CORERelocationMetadata(ins *asm.Instruction) *CORERelocation {
	relo, _ := ins.Metadata.Get(coreRelocationMeta{}).(*CORERelocation)
	return relo
}

// CORERelocationInfos contains a sorted list of co:re relocation infos.
type CORERelocationInfos struct {
	infos []coreRelocationInfo
}

type coreRelocationInfo struct {
	relo   *CORERelocation
	offset asm.RawInstructionOffset
}

func newRelocationInfo(relo bpfCORERelo, spec *Spec, strings *stringTable) (*coreRelocationInfo, error) {
	typ, err := spec.TypeByID(relo.TypeID)
	if err != nil {
		return nil, err
	}

	accessorStr, err := strings.Lookup(relo.AccessStrOff)
	if err != nil {
		return nil, err
	}

	accessor, err := parseCOREAccessor(accessorStr)
	if err != nil {
		return nil, fmt.Errorf("accessor %q: %s", accessorStr, err)
	}

	return &coreRelocationInfo{
		&CORERelocation{
			typ,
			accessor,
			relo.Kind,
			relo.TypeID,
		},
		asm.RawInstructionOffset(relo.InsnOff),
	}, nil
}

func newRelocationInfos(brs []bpfCORERelo, spec *Spec, strings *stringTable) (CORERelocationInfos, error) {
	rs := CORERelocationInfos{
		infos: make([]coreRelocationInfo, 0, len(brs)),
	}
	for _, br := range brs {
		relo, err := newRelocationInfo(br, spec, strings)
		if err != nil {
			return CORERelocationInfos{}, fmt.Errorf("offset %d: %w", br.InsnOff, err)
		}
		rs.infos = append(rs.infos, *relo)
	}
	sort.Slice(rs.infos, func(i, j int) bool {
		return rs.infos[i].offset < rs.infos[j].offset
	})
	return rs, nil
}

var extInfoReloSize = binary.Size(bpfCORERelo{})

// parseCORERelos parses a core_relos sub-section within .BTF.ext ito a map of
// CO-RE relocations indexed by section name.
func parseCORERelos(r io.Reader, bo binary.ByteOrder, strings *stringTable) (map[string][]bpfCORERelo, error) {
	recordSize, err := parseExtInfoRecordSize(r, bo)
	if err != nil {
		return nil, err
	}

	if recordSize != uint32(extInfoReloSize) {
		return nil, fmt.Errorf("expected record size %d, got %d", extInfoReloSize, recordSize)
	}

	result := make(map[string][]bpfCORERelo)
	for {
		secName, infoHeader, err := parseExtInfoSec(r, bo, strings)
		if errors.Is(err, io.EOF) {
			return result, nil
		}
		if err != nil {
			return nil, err
		}

		records, err := parseCOREReloRecords(r, bo, recordSize, infoHeader.NumInfo)
		if err != nil {
			return nil, fmt.Errorf("section %v: %w", secName, err)
		}

		result[secName] = records
	}
}

// parseCOREReloRecords parses a stream of CO-RE relocation entries into a
// coreRelos. These records appear after a btf_ext_info_sec header in the
// core_relos sub-section of .BTF.ext.
func parseCOREReloRecords(r io.Reader, bo binary.ByteOrder, recordSize uint32, recordNum uint32) ([]bpfCORERelo, error) {
	var out []bpfCORERelo

	var relo bpfCORERelo
	for i := uint32(0); i < recordNum; i++ {
		if err := binary.Read(r, bo, &relo); err != nil {
			return nil, fmt.Errorf("can't read CO-RE relocation: %v", err)
		}

		if relo.InsnOff%asm.InstructionSize != 0 {
			return nil, fmt.Errorf("offset %v is not aligned with instruction size", relo.InsnOff)
		}

		// ELF tracks offset in bytes, the kernel expects raw BPF instructions.
		// Convert as early as possible.
		relo.InsnOff /= asm.InstructionSize

		out = append(out, relo)
	}

	return out, nil
}

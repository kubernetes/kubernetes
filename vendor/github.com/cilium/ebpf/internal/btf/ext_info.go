package btf

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"io/ioutil"

	"github.com/cilium/ebpf/asm"
	"github.com/cilium/ebpf/internal"
)

type btfExtHeader struct {
	Magic   uint16
	Version uint8
	Flags   uint8
	HdrLen  uint32

	FuncInfoOff uint32
	FuncInfoLen uint32
	LineInfoOff uint32
	LineInfoLen uint32
}

type btfExtCoreHeader struct {
	CoreReloOff uint32
	CoreReloLen uint32
}

func parseExtInfos(r io.ReadSeeker, bo binary.ByteOrder, strings stringTable) (funcInfo, lineInfo map[string]extInfo, relos map[string]coreRelos, err error) {
	var header btfExtHeader
	var coreHeader btfExtCoreHeader
	if err := binary.Read(r, bo, &header); err != nil {
		return nil, nil, nil, fmt.Errorf("can't read header: %v", err)
	}

	if header.Magic != btfMagic {
		return nil, nil, nil, fmt.Errorf("incorrect magic value %v", header.Magic)
	}

	if header.Version != 1 {
		return nil, nil, nil, fmt.Errorf("unexpected version %v", header.Version)
	}

	if header.Flags != 0 {
		return nil, nil, nil, fmt.Errorf("unsupported flags %v", header.Flags)
	}

	remainder := int64(header.HdrLen) - int64(binary.Size(&header))
	if remainder < 0 {
		return nil, nil, nil, errors.New("header is too short")
	}

	coreHdrSize := int64(binary.Size(&coreHeader))
	if remainder >= coreHdrSize {
		if err := binary.Read(r, bo, &coreHeader); err != nil {
			return nil, nil, nil, fmt.Errorf("can't read CO-RE relocation header: %v", err)
		}
		remainder -= coreHdrSize
	}

	// Of course, the .BTF.ext header has different semantics than the
	// .BTF ext header. We need to ignore non-null values.
	_, err = io.CopyN(ioutil.Discard, r, remainder)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("header padding: %v", err)
	}

	if _, err := r.Seek(int64(header.HdrLen+header.FuncInfoOff), io.SeekStart); err != nil {
		return nil, nil, nil, fmt.Errorf("can't seek to function info section: %v", err)
	}

	buf := bufio.NewReader(io.LimitReader(r, int64(header.FuncInfoLen)))
	funcInfo, err = parseExtInfo(buf, bo, strings)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("function info: %w", err)
	}

	if _, err := r.Seek(int64(header.HdrLen+header.LineInfoOff), io.SeekStart); err != nil {
		return nil, nil, nil, fmt.Errorf("can't seek to line info section: %v", err)
	}

	buf = bufio.NewReader(io.LimitReader(r, int64(header.LineInfoLen)))
	lineInfo, err = parseExtInfo(buf, bo, strings)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("line info: %w", err)
	}

	if coreHeader.CoreReloOff > 0 && coreHeader.CoreReloLen > 0 {
		if _, err := r.Seek(int64(header.HdrLen+coreHeader.CoreReloOff), io.SeekStart); err != nil {
			return nil, nil, nil, fmt.Errorf("can't seek to CO-RE relocation section: %v", err)
		}

		relos, err = parseExtInfoRelos(io.LimitReader(r, int64(coreHeader.CoreReloLen)), bo, strings)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("CO-RE relocation info: %w", err)
		}
	}

	return funcInfo, lineInfo, relos, nil
}

type btfExtInfoSec struct {
	SecNameOff uint32
	NumInfo    uint32
}

type extInfoRecord struct {
	InsnOff uint64
	Opaque  []byte
}

type extInfo struct {
	recordSize uint32
	records    []extInfoRecord
}

func (ei extInfo) append(other extInfo, offset uint64) (extInfo, error) {
	if other.recordSize != ei.recordSize {
		return extInfo{}, fmt.Errorf("ext_info record size mismatch, want %d (got %d)", ei.recordSize, other.recordSize)
	}

	records := make([]extInfoRecord, 0, len(ei.records)+len(other.records))
	records = append(records, ei.records...)
	for _, info := range other.records {
		records = append(records, extInfoRecord{
			InsnOff: info.InsnOff + offset,
			Opaque:  info.Opaque,
		})
	}
	return extInfo{ei.recordSize, records}, nil
}

func (ei extInfo) MarshalBinary() ([]byte, error) {
	if len(ei.records) == 0 {
		return nil, nil
	}

	buf := bytes.NewBuffer(make([]byte, 0, int(ei.recordSize)*len(ei.records)))
	for _, info := range ei.records {
		// The kernel expects offsets in number of raw bpf instructions,
		// while the ELF tracks it in bytes.
		insnOff := uint32(info.InsnOff / asm.InstructionSize)
		if err := binary.Write(buf, internal.NativeEndian, insnOff); err != nil {
			return nil, fmt.Errorf("can't write instruction offset: %v", err)
		}

		buf.Write(info.Opaque)
	}

	return buf.Bytes(), nil
}

func parseExtInfo(r io.Reader, bo binary.ByteOrder, strings stringTable) (map[string]extInfo, error) {
	const maxRecordSize = 256

	var recordSize uint32
	if err := binary.Read(r, bo, &recordSize); err != nil {
		return nil, fmt.Errorf("can't read record size: %v", err)
	}

	if recordSize < 4 {
		// Need at least insnOff
		return nil, errors.New("record size too short")
	}
	if recordSize > maxRecordSize {
		return nil, fmt.Errorf("record size %v exceeds %v", recordSize, maxRecordSize)
	}

	result := make(map[string]extInfo)
	for {
		secName, infoHeader, err := parseExtInfoHeader(r, bo, strings)
		if errors.Is(err, io.EOF) {
			return result, nil
		}

		var records []extInfoRecord
		for i := uint32(0); i < infoHeader.NumInfo; i++ {
			var byteOff uint32
			if err := binary.Read(r, bo, &byteOff); err != nil {
				return nil, fmt.Errorf("section %v: can't read extended info offset: %v", secName, err)
			}

			buf := make([]byte, int(recordSize-4))
			if _, err := io.ReadFull(r, buf); err != nil {
				return nil, fmt.Errorf("section %v: can't read record: %v", secName, err)
			}

			if byteOff%asm.InstructionSize != 0 {
				return nil, fmt.Errorf("section %v: offset %v is not aligned with instruction size", secName, byteOff)
			}

			records = append(records, extInfoRecord{uint64(byteOff), buf})
		}

		result[secName] = extInfo{
			recordSize,
			records,
		}
	}
}

// bpfCoreRelo matches `struct bpf_core_relo` from the kernel
type bpfCoreRelo struct {
	InsnOff      uint32
	TypeID       TypeID
	AccessStrOff uint32
	Kind         COREKind
}

type coreRelo struct {
	insnOff  uint32
	typeID   TypeID
	accessor coreAccessor
	kind     COREKind
}

type coreRelos []coreRelo

// append two slices of extInfoRelo to each other. The InsnOff of b are adjusted
// by offset.
func (r coreRelos) append(other coreRelos, offset uint64) coreRelos {
	result := make([]coreRelo, 0, len(r)+len(other))
	result = append(result, r...)
	for _, relo := range other {
		relo.insnOff += uint32(offset)
		result = append(result, relo)
	}
	return result
}

var extInfoReloSize = binary.Size(bpfCoreRelo{})

func parseExtInfoRelos(r io.Reader, bo binary.ByteOrder, strings stringTable) (map[string]coreRelos, error) {
	var recordSize uint32
	if err := binary.Read(r, bo, &recordSize); err != nil {
		return nil, fmt.Errorf("read record size: %v", err)
	}

	if recordSize != uint32(extInfoReloSize) {
		return nil, fmt.Errorf("expected record size %d, got %d", extInfoReloSize, recordSize)
	}

	result := make(map[string]coreRelos)
	for {
		secName, infoHeader, err := parseExtInfoHeader(r, bo, strings)
		if errors.Is(err, io.EOF) {
			return result, nil
		}

		var relos coreRelos
		for i := uint32(0); i < infoHeader.NumInfo; i++ {
			var relo bpfCoreRelo
			if err := binary.Read(r, bo, &relo); err != nil {
				return nil, fmt.Errorf("section %v: read record: %v", secName, err)
			}

			if relo.InsnOff%asm.InstructionSize != 0 {
				return nil, fmt.Errorf("section %v: offset %v is not aligned with instruction size", secName, relo.InsnOff)
			}

			accessorStr, err := strings.Lookup(relo.AccessStrOff)
			if err != nil {
				return nil, err
			}

			accessor, err := parseCoreAccessor(accessorStr)
			if err != nil {
				return nil, fmt.Errorf("accessor %q: %s", accessorStr, err)
			}

			relos = append(relos, coreRelo{
				relo.InsnOff,
				relo.TypeID,
				accessor,
				relo.Kind,
			})
		}

		result[secName] = relos
	}
}

func parseExtInfoHeader(r io.Reader, bo binary.ByteOrder, strings stringTable) (string, *btfExtInfoSec, error) {
	var infoHeader btfExtInfoSec
	if err := binary.Read(r, bo, &infoHeader); err != nil {
		return "", nil, fmt.Errorf("read ext info header: %w", err)
	}

	secName, err := strings.Lookup(infoHeader.SecNameOff)
	if err != nil {
		return "", nil, fmt.Errorf("get section name: %w", err)
	}

	if infoHeader.NumInfo == 0 {
		return "", nil, fmt.Errorf("section %s has zero records", secName)
	}

	return secName, &infoHeader, nil
}

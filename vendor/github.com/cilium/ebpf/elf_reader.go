package ebpf

import (
	"bytes"
	"debug/elf"
	"encoding/binary"
	"io"
	"os"
	"strings"

	"github.com/cilium/ebpf/asm"
	"github.com/cilium/ebpf/internal"
	"github.com/cilium/ebpf/internal/btf"

	"github.com/pkg/errors"
)

type elfCode struct {
	*elf.File
	symbols           []elf.Symbol
	symbolsPerSection map[elf.SectionIndex]map[uint64]string
	license           string
	version           uint32
}

// LoadCollectionSpec parses an ELF file into a CollectionSpec.
func LoadCollectionSpec(file string) (*CollectionSpec, error) {
	f, err := os.Open(file)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	spec, err := LoadCollectionSpecFromReader(f)
	return spec, errors.Wrapf(err, "file %s", file)
}

// LoadCollectionSpecFromReader parses an ELF file into a CollectionSpec.
func LoadCollectionSpecFromReader(rd io.ReaderAt) (*CollectionSpec, error) {
	f, err := elf.NewFile(rd)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	symbols, err := f.Symbols()
	if err != nil {
		return nil, errors.Wrap(err, "load symbols")
	}

	ec := &elfCode{f, symbols, symbolsPerSection(symbols), "", 0}

	var (
		licenseSection *elf.Section
		versionSection *elf.Section
		btfMaps        = make(map[elf.SectionIndex]*elf.Section)
		progSections   = make(map[elf.SectionIndex]*elf.Section)
		relSections    = make(map[elf.SectionIndex]*elf.Section)
		mapSections    = make(map[elf.SectionIndex]*elf.Section)
	)

	for i, sec := range ec.Sections {
		switch {
		case strings.HasPrefix(sec.Name, "license"):
			licenseSection = sec
		case strings.HasPrefix(sec.Name, "version"):
			versionSection = sec
		case strings.HasPrefix(sec.Name, "maps"):
			mapSections[elf.SectionIndex(i)] = sec
		case sec.Name == ".maps":
			btfMaps[elf.SectionIndex(i)] = sec
		case sec.Type == elf.SHT_REL:
			if int(sec.Info) >= len(ec.Sections) {
				return nil, errors.Errorf("found relocation section %v for missing section %v", i, sec.Info)
			}

			// Store relocations under the section index of the target
			idx := elf.SectionIndex(sec.Info)
			if relSections[idx] != nil {
				return nil, errors.Errorf("section %d has multiple relocation sections", sec.Info)
			}
			relSections[idx] = sec
		case sec.Type == elf.SHT_PROGBITS && (sec.Flags&elf.SHF_EXECINSTR) != 0 && sec.Size > 0:
			progSections[elf.SectionIndex(i)] = sec
		}
	}

	ec.license, err = loadLicense(licenseSection)
	if err != nil {
		return nil, errors.Wrap(err, "load license")
	}

	ec.version, err = loadVersion(versionSection, ec.ByteOrder)
	if err != nil {
		return nil, errors.Wrap(err, "load version")
	}

	btf, err := btf.LoadSpecFromReader(rd)
	if err != nil {
		return nil, errors.Wrap(err, "load BTF")
	}

	maps := make(map[string]*MapSpec)

	if err := ec.loadMaps(maps, mapSections); err != nil {
		return nil, errors.Wrap(err, "load maps")
	}

	if len(btfMaps) > 0 {
		if err := ec.loadBTFMaps(maps, btfMaps, btf); err != nil {
			return nil, errors.Wrap(err, "load BTF maps")
		}
	}

	progs, err := ec.loadPrograms(progSections, relSections, btf)
	if err != nil {
		return nil, errors.Wrap(err, "load programs")
	}

	return &CollectionSpec{maps, progs}, nil
}

func loadLicense(sec *elf.Section) (string, error) {
	if sec == nil {
		return "", errors.Errorf("missing license section")
	}
	data, err := sec.Data()
	if err != nil {
		return "", errors.Wrapf(err, "section %s", sec.Name)
	}
	return string(bytes.TrimRight(data, "\000")), nil
}

func loadVersion(sec *elf.Section, bo binary.ByteOrder) (uint32, error) {
	if sec == nil {
		return 0, nil
	}

	var version uint32
	err := binary.Read(sec.Open(), bo, &version)
	return version, errors.Wrapf(err, "section %s", sec.Name)
}

func (ec *elfCode) loadPrograms(progSections, relSections map[elf.SectionIndex]*elf.Section, btf *btf.Spec) (map[string]*ProgramSpec, error) {
	var (
		progs []*ProgramSpec
		libs  []*ProgramSpec
	)

	for idx, prog := range progSections {
		syms := ec.symbolsPerSection[idx]
		if len(syms) == 0 {
			return nil, errors.Errorf("section %v: missing symbols", prog.Name)
		}

		funcSym := syms[0]
		if funcSym == "" {
			return nil, errors.Errorf("section %v: no label at start", prog.Name)
		}

		rels, err := ec.loadRelocations(relSections[idx])
		if err != nil {
			return nil, errors.Wrapf(err, "program %s: can't load relocations", funcSym)
		}

		insns, length, err := ec.loadInstructions(prog, syms, rels)
		if err != nil {
			return nil, errors.Wrapf(err, "program %s: can't unmarshal instructions", funcSym)
		}

		progType, attachType := getProgType(prog.Name)

		spec := &ProgramSpec{
			Name:          funcSym,
			Type:          progType,
			AttachType:    attachType,
			License:       ec.license,
			KernelVersion: ec.version,
			Instructions:  insns,
		}

		if btf != nil {
			spec.BTF, err = btf.Program(prog.Name, length)
			if err != nil {
				return nil, errors.Wrapf(err, "BTF for section %s (program %s)", prog.Name, funcSym)
			}
		}

		if spec.Type == UnspecifiedProgram {
			// There is no single name we can use for "library" sections,
			// since they may contain multiple functions. We'll decode the
			// labels they contain later on, and then link sections that way.
			libs = append(libs, spec)
		} else {
			progs = append(progs, spec)
		}
	}

	res := make(map[string]*ProgramSpec, len(progs))
	for _, prog := range progs {
		err := link(prog, libs)
		if err != nil {
			return nil, errors.Wrapf(err, "program %s", prog.Name)
		}
		res[prog.Name] = prog
	}

	return res, nil
}

func (ec *elfCode) loadInstructions(section *elf.Section, symbols, relocations map[uint64]string) (asm.Instructions, uint64, error) {
	var (
		r      = section.Open()
		insns  asm.Instructions
		ins    asm.Instruction
		offset uint64
	)
	for {
		n, err := ins.Unmarshal(r, ec.ByteOrder)
		if err == io.EOF {
			return insns, offset, nil
		}
		if err != nil {
			return nil, 0, errors.Wrapf(err, "offset %d", offset)
		}

		ins.Symbol = symbols[offset]
		ins.Reference = relocations[offset]

		insns = append(insns, ins)
		offset += n
	}
}

func (ec *elfCode) loadMaps(maps map[string]*MapSpec, mapSections map[elf.SectionIndex]*elf.Section) error {
	for idx, sec := range mapSections {
		syms := ec.symbolsPerSection[idx]
		if len(syms) == 0 {
			return errors.Errorf("section %v: no symbols", sec.Name)
		}

		if sec.Size%uint64(len(syms)) != 0 {
			return errors.Errorf("section %v: map descriptors are not of equal size", sec.Name)
		}

		var (
			r    = sec.Open()
			size = sec.Size / uint64(len(syms))
		)
		for i, offset := 0, uint64(0); i < len(syms); i, offset = i+1, offset+size {
			mapSym := syms[offset]
			if mapSym == "" {
				return errors.Errorf("section %s: missing symbol for map at offset %d", sec.Name, offset)
			}

			if maps[mapSym] != nil {
				return errors.Errorf("section %v: map %v already exists", sec.Name, mapSym)
			}

			lr := io.LimitReader(r, int64(size))

			var spec MapSpec
			switch {
			case binary.Read(lr, ec.ByteOrder, &spec.Type) != nil:
				return errors.Errorf("map %v: missing type", mapSym)
			case binary.Read(lr, ec.ByteOrder, &spec.KeySize) != nil:
				return errors.Errorf("map %v: missing key size", mapSym)
			case binary.Read(lr, ec.ByteOrder, &spec.ValueSize) != nil:
				return errors.Errorf("map %v: missing value size", mapSym)
			case binary.Read(lr, ec.ByteOrder, &spec.MaxEntries) != nil:
				return errors.Errorf("map %v: missing max entries", mapSym)
			case binary.Read(lr, ec.ByteOrder, &spec.Flags) != nil:
				return errors.Errorf("map %v: missing flags", mapSym)
			}

			if _, err := io.Copy(internal.DiscardZeroes{}, lr); err != nil {
				return errors.Errorf("map %v: unknown and non-zero fields in definition", mapSym)
			}

			maps[mapSym] = &spec
		}
	}

	return nil
}

func (ec *elfCode) loadBTFMaps(maps map[string]*MapSpec, mapSections map[elf.SectionIndex]*elf.Section, spec *btf.Spec) error {

	if spec == nil {
		return errors.Errorf("missing BTF")
	}

	for idx, sec := range mapSections {
		syms := ec.symbolsPerSection[idx]
		if len(syms) == 0 {
			return errors.Errorf("section %v: no symbols", sec.Name)
		}

		for _, sym := range syms {
			if maps[sym] != nil {
				return errors.Errorf("section %v: map %v already exists", sec.Name, sym)
			}

			btfMap, err := spec.Map(sym)
			if err != nil {
				return errors.Wrapf(err, "map %v: can't get BTF", sym)
			}

			spec, err := mapSpecFromBTF(btfMap)
			if err != nil {
				return errors.Wrapf(err, "map %v", sym)
			}

			maps[sym] = spec
		}
	}

	return nil
}

func mapSpecFromBTF(btfMap *btf.Map) (*MapSpec, error) {
	var (
		mapType, flags, maxEntries uint32
		err                        error
	)
	for _, member := range btf.MapType(btfMap).Members {
		switch member.Name {
		case "type":
			mapType, err = uintFromBTF(member.Type)
			if err != nil {
				return nil, errors.Wrap(err, "can't get type")
			}

		case "map_flags":
			flags, err = uintFromBTF(member.Type)
			if err != nil {
				return nil, errors.Wrap(err, "can't get BTF map flags")
			}

		case "max_entries":
			maxEntries, err = uintFromBTF(member.Type)
			if err != nil {
				return nil, errors.Wrap(err, "can't get BTF map max entries")
			}

		case "key":
		case "value":
		default:
			return nil, errors.Errorf("unrecognized field %s in BTF map definition", member.Name)
		}
	}

	keySize, err := btf.Sizeof(btf.MapKey(btfMap))
	if err != nil {
		return nil, errors.Wrap(err, "can't get size of BTF key")
	}

	valueSize, err := btf.Sizeof(btf.MapValue(btfMap))
	if err != nil {
		return nil, errors.Wrap(err, "can't get size of BTF value")
	}

	return &MapSpec{
		Type:       MapType(mapType),
		KeySize:    uint32(keySize),
		ValueSize:  uint32(valueSize),
		MaxEntries: maxEntries,
		Flags:      flags,
		BTF:        btfMap,
	}, nil
}

// uintFromBTF resolves the __uint macro, which is a pointer to a sized
// array, e.g. for int (*foo)[10], this function will return 10.
func uintFromBTF(typ btf.Type) (uint32, error) {
	ptr, ok := typ.(*btf.Pointer)
	if !ok {
		return 0, errors.Errorf("not a pointer: %v", typ)
	}

	arr, ok := ptr.Target.(*btf.Array)
	if !ok {
		return 0, errors.Errorf("not a pointer to array: %v", typ)
	}

	return arr.Nelems, nil
}

func getProgType(v string) (ProgramType, AttachType) {
	types := map[string]ProgramType{
		// From https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/tools/lib/bpf/libbpf.c#n3568
		"socket":          SocketFilter,
		"seccomp":         SocketFilter,
		"kprobe/":         Kprobe,
		"uprobe/":         Kprobe,
		"kretprobe/":      Kprobe,
		"uretprobe/":      Kprobe,
		"tracepoint/":     TracePoint,
		"raw_tracepoint/": RawTracepoint,
		"xdp":             XDP,
		"perf_event":      PerfEvent,
		"lwt_in":          LWTIn,
		"lwt_out":         LWTOut,
		"lwt_xmit":        LWTXmit,
		"lwt_seg6local":   LWTSeg6Local,
		"sockops":         SockOps,
		"sk_skb":          SkSKB,
		"sk_msg":          SkMsg,
		"lirc_mode2":      LircMode2,
		"flow_dissector":  FlowDissector,

		"cgroup_skb/":       CGroupSKB,
		"cgroup/dev":        CGroupDevice,
		"cgroup/skb":        CGroupSKB,
		"cgroup/sock":       CGroupSock,
		"cgroup/post_bind":  CGroupSock,
		"cgroup/bind":       CGroupSockAddr,
		"cgroup/connect":    CGroupSockAddr,
		"cgroup/sendmsg":    CGroupSockAddr,
		"cgroup/recvmsg":    CGroupSockAddr,
		"cgroup/sysctl":     CGroupSysctl,
		"cgroup/getsockopt": CGroupSockopt,
		"cgroup/setsockopt": CGroupSockopt,
		"classifier":        SchedCLS,
		"action":            SchedACT,
	}
	attachTypes := map[string]AttachType{
		"cgroup_skb/ingress":    AttachCGroupInetIngress,
		"cgroup_skb/egress":     AttachCGroupInetEgress,
		"cgroup/sock":           AttachCGroupInetSockCreate,
		"cgroup/post_bind4":     AttachCGroupInet4PostBind,
		"cgroup/post_bind6":     AttachCGroupInet6PostBind,
		"cgroup/dev":            AttachCGroupDevice,
		"sockops":               AttachCGroupSockOps,
		"sk_skb/stream_parser":  AttachSkSKBStreamParser,
		"sk_skb/stream_verdict": AttachSkSKBStreamVerdict,
		"sk_msg":                AttachSkSKBStreamVerdict,
		"lirc_mode2":            AttachLircMode2,
		"flow_dissector":        AttachFlowDissector,
		"cgroup/bind4":          AttachCGroupInet4Bind,
		"cgroup/bind6":          AttachCGroupInet6Bind,
		"cgroup/connect4":       AttachCGroupInet4Connect,
		"cgroup/connect6":       AttachCGroupInet6Connect,
		"cgroup/sendmsg4":       AttachCGroupUDP4Sendmsg,
		"cgroup/sendmsg6":       AttachCGroupUDP6Sendmsg,
		"cgroup/recvmsg4":       AttachCGroupUDP4Recvmsg,
		"cgroup/recvmsg6":       AttachCGroupUDP6Recvmsg,
		"cgroup/sysctl":         AttachCGroupSysctl,
		"cgroup/getsockopt":     AttachCGroupGetsockopt,
		"cgroup/setsockopt":     AttachCGroupSetsockopt,
	}
	attachType := AttachNone
	for k, t := range attachTypes {
		if strings.HasPrefix(v, k) {
			attachType = t
		}
	}

	for k, t := range types {
		if strings.HasPrefix(v, k) {
			return t, attachType
		}
	}
	return UnspecifiedProgram, AttachNone
}

func (ec *elfCode) loadRelocations(sec *elf.Section) (map[uint64]string, error) {
	rels := make(map[uint64]string)
	if sec == nil {
		return rels, nil
	}

	if sec.Entsize < 16 {
		return nil, errors.New("rels are less than 16 bytes")
	}

	r := sec.Open()
	for off := uint64(0); off < sec.Size; off += sec.Entsize {
		ent := io.LimitReader(r, int64(sec.Entsize))

		var rel elf.Rel64
		if binary.Read(ent, ec.ByteOrder, &rel) != nil {
			return nil, errors.Errorf("can't parse relocation at offset %v", off)
		}

		symNo := int(elf.R_SYM64(rel.Info) - 1)
		if symNo >= len(ec.symbols) {
			return nil, errors.Errorf("relocation at offset %d: symbol %v doesnt exist", off, symNo)
		}

		rels[rel.Off] = ec.symbols[symNo].Name
	}
	return rels, nil
}

func symbolsPerSection(symbols []elf.Symbol) map[elf.SectionIndex]map[uint64]string {
	result := make(map[elf.SectionIndex]map[uint64]string)
	for i, sym := range symbols {
		switch elf.ST_TYPE(sym.Info) {
		case elf.STT_NOTYPE:
			// Older versions of LLVM doesn't tag
			// symbols correctly.
			break
		case elf.STT_OBJECT:
			break
		case elf.STT_FUNC:
			break
		default:
			continue
		}

		if sym.Name == "" {
			continue
		}

		idx := sym.Section
		if _, ok := result[idx]; !ok {
			result[idx] = make(map[uint64]string)
		}
		result[idx][sym.Value] = symbols[i].Name
	}
	return result
}

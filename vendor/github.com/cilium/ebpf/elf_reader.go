package ebpf

import (
	"bytes"
	"debug/elf"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"strings"

	"github.com/cilium/ebpf/asm"
	"github.com/cilium/ebpf/internal"
	"github.com/cilium/ebpf/internal/btf"
	"github.com/cilium/ebpf/internal/unix"
)

type elfCode struct {
	*elf.File
	symbols           []elf.Symbol
	symbolsPerSection map[elf.SectionIndex]map[uint64]elf.Symbol
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
	if err != nil {
		return nil, fmt.Errorf("file %s: %w", file, err)
	}
	return spec, nil
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
		return nil, fmt.Errorf("load symbols: %v", err)
	}

	ec := &elfCode{f, symbols, symbolsPerSection(symbols), "", 0}

	var (
		licenseSection *elf.Section
		versionSection *elf.Section
		btfMaps        = make(map[elf.SectionIndex]*elf.Section)
		progSections   = make(map[elf.SectionIndex]*elf.Section)
		relSections    = make(map[elf.SectionIndex]*elf.Section)
		mapSections    = make(map[elf.SectionIndex]*elf.Section)
		dataSections   = make(map[elf.SectionIndex]*elf.Section)
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
		case sec.Name == ".bss" || sec.Name == ".rodata" || sec.Name == ".data":
			dataSections[elf.SectionIndex(i)] = sec
		case sec.Type == elf.SHT_REL:
			if int(sec.Info) >= len(ec.Sections) {
				return nil, fmt.Errorf("found relocation section %v for missing section %v", i, sec.Info)
			}

			// Store relocations under the section index of the target
			idx := elf.SectionIndex(sec.Info)
			if relSections[idx] != nil {
				return nil, fmt.Errorf("section %d has multiple relocation sections", sec.Info)
			}
			relSections[idx] = sec
		case sec.Type == elf.SHT_PROGBITS && (sec.Flags&elf.SHF_EXECINSTR) != 0 && sec.Size > 0:
			progSections[elf.SectionIndex(i)] = sec
		}
	}

	ec.license, err = loadLicense(licenseSection)
	if err != nil {
		return nil, fmt.Errorf("load license: %w", err)
	}

	ec.version, err = loadVersion(versionSection, ec.ByteOrder)
	if err != nil {
		return nil, fmt.Errorf("load version: %w", err)
	}

	btfSpec, err := btf.LoadSpecFromReader(rd)
	if err != nil {
		return nil, fmt.Errorf("load BTF: %w", err)
	}

	relocations, referencedSections, err := ec.loadRelocations(relSections)
	if err != nil {
		return nil, fmt.Errorf("load relocations: %w", err)
	}

	maps := make(map[string]*MapSpec)
	if err := ec.loadMaps(maps, mapSections); err != nil {
		return nil, fmt.Errorf("load maps: %w", err)
	}

	if len(btfMaps) > 0 {
		if err := ec.loadBTFMaps(maps, btfMaps, btfSpec); err != nil {
			return nil, fmt.Errorf("load BTF maps: %w", err)
		}
	}

	if len(dataSections) > 0 {
		for idx := range dataSections {
			if !referencedSections[idx] {
				// Prune data sections which are not referenced by any
				// instructions.
				delete(dataSections, idx)
			}
		}

		if err := ec.loadDataSections(maps, dataSections, btfSpec); err != nil {
			return nil, fmt.Errorf("load data sections: %w", err)
		}
	}

	progs, err := ec.loadPrograms(progSections, relocations, btfSpec)
	if err != nil {
		return nil, fmt.Errorf("load programs: %w", err)
	}

	return &CollectionSpec{maps, progs}, nil
}

func loadLicense(sec *elf.Section) (string, error) {
	if sec == nil {
		return "", nil
	}

	data, err := sec.Data()
	if err != nil {
		return "", fmt.Errorf("section %s: %v", sec.Name, err)
	}
	return string(bytes.TrimRight(data, "\000")), nil
}

func loadVersion(sec *elf.Section, bo binary.ByteOrder) (uint32, error) {
	if sec == nil {
		return 0, nil
	}

	var version uint32
	if err := binary.Read(sec.Open(), bo, &version); err != nil {
		return 0, fmt.Errorf("section %s: %v", sec.Name, err)
	}
	return version, nil
}

func (ec *elfCode) loadPrograms(progSections map[elf.SectionIndex]*elf.Section, relocations map[elf.SectionIndex]map[uint64]elf.Symbol, btfSpec *btf.Spec) (map[string]*ProgramSpec, error) {
	var (
		progs []*ProgramSpec
		libs  []*ProgramSpec
	)

	for idx, sec := range progSections {
		syms := ec.symbolsPerSection[idx]
		if len(syms) == 0 {
			return nil, fmt.Errorf("section %v: missing symbols", sec.Name)
		}

		funcSym, ok := syms[0]
		if !ok {
			return nil, fmt.Errorf("section %v: no label at start", sec.Name)
		}

		insns, length, err := ec.loadInstructions(sec, syms, relocations[idx])
		if err != nil {
			return nil, fmt.Errorf("program %s: can't unmarshal instructions: %w", funcSym.Name, err)
		}

		progType, attachType, attachTo := getProgType(sec.Name)

		spec := &ProgramSpec{
			Name:          funcSym.Name,
			Type:          progType,
			AttachType:    attachType,
			AttachTo:      attachTo,
			License:       ec.license,
			KernelVersion: ec.version,
			Instructions:  insns,
			ByteOrder:     ec.ByteOrder,
		}

		if btfSpec != nil {
			spec.BTF, err = btfSpec.Program(sec.Name, length)
			if err != nil && !errors.Is(err, btf.ErrNoExtendedInfo) {
				return nil, fmt.Errorf("program %s: %w", funcSym.Name, err)
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
			return nil, fmt.Errorf("program %s: %w", prog.Name, err)
		}
		res[prog.Name] = prog
	}

	return res, nil
}

func (ec *elfCode) loadInstructions(section *elf.Section, symbols, relocations map[uint64]elf.Symbol) (asm.Instructions, uint64, error) {
	var (
		r      = section.Open()
		insns  asm.Instructions
		offset uint64
	)
	for {
		var ins asm.Instruction
		n, err := ins.Unmarshal(r, ec.ByteOrder)
		if err == io.EOF {
			return insns, offset, nil
		}
		if err != nil {
			return nil, 0, fmt.Errorf("offset %d: %w", offset, err)
		}

		ins.Symbol = symbols[offset].Name

		if rel, ok := relocations[offset]; ok {
			if err = ec.relocateInstruction(&ins, rel); err != nil {
				return nil, 0, fmt.Errorf("offset %d: can't relocate instruction: %w", offset, err)
			}
		}

		insns = append(insns, ins)
		offset += n
	}
}

func (ec *elfCode) relocateInstruction(ins *asm.Instruction, rel elf.Symbol) error {
	var (
		typ  = elf.ST_TYPE(rel.Info)
		bind = elf.ST_BIND(rel.Info)
		name = rel.Name
	)

	if typ == elf.STT_SECTION {
		// Symbols with section type do not have a name set. Get it
		// from the section itself.
		idx := int(rel.Section)
		if idx > len(ec.Sections) {
			return errors.New("out-of-bounds section index")
		}

		name = ec.Sections[idx].Name
	}

outer:
	switch {
	case ins.OpCode == asm.LoadImmOp(asm.DWord):
		// There are two distinct types of a load from a map:
		// a direct one, where the value is extracted without
		// a call to map_lookup_elem in eBPF, and an indirect one
		// that goes via the helper. They are distinguished by
		// different relocations.
		switch typ {
		case elf.STT_SECTION:
			// This is a direct load since the referenced symbol is a
			// section. Weirdly, the offset of the real symbol in the
			// section is encoded in the instruction stream.
			if bind != elf.STB_LOCAL {
				return fmt.Errorf("direct load: %s: unsupported relocation %s", name, bind)
			}

			// For some reason, clang encodes the offset of the symbol its
			// section in the first basic BPF instruction, while the kernel
			// expects it in the second one.
			ins.Constant <<= 32
			ins.Src = asm.PseudoMapValue

		case elf.STT_NOTYPE:
			if bind == elf.STB_GLOBAL && rel.Section == elf.SHN_UNDEF {
				// This is a relocation generated by inline assembly.
				// We can't do more than assigning ins.Reference.
				break outer
			}

			// This is an ELF generated on clang < 8, which doesn't tag
			// relocations appropriately.
			fallthrough

		case elf.STT_OBJECT:
			if bind != elf.STB_GLOBAL {
				return fmt.Errorf("load: %s: unsupported binding: %s", name, bind)
			}

			ins.Src = asm.PseudoMapFD

		default:
			return fmt.Errorf("load: %s: unsupported relocation: %s", name, typ)
		}

		// Mark the instruction as needing an update when creating the
		// collection.
		if err := ins.RewriteMapPtr(-1); err != nil {
			return err
		}

	case ins.OpCode.JumpOp() == asm.Call:
		if ins.Src != asm.PseudoCall {
			return fmt.Errorf("call: %s: incorrect source register", name)
		}

		switch typ {
		case elf.STT_NOTYPE, elf.STT_FUNC:
			if bind != elf.STB_GLOBAL {
				return fmt.Errorf("call: %s: unsupported binding: %s", name, bind)
			}

		case elf.STT_SECTION:
			if bind != elf.STB_LOCAL {
				return fmt.Errorf("call: %s: unsupported binding: %s", name, bind)
			}

			// The function we want to call is in the indicated section,
			// at the offset encoded in the instruction itself. Reverse
			// the calculation to find the real function we're looking for.
			// A value of -1 references the first instruction in the section.
			offset := int64(int32(ins.Constant)+1) * asm.InstructionSize
			if offset < 0 {
				return fmt.Errorf("call: %s: invalid offset %d", name, offset)
			}

			sym, ok := ec.symbolsPerSection[rel.Section][uint64(offset)]
			if !ok {
				return fmt.Errorf("call: %s: no symbol at offset %d", name, offset)
			}

			ins.Constant = -1
			name = sym.Name

		default:
			return fmt.Errorf("call: %s: invalid symbol type %s", name, typ)
		}

	default:
		return fmt.Errorf("relocation for unsupported instruction: %s", ins.OpCode)
	}

	ins.Reference = name
	return nil
}

func (ec *elfCode) loadMaps(maps map[string]*MapSpec, mapSections map[elf.SectionIndex]*elf.Section) error {
	for idx, sec := range mapSections {
		syms := ec.symbolsPerSection[idx]
		if len(syms) == 0 {
			return fmt.Errorf("section %v: no symbols", sec.Name)
		}

		if sec.Size%uint64(len(syms)) != 0 {
			return fmt.Errorf("section %v: map descriptors are not of equal size", sec.Name)
		}

		var (
			r    = sec.Open()
			size = sec.Size / uint64(len(syms))
		)
		for i, offset := 0, uint64(0); i < len(syms); i, offset = i+1, offset+size {
			mapSym, ok := syms[offset]
			if !ok {
				return fmt.Errorf("section %s: missing symbol for map at offset %d", sec.Name, offset)
			}

			if maps[mapSym.Name] != nil {
				return fmt.Errorf("section %v: map %v already exists", sec.Name, mapSym)
			}

			lr := io.LimitReader(r, int64(size))

			spec := MapSpec{
				Name: SanitizeName(mapSym.Name, -1),
			}
			switch {
			case binary.Read(lr, ec.ByteOrder, &spec.Type) != nil:
				return fmt.Errorf("map %v: missing type", mapSym)
			case binary.Read(lr, ec.ByteOrder, &spec.KeySize) != nil:
				return fmt.Errorf("map %v: missing key size", mapSym)
			case binary.Read(lr, ec.ByteOrder, &spec.ValueSize) != nil:
				return fmt.Errorf("map %v: missing value size", mapSym)
			case binary.Read(lr, ec.ByteOrder, &spec.MaxEntries) != nil:
				return fmt.Errorf("map %v: missing max entries", mapSym)
			case binary.Read(lr, ec.ByteOrder, &spec.Flags) != nil:
				return fmt.Errorf("map %v: missing flags", mapSym)
			}

			if _, err := io.Copy(internal.DiscardZeroes{}, lr); err != nil {
				return fmt.Errorf("map %v: unknown and non-zero fields in definition", mapSym)
			}

			maps[mapSym.Name] = &spec
		}
	}

	return nil
}

func (ec *elfCode) loadBTFMaps(maps map[string]*MapSpec, mapSections map[elf.SectionIndex]*elf.Section, spec *btf.Spec) error {
	if spec == nil {
		return fmt.Errorf("missing BTF")
	}

	for idx, sec := range mapSections {
		syms := ec.symbolsPerSection[idx]
		if len(syms) == 0 {
			return fmt.Errorf("section %v: no symbols", sec.Name)
		}

		for _, sym := range syms {
			name := sym.Name
			if maps[name] != nil {
				return fmt.Errorf("section %v: map %v already exists", sec.Name, sym)
			}

			mapSpec, err := mapSpecFromBTF(spec, name)
			if err != nil {
				return fmt.Errorf("map %v: %w", name, err)
			}

			maps[name] = mapSpec
		}
	}

	return nil
}

func mapSpecFromBTF(spec *btf.Spec, name string) (*MapSpec, error) {
	btfMap, btfMapMembers, err := spec.Map(name)
	if err != nil {
		return nil, fmt.Errorf("can't get BTF: %w", err)
	}

	keyType := btf.MapKey(btfMap)
	size, err := btf.Sizeof(keyType)
	if err != nil {
		return nil, fmt.Errorf("can't get size of BTF key: %w", err)
	}
	keySize := uint32(size)

	valueType := btf.MapValue(btfMap)
	size, err = btf.Sizeof(valueType)
	if err != nil {
		return nil, fmt.Errorf("can't get size of BTF value: %w", err)
	}
	valueSize := uint32(size)

	var (
		mapType, flags, maxEntries uint32
	)
	for _, member := range btfMapMembers {
		switch member.Name {
		case "type":
			mapType, err = uintFromBTF(member.Type)
			if err != nil {
				return nil, fmt.Errorf("can't get type: %w", err)
			}

		case "map_flags":
			flags, err = uintFromBTF(member.Type)
			if err != nil {
				return nil, fmt.Errorf("can't get BTF map flags: %w", err)
			}

		case "max_entries":
			maxEntries, err = uintFromBTF(member.Type)
			if err != nil {
				return nil, fmt.Errorf("can't get BTF map max entries: %w", err)
			}

		case "key_size":
			if _, isVoid := keyType.(*btf.Void); !isVoid {
				return nil, errors.New("both key and key_size given")
			}

			keySize, err = uintFromBTF(member.Type)
			if err != nil {
				return nil, fmt.Errorf("can't get BTF key size: %w", err)
			}

		case "value_size":
			if _, isVoid := valueType.(*btf.Void); !isVoid {
				return nil, errors.New("both value and value_size given")
			}

			valueSize, err = uintFromBTF(member.Type)
			if err != nil {
				return nil, fmt.Errorf("can't get BTF value size: %w", err)
			}

		case "pinning":
			pinning, err := uintFromBTF(member.Type)
			if err != nil {
				return nil, fmt.Errorf("can't get pinning: %w", err)
			}

			if pinning != 0 {
				return nil, fmt.Errorf("'pinning' attribute not supported: %w", ErrNotSupported)
			}

		case "key", "value":
		default:
			return nil, fmt.Errorf("unrecognized field %s in BTF map definition", member.Name)
		}
	}

	return &MapSpec{
		Type:       MapType(mapType),
		KeySize:    keySize,
		ValueSize:  valueSize,
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
		return 0, fmt.Errorf("not a pointer: %v", typ)
	}

	arr, ok := ptr.Target.(*btf.Array)
	if !ok {
		return 0, fmt.Errorf("not a pointer to array: %v", typ)
	}

	return arr.Nelems, nil
}

func (ec *elfCode) loadDataSections(maps map[string]*MapSpec, dataSections map[elf.SectionIndex]*elf.Section, spec *btf.Spec) error {
	if spec == nil {
		return errors.New("data sections require BTF, make sure all consts are marked as static")
	}

	for _, sec := range dataSections {
		btfMap, err := spec.Datasec(sec.Name)
		if err != nil {
			return err
		}

		data, err := sec.Data()
		if err != nil {
			return fmt.Errorf("data section %s: can't get contents: %w", sec.Name, err)
		}

		if uint64(len(data)) > math.MaxUint32 {
			return fmt.Errorf("data section %s: contents exceed maximum size", sec.Name)
		}

		mapSpec := &MapSpec{
			Name:       SanitizeName(sec.Name, -1),
			Type:       Array,
			KeySize:    4,
			ValueSize:  uint32(len(data)),
			MaxEntries: 1,
			Contents:   []MapKV{{uint32(0), data}},
			BTF:        btfMap,
		}

		switch sec.Name {
		case ".rodata":
			mapSpec.Flags = unix.BPF_F_RDONLY_PROG
			mapSpec.Freeze = true
		case ".bss":
			// The kernel already zero-initializes the map
			mapSpec.Contents = nil
		}

		maps[sec.Name] = mapSpec
	}
	return nil
}

func getProgType(sectionName string) (ProgramType, AttachType, string) {
	types := map[string]struct {
		progType   ProgramType
		attachType AttachType
	}{
		// From https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/tools/lib/bpf/libbpf.c
		"socket":                {SocketFilter, AttachNone},
		"seccomp":               {SocketFilter, AttachNone},
		"kprobe/":               {Kprobe, AttachNone},
		"uprobe/":               {Kprobe, AttachNone},
		"kretprobe/":            {Kprobe, AttachNone},
		"uretprobe/":            {Kprobe, AttachNone},
		"tracepoint/":           {TracePoint, AttachNone},
		"raw_tracepoint/":       {RawTracepoint, AttachNone},
		"xdp":                   {XDP, AttachNone},
		"perf_event":            {PerfEvent, AttachNone},
		"lwt_in":                {LWTIn, AttachNone},
		"lwt_out":               {LWTOut, AttachNone},
		"lwt_xmit":              {LWTXmit, AttachNone},
		"lwt_seg6local":         {LWTSeg6Local, AttachNone},
		"sockops":               {SockOps, AttachCGroupSockOps},
		"sk_skb/stream_parser":  {SkSKB, AttachSkSKBStreamParser},
		"sk_skb/stream_verdict": {SkSKB, AttachSkSKBStreamParser},
		"sk_msg":                {SkMsg, AttachSkSKBStreamVerdict},
		"lirc_mode2":            {LircMode2, AttachLircMode2},
		"flow_dissector":        {FlowDissector, AttachFlowDissector},
		"iter/":                 {Tracing, AttachTraceIter},

		"cgroup_skb/ingress": {CGroupSKB, AttachCGroupInetIngress},
		"cgroup_skb/egress":  {CGroupSKB, AttachCGroupInetEgress},
		"cgroup/dev":         {CGroupDevice, AttachCGroupDevice},
		"cgroup/skb":         {CGroupSKB, AttachNone},
		"cgroup/sock":        {CGroupSock, AttachCGroupInetSockCreate},
		"cgroup/post_bind4":  {CGroupSock, AttachCGroupInet4PostBind},
		"cgroup/post_bind6":  {CGroupSock, AttachCGroupInet6PostBind},
		"cgroup/bind4":       {CGroupSockAddr, AttachCGroupInet4Bind},
		"cgroup/bind6":       {CGroupSockAddr, AttachCGroupInet6Bind},
		"cgroup/connect4":    {CGroupSockAddr, AttachCGroupInet4Connect},
		"cgroup/connect6":    {CGroupSockAddr, AttachCGroupInet6Connect},
		"cgroup/sendmsg4":    {CGroupSockAddr, AttachCGroupUDP4Sendmsg},
		"cgroup/sendmsg6":    {CGroupSockAddr, AttachCGroupUDP6Sendmsg},
		"cgroup/recvmsg4":    {CGroupSockAddr, AttachCGroupUDP4Recvmsg},
		"cgroup/recvmsg6":    {CGroupSockAddr, AttachCGroupUDP6Recvmsg},
		"cgroup/sysctl":      {CGroupSysctl, AttachCGroupSysctl},
		"cgroup/getsockopt":  {CGroupSockopt, AttachCGroupGetsockopt},
		"cgroup/setsockopt":  {CGroupSockopt, AttachCGroupSetsockopt},
		"classifier":         {SchedCLS, AttachNone},
		"action":             {SchedACT, AttachNone},
	}

	for prefix, t := range types {
		if !strings.HasPrefix(sectionName, prefix) {
			continue
		}

		if !strings.HasSuffix(prefix, "/") {
			return t.progType, t.attachType, ""
		}

		return t.progType, t.attachType, sectionName[len(prefix):]
	}

	return UnspecifiedProgram, AttachNone, ""
}

func (ec *elfCode) loadRelocations(sections map[elf.SectionIndex]*elf.Section) (map[elf.SectionIndex]map[uint64]elf.Symbol, map[elf.SectionIndex]bool, error) {
	result := make(map[elf.SectionIndex]map[uint64]elf.Symbol)
	targets := make(map[elf.SectionIndex]bool)
	for idx, sec := range sections {
		rels := make(map[uint64]elf.Symbol)

		if sec.Entsize < 16 {
			return nil, nil, fmt.Errorf("section %s: relocations are less than 16 bytes", sec.Name)
		}

		r := sec.Open()
		for off := uint64(0); off < sec.Size; off += sec.Entsize {
			ent := io.LimitReader(r, int64(sec.Entsize))

			var rel elf.Rel64
			if binary.Read(ent, ec.ByteOrder, &rel) != nil {
				return nil, nil, fmt.Errorf("can't parse relocation at offset %v", off)
			}

			symNo := int(elf.R_SYM64(rel.Info) - 1)
			if symNo >= len(ec.symbols) {
				return nil, nil, fmt.Errorf("relocation at offset %d: symbol %v doesnt exist", off, symNo)
			}

			symbol := ec.symbols[symNo]
			targets[symbol.Section] = true
			rels[rel.Off] = ec.symbols[symNo]
		}

		result[idx] = rels
	}
	return result, targets, nil
}

func symbolsPerSection(symbols []elf.Symbol) map[elf.SectionIndex]map[uint64]elf.Symbol {
	result := make(map[elf.SectionIndex]map[uint64]elf.Symbol)
	for _, sym := range symbols {
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

		if sym.Section == elf.SHN_UNDEF || sym.Section >= elf.SHN_LORESERVE {
			continue
		}

		if sym.Name == "" {
			continue
		}

		idx := sym.Section
		if _, ok := result[idx]; !ok {
			result[idx] = make(map[uint64]elf.Symbol)
		}
		result[idx][sym.Value] = sym
	}
	return result
}

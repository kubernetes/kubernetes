package ebpf

import (
	"bufio"
	"bytes"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"syscall"
	"time"
	"unsafe"

	"github.com/cilium/ebpf/asm"
	"github.com/cilium/ebpf/btf"
	"github.com/cilium/ebpf/internal"
	"github.com/cilium/ebpf/internal/sys"
	"github.com/cilium/ebpf/internal/unix"
)

// MapInfo describes a map.
type MapInfo struct {
	Type       MapType
	id         MapID
	KeySize    uint32
	ValueSize  uint32
	MaxEntries uint32
	Flags      uint32
	// Name as supplied by user space at load time. Available from 4.15.
	Name string
}

func newMapInfoFromFd(fd *sys.FD) (*MapInfo, error) {
	var info sys.MapInfo
	err := sys.ObjInfo(fd, &info)
	if errors.Is(err, syscall.EINVAL) {
		return newMapInfoFromProc(fd)
	}
	if err != nil {
		return nil, err
	}

	return &MapInfo{
		MapType(info.Type),
		MapID(info.Id),
		info.KeySize,
		info.ValueSize,
		info.MaxEntries,
		uint32(info.MapFlags),
		unix.ByteSliceToString(info.Name[:]),
	}, nil
}

func newMapInfoFromProc(fd *sys.FD) (*MapInfo, error) {
	var mi MapInfo
	err := scanFdInfo(fd, map[string]interface{}{
		"map_type":    &mi.Type,
		"key_size":    &mi.KeySize,
		"value_size":  &mi.ValueSize,
		"max_entries": &mi.MaxEntries,
		"map_flags":   &mi.Flags,
	})
	if err != nil {
		return nil, err
	}
	return &mi, nil
}

// ID returns the map ID.
//
// Available from 4.13.
//
// The bool return value indicates whether this optional field is available.
func (mi *MapInfo) ID() (MapID, bool) {
	return mi.id, mi.id > 0
}

// programStats holds statistics of a program.
type programStats struct {
	// Total accumulated runtime of the program ins ns.
	runtime time.Duration
	// Total number of times the program was called.
	runCount uint64
}

// ProgramInfo describes a program.
type ProgramInfo struct {
	Type ProgramType
	id   ProgramID
	// Truncated hash of the BPF bytecode. Available from 4.13.
	Tag string
	// Name as supplied by user space at load time. Available from 4.15.
	Name string

	createdByUID     uint32
	haveCreatedByUID bool
	btf              btf.ID
	stats            *programStats

	maps  []MapID
	insns []byte
}

func newProgramInfoFromFd(fd *sys.FD) (*ProgramInfo, error) {
	var info sys.ProgInfo
	err := sys.ObjInfo(fd, &info)
	if errors.Is(err, syscall.EINVAL) {
		return newProgramInfoFromProc(fd)
	}
	if err != nil {
		return nil, err
	}

	pi := ProgramInfo{
		Type: ProgramType(info.Type),
		id:   ProgramID(info.Id),
		Tag:  hex.EncodeToString(info.Tag[:]),
		Name: unix.ByteSliceToString(info.Name[:]),
		btf:  btf.ID(info.BtfId),
		stats: &programStats{
			runtime:  time.Duration(info.RunTimeNs),
			runCount: info.RunCnt,
		},
	}

	// Start with a clean struct for the second call, otherwise we may get EFAULT.
	var info2 sys.ProgInfo

	if info.NrMapIds > 0 {
		pi.maps = make([]MapID, info.NrMapIds)
		info2.NrMapIds = info.NrMapIds
		info2.MapIds = sys.NewPointer(unsafe.Pointer(&pi.maps[0]))
	} else if haveProgramInfoMapIDs() == nil {
		// This program really has no associated maps.
		pi.maps = make([]MapID, 0)
	} else {
		// The kernel doesn't report associated maps.
		pi.maps = nil
	}

	// createdByUID and NrMapIds were introduced in the same kernel version.
	if pi.maps != nil {
		pi.createdByUID = info.CreatedByUid
		pi.haveCreatedByUID = true
	}

	if info.XlatedProgLen > 0 {
		pi.insns = make([]byte, info.XlatedProgLen)
		info2.XlatedProgLen = info.XlatedProgLen
		info2.XlatedProgInsns = sys.NewSlicePointer(pi.insns)
	}

	if info.NrMapIds > 0 || info.XlatedProgLen > 0 {
		if err := sys.ObjInfo(fd, &info2); err != nil {
			return nil, err
		}
	}

	return &pi, nil
}

func newProgramInfoFromProc(fd *sys.FD) (*ProgramInfo, error) {
	var info ProgramInfo
	err := scanFdInfo(fd, map[string]interface{}{
		"prog_type": &info.Type,
		"prog_tag":  &info.Tag,
	})
	if errors.Is(err, errMissingFields) {
		return nil, &internal.UnsupportedFeatureError{
			Name:           "reading program info from /proc/self/fdinfo",
			MinimumVersion: internal.Version{4, 10, 0},
		}
	}
	if err != nil {
		return nil, err
	}

	return &info, nil
}

// ID returns the program ID.
//
// Available from 4.13.
//
// The bool return value indicates whether this optional field is available.
func (pi *ProgramInfo) ID() (ProgramID, bool) {
	return pi.id, pi.id > 0
}

// CreatedByUID returns the Uid that created the program.
//
// Available from 4.15.
//
// The bool return value indicates whether this optional field is available.
func (pi *ProgramInfo) CreatedByUID() (uint32, bool) {
	return pi.createdByUID, pi.haveCreatedByUID
}

// BTFID returns the BTF ID associated with the program.
//
// The ID is only valid as long as the associated program is kept alive.
// Available from 5.0.
//
// The bool return value indicates whether this optional field is available and
// populated. (The field may be available but not populated if the kernel
// supports the field but the program was loaded without BTF information.)
func (pi *ProgramInfo) BTFID() (btf.ID, bool) {
	return pi.btf, pi.btf > 0
}

// RunCount returns the total number of times the program was called.
//
// Can return 0 if the collection of statistics is not enabled. See EnableStats().
// The bool return value indicates whether this optional field is available.
func (pi *ProgramInfo) RunCount() (uint64, bool) {
	if pi.stats != nil {
		return pi.stats.runCount, true
	}
	return 0, false
}

// Runtime returns the total accumulated runtime of the program.
//
// Can return 0 if the collection of statistics is not enabled. See EnableStats().
// The bool return value indicates whether this optional field is available.
func (pi *ProgramInfo) Runtime() (time.Duration, bool) {
	if pi.stats != nil {
		return pi.stats.runtime, true
	}
	return time.Duration(0), false
}

// Instructions returns the 'xlated' instruction stream of the program
// after it has been verified and rewritten by the kernel. These instructions
// cannot be loaded back into the kernel as-is, this is mainly used for
// inspecting loaded programs for troubleshooting, dumping, etc.
//
// For example, map accesses are made to reference their kernel map IDs,
// not the FDs they had when the program was inserted. Note that before
// the introduction of bpf_insn_prepare_dump in kernel 4.16, xlated
// instructions were not sanitized, making the output even less reusable
// and less likely to round-trip or evaluate to the same program Tag.
//
// The first instruction is marked as a symbol using the Program's name.
//
// Available from 4.13. Requires CAP_BPF or equivalent.
func (pi *ProgramInfo) Instructions() (asm.Instructions, error) {
	// If the calling process is not BPF-capable or if the kernel doesn't
	// support getting xlated instructions, the field will be zero.
	if len(pi.insns) == 0 {
		return nil, fmt.Errorf("insufficient permissions or unsupported kernel: %w", ErrNotSupported)
	}

	r := bytes.NewReader(pi.insns)
	var insns asm.Instructions
	if err := insns.Unmarshal(r, internal.NativeEndian); err != nil {
		return nil, fmt.Errorf("unmarshaling instructions: %w", err)
	}

	// Tag the first instruction with the name of the program, if available.
	insns[0] = insns[0].WithSymbol(pi.Name)

	return insns, nil
}

// MapIDs returns the maps related to the program.
//
// Available from 4.15.
//
// The bool return value indicates whether this optional field is available.
func (pi *ProgramInfo) MapIDs() ([]MapID, bool) {
	return pi.maps, pi.maps != nil
}

func scanFdInfo(fd *sys.FD, fields map[string]interface{}) error {
	fh, err := os.Open(fmt.Sprintf("/proc/self/fdinfo/%d", fd.Int()))
	if err != nil {
		return err
	}
	defer fh.Close()

	if err := scanFdInfoReader(fh, fields); err != nil {
		return fmt.Errorf("%s: %w", fh.Name(), err)
	}
	return nil
}

var errMissingFields = errors.New("missing fields")

func scanFdInfoReader(r io.Reader, fields map[string]interface{}) error {
	var (
		scanner = bufio.NewScanner(r)
		scanned int
	)

	for scanner.Scan() {
		parts := strings.SplitN(scanner.Text(), "\t", 2)
		if len(parts) != 2 {
			continue
		}

		name := strings.TrimSuffix(parts[0], ":")
		field, ok := fields[string(name)]
		if !ok {
			continue
		}

		if n, err := fmt.Sscanln(parts[1], field); err != nil || n != 1 {
			return fmt.Errorf("can't parse field %s: %v", name, err)
		}

		scanned++
	}

	if err := scanner.Err(); err != nil {
		return err
	}

	if len(fields) > 0 && scanned == 0 {
		return ErrNotSupported
	}

	if scanned != len(fields) {
		return errMissingFields
	}

	return nil
}

// EnableStats starts the measuring of the runtime
// and run counts of eBPF programs.
//
// Collecting statistics can have an impact on the performance.
//
// Requires at least 5.8.
func EnableStats(which uint32) (io.Closer, error) {
	fd, err := sys.EnableStats(&sys.EnableStatsAttr{
		Type: which,
	})
	if err != nil {
		return nil, err
	}
	return fd, nil
}

var haveProgramInfoMapIDs = internal.NewFeatureTest("map IDs in program info", "4.15", func() error {
	prog, err := progLoad(asm.Instructions{
		asm.LoadImm(asm.R0, 0, asm.DWord),
		asm.Return(),
	}, SocketFilter, "MIT")
	if err != nil {
		return err
	}
	defer prog.Close()

	err = sys.ObjInfo(prog, &sys.ProgInfo{
		// NB: Don't need to allocate MapIds since the program isn't using
		// any maps.
		NrMapIds: 1,
	})
	if errors.Is(err, unix.EINVAL) {
		// Most likely the syscall doesn't exist.
		return internal.ErrNotSupported
	}
	if errors.Is(err, unix.E2BIG) {
		// We've hit check_uarg_tail_zero on older kernels.
		return internal.ErrNotSupported
	}

	return err
})

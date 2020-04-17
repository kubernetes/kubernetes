package ebpf

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"os"
	"syscall"

	"github.com/pkg/errors"
)

// MapABI are the attributes of a Map which are available across all supported kernels.
type MapABI struct {
	Type       MapType
	KeySize    uint32
	ValueSize  uint32
	MaxEntries uint32
	Flags      uint32
}

func newMapABIFromSpec(spec *MapSpec) *MapABI {
	return &MapABI{
		spec.Type,
		spec.KeySize,
		spec.ValueSize,
		spec.MaxEntries,
		spec.Flags,
	}
}

func newMapABIFromFd(fd *bpfFD) (string, *MapABI, error) {
	info, err := bpfGetMapInfoByFD(fd)
	if err != nil {
		if errors.Cause(err) == syscall.EINVAL {
			abi, err := newMapABIFromProc(fd)
			return "", abi, err
		}
		return "", nil, err
	}

	return "", &MapABI{
		MapType(info.mapType),
		info.keySize,
		info.valueSize,
		info.maxEntries,
		info.flags,
	}, nil
}

func newMapABIFromProc(fd *bpfFD) (*MapABI, error) {
	var abi MapABI
	err := scanFdInfo(fd, map[string]interface{}{
		"map_type":    &abi.Type,
		"key_size":    &abi.KeySize,
		"value_size":  &abi.ValueSize,
		"max_entries": &abi.MaxEntries,
		"map_flags":   &abi.Flags,
	})
	if err != nil {
		return nil, err
	}
	return &abi, nil
}

// Equal returns true if two ABIs have the same values.
func (abi *MapABI) Equal(other *MapABI) bool {
	switch {
	case abi.Type != other.Type:
		return false
	case abi.KeySize != other.KeySize:
		return false
	case abi.ValueSize != other.ValueSize:
		return false
	case abi.MaxEntries != other.MaxEntries:
		return false
	case abi.Flags != other.Flags:
		return false
	default:
		return true
	}
}

// ProgramABI are the attributes of a Program which are available across all supported kernels.
type ProgramABI struct {
	Type ProgramType
}

func newProgramABIFromSpec(spec *ProgramSpec) *ProgramABI {
	return &ProgramABI{
		spec.Type,
	}
}

func newProgramABIFromFd(fd *bpfFD) (string, *ProgramABI, error) {
	info, err := bpfGetProgInfoByFD(fd)
	if err != nil {
		if errors.Cause(err) == syscall.EINVAL {
			return newProgramABIFromProc(fd)
		}

		return "", nil, err
	}

	var name string
	if bpfName := convertCString(info.name[:]); bpfName != "" {
		name = bpfName
	} else {
		name = convertCString(info.tag[:])
	}

	return name, &ProgramABI{
		Type: ProgramType(info.progType),
	}, nil
}

func newProgramABIFromProc(fd *bpfFD) (string, *ProgramABI, error) {
	var (
		abi  ProgramABI
		name string
	)

	err := scanFdInfo(fd, map[string]interface{}{
		"prog_type": &abi.Type,
		"prog_tag":  &name,
	})
	if err != nil {
		return "", nil, err
	}

	return name, &abi, nil
}

func scanFdInfo(fd *bpfFD, fields map[string]interface{}) error {
	raw, err := fd.value()
	if err != nil {
		return err
	}

	fh, err := os.Open(fmt.Sprintf("/proc/self/fdinfo/%d", raw))
	if err != nil {
		return err
	}
	defer fh.Close()

	return errors.Wrap(scanFdInfoReader(fh, fields), fh.Name())
}

func scanFdInfoReader(r io.Reader, fields map[string]interface{}) error {
	var (
		scanner = bufio.NewScanner(r)
		scanned int
	)

	for scanner.Scan() {
		parts := bytes.SplitN(scanner.Bytes(), []byte("\t"), 2)
		if len(parts) != 2 {
			continue
		}

		name := bytes.TrimSuffix(parts[0], []byte(":"))
		field, ok := fields[string(name)]
		if !ok {
			continue
		}

		if n, err := fmt.Fscanln(bytes.NewReader(parts[1]), field); err != nil || n != 1 {
			return errors.Wrapf(err, "can't parse field %s", name)
		}

		scanned++
	}

	if err := scanner.Err(); err != nil {
		return err
	}

	if scanned != len(fields) {
		return errors.Errorf("parsed %d instead of %d fields", scanned, len(fields))
	}

	return nil
}

// Equal returns true if two ABIs have the same values.
func (abi *ProgramABI) Equal(other *ProgramABI) bool {
	switch {
	case abi.Type != other.Type:
		return false
	default:
		return true
	}
}

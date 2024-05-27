package kconfig

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"

	"github.com/cilium/ebpf/btf"
	"github.com/cilium/ebpf/internal"
)

// Find find a kconfig file on the host.
// It first reads from /boot/config- of the current running kernel and tries
// /proc/config.gz if nothing was found in /boot.
// If none of the file provide a kconfig, it returns an error.
func Find() (*os.File, error) {
	kernelRelease, err := internal.KernelRelease()
	if err != nil {
		return nil, fmt.Errorf("cannot get kernel release: %w", err)
	}

	path := "/boot/config-" + kernelRelease
	f, err := os.Open(path)
	if err == nil {
		return f, nil
	}

	f, err = os.Open("/proc/config.gz")
	if err == nil {
		return f, nil
	}

	return nil, fmt.Errorf("neither %s nor /proc/config.gz provide a kconfig", path)
}

// Parse parses the kconfig file for which a reader is given.
// All the CONFIG_* which are in filter and which are set set will be
// put in the returned map as key with their corresponding value as map value.
// If filter is nil, no filtering will occur.
// If the kconfig file is not valid, error will be returned.
func Parse(source io.ReaderAt, filter map[string]struct{}) (map[string]string, error) {
	var r io.Reader
	zr, err := gzip.NewReader(io.NewSectionReader(source, 0, math.MaxInt64))
	if err != nil {
		r = io.NewSectionReader(source, 0, math.MaxInt64)
	} else {
		// Source is gzip compressed, transparently decompress.
		r = zr
	}

	ret := make(map[string]string, len(filter))

	s := bufio.NewScanner(r)

	for s.Scan() {
		line := s.Bytes()
		err = processKconfigLine(line, ret, filter)
		if err != nil {
			return nil, fmt.Errorf("cannot parse line: %w", err)
		}

		if filter != nil && len(ret) == len(filter) {
			break
		}
	}

	if err := s.Err(); err != nil {
		return nil, fmt.Errorf("cannot parse: %w", err)
	}

	if zr != nil {
		return ret, zr.Close()
	}

	return ret, nil
}

// Golang translation of libbpf bpf_object__process_kconfig_line():
// https://github.com/libbpf/libbpf/blob/fbd60dbff51c870f5e80a17c4f2fd639eb80af90/src/libbpf.c#L1874
// It does the same checks but does not put the data inside the BPF map.
func processKconfigLine(line []byte, m map[string]string, filter map[string]struct{}) error {
	// Ignore empty lines and "# CONFIG_* is not set".
	if !bytes.HasPrefix(line, []byte("CONFIG_")) {
		return nil
	}

	key, value, found := bytes.Cut(line, []byte{'='})
	if !found {
		return fmt.Errorf("line %q does not contain separator '='", line)
	}

	if len(value) == 0 {
		return fmt.Errorf("line %q has no value", line)
	}

	if filter != nil {
		// NB: map[string(key)] gets special optimisation help from the compiler
		// and doesn't allocate. Don't turn this into a variable.
		_, ok := filter[string(key)]
		if !ok {
			return nil
		}
	}

	// This can seem odd, but libbpf only sets the value the first time the key is
	// met:
	// https://github.com/torvalds/linux/blob/0d85b27b0cc6/tools/lib/bpf/libbpf.c#L1906-L1908
	_, ok := m[string(key)]
	if !ok {
		m[string(key)] = string(value)
	}

	return nil
}

// PutValue translates the value given as parameter depending on the BTF
// type, the translated value is then written to the byte array.
func PutValue(data []byte, typ btf.Type, value string) error {
	typ = btf.UnderlyingType(typ)

	switch value {
	case "y", "n", "m":
		return putValueTri(data, typ, value)
	default:
		if strings.HasPrefix(value, `"`) {
			return putValueString(data, typ, value)
		}
		return putValueNumber(data, typ, value)
	}
}

// Golang translation of libbpf_tristate enum:
// https://github.com/libbpf/libbpf/blob/fbd60dbff51c870f5e80a17c4f2fd639eb80af90/src/bpf_helpers.h#L169
type triState int

const (
	TriNo     triState = 0
	TriYes    triState = 1
	TriModule triState = 2
)

func putValueTri(data []byte, typ btf.Type, value string) error {
	switch v := typ.(type) {
	case *btf.Int:
		if v.Encoding != btf.Bool {
			return fmt.Errorf("cannot add tri value, expected btf.Bool, got: %v", v.Encoding)
		}

		if v.Size != 1 {
			return fmt.Errorf("cannot add tri value, expected size of 1 byte, got: %d", v.Size)
		}

		switch value {
		case "y":
			data[0] = 1
		case "n":
			data[0] = 0
		default:
			return fmt.Errorf("cannot use %q for btf.Bool", value)
		}
	case *btf.Enum:
		if v.Name != "libbpf_tristate" {
			return fmt.Errorf("cannot use enum %q, only libbpf_tristate is supported", v.Name)
		}

		var tri triState
		switch value {
		case "y":
			tri = TriYes
		case "m":
			tri = TriModule
		case "n":
			tri = TriNo
		default:
			return fmt.Errorf("value %q is not support for libbpf_tristate", value)
		}

		internal.NativeEndian.PutUint64(data, uint64(tri))
	default:
		return fmt.Errorf("cannot add number value, expected btf.Int or btf.Enum, got: %T", v)
	}

	return nil
}

func putValueString(data []byte, typ btf.Type, value string) error {
	array, ok := typ.(*btf.Array)
	if !ok {
		return fmt.Errorf("cannot add string value, expected btf.Array, got %T", array)
	}

	contentType, ok := btf.UnderlyingType(array.Type).(*btf.Int)
	if !ok {
		return fmt.Errorf("cannot add string value, expected array of btf.Int, got %T", contentType)
	}

	// Any Int, which is not bool, of one byte could be used to store char:
	// https://github.com/torvalds/linux/blob/1a5304fecee5/tools/lib/bpf/libbpf.c#L3637-L3638
	if contentType.Size != 1 && contentType.Encoding != btf.Bool {
		return fmt.Errorf("cannot add string value, expected array of btf.Int of size 1, got array of btf.Int of size: %v", contentType.Size)
	}

	if !strings.HasPrefix(value, `"`) || !strings.HasSuffix(value, `"`) {
		return fmt.Errorf(`value %q must start and finish with '"'`, value)
	}

	str := strings.Trim(value, `"`)

	// We need to trim string if the bpf array is smaller.
	if uint32(len(str)) >= array.Nelems {
		str = str[:array.Nelems]
	}

	// Write the string content to .kconfig.
	copy(data, str)

	return nil
}

func putValueNumber(data []byte, typ btf.Type, value string) error {
	integer, ok := typ.(*btf.Int)
	if !ok {
		return fmt.Errorf("cannot add number value, expected *btf.Int, got: %T", integer)
	}

	size := integer.Size
	sizeInBits := size * 8

	var n uint64
	var err error
	if integer.Encoding == btf.Signed {
		parsed, e := strconv.ParseInt(value, 0, int(sizeInBits))

		n = uint64(parsed)
		err = e
	} else {
		parsed, e := strconv.ParseUint(value, 0, int(sizeInBits))

		n = uint64(parsed)
		err = e
	}

	if err != nil {
		return fmt.Errorf("cannot parse value: %w", err)
	}

	return PutInteger(data, integer, n)
}

// PutInteger writes n into data.
//
// integer determines how much is written into data and what the valid values
// are.
func PutInteger(data []byte, integer *btf.Int, n uint64) error {
	// This function should match set_kcfg_value_num in libbpf.
	if integer.Encoding == btf.Bool && n > 1 {
		return fmt.Errorf("invalid boolean value: %d", n)
	}

	if len(data) < int(integer.Size) {
		return fmt.Errorf("can't fit an integer of size %d into a byte slice of length %d", integer.Size, len(data))
	}

	switch integer.Size {
	case 1:
		if integer.Encoding == btf.Signed && (int64(n) > math.MaxInt8 || int64(n) < math.MinInt8) {
			return fmt.Errorf("can't represent %d as a signed integer of size %d", int64(n), integer.Size)
		}
		data[0] = byte(n)
	case 2:
		if integer.Encoding == btf.Signed && (int64(n) > math.MaxInt16 || int64(n) < math.MinInt16) {
			return fmt.Errorf("can't represent %d as a signed integer of size %d", int64(n), integer.Size)
		}
		internal.NativeEndian.PutUint16(data, uint16(n))
	case 4:
		if integer.Encoding == btf.Signed && (int64(n) > math.MaxInt32 || int64(n) < math.MinInt32) {
			return fmt.Errorf("can't represent %d as a signed integer of size %d", int64(n), integer.Size)
		}
		internal.NativeEndian.PutUint32(data, uint32(n))
	case 8:
		internal.NativeEndian.PutUint64(data, uint64(n))
	default:
		return fmt.Errorf("size (%d) is not valid, expected: 1, 2, 4 or 8", integer.Size)
	}

	return nil
}

package internal

import (
	"debug/elf"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"

	"github.com/cilium/ebpf/internal/unix"
)

var (
	errAuxvNoVDSO = errors.New("no vdso address found in auxv")
)

// vdsoVersion returns the LINUX_VERSION_CODE embedded in the vDSO library
// linked into the current process image.
func vdsoVersion() (uint32, error) {
	// Read data from the auxiliary vector, which is normally passed directly
	// to the process. Go does not expose that data, so we must read it from procfs.
	// https://man7.org/linux/man-pages/man3/getauxval.3.html
	av, err := os.Open("/proc/self/auxv")
	if errors.Is(err, unix.EACCES) {
		return 0, fmt.Errorf("opening auxv: %w (process may not be dumpable due to file capabilities)", err)
	}
	if err != nil {
		return 0, fmt.Errorf("opening auxv: %w", err)
	}
	defer av.Close()

	vdsoAddr, err := vdsoMemoryAddress(av)
	if err != nil {
		return 0, fmt.Errorf("finding vDSO memory address: %w", err)
	}

	// Use /proc/self/mem rather than unsafe.Pointer tricks.
	mem, err := os.Open("/proc/self/mem")
	if err != nil {
		return 0, fmt.Errorf("opening mem: %w", err)
	}
	defer mem.Close()

	// Open ELF at provided memory address, as offset into /proc/self/mem.
	c, err := vdsoLinuxVersionCode(io.NewSectionReader(mem, int64(vdsoAddr), math.MaxInt64))
	if err != nil {
		return 0, fmt.Errorf("reading linux version code: %w", err)
	}

	return c, nil
}

// vdsoMemoryAddress returns the memory address of the vDSO library
// linked into the current process image. r is an io.Reader into an auxv blob.
func vdsoMemoryAddress(r io.Reader) (uint64, error) {
	const (
		_AT_NULL         = 0  // End of vector
		_AT_SYSINFO_EHDR = 33 // Offset to vDSO blob in process image
	)

	// Loop through all tag/value pairs in auxv until we find `AT_SYSINFO_EHDR`,
	// the address of a page containing the virtual Dynamic Shared Object (vDSO).
	aux := struct{ Tag, Val uint64 }{}
	for {
		if err := binary.Read(r, NativeEndian, &aux); err != nil {
			return 0, fmt.Errorf("reading auxv entry: %w", err)
		}

		switch aux.Tag {
		case _AT_SYSINFO_EHDR:
			if aux.Val != 0 {
				return aux.Val, nil
			}
			return 0, fmt.Errorf("invalid vDSO address in auxv")
		// _AT_NULL is always the last tag/val pair in the aux vector
		// and can be treated like EOF.
		case _AT_NULL:
			return 0, errAuxvNoVDSO
		}
	}
}

// format described at https://www.man7.org/linux/man-pages/man5/elf.5.html in section 'Notes (Nhdr)'
type elfNoteHeader struct {
	NameSize int32
	DescSize int32
	Type     int32
}

// vdsoLinuxVersionCode returns the LINUX_VERSION_CODE embedded in
// the ELF notes section of the binary provided by the reader.
func vdsoLinuxVersionCode(r io.ReaderAt) (uint32, error) {
	hdr, err := NewSafeELFFile(r)
	if err != nil {
		return 0, fmt.Errorf("reading vDSO ELF: %w", err)
	}

	sections := hdr.SectionsByType(elf.SHT_NOTE)
	if len(sections) == 0 {
		return 0, fmt.Errorf("no note section found in vDSO ELF")
	}

	for _, sec := range sections {
		sr := sec.Open()
		var n elfNoteHeader

		// Read notes until we find one named 'Linux'.
		for {
			if err := binary.Read(sr, hdr.ByteOrder, &n); err != nil {
				if errors.Is(err, io.EOF) {
					// We looked at all the notes in this section
					break
				}
				return 0, fmt.Errorf("reading note header: %w", err)
			}

			// If a note name is defined, it follows the note header.
			var name string
			if n.NameSize > 0 {
				// Read the note name, aligned to 4 bytes.
				buf := make([]byte, Align(n.NameSize, 4))
				if err := binary.Read(sr, hdr.ByteOrder, &buf); err != nil {
					return 0, fmt.Errorf("reading note name: %w", err)
				}

				// Read nul-terminated string.
				name = unix.ByteSliceToString(buf[:n.NameSize])
			}

			// If a note descriptor is defined, it follows the name.
			// It is possible for a note to have a descriptor but not a name.
			if n.DescSize > 0 {
				// LINUX_VERSION_CODE is a uint32 value.
				if name == "Linux" && n.DescSize == 4 && n.Type == 0 {
					var version uint32
					if err := binary.Read(sr, hdr.ByteOrder, &version); err != nil {
						return 0, fmt.Errorf("reading note descriptor: %w", err)
					}
					return version, nil
				}

				// Discard the note descriptor if it exists but we're not interested in it.
				if _, err := io.CopyN(io.Discard, sr, int64(Align(n.DescSize, 4))); err != nil {
					return 0, err
				}
			}
		}
	}

	return 0, fmt.Errorf("no Linux note in ELF")
}

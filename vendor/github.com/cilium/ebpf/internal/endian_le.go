//go:build 386 || amd64 || amd64p32 || arm || arm64 || loong64 || mipsle || mips64le || mips64p32le || ppc64le || riscv64

package internal

import "encoding/binary"

// NativeEndian is set to either binary.BigEndian or binary.LittleEndian,
// depending on the host's endianness.
var NativeEndian = binary.LittleEndian

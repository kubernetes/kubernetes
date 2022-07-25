package vhd

import (
	"encoding/hex"
	"fmt"
	"os"
	"strings"
)

// https://groups.google.com/forum/#!msg/golang-nuts/d0nF_k4dSx4/rPGgfXv6QCoJ
func uuidgen() string {
	b := uuidgenBytes()
	return fmt.Sprintf("%x-%x-%x-%x-%x",
		b[0:4], b[4:6], b[6:8], b[8:10], b[10:])
}

func fmtField(name, value string) {
	fmt.Printf("%-25s%s\n", name+":", value)
}

func uuidgenBytes() []byte {
	f, err := os.Open("/dev/urandom")
	check(err)
	b := make([]byte, 16)
	f.Read(b)
	return b
}

func check(e error) {
	if e != nil {
		panic(e)
	}
}

func hexs(a []byte) string {
	return "0x" + hex.EncodeToString(a[:])
}

func uuid(a []byte) string {
	return fmt.Sprintf("%08x-%04x-%04x-%04x-%04x",
		a[:4],
		a[4:6],
		a[6:8],
		a[8:10],
		a[10:16])
}

func uuidToBytes(uuid string) []byte {
	s := strings.Replace(uuid, "-", "", -1)
	h, err := hex.DecodeString(s)
	check(err)

	return h
}

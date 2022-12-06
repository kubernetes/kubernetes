package dbus

import (
	"encoding/binary"
	"errors"
	"io"
	"unsafe"
)

var nativeEndian binary.ByteOrder

func detectEndianness() binary.ByteOrder {
	var x uint32 = 0x01020304
	if *(*byte)(unsafe.Pointer(&x)) == 0x01 {
		return binary.BigEndian
	}
	return binary.LittleEndian
}

func init() {
	nativeEndian = detectEndianness()
}

type genericTransport struct {
	io.ReadWriteCloser
}

func (t genericTransport) SendNullByte() error {
	_, err := t.Write([]byte{0})
	return err
}

func (t genericTransport) SupportsUnixFDs() bool {
	return false
}

func (t genericTransport) EnableUnixFDs() {}

func (t genericTransport) ReadMessage() (*Message, error) {
	return DecodeMessage(t)
}

func (t genericTransport) SendMessage(msg *Message) error {
	fds, err := msg.CountFds()
	if err != nil {
		return err
	}
	if fds != 0 {
		return errors.New("dbus: unix fd passing not enabled")
	}
	return msg.EncodeTo(t, nativeEndian)
}

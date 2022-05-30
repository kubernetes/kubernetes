// Copyright (C) 2013-2018 by Maxim Bublis <b@codemonkey.ru>
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

package uuid

import (
	"crypto/md5"
	"crypto/rand"
	"crypto/sha1"
	"encoding/binary"
	"errors"
	"fmt"
	"hash"
	"io"
	"net"
	"sync"
	"time"
)

// Difference in 100-nanosecond intervals between
// UUID epoch (October 15, 1582) and Unix epoch (January 1, 1970).
const epochStart = 122192928000000000

type epochFunc func() time.Time

// HWAddrFunc is the function type used to provide hardware (MAC) addresses.
type HWAddrFunc func() (net.HardwareAddr, error)

// DefaultGenerator is the default UUID Generator used by this package.
var DefaultGenerator Generator = NewGen()

// NewV1 returns a UUID based on the current timestamp and MAC address.
func NewV1() (UUID, error) {
	return DefaultGenerator.NewV1()
}

// NewV3 returns a UUID based on the MD5 hash of the namespace UUID and name.
func NewV3(ns UUID, name string) UUID {
	return DefaultGenerator.NewV3(ns, name)
}

// NewV4 returns a randomly generated UUID.
func NewV4() (UUID, error) {
	return DefaultGenerator.NewV4()
}

// NewV5 returns a UUID based on SHA-1 hash of the namespace UUID and name.
func NewV5(ns UUID, name string) UUID {
	return DefaultGenerator.NewV5(ns, name)
}

// NewV6 returns a k-sortable UUID based on a timestamp and 48 bits of
// pseudorandom data. The timestamp in a V6 UUID is the same as V1, with the bit
// order being adjusted to allow the UUID to be k-sortable.
//
// This is implemented based on revision 02 of the Peabody UUID draft, and may
// be subject to change pending further revisions. Until the final specification
// revision is finished, changes required to implement updates to the spec will
// not be considered a breaking change. They will happen as a minor version
// releases until the spec is final.
func NewV6() (UUID, error) {
	return DefaultGenerator.NewV6()
}

// NewV7 returns a k-sortable UUID based on the current UNIX epoch, with the
// ability to configure the timestamp's precision from millisecond all the way
// to nanosecond. The additional precision is supported by reducing the amount
// of pseudorandom data that makes up the rest of the UUID.
//
// If an unknown Precision argument is passed to this method it will panic. As
// such it's strongly encouraged to use the package-provided constants for this
// value.
//
// This is implemented based on revision 02 of the Peabody UUID draft, and may
// be subject to change pending further revisions. Until the final specification
// revision is finished, changes required to implement updates to the spec will
// not be considered a breaking change. They will happen as a minor version
// releases until the spec is final.
func NewV7(p Precision) (UUID, error) {
	return DefaultGenerator.NewV7(p)
}

// Generator provides an interface for generating UUIDs.
type Generator interface {
	NewV1() (UUID, error)
	NewV3(ns UUID, name string) UUID
	NewV4() (UUID, error)
	NewV5(ns UUID, name string) UUID
	NewV6() (UUID, error)
	NewV7(Precision) (UUID, error)
}

// Gen is a reference UUID generator based on the specifications laid out in
// RFC-4122 and DCE 1.1: Authentication and Security Services. This type
// satisfies the Generator interface as defined in this package.
//
// For consumers who are generating V1 UUIDs, but don't want to expose the MAC
// address of the node generating the UUIDs, the NewGenWithHWAF() function has been
// provided as a convenience. See the function's documentation for more info.
//
// The authors of this package do not feel that the majority of users will need
// to obfuscate their MAC address, and so we recommend using NewGen() to create
// a new generator.
type Gen struct {
	clockSequenceOnce sync.Once
	hardwareAddrOnce  sync.Once
	storageMutex      sync.Mutex

	rand io.Reader

	epochFunc     epochFunc
	hwAddrFunc    HWAddrFunc
	lastTime      uint64
	clockSequence uint16
	hardwareAddr  [6]byte

	v7LastTime      uint64
	v7LastSubsec    uint64
	v7ClockSequence uint16
}

// interface check -- build will fail if *Gen doesn't satisfy Generator
var _ Generator = (*Gen)(nil)

// NewGen returns a new instance of Gen with some default values set. Most
// people should use this.
func NewGen() *Gen {
	return NewGenWithHWAF(defaultHWAddrFunc)
}

// NewGenWithHWAF builds a new UUID generator with the HWAddrFunc provided. Most
// consumers should use NewGen() instead.
//
// This is used so that consumers can generate their own MAC addresses, for use
// in the generated UUIDs, if there is some concern about exposing the physical
// address of the machine generating the UUID.
//
// The Gen generator will only invoke the HWAddrFunc once, and cache that MAC
// address for all the future UUIDs generated by it. If you'd like to switch the
// MAC address being used, you'll need to create a new generator using this
// function.
func NewGenWithHWAF(hwaf HWAddrFunc) *Gen {
	return &Gen{
		epochFunc:  time.Now,
		hwAddrFunc: hwaf,
		rand:       rand.Reader,
	}
}

// NewV1 returns a UUID based on the current timestamp and MAC address.
func (g *Gen) NewV1() (UUID, error) {
	u := UUID{}

	timeNow, clockSeq, err := g.getClockSequence()
	if err != nil {
		return Nil, err
	}
	binary.BigEndian.PutUint32(u[0:], uint32(timeNow))
	binary.BigEndian.PutUint16(u[4:], uint16(timeNow>>32))
	binary.BigEndian.PutUint16(u[6:], uint16(timeNow>>48))
	binary.BigEndian.PutUint16(u[8:], clockSeq)

	hardwareAddr, err := g.getHardwareAddr()
	if err != nil {
		return Nil, err
	}
	copy(u[10:], hardwareAddr)

	u.SetVersion(V1)
	u.SetVariant(VariantRFC4122)

	return u, nil
}

// NewV3 returns a UUID based on the MD5 hash of the namespace UUID and name.
func (g *Gen) NewV3(ns UUID, name string) UUID {
	u := newFromHash(md5.New(), ns, name)
	u.SetVersion(V3)
	u.SetVariant(VariantRFC4122)

	return u
}

// NewV4 returns a randomly generated UUID.
func (g *Gen) NewV4() (UUID, error) {
	u := UUID{}
	if _, err := io.ReadFull(g.rand, u[:]); err != nil {
		return Nil, err
	}
	u.SetVersion(V4)
	u.SetVariant(VariantRFC4122)

	return u, nil
}

// NewV5 returns a UUID based on SHA-1 hash of the namespace UUID and name.
func (g *Gen) NewV5(ns UUID, name string) UUID {
	u := newFromHash(sha1.New(), ns, name)
	u.SetVersion(V5)
	u.SetVariant(VariantRFC4122)

	return u
}

// NewV6 returns a k-sortable UUID based on a timestamp and 48 bits of
// pseudorandom data. The timestamp in a V6 UUID is the same as V1, with the bit
// order being adjusted to allow the UUID to be k-sortable.
//
// This is implemented based on revision 02 of the Peabody UUID draft, and may
// be subject to change pending further revisions. Until the final specification
// revision is finished, changes required to implement updates to the spec will
// not be considered a breaking change. They will happen as a minor version
// releases until the spec is final.
func (g *Gen) NewV6() (UUID, error) {
	var u UUID

	if _, err := io.ReadFull(g.rand, u[10:]); err != nil {
		return Nil, err
	}

	timeNow, clockSeq, err := g.getClockSequence()
	if err != nil {
		return Nil, err
	}

	binary.BigEndian.PutUint32(u[0:], uint32(timeNow>>28))   // set time_high
	binary.BigEndian.PutUint16(u[4:], uint16(timeNow>>12))   // set time_mid
	binary.BigEndian.PutUint16(u[6:], uint16(timeNow&0xfff)) // set time_low (minus four version bits)
	binary.BigEndian.PutUint16(u[8:], clockSeq&0x3fff)       // set clk_seq_hi_res (minus two variant bits)

	u.SetVersion(V6)
	u.SetVariant(VariantRFC4122)

	return u, nil
}

// getClockSequence returns the epoch and clock sequence for V1 and V6 UUIDs.
func (g *Gen) getClockSequence() (uint64, uint16, error) {
	var err error
	g.clockSequenceOnce.Do(func() {
		buf := make([]byte, 2)
		if _, err = io.ReadFull(g.rand, buf); err != nil {
			return
		}
		g.clockSequence = binary.BigEndian.Uint16(buf)
	})
	if err != nil {
		return 0, 0, err
	}

	g.storageMutex.Lock()
	defer g.storageMutex.Unlock()

	timeNow := g.getEpoch()
	// Clock didn't change since last UUID generation.
	// Should increase clock sequence.
	if timeNow <= g.lastTime {
		g.clockSequence++
	}
	g.lastTime = timeNow

	return timeNow, g.clockSequence, nil
}

// Precision is used to configure the V7 generator, to specify how precise the
// timestamp within the UUID should be.
type Precision byte

const (
	NanosecondPrecision Precision = iota
	MicrosecondPrecision
	MillisecondPrecision
)

func (p Precision) String() string {
	switch p {
	case NanosecondPrecision:
		return "nanosecond"

	case MicrosecondPrecision:
		return "microsecond"

	case MillisecondPrecision:
		return "millisecond"

	default:
		return "unknown"
	}
}

// Duration returns the time.Duration for a specific precision. If the Precision
// value is not known, this returns 0.
func (p Precision) Duration() time.Duration {
	switch p {
	case NanosecondPrecision:
		return time.Nanosecond

	case MicrosecondPrecision:
		return time.Microsecond

	case MillisecondPrecision:
		return time.Millisecond

	default:
		return 0
	}
}

// NewV7 returns a k-sortable UUID based on the current UNIX epoch, with the
// ability to configure the timestamp's precision from millisecond all the way
// to nanosecond. The additional precision is supported by reducing the amount
// of pseudorandom data that makes up the rest of the UUID.
//
// If an unknown Precision argument is passed to this method it will panic. As
// such it's strongly encouraged to use the package-provided constants for this
// value.
//
// This is implemented based on revision 02 of the Peabody UUID draft, and may
// be subject to change pending further revisions. Until the final specification
// revision is finished, changes required to implement updates to the spec will
// not be considered a breaking change. They will happen as a minor version
// releases until the spec is final.
func (g *Gen) NewV7(p Precision) (UUID, error) {
	var u UUID
	var err error

	switch p {
	case NanosecondPrecision:
		u, err = g.newV7Nano()

	case MicrosecondPrecision:
		u, err = g.newV7Micro()

	case MillisecondPrecision:
		u, err = g.newV7Milli()

	default:
		panic(fmt.Sprintf("unknown precision value %d", p))
	}

	if err != nil {
		return Nil, err
	}

	u.SetVersion(V7)
	u.SetVariant(VariantRFC4122)

	return u, nil
}

func (g *Gen) newV7Milli() (UUID, error) {
	var u UUID

	if _, err := io.ReadFull(g.rand, u[8:]); err != nil {
		return Nil, err
	}

	sec, nano, seq, err := g.getV7ClockSequence(MillisecondPrecision)
	if err != nil {
		return Nil, err
	}

	msec := (nano / 1000000) & 0xfff

	d := (sec << 28)           // set unixts field
	d |= (msec << 16)          // set msec field
	d |= (uint64(seq) & 0xfff) // set seq field

	binary.BigEndian.PutUint64(u[:], d)

	return u, nil
}

func (g *Gen) newV7Micro() (UUID, error) {
	var u UUID

	if _, err := io.ReadFull(g.rand, u[10:]); err != nil {
		return Nil, err
	}

	sec, nano, seq, err := g.getV7ClockSequence(MicrosecondPrecision)
	if err != nil {
		return Nil, err
	}

	usec := nano / 1000
	usech := (usec << 4) & 0xfff0000
	usecl := usec & 0xfff

	d := (sec << 28)   // set unixts field
	d |= usech | usecl // set usec fields

	binary.BigEndian.PutUint64(u[:], d)
	binary.BigEndian.PutUint16(u[8:], seq)

	return u, nil
}

func (g *Gen) newV7Nano() (UUID, error) {
	var u UUID

	if _, err := io.ReadFull(g.rand, u[11:]); err != nil {
		return Nil, err
	}

	sec, nano, seq, err := g.getV7ClockSequence(NanosecondPrecision)
	if err != nil {
		return Nil, err
	}

	nano &= 0x3fffffffff
	nanoh := nano >> 26
	nanom := (nano >> 14) & 0xfff
	nanol := uint16(nano & 0x3fff)

	d := (sec << 28)           // set unixts field
	d |= (nanoh << 16) | nanom // set nsec high and med fields

	binary.BigEndian.PutUint64(u[:], d)
	binary.BigEndian.PutUint16(u[8:], nanol) // set nsec low field

	u[10] = byte(seq) // set seq field

	return u, nil
}

const (
	maxSeq14 = (1 << 14) - 1
	maxSeq12 = (1 << 12) - 1
	maxSeq8  = (1 << 8) - 1
)

// getV7ClockSequence returns the unix epoch, nanoseconds of current second, and
// the sequence for V7 UUIDs.
func (g *Gen) getV7ClockSequence(p Precision) (epoch uint64, nano uint64, seq uint16, err error) {
	g.storageMutex.Lock()
	defer g.storageMutex.Unlock()

	tn := g.epochFunc()
	unix := uint64(tn.Unix())
	nsec := uint64(tn.Nanosecond())

	// V7 UUIDs have more precise requirements around how the clock sequence
	// value is generated and used. Specifically they require that the sequence
	// be zero, unless we've already generated a UUID within this unit of time
	// (millisecond, microsecond, or nanosecond) at which point you should
	// increment the sequence. Likewise if time has warped backwards for some reason (NTP
	// adjustment?), we also increment the clock sequence to reduce the risk of a
	// collision.
	switch {
	case unix < g.v7LastTime:
		g.v7ClockSequence++

	case unix > g.v7LastTime:
		g.v7ClockSequence = 0

	case unix == g.v7LastTime:
		switch p {
		case NanosecondPrecision:
			if nsec <= g.v7LastSubsec {
				if g.v7ClockSequence >= maxSeq8 {
					return 0, 0, 0, errors.New("generating nanosecond precision UUIDv7s too fast: internal clock sequence would roll over")
				}

				g.v7ClockSequence++
			} else {
				g.v7ClockSequence = 0
			}

		case MicrosecondPrecision:
			if nsec/1000 <= g.v7LastSubsec/1000 {
				if g.v7ClockSequence >= maxSeq14 {
					return 0, 0, 0, errors.New("generating microsecond precision UUIDv7s too fast: internal clock sequence would roll over")
				}

				g.v7ClockSequence++
			} else {
				g.v7ClockSequence = 0
			}

		case MillisecondPrecision:
			if nsec/1000000 <= g.v7LastSubsec/1000000 {
				if g.v7ClockSequence >= maxSeq12 {
					return 0, 0, 0, errors.New("generating millisecond precision UUIDv7s too fast: internal clock sequence would roll over")
				}

				g.v7ClockSequence++
			} else {
				g.v7ClockSequence = 0
			}

		default:
			panic(fmt.Sprintf("unknown precision value %d", p))
		}
	}

	g.v7LastTime = unix
	g.v7LastSubsec = nsec

	return unix, nsec, g.v7ClockSequence, nil
}

// Returns the hardware address.
func (g *Gen) getHardwareAddr() ([]byte, error) {
	var err error
	g.hardwareAddrOnce.Do(func() {
		var hwAddr net.HardwareAddr
		if hwAddr, err = g.hwAddrFunc(); err == nil {
			copy(g.hardwareAddr[:], hwAddr)
			return
		}

		// Initialize hardwareAddr randomly in case
		// of real network interfaces absence.
		if _, err = io.ReadFull(g.rand, g.hardwareAddr[:]); err != nil {
			return
		}
		// Set multicast bit as recommended by RFC-4122
		g.hardwareAddr[0] |= 0x01
	})
	if err != nil {
		return []byte{}, err
	}
	return g.hardwareAddr[:], nil
}

// Returns the difference between UUID epoch (October 15, 1582)
// and current time in 100-nanosecond intervals.
func (g *Gen) getEpoch() uint64 {
	return epochStart + uint64(g.epochFunc().UnixNano()/100)
}

// Returns the UUID based on the hashing of the namespace UUID and name.
func newFromHash(h hash.Hash, ns UUID, name string) UUID {
	u := UUID{}
	h.Write(ns[:])
	h.Write([]byte(name))
	copy(u[:], h.Sum(nil))

	return u
}

// Returns the hardware address.
func defaultHWAddrFunc() (net.HardwareAddr, error) {
	ifaces, err := net.Interfaces()
	if err != nil {
		return []byte{}, err
	}
	for _, iface := range ifaces {
		if len(iface.HardwareAddr) >= 6 {
			return iface.HardwareAddr, nil
		}
	}
	return []byte{}, fmt.Errorf("uuid: no HW address found")
}

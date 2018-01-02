package core

import (
	"crypto/rand"
	"encoding/binary"
	"io"
	mrand "math/rand"

	"github.com/cloudflare/cfssl/log"
)

var seeded bool

func seed() error {
	if seeded {
		return nil
	}

	var buf [8]byte
	_, err := io.ReadFull(rand.Reader, buf[:])
	if err != nil {
		return err
	}

	n := int64(binary.LittleEndian.Uint64(buf[:]))
	mrand.Seed(n)
	seeded = true
	return nil
}

func init() {
	err := seed()
	if err != nil {
		log.Errorf("seeding mrand failed: %v", err)
	}
}

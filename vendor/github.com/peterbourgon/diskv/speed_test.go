package diskv

import (
	"fmt"
	"math/rand"
	"testing"
)

func shuffle(keys []string) {
	ints := rand.Perm(len(keys))
	for i := range keys {
		keys[i], keys[ints[i]] = keys[ints[i]], keys[i]
	}
}

func genValue(size int) []byte {
	v := make([]byte, size)
	for i := 0; i < size; i++ {
		v[i] = uint8((rand.Int() % 26) + 97) // a-z
	}
	return v
}

const (
	keyCount = 1000
)

func genKeys() []string {
	keys := make([]string, keyCount)
	for i := 0; i < keyCount; i++ {
		keys[i] = fmt.Sprintf("%d", i)
	}
	return keys
}

func (d *Diskv) load(keys []string, val []byte) {
	for _, key := range keys {
		d.Write(key, val)
	}
}

func benchRead(b *testing.B, size, cachesz int) {
	b.StopTimer()
	d := New(Options{
		BasePath:     "speed-test",
		Transform:    func(string) []string { return []string{} },
		CacheSizeMax: uint64(cachesz),
	})
	defer d.EraseAll()

	keys := genKeys()
	value := genValue(size)
	d.load(keys, value)
	shuffle(keys)
	b.SetBytes(int64(size))

	b.StartTimer()
	for i := 0; i < b.N; i++ {
		_, _ = d.Read(keys[i%len(keys)])
	}
	b.StopTimer()
}

func benchWrite(b *testing.B, size int, withIndex bool) {
	b.StopTimer()

	options := Options{
		BasePath:     "speed-test",
		Transform:    func(string) []string { return []string{} },
		CacheSizeMax: 0,
	}
	if withIndex {
		options.Index = &BTreeIndex{}
		options.IndexLess = strLess
	}

	d := New(options)
	defer d.EraseAll()
	keys := genKeys()
	value := genValue(size)
	shuffle(keys)
	b.SetBytes(int64(size))

	b.StartTimer()
	for i := 0; i < b.N; i++ {
		d.Write(keys[i%len(keys)], value)
	}
	b.StopTimer()
}

func BenchmarkWrite__32B_NoIndex(b *testing.B) {
	benchWrite(b, 32, false)
}

func BenchmarkWrite__1KB_NoIndex(b *testing.B) {
	benchWrite(b, 1024, false)
}

func BenchmarkWrite__4KB_NoIndex(b *testing.B) {
	benchWrite(b, 4096, false)
}

func BenchmarkWrite_10KB_NoIndex(b *testing.B) {
	benchWrite(b, 10240, false)
}

func BenchmarkWrite__32B_WithIndex(b *testing.B) {
	benchWrite(b, 32, true)
}

func BenchmarkWrite__1KB_WithIndex(b *testing.B) {
	benchWrite(b, 1024, true)
}

func BenchmarkWrite__4KB_WithIndex(b *testing.B) {
	benchWrite(b, 4096, true)
}

func BenchmarkWrite_10KB_WithIndex(b *testing.B) {
	benchWrite(b, 10240, true)
}

func BenchmarkRead__32B_NoCache(b *testing.B) {
	benchRead(b, 32, 0)
}

func BenchmarkRead__1KB_NoCache(b *testing.B) {
	benchRead(b, 1024, 0)
}

func BenchmarkRead__4KB_NoCache(b *testing.B) {
	benchRead(b, 4096, 0)
}

func BenchmarkRead_10KB_NoCache(b *testing.B) {
	benchRead(b, 10240, 0)
}

func BenchmarkRead__32B_WithCache(b *testing.B) {
	benchRead(b, 32, keyCount*32*2)
}

func BenchmarkRead__1KB_WithCache(b *testing.B) {
	benchRead(b, 1024, keyCount*1024*2)
}

func BenchmarkRead__4KB_WithCache(b *testing.B) {
	benchRead(b, 4096, keyCount*4096*2)
}

func BenchmarkRead_10KB_WithCache(b *testing.B) {
	benchRead(b, 10240, keyCount*4096*2)
}

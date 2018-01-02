package goproxy_test

import (
	"bytes"
	"crypto/rsa"
	"encoding/binary"
	"github.com/elazarl/goproxy"
	"io"
	"math"
	"math/rand"
	"testing"
)

type RandSeedReader struct {
	r rand.Rand
}

func (r *RandSeedReader) Read(b []byte) (n int, err error) {
	for i := range b {
		b[i] = byte(r.r.Int() & 0xFF)
	}
	return len(b), nil
}

func TestCounterEncDifferentConsecutive(t *testing.T) {
	k, err := rsa.GenerateKey(&RandSeedReader{*rand.New(rand.NewSource(0xFF43109))}, 128)
	fatalOnErr(err, "rsa.GenerateKey", t)
	c, err := goproxy.NewCounterEncryptorRandFromKey(k, []byte("the quick brown fox run over the lazy dog"))
	fatalOnErr(err, "NewCounterEncryptorRandFromKey", t)
	for i := 0; i < 100*1000; i++ {
		var a, b int64
		binary.Read(&c, binary.BigEndian, &a)
		binary.Read(&c, binary.BigEndian, &b)
		if a == b {
			t.Fatal("two consecutive equal int64", a, b)
		}
	}
}

func TestCounterEncIdenticalStreams(t *testing.T) {
	k, err := rsa.GenerateKey(&RandSeedReader{*rand.New(rand.NewSource(0xFF43109))}, 128)
	fatalOnErr(err, "rsa.GenerateKey", t)
	c1, err := goproxy.NewCounterEncryptorRandFromKey(k, []byte("the quick brown fox run over the lazy dog"))
	fatalOnErr(err, "NewCounterEncryptorRandFromKey", t)
	c2, err := goproxy.NewCounterEncryptorRandFromKey(k, []byte("the quick brown fox run over the lazy dog"))
	fatalOnErr(err, "NewCounterEncryptorRandFromKey", t)
	nout := 1000
	out1, out2 := make([]byte, nout), make([]byte, nout)
	io.ReadFull(&c1, out1)
	tmp := out2[:]
	rand.Seed(0xFF43109)
	for len(tmp) > 0 {
		n := 1 + rand.Intn(256)
		if n > len(tmp) {
			n = len(tmp)
		}
		n, err := c2.Read(tmp[:n])
		fatalOnErr(err, "CounterEncryptorRand.Read", t)
		tmp = tmp[n:]
	}
	if !bytes.Equal(out1, out2) {
		t.Error("identical CSPRNG does not produce the same output")
	}
}

func stddev(data []int) float64 {
	var sum, sum_sqr float64 = 0, 0
	for _, h := range data {
		sum += float64(h)
		sum_sqr += float64(h) * float64(h)
	}
	n := float64(len(data))
	variance := (sum_sqr - ((sum * sum) / n)) / (n - 1)
	return math.Sqrt(variance)
}

func TestCounterEncStreamHistogram(t *testing.T) {
	k, err := rsa.GenerateKey(&RandSeedReader{*rand.New(rand.NewSource(0xFF43109))}, 128)
	fatalOnErr(err, "rsa.GenerateKey", t)
	c, err := goproxy.NewCounterEncryptorRandFromKey(k, []byte("the quick brown fox run over the lazy dog"))
	fatalOnErr(err, "NewCounterEncryptorRandFromKey", t)
	nout := 100 * 1000
	out := make([]byte, nout)
	io.ReadFull(&c, out)
	refhist := make([]int, 256)
	for i := 0; i < nout; i++ {
		refhist[rand.Intn(256)]++
	}
	hist := make([]int, 256)
	for _, b := range out {
		hist[int(b)]++
	}
	refstddev, stddev := stddev(refhist), stddev(hist)
	// due to lack of time, I guestimate
	t.Logf("ref:%v - act:%v = %v", refstddev, stddev, math.Abs(refstddev-stddev))
	if math.Abs(refstddev-stddev) >= 1 {
		t.Errorf("stddev of ref histogram different than regular PRNG: %v %v", refstddev, stddev)
	}
}

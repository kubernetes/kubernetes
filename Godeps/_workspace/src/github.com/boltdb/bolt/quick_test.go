package bolt_test

import (
	"bytes"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"reflect"
	"testing/quick"
	"time"
)

// testing/quick defaults to 5 iterations and a random seed.
// You can override these settings from the command line:
//
//   -quick.count     The number of iterations to perform.
//   -quick.seed      The seed to use for randomizing.
//   -quick.maxitems  The maximum number of items to insert into a DB.
//   -quick.maxksize  The maximum size of a key.
//   -quick.maxvsize  The maximum size of a value.
//

var qcount, qseed, qmaxitems, qmaxksize, qmaxvsize int

func init() {
	flag.IntVar(&qcount, "quick.count", 5, "")
	flag.IntVar(&qseed, "quick.seed", int(time.Now().UnixNano())%100000, "")
	flag.IntVar(&qmaxitems, "quick.maxitems", 1000, "")
	flag.IntVar(&qmaxksize, "quick.maxksize", 1024, "")
	flag.IntVar(&qmaxvsize, "quick.maxvsize", 1024, "")
	flag.Parse()
	fmt.Fprintln(os.Stderr, "seed:", qseed)
	fmt.Fprintf(os.Stderr, "quick settings: count=%v, items=%v, ksize=%v, vsize=%v\n", qcount, qmaxitems, qmaxksize, qmaxvsize)
}

func qconfig() *quick.Config {
	return &quick.Config{
		MaxCount: qcount,
		Rand:     rand.New(rand.NewSource(int64(qseed))),
	}
}

type testdata []testdataitem

func (t testdata) Len() int           { return len(t) }
func (t testdata) Swap(i, j int)      { t[i], t[j] = t[j], t[i] }
func (t testdata) Less(i, j int) bool { return bytes.Compare(t[i].Key, t[j].Key) == -1 }

func (t testdata) Generate(rand *rand.Rand, size int) reflect.Value {
	n := rand.Intn(qmaxitems-1) + 1
	items := make(testdata, n)
	for i := 0; i < n; i++ {
		item := &items[i]
		item.Key = randByteSlice(rand, 1, qmaxksize)
		item.Value = randByteSlice(rand, 0, qmaxvsize)
	}
	return reflect.ValueOf(items)
}

type revtestdata []testdataitem

func (t revtestdata) Len() int           { return len(t) }
func (t revtestdata) Swap(i, j int)      { t[i], t[j] = t[j], t[i] }
func (t revtestdata) Less(i, j int) bool { return bytes.Compare(t[i].Key, t[j].Key) == 1 }

type testdataitem struct {
	Key   []byte
	Value []byte
}

func randByteSlice(rand *rand.Rand, minSize, maxSize int) []byte {
	n := rand.Intn(maxSize-minSize) + minSize
	b := make([]byte, n)
	for i := 0; i < n; i++ {
		b[i] = byte(rand.Intn(255))
	}
	return b
}

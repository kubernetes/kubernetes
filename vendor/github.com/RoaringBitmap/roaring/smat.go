// +build gofuzz

/*
# Instructions for smat testing for roaring

[smat](https://github.com/mschoch/smat) is a framework that provides
state machine assisted fuzz testing.

To run the smat tests for roaring...

## Prerequisites

    $ go get github.com/dvyukov/go-fuzz/go-fuzz
    $ go get github.com/dvyukov/go-fuzz/go-fuzz-build

## Steps

1.  Generate initial smat corpus:
```
    go test -tags=gofuzz -run=TestGenerateSmatCorpus
```

2.  Build go-fuzz test program with instrumentation:
```
    go-fuzz-build -func FuzzSmat github.com/RoaringBitmap/roaring
```

3.  Run go-fuzz:
```
    go-fuzz -bin=./roaring-fuzz.zip -workdir=workdir/ -timeout=200
```

You should see output like...
```
2016/09/16 13:58:35 slaves: 8, corpus: 1 (3s ago), crashers: 0, restarts: 1/0, execs: 0 (0/sec), cover: 0, uptime: 3s
2016/09/16 13:58:38 slaves: 8, corpus: 1 (6s ago), crashers: 0, restarts: 1/0, execs: 0 (0/sec), cover: 0, uptime: 6s
2016/09/16 13:58:41 slaves: 8, corpus: 1 (9s ago), crashers: 0, restarts: 1/44, execs: 44 (5/sec), cover: 0, uptime: 9s
2016/09/16 13:58:44 slaves: 8, corpus: 1 (12s ago), crashers: 0, restarts: 1/45, execs: 45 (4/sec), cover: 0, uptime: 12s
2016/09/16 13:58:47 slaves: 8, corpus: 1 (15s ago), crashers: 0, restarts: 1/46, execs: 46 (3/sec), cover: 0, uptime: 15s
2016/09/16 13:58:50 slaves: 8, corpus: 1 (18s ago), crashers: 0, restarts: 1/47, execs: 47 (3/sec), cover: 0, uptime: 18s
2016/09/16 13:58:53 slaves: 8, corpus: 1 (21s ago), crashers: 0, restarts: 1/63, execs: 63 (3/sec), cover: 0, uptime: 21s
2016/09/16 13:58:56 slaves: 8, corpus: 1 (24s ago), crashers: 0, restarts: 1/65, execs: 65 (3/sec), cover: 0, uptime: 24s
2016/09/16 13:58:59 slaves: 8, corpus: 1 (27s ago), crashers: 0, restarts: 1/66, execs: 66 (2/sec), cover: 0, uptime: 27s
2016/09/16 13:59:02 slaves: 8, corpus: 1 (30s ago), crashers: 0, restarts: 1/67, execs: 67 (2/sec), cover: 0, uptime: 30s
2016/09/16 13:59:05 slaves: 8, corpus: 1 (33s ago), crashers: 0, restarts: 1/83, execs: 83 (3/sec), cover: 0, uptime: 33s
2016/09/16 13:59:08 slaves: 8, corpus: 1 (36s ago), crashers: 0, restarts: 1/84, execs: 84 (2/sec), cover: 0, uptime: 36s
2016/09/16 13:59:11 slaves: 8, corpus: 2 (0s ago), crashers: 0, restarts: 1/85, execs: 85 (2/sec), cover: 0, uptime: 39s
2016/09/16 13:59:14 slaves: 8, corpus: 17 (2s ago), crashers: 0, restarts: 1/86, execs: 86 (2/sec), cover: 480, uptime: 42s
2016/09/16 13:59:17 slaves: 8, corpus: 17 (5s ago), crashers: 0, restarts: 1/66, execs: 132 (3/sec), cover: 487, uptime: 45s
2016/09/16 13:59:20 slaves: 8, corpus: 17 (8s ago), crashers: 0, restarts: 1/440, execs: 2645 (55/sec), cover: 487, uptime: 48s

```

Let it run, and if the # of crashers is > 0, check out the reports in
the workdir where you should be able to find the panic goroutine stack
traces.
*/

package roaring

import (
	"fmt"
	"sort"

	"github.com/mschoch/smat"
	"github.com/willf/bitset"
)

// fuzz test using state machine driven by byte stream.
func FuzzSmat(data []byte) int {
	return smat.Fuzz(&smatContext{}, smat.ActionID('S'), smat.ActionID('T'),
		smatActionMap, data)
}

var smatDebug = false

func smatLog(prefix, format string, args ...interface{}) {
	if smatDebug {
		fmt.Print(prefix)
		fmt.Printf(format, args...)
	}
}

type smatContext struct {
	pairs []*smatPair

	// Two registers, x & y.
	x int
	y int

	actions int
}

type smatPair struct {
	bm *Bitmap
	bs *bitset.BitSet
}

// ------------------------------------------------------------------

var smatActionMap = smat.ActionMap{
	smat.ActionID('X'): smatAction("x++", smatWrap(func(c *smatContext) { c.x++ })),
	smat.ActionID('x'): smatAction("x--", smatWrap(func(c *smatContext) { c.x-- })),
	smat.ActionID('Y'): smatAction("y++", smatWrap(func(c *smatContext) { c.y++ })),
	smat.ActionID('y'): smatAction("y--", smatWrap(func(c *smatContext) { c.y-- })),
	smat.ActionID('*'): smatAction("x*y", smatWrap(func(c *smatContext) { c.x = c.x * c.y })),
	smat.ActionID('<'): smatAction("x<<", smatWrap(func(c *smatContext) { c.x = c.x << 1 })),

	smat.ActionID('^'): smatAction("swap", smatWrap(func(c *smatContext) { c.x, c.y = c.y, c.x })),

	smat.ActionID('['): smatAction(" pushPair", smatWrap(smatPushPair)),
	smat.ActionID(']'): smatAction(" popPair", smatWrap(smatPopPair)),

	smat.ActionID('B'): smatAction(" setBit", smatWrap(smatSetBit)),
	smat.ActionID('b'): smatAction(" removeBit", smatWrap(smatRemoveBit)),

	smat.ActionID('o'): smatAction(" or", smatWrap(smatOr)),
	smat.ActionID('a'): smatAction(" and", smatWrap(smatAnd)),

	smat.ActionID('#'): smatAction(" cardinality", smatWrap(smatCardinality)),

	smat.ActionID('O'): smatAction(" orCardinality", smatWrap(smatOrCardinality)),
	smat.ActionID('A'): smatAction(" andCardinality", smatWrap(smatAndCardinality)),

	smat.ActionID('c'): smatAction(" clear", smatWrap(smatClear)),
	smat.ActionID('r'): smatAction(" runOptimize", smatWrap(smatRunOptimize)),

	smat.ActionID('e'): smatAction(" isEmpty", smatWrap(smatIsEmpty)),

	smat.ActionID('i'): smatAction(" intersects", smatWrap(smatIntersects)),

	smat.ActionID('f'): smatAction(" flip", smatWrap(smatFlip)),

	smat.ActionID('-'): smatAction(" difference", smatWrap(smatDifference)),
}

var smatRunningPercentActions []smat.PercentAction

func init() {
	var ids []int
	for actionId := range smatActionMap {
		ids = append(ids, int(actionId))
	}
	sort.Ints(ids)

	pct := 100 / len(smatActionMap)
	for _, actionId := range ids {
		smatRunningPercentActions = append(smatRunningPercentActions,
			smat.PercentAction{pct, smat.ActionID(actionId)})
	}

	smatActionMap[smat.ActionID('S')] = smatAction("SETUP", smatSetupFunc)
	smatActionMap[smat.ActionID('T')] = smatAction("TEARDOWN", smatTeardownFunc)
}

// We only have one smat state: running.
func smatRunning(next byte) smat.ActionID {
	return smat.PercentExecute(next, smatRunningPercentActions...)
}

func smatAction(name string, f func(ctx smat.Context) (smat.State, error)) func(smat.Context) (smat.State, error) {
	return func(ctx smat.Context) (smat.State, error) {
		c := ctx.(*smatContext)
		c.actions++

		smatLog("  ", "%s\n", name)

		return f(ctx)
	}
}

// Creates an smat action func based on a simple callback.
func smatWrap(cb func(c *smatContext)) func(smat.Context) (next smat.State, err error) {
	return func(ctx smat.Context) (next smat.State, err error) {
		c := ctx.(*smatContext)
		cb(c)
		return smatRunning, nil
	}
}

// Invokes a callback function with the input v bounded to len(c.pairs).
func (c *smatContext) withPair(v int, cb func(*smatPair)) {
	if len(c.pairs) > 0 {
		if v < 0 {
			v = -v
		}
		v = v % len(c.pairs)
		cb(c.pairs[v])
	}
}

// ------------------------------------------------------------------

func smatSetupFunc(ctx smat.Context) (next smat.State, err error) {
	return smatRunning, nil
}

func smatTeardownFunc(ctx smat.Context) (next smat.State, err error) {
	return nil, err
}

// ------------------------------------------------------------------

func smatPushPair(c *smatContext) {
	c.pairs = append(c.pairs, &smatPair{
		bm: NewBitmap(),
		bs: bitset.New(100),
	})
}

func smatPopPair(c *smatContext) {
	if len(c.pairs) > 0 {
		c.pairs = c.pairs[0 : len(c.pairs)-1]
	}
}

func smatSetBit(c *smatContext) {
	c.withPair(c.x, func(p *smatPair) {
		y := uint32(c.y)
		p.bm.AddInt(int(y))
		p.bs.Set(uint(y))
		p.checkEquals()
	})
}

func smatRemoveBit(c *smatContext) {
	c.withPair(c.x, func(p *smatPair) {
		y := uint32(c.y)
		p.bm.Remove(y)
		p.bs.Clear(uint(y))
		p.checkEquals()
	})
}

func smatAnd(c *smatContext) {
	c.withPair(c.x, func(px *smatPair) {
		c.withPair(c.y, func(py *smatPair) {
			px.bm.And(py.bm)
			px.bs = px.bs.Intersection(py.bs)
			px.checkEquals()
			py.checkEquals()
		})
	})
}

func smatOr(c *smatContext) {
	c.withPair(c.x, func(px *smatPair) {
		c.withPair(c.y, func(py *smatPair) {
			px.bm.Or(py.bm)
			px.bs = px.bs.Union(py.bs)
			px.checkEquals()
			py.checkEquals()
		})
	})
}

func smatAndCardinality(c *smatContext) {
	c.withPair(c.x, func(px *smatPair) {
		c.withPair(c.y, func(py *smatPair) {
			c0 := px.bm.AndCardinality(py.bm)
			c1 := px.bs.IntersectionCardinality(py.bs)
			if c0 != uint64(c1) {
				panic("expected same add cardinality")
			}
			px.checkEquals()
			py.checkEquals()
		})
	})
}

func smatOrCardinality(c *smatContext) {
	c.withPair(c.x, func(px *smatPair) {
		c.withPair(c.y, func(py *smatPair) {
			c0 := px.bm.OrCardinality(py.bm)
			c1 := px.bs.UnionCardinality(py.bs)
			if c0 != uint64(c1) {
				panic("expected same or cardinality")
			}
			px.checkEquals()
			py.checkEquals()
		})
	})
}

func smatRunOptimize(c *smatContext) {
	c.withPair(c.x, func(px *smatPair) {
		px.bm.RunOptimize()
		px.checkEquals()
	})
}

func smatClear(c *smatContext) {
	c.withPair(c.x, func(px *smatPair) {
		px.bm.Clear()
		px.bs = px.bs.ClearAll()
		px.checkEquals()
	})
}

func smatCardinality(c *smatContext) {
	c.withPair(c.x, func(px *smatPair) {
		c0 := px.bm.GetCardinality()
		c1 := px.bs.Count()
		if c0 != uint64(c1) {
			panic("expected same cardinality")
		}
	})
}

func smatIsEmpty(c *smatContext) {
	c.withPair(c.x, func(px *smatPair) {
		c0 := px.bm.IsEmpty()
		c1 := px.bs.None()
		if c0 != c1 {
			panic("expected same is empty")
		}
	})
}

func smatIntersects(c *smatContext) {
	c.withPair(c.x, func(px *smatPair) {
		c.withPair(c.y, func(py *smatPair) {
			v0 := px.bm.Intersects(py.bm)
			v1 := px.bs.IntersectionCardinality(py.bs) > 0
			if v0 != v1 {
				panic("intersects not equal")
			}

			px.checkEquals()
			py.checkEquals()
		})
	})
}

func smatFlip(c *smatContext) {
	c.withPair(c.x, func(p *smatPair) {
		y := uint32(c.y)
		p.bm.Flip(uint64(y), uint64(y)+1)
		p.bs = p.bs.Flip(uint(y))
		p.checkEquals()
	})
}

func smatDifference(c *smatContext) {
	c.withPair(c.x, func(px *smatPair) {
		c.withPair(c.y, func(py *smatPair) {
			px.bm.AndNot(py.bm)
			px.bs = px.bs.Difference(py.bs)
			px.checkEquals()
			py.checkEquals()
		})
	})
}

func (p *smatPair) checkEquals() {
	if !p.equalsBitSet(p.bs, p.bm) {
		panic("bitset mismatch")
	}
}

func (p *smatPair) equalsBitSet(a *bitset.BitSet, b *Bitmap) bool {
	for i, e := a.NextSet(0); e; i, e = a.NextSet(i + 1) {
		if !b.ContainsInt(int(i)) {
			fmt.Printf("in a bitset, not b bitmap, i: %d\n", i)
			fmt.Printf("  a bitset: %s\n  b bitmap: %s\n",
				a.String(), b.String())
			return false
		}
	}

	i := b.Iterator()
	for i.HasNext() {
		v := i.Next()
		if !a.Test(uint(v)) {
			fmt.Printf("in b bitmap, not a bitset, v: %d\n", v)
			fmt.Printf("  a bitset: %s\n  b bitmap: %s\n",
				a.String(), b.String())
			return false
		}
	}

	return true
}

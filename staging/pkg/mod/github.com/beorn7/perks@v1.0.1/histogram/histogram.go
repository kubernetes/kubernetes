// Package histogram provides a Go implementation of BigML's histogram package
// for Clojure/Java. It is currently experimental.
package histogram

import (
	"container/heap"
	"math"
	"sort"
)

type Bin struct {
	Count int
	Sum   float64
}

func (b *Bin) Update(x *Bin) {
	b.Count += x.Count
	b.Sum += x.Sum
}

func (b *Bin) Mean() float64 {
	return b.Sum / float64(b.Count)
}

type Bins []*Bin

func (bs Bins) Len() int           { return len(bs) }
func (bs Bins) Less(i, j int) bool { return bs[i].Mean() < bs[j].Mean() }
func (bs Bins) Swap(i, j int)      { bs[i], bs[j] = bs[j], bs[i] }

func (bs *Bins) Push(x interface{}) {
	*bs = append(*bs, x.(*Bin))
}

func (bs *Bins) Pop() interface{} {
	return bs.remove(len(*bs) - 1)
}

func (bs *Bins) remove(n int) *Bin {
	if n < 0 || len(*bs) < n {
		return nil
	}
	x := (*bs)[n]
	*bs = append((*bs)[:n], (*bs)[n+1:]...)
	return x
}

type Histogram struct {
	res *reservoir
}

func New(maxBins int) *Histogram {
	return &Histogram{res: newReservoir(maxBins)}
}

func (h *Histogram) Insert(f float64) {
	h.res.insert(&Bin{1, f})
	h.res.compress()
}

func (h *Histogram) Bins() Bins {
	return h.res.bins
}

type reservoir struct {
	n       int
	maxBins int
	bins    Bins
}

func newReservoir(maxBins int) *reservoir {
	return &reservoir{maxBins: maxBins}
}

func (r *reservoir) insert(bin *Bin) {
	r.n += bin.Count
	i := sort.Search(len(r.bins), func(i int) bool {
		return r.bins[i].Mean() >= bin.Mean()
	})
	if i < 0 || i == r.bins.Len() {
		// TODO(blake): Maybe use an .insert(i, bin) instead of
		// performing the extra work of a heap.Push.
		heap.Push(&r.bins, bin)
		return
	}
	r.bins[i].Update(bin)
}

func (r *reservoir) compress() {
	for r.bins.Len() > r.maxBins {
		minGapIndex := -1
		minGap := math.MaxFloat64
		for i := 0; i < r.bins.Len()-1; i++ {
			gap := gapWeight(r.bins[i], r.bins[i+1])
			if minGap > gap {
				minGap = gap
				minGapIndex = i
			}
		}
		prev := r.bins[minGapIndex]
		next := r.bins.remove(minGapIndex + 1)
		prev.Update(next)
	}
}

func gapWeight(prev, next *Bin) float64 {
	return next.Mean() - prev.Mean()
}

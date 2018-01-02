package ccache

import (
	"strconv"
	"testing"
	"time"

	. "github.com/karlseguin/expect"
)

type CacheTests struct{}

func Test_Cache(t *testing.T) {
	Expectify(new(CacheTests), t)
}

func (_ CacheTests) DeletesAValue() {
	cache := New(Configure())
	cache.Set("spice", "flow", time.Minute)
	cache.Set("worm", "sand", time.Minute)
	cache.Delete("spice")
	Expect(cache.Get("spice")).To.Equal(nil)
	Expect(cache.Get("worm").Value()).To.Equal("sand")
}

func (_ CacheTests) FetchesExpiredItems() {
	cache := New(Configure())
	fn := func() (interface{}, error) { return "moo-moo", nil }

	cache.Set("beef", "moo", time.Second*-1)
	Expect(cache.Get("beef").Value()).To.Equal("moo")

	out, _ := cache.Fetch("beef", time.Second, fn)
	Expect(out.Value()).To.Equal("moo-moo")
}

func (_ CacheTests) GCsTheOldestItems() {
	cache := New(Configure().ItemsToPrune(10))
	for i := 0; i < 500; i++ {
		cache.Set(strconv.Itoa(i), i, time.Minute)
	}
	//let the items get promoted (and added to our list)
	time.Sleep(time.Millisecond * 10)
	gcCache(cache)
	Expect(cache.Get("9")).To.Equal(nil)
	Expect(cache.Get("10").Value()).To.Equal(10)
}

func (_ CacheTests) PromotedItemsDontGetPruned() {
	cache := New(Configure().ItemsToPrune(10).GetsPerPromote(1))
	for i := 0; i < 500; i++ {
		cache.Set(strconv.Itoa(i), i, time.Minute)
	}
	time.Sleep(time.Millisecond * 10) //run the worker once to init the list
	cache.Get("9")
	time.Sleep(time.Millisecond * 10)
	gcCache(cache)
	Expect(cache.Get("9").Value()).To.Equal(9)
	Expect(cache.Get("10")).To.Equal(nil)
	Expect(cache.Get("11").Value()).To.Equal(11)
}

func (_ CacheTests) TrackerDoesNotCleanupHeldInstance() {
	cache := New(Configure().ItemsToPrune(10).Track())
	for i := 0; i < 10; i++ {
		cache.Set(strconv.Itoa(i), i, time.Minute)
	}
	item := cache.TrackingGet("0")
	time.Sleep(time.Millisecond * 10)
	gcCache(cache)
	Expect(cache.Get("0").Value()).To.Equal(0)
	Expect(cache.Get("1")).To.Equal(nil)
	item.Release()
	gcCache(cache)
	Expect(cache.Get("0")).To.Equal(nil)
}

func (_ CacheTests) RemovesOldestItemWhenFull() {
	cache := New(Configure().MaxSize(5).ItemsToPrune(1))
	for i := 0; i < 7; i++ {
		cache.Set(strconv.Itoa(i), i, time.Minute)
	}
	time.Sleep(time.Millisecond * 10)
	Expect(cache.Get("0")).To.Equal(nil)
	Expect(cache.Get("1")).To.Equal(nil)
	Expect(cache.Get("2").Value()).To.Equal(2)
}

func (_ CacheTests) RemovesOldestItemWhenFullBySizer() {
	cache := New(Configure().MaxSize(9).ItemsToPrune(2))
	for i := 0; i < 7; i++ {
		cache.Set(strconv.Itoa(i), &SizedItem{i, 2}, time.Minute)
	}
	time.Sleep(time.Millisecond * 10)
	Expect(cache.Get("0")).To.Equal(nil)
	Expect(cache.Get("1")).To.Equal(nil)
	Expect(cache.Get("2")).To.Equal(nil)
	Expect(cache.Get("3")).To.Equal(nil)
	Expect(cache.Get("4").Value().(*SizedItem).id).To.Equal(4)
}

func (_ CacheTests) SetUpdatesSizeOnDelta() {
	cache := New(Configure())
	cache.Set("a", &SizedItem{0, 2}, time.Minute)
	cache.Set("b", &SizedItem{0, 3}, time.Minute)
	time.Sleep(time.Millisecond * 5)
	checkSize(cache, 5)
	cache.Set("b", &SizedItem{0, 3}, time.Minute)
	time.Sleep(time.Millisecond * 5)
	checkSize(cache, 5)
	cache.Set("b", &SizedItem{0, 4}, time.Minute)
	time.Sleep(time.Millisecond * 5)
	checkSize(cache, 6)
	cache.Set("b", &SizedItem{0, 2}, time.Minute)
	time.Sleep(time.Millisecond * 5)
	checkSize(cache, 4)
	cache.Delete("b")
	time.Sleep(time.Millisecond * 100)
	checkSize(cache, 2)
}

func (_ CacheTests) ReplaceDoesNotchangeSizeIfNotSet() {
	cache := New(Configure())
	cache.Set("1", &SizedItem{1, 2}, time.Minute)
	cache.Set("2", &SizedItem{1, 2}, time.Minute)
	cache.Set("3", &SizedItem{1, 2}, time.Minute)
	cache.Replace("4", &SizedItem{1, 2})
	time.Sleep(time.Millisecond * 5)
	checkSize(cache, 6)
}

func (_ CacheTests) ReplaceChangesSize() {
	cache := New(Configure())
	cache.Set("1", &SizedItem{1, 2}, time.Minute)
	cache.Set("2", &SizedItem{1, 2}, time.Minute)

	cache.Replace("2", &SizedItem{1, 2})
	time.Sleep(time.Millisecond * 5)
	checkSize(cache, 4)

	cache.Replace("2", &SizedItem{1, 1})
	time.Sleep(time.Millisecond * 5)
	checkSize(cache, 3)

	cache.Replace("2", &SizedItem{1, 3})
	time.Sleep(time.Millisecond * 5)
	checkSize(cache, 5)
}

type SizedItem struct {
	id int
	s  int64
}

func (s *SizedItem) Size() int64 {
	return s.s
}

func checkSize(cache *Cache, sz int64) {
	cache.Stop()
	Expect(cache.size).To.Equal(sz)
	cache.restart()
}

func gcCache(cache *Cache) {
	cache.Stop()
	cache.gc()
	cache.restart()
}

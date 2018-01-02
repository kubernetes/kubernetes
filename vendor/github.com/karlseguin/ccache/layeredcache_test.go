package ccache

import (
	. "github.com/karlseguin/expect"
	"strconv"
	"testing"
	"time"
)

type LayeredCacheTests struct{}

func Test_LayeredCache(t *testing.T) {
	Expectify(new(LayeredCacheTests), t)
}

func (_ *LayeredCacheTests) GetsANonExistantValue() {
	cache := newLayered()
	Expect(cache.Get("spice", "flow")).To.Equal(nil)
}

func (_ *LayeredCacheTests) SetANewValue() {
	cache := newLayered()
	cache.Set("spice", "flow", "a value", time.Minute)
	Expect(cache.Get("spice", "flow").Value()).To.Equal("a value")
	Expect(cache.Get("spice", "stop")).To.Equal(nil)
}

func (_ *LayeredCacheTests) SetsMultipleValueWithinTheSameLayer() {
	cache := newLayered()
	cache.Set("spice", "flow", "value-a", time.Minute)
	cache.Set("spice", "must", "value-b", time.Minute)
	cache.Set("leto", "sister", "ghanima", time.Minute)
	Expect(cache.Get("spice", "flow").Value()).To.Equal("value-a")
	Expect(cache.Get("spice", "must").Value()).To.Equal("value-b")
	Expect(cache.Get("spice", "worm")).To.Equal(nil)

	Expect(cache.Get("leto", "sister").Value()).To.Equal("ghanima")
	Expect(cache.Get("leto", "brother")).To.Equal(nil)
	Expect(cache.Get("baron", "friend")).To.Equal(nil)
}

func (_ *LayeredCacheTests) ReplaceDoesNothingIfKeyDoesNotExist() {
	cache := newLayered()
	Expect(cache.Replace("spice", "flow", "value-a")).To.Equal(false)
	Expect(cache.Get("spice", "flow")).To.Equal(nil)
}

func (_ *LayeredCacheTests) ReplaceUpdatesTheValue() {
	cache := newLayered()
	cache.Set("spice", "flow", "value-a", time.Minute)
	Expect(cache.Replace("spice", "flow", "value-b")).To.Equal(true)
	Expect(cache.Get("spice", "flow").Value().(string)).To.Equal("value-b")
	//not sure how to test that the TTL hasn't changed sort of a sleep..
}

func (_ *LayeredCacheTests) DeletesAValue() {
	cache := newLayered()
	cache.Set("spice", "flow", "value-a", time.Minute)
	cache.Set("spice", "must", "value-b", time.Minute)
	cache.Set("leto", "sister", "ghanima", time.Minute)
	cache.Delete("spice", "flow")
	Expect(cache.Get("spice", "flow")).To.Equal(nil)
	Expect(cache.Get("spice", "must").Value()).To.Equal("value-b")
	Expect(cache.Get("spice", "worm")).To.Equal(nil)
	Expect(cache.Get("leto", "sister").Value()).To.Equal("ghanima")
}

func (_ *LayeredCacheTests) DeletesALayer() {
	cache := newLayered()
	cache.Set("spice", "flow", "value-a", time.Minute)
	cache.Set("spice", "must", "value-b", time.Minute)
	cache.Set("leto", "sister", "ghanima", time.Minute)
	cache.DeleteAll("spice")
	Expect(cache.Get("spice", "flow")).To.Equal(nil)
	Expect(cache.Get("spice", "must")).To.Equal(nil)
	Expect(cache.Get("spice", "worm")).To.Equal(nil)
	Expect(cache.Get("leto", "sister").Value()).To.Equal("ghanima")
}

func (_ LayeredCacheTests) GCsTheOldestItems() {
	cache := Layered(Configure().ItemsToPrune(10))
	cache.Set("xx", "a", 23, time.Minute)
	for i := 0; i < 500; i++ {
		cache.Set(strconv.Itoa(i), "a", i, time.Minute)
	}
	cache.Set("xx", "b", 9001, time.Minute)
	//let the items get promoted (and added to our list)
	time.Sleep(time.Millisecond * 10)
	gcLayeredCache(cache)
	Expect(cache.Get("xx", "a")).To.Equal(nil)
	Expect(cache.Get("xx", "b").Value()).To.Equal(9001)
	Expect(cache.Get("8", "a")).To.Equal(nil)
	Expect(cache.Get("9", "a").Value()).To.Equal(9)
	Expect(cache.Get("10", "a").Value()).To.Equal(10)
}

func (_ LayeredCacheTests) PromotedItemsDontGetPruned() {
	cache := Layered(Configure().ItemsToPrune(10).GetsPerPromote(1))
	for i := 0; i < 500; i++ {
		cache.Set(strconv.Itoa(i), "a", i, time.Minute)
	}
	time.Sleep(time.Millisecond * 10) //run the worker once to init the list
	cache.Get("9", "a")
	time.Sleep(time.Millisecond * 10)
	gcLayeredCache(cache)
	Expect(cache.Get("9", "a").Value()).To.Equal(9)
	Expect(cache.Get("10", "a")).To.Equal(nil)
	Expect(cache.Get("11", "a").Value()).To.Equal(11)
}

func (_ LayeredCacheTests) TrackerDoesNotCleanupHeldInstance() {
	cache := Layered(Configure().ItemsToPrune(10).Track())
	for i := 0; i < 10; i++ {
		cache.Set(strconv.Itoa(i), "a", i, time.Minute)
	}
	item := cache.TrackingGet("0", "a")
	time.Sleep(time.Millisecond * 10)
	gcLayeredCache(cache)
	Expect(cache.Get("0", "a").Value()).To.Equal(0)
	Expect(cache.Get("1", "a")).To.Equal(nil)
	item.Release()
	gcLayeredCache(cache)
	Expect(cache.Get("0", "a")).To.Equal(nil)
}

func (_ LayeredCacheTests) RemovesOldestItemWhenFull() {
	cache := Layered(Configure().MaxSize(5).ItemsToPrune(1))
	cache.Set("xx", "a", 23, time.Minute)
	for i := 0; i < 7; i++ {
		cache.Set(strconv.Itoa(i), "a", i, time.Minute)
	}
	cache.Set("xx", "b", 9001, time.Minute)
	time.Sleep(time.Millisecond * 10)
	Expect(cache.Get("xx", "a")).To.Equal(nil)
	Expect(cache.Get("0", "a")).To.Equal(nil)
	Expect(cache.Get("1", "a")).To.Equal(nil)
	Expect(cache.Get("2", "a")).To.Equal(nil)
	Expect(cache.Get("3", "a").Value()).To.Equal(3)
	Expect(cache.Get("xx", "b").Value()).To.Equal(9001)
}

func newLayered() *LayeredCache {
	return Layered(Configure())
}

func (_ LayeredCacheTests) RemovesOldestItemWhenFullBySizer() {
	cache := Layered(Configure().MaxSize(9).ItemsToPrune(2))
	for i := 0; i < 7; i++ {
		cache.Set("pri", strconv.Itoa(i), &SizedItem{i, 2}, time.Minute)
	}
	time.Sleep(time.Millisecond * 10)
	Expect(cache.Get("pri", "0")).To.Equal(nil)
	Expect(cache.Get("pri", "1")).To.Equal(nil)
	Expect(cache.Get("pri", "2")).To.Equal(nil)
	Expect(cache.Get("pri", "3")).To.Equal(nil)
	Expect(cache.Get("pri", "4").Value().(*SizedItem).id).To.Equal(4)
}

func (_ LayeredCacheTests) SetUpdatesSizeOnDelta() {
	cache := Layered(Configure())
	cache.Set("pri", "a", &SizedItem{0, 2}, time.Minute)
	cache.Set("pri", "b", &SizedItem{0, 3}, time.Minute)
	time.Sleep(time.Millisecond * 5)
	checkLayeredSize(cache, 5)
	cache.Set("pri", "b", &SizedItem{0, 3}, time.Minute)
	time.Sleep(time.Millisecond * 5)
	checkLayeredSize(cache, 5)
	cache.Set("pri", "b", &SizedItem{0, 4}, time.Minute)
	time.Sleep(time.Millisecond * 5)
	checkLayeredSize(cache, 6)
	cache.Set("pri", "b", &SizedItem{0, 2}, time.Minute)
	cache.Set("sec", "b", &SizedItem{0, 3}, time.Minute)
	time.Sleep(time.Millisecond * 5)
	checkLayeredSize(cache, 7)
	cache.Delete("pri", "b")
	time.Sleep(time.Millisecond * 10)
	checkLayeredSize(cache, 5)
}

func (_ LayeredCacheTests) ReplaceDoesNotchangeSizeIfNotSet() {
	cache := Layered(Configure())
	cache.Set("pri", "1", &SizedItem{1, 2}, time.Minute)
	cache.Set("pri", "2", &SizedItem{1, 2}, time.Minute)
	cache.Set("pri", "3", &SizedItem{1, 2}, time.Minute)
	cache.Replace("sec", "3", &SizedItem{1, 2})
	time.Sleep(time.Millisecond * 5)
	checkLayeredSize(cache, 6)
}

func (_ LayeredCacheTests) ReplaceChangesSize() {
	cache := Layered(Configure())
	cache.Set("pri", "1", &SizedItem{1, 2}, time.Minute)
	cache.Set("pri", "2", &SizedItem{1, 2}, time.Minute)

	cache.Replace("pri", "2", &SizedItem{1, 2})
	time.Sleep(time.Millisecond * 5)
	checkLayeredSize(cache, 4)

	cache.Replace("pri", "2", &SizedItem{1, 1})
	time.Sleep(time.Millisecond * 5)
	checkLayeredSize(cache, 3)

	cache.Replace("pri", "2", &SizedItem{1, 3})
	time.Sleep(time.Millisecond * 5)
	checkLayeredSize(cache, 5)
}

func checkLayeredSize(cache *LayeredCache, sz int64) {
	cache.Stop()
	Expect(cache.size).To.Equal(sz)
	cache.restart()
}

func gcLayeredCache(cache *LayeredCache) {
	cache.Stop()
	cache.gc()
	cache.restart()
}

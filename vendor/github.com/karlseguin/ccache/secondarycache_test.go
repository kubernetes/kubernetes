package ccache

import (
	. "github.com/karlseguin/expect"
	"testing"
	"time"
	"strconv"
)

type SecondaryCacheTests struct{}

func Test_SecondaryCache(t *testing.T) {
	Expectify(new(SecondaryCacheTests), t)
}

func (_ SecondaryCacheTests) GetsANonExistantValue() {
	cache := newLayered().GetOrCreateSecondaryCache("foo")
	Expect(cache).Not.To.Equal(nil)
}

func (_ SecondaryCacheTests) SetANewValue() {
	cache := newLayered()
	cache.Set("spice", "flow", "a value", time.Minute)
	sCache := cache.GetOrCreateSecondaryCache("spice")
	Expect(sCache.Get("flow").Value()).To.Equal("a value")
	Expect(sCache.Get("stop")).To.Equal(nil)
}

func (_ SecondaryCacheTests) ValueCanBeSeenInBothCaches1() {
	cache := newLayered()
	cache.Set("spice", "flow", "a value", time.Minute)
	sCache := cache.GetOrCreateSecondaryCache("spice")
	sCache.Set("orinoco", "another value", time.Minute)
	Expect(sCache.Get("orinoco").Value()).To.Equal("another value")
	Expect(cache.Get("spice", "orinoco").Value()).To.Equal("another value")
}

func (_ SecondaryCacheTests) ValueCanBeSeenInBothCaches2() {
	cache := newLayered()
	sCache := cache.GetOrCreateSecondaryCache("spice")
	sCache.Set("flow", "a value", time.Minute)
	Expect(sCache.Get("flow").Value()).To.Equal("a value")
	Expect(cache.Get("spice", "flow").Value()).To.Equal("a value")
}

func (_ SecondaryCacheTests) DeletesAreReflectedInBothCaches() {
	cache := newLayered()
	cache.Set("spice", "flow", "a value", time.Minute)
	cache.Set("spice", "sister", "ghanima", time.Minute)
	sCache := cache.GetOrCreateSecondaryCache("spice")

	cache.Delete("spice", "flow")
	Expect(cache.Get("spice", "flow")).To.Equal(nil)
	Expect(sCache.Get("flow")).To.Equal(nil)

	sCache.Delete("sister")
	Expect(cache.Get("spice", "sister")).To.Equal(nil)
	Expect(sCache.Get("sister")).To.Equal(nil)
}

func (_ SecondaryCacheTests) ReplaceDoesNothingIfKeyDoesNotExist() {
	cache := newLayered()
	sCache := cache.GetOrCreateSecondaryCache("spice")
	Expect(sCache.Replace("flow", "value-a")).To.Equal(false)
	Expect(cache.Get("spice", "flow")).To.Equal(nil)
}

func (_ SecondaryCacheTests) ReplaceUpdatesTheValue() {
	cache := newLayered()
	cache.Set("spice", "flow", "value-a", time.Minute)
	sCache := cache.GetOrCreateSecondaryCache("spice")
	Expect(sCache.Replace("flow", "value-b")).To.Equal(true)
	Expect(cache.Get("spice", "flow").Value().(string)).To.Equal("value-b")
}

func (_ SecondaryCacheTests) FetchReturnsAnExistingValue() {
	cache := newLayered()
	cache.Set("spice", "flow", "value-a", time.Minute)
	sCache := cache.GetOrCreateSecondaryCache("spice")
	val, _ := sCache.Fetch("flow", time.Minute, func() (interface{}, error) {return "a fetched value", nil})
	Expect(val.Value().(string)).To.Equal("value-a")
}

func (_ SecondaryCacheTests) FetchReturnsANewValue() {
	cache := newLayered()
	sCache := cache.GetOrCreateSecondaryCache("spice")
	val, _ := sCache.Fetch("flow", time.Minute, func() (interface{}, error) {return "a fetched value", nil})
	Expect(val.Value().(string)).To.Equal("a fetched value")
}

func (_ SecondaryCacheTests) TrackerDoesNotCleanupHeldInstance() {
	cache := Layered(Configure().ItemsToPrune(10).Track())
	for i := 0; i < 10; i++ {
		cache.Set(strconv.Itoa(i), "a", i, time.Minute)
	}
	sCache := cache.GetOrCreateSecondaryCache("0")
	item := sCache.TrackingGet("a")
	time.Sleep(time.Millisecond * 10)
	gcLayeredCache(cache)
	Expect(cache.Get("0", "a").Value()).To.Equal(0)
	Expect(cache.Get("1", "a")).To.Equal(nil)
	item.Release()
	gcLayeredCache(cache)
	Expect(cache.Get("0", "a")).To.Equal(nil)
}

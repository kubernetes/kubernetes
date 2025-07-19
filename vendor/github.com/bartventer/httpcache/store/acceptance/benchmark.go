//go:build httpcache_acceptance_benchmarks
// +build httpcache_acceptance_benchmarks

package acceptance

import (
	"errors"
	"testing"

	"github.com/bartventer/httpcache/store/driver"
)

// RunB runs the acceptance tests for a Cache implementation in a benchmark context.
func RunB(b *testing.B, factory Factory) {
	b.Helper()

	key := "benchmark_key"
	value := []byte("benchmark_value")
	b.Run("Get", func(b *testing.B) { benchmarkGet(b, factory.Make, key) })
	b.Run("Set", func(b *testing.B) { benchmarkSet(b, factory.Make, key, value) })
	b.Run("Delete", func(b *testing.B) { benchmarkDelete(b, factory.Make, key) })
}

func benchmarkGet(b *testing.B, factory FactoryFunc, key string) {
	cache, cleanup := factory.Make()
	defer cleanup()

	// Pre-populate the cache
	if err := cache.Set(key, []byte("value")); err != nil {
		b.Fatalf("Set failed: %v", err)
	}

	for b.Loop() {
		if _, err := cache.Get(key); err != nil {
			b.Errorf("Get failed: %v", err)
		}
	}
}

func benchmarkSet(b *testing.B, factory FactoryFunc, key string, value []byte) {
	cache, cleanup := factory.Make()
	defer cleanup()

	for b.Loop() {
		if err := cache.Set(key, value); err != nil {
			b.Errorf("Set failed: %v", err)
		}
	}
}
func benchmarkDelete(b *testing.B, factory FactoryFunc, key string) {
	cache, cleanup := factory.Make()
	defer cleanup()

	// Pre-populate the cache
	if err := cache.Set(key, []byte("value")); err != nil {
		b.Fatalf("Set failed: %v", err)
	}

	for b.Loop() {
		if err := cache.Delete(key); err != nil && !errors.Is(err, driver.ErrNotExist) {
			b.Errorf("Delete failed: %v", err)
		}
	}
}

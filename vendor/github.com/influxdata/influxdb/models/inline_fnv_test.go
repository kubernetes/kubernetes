package models_test

import (
	"hash/fnv"
	"testing"
	"testing/quick"

	"github.com/influxdata/influxdb/models"
)

func TestInlineFNV64aEquivalenceFuzz(t *testing.T) {
	f := func(data []byte) bool {
		stdlibFNV := fnv.New64a()
		stdlibFNV.Write(data)
		want := stdlibFNV.Sum64()

		inlineFNV := models.NewInlineFNV64a()
		inlineFNV.Write(data)
		got := inlineFNV.Sum64()

		return want == got
	}
	cfg := &quick.Config{
		MaxCount: 10000,
	}
	if err := quick.Check(f, cfg); err != nil {
		t.Fatal(err)
	}
}

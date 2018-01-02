package internal

import "testing"

func TestLookupCommandInfo(t *testing.T) {
	for _, n := range []string{"watch", "WATCH", "wAtch"} {
		if LookupCommandInfo(n) == (CommandInfo{}) {
			t.Errorf("LookupCommandInfo(%q) = CommandInfo{}, expected non-zero value", n)
		}
	}
}

func benchmarkLookupCommandInfo(b *testing.B, names ...string) {
	for i := 0; i < b.N; i++ {
		for _, c := range names {
			LookupCommandInfo(c)
		}
	}
}

func BenchmarkLookupCommandInfoCorrectCase(b *testing.B) {
	benchmarkLookupCommandInfo(b, "watch", "WATCH", "monitor", "MONITOR")
}

func BenchmarkLookupCommandInfoMixedCase(b *testing.B) {
	benchmarkLookupCommandInfo(b, "wAtch", "WeTCH", "monItor", "MONiTOR")
}

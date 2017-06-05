package purell

import (
	"testing"
)

var (
	safeUrl        = "HttPS://..iaMHost..Test:443/paTh^A%ef//./%41PaTH/..//?"
	usuallySafeUrl = "HttPS://..iaMHost..Test:443/paTh^A%ef//./%41PaTH/../final/"
	unsafeUrl      = "HttPS://..www.iaMHost..Test:443/paTh^A%ef//./%41PaTH/../final/index.html?t=val1&a=val4&z=val5&a=val1#fragment"
	allDWORDUrl    = "HttPS://1113982867:/paTh^A%ef//./%41PaTH/../final/index.html?t=val1&a=val4&z=val5&a=val1#fragment"
	allOctalUrl    = "HttPS://0102.0146.07.0223:/paTh^A%ef//./%41PaTH/../final/index.html?t=val1&a=val4&z=val5&a=val1#fragment"
	allHexUrl      = "HttPS://0x42660793:/paTh^A%ef//./%41PaTH/../final/index.html?t=val1&a=val4&z=val5&a=val1#fragment"
	allCombinedUrl = "HttPS://..0x42660793.:/paTh^A%ef//./%41PaTH/../final/index.html?t=val1&a=val4&z=val5&a=val1#fragment"
)

func BenchmarkSafe(b *testing.B) {
	for i := 0; i < b.N; i++ {
		NormalizeURLString(safeUrl, FlagsSafe)
	}
}

func BenchmarkUsuallySafe(b *testing.B) {
	for i := 0; i < b.N; i++ {
		NormalizeURLString(usuallySafeUrl, FlagsUsuallySafeGreedy)
	}
}

func BenchmarkUnsafe(b *testing.B) {
	for i := 0; i < b.N; i++ {
		NormalizeURLString(unsafeUrl, FlagsUnsafeGreedy)
	}
}

func BenchmarkAllDWORD(b *testing.B) {
	for i := 0; i < b.N; i++ {
		NormalizeURLString(allDWORDUrl, FlagsAllGreedy)
	}
}

func BenchmarkAllOctal(b *testing.B) {
	for i := 0; i < b.N; i++ {
		NormalizeURLString(allOctalUrl, FlagsAllGreedy)
	}
}

func BenchmarkAllHex(b *testing.B) {
	for i := 0; i < b.N; i++ {
		NormalizeURLString(allHexUrl, FlagsAllGreedy)
	}
}

func BenchmarkAllCombined(b *testing.B) {
	for i := 0; i < b.N; i++ {
		NormalizeURLString(allCombinedUrl, FlagsAllGreedy)
	}
}

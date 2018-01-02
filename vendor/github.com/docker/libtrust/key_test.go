package libtrust

import (
	"testing"
)

type generateFunc func() (PrivateKey, error)

func runGenerateBench(b *testing.B, f generateFunc, name string) {
	for i := 0; i < b.N; i++ {
		_, err := f()
		if err != nil {
			b.Fatalf("Error generating %s: %s", name, err)
		}
	}
}

func runFingerprintBench(b *testing.B, f generateFunc, name string) {
	b.StopTimer()
	// Don't count this relatively slow generation call.
	key, err := f()
	if err != nil {
		b.Fatalf("Error generating %s: %s", name, err)
	}
	b.StartTimer()

	for i := 0; i < b.N; i++ {
		if key.KeyID() == "" {
			b.Fatalf("Error generating key ID for %s", name)
		}
	}
}

func BenchmarkECP256Generate(b *testing.B) {
	runGenerateBench(b, GenerateECP256PrivateKey, "P256")
}

func BenchmarkECP384Generate(b *testing.B) {
	runGenerateBench(b, GenerateECP384PrivateKey, "P384")
}

func BenchmarkECP521Generate(b *testing.B) {
	runGenerateBench(b, GenerateECP521PrivateKey, "P521")
}

func BenchmarkRSA2048Generate(b *testing.B) {
	runGenerateBench(b, GenerateRSA2048PrivateKey, "RSA2048")
}

func BenchmarkRSA3072Generate(b *testing.B) {
	runGenerateBench(b, GenerateRSA3072PrivateKey, "RSA3072")
}

func BenchmarkRSA4096Generate(b *testing.B) {
	runGenerateBench(b, GenerateRSA4096PrivateKey, "RSA4096")
}

func BenchmarkECP256Fingerprint(b *testing.B) {
	runFingerprintBench(b, GenerateECP256PrivateKey, "P256")
}

func BenchmarkECP384Fingerprint(b *testing.B) {
	runFingerprintBench(b, GenerateECP384PrivateKey, "P384")
}

func BenchmarkECP521Fingerprint(b *testing.B) {
	runFingerprintBench(b, GenerateECP521PrivateKey, "P521")
}

func BenchmarkRSA2048Fingerprint(b *testing.B) {
	runFingerprintBench(b, GenerateRSA2048PrivateKey, "RSA2048")
}

func BenchmarkRSA3072Fingerprint(b *testing.B) {
	runFingerprintBench(b, GenerateRSA3072PrivateKey, "RSA3072")
}

func BenchmarkRSA4096Fingerprint(b *testing.B) {
	runFingerprintBench(b, GenerateRSA4096PrivateKey, "RSA4096")
}

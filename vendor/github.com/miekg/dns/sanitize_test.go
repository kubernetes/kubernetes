package dns

import "testing"

func TestDedup(t *testing.T) {
	// make it []string
	testcases := map[[3]RR][]string{
		[...]RR{
			newRR(t, "mIek.nl. IN A 127.0.0.1"),
			newRR(t, "mieK.nl. IN A 127.0.0.1"),
			newRR(t, "miek.Nl. IN A 127.0.0.1"),
		}: {"mIek.nl.\t3600\tIN\tA\t127.0.0.1"},
		[...]RR{
			newRR(t, "miEk.nl. 2000 IN A 127.0.0.1"),
			newRR(t, "mieK.Nl. 1000 IN A 127.0.0.1"),
			newRR(t, "Miek.nL. 500 IN A 127.0.0.1"),
		}: {"miEk.nl.\t500\tIN\tA\t127.0.0.1"},
		[...]RR{
			newRR(t, "miek.nl. IN A 127.0.0.1"),
			newRR(t, "miek.nl. CH A 127.0.0.1"),
			newRR(t, "miek.nl. IN A 127.0.0.1"),
		}: {"miek.nl.\t3600\tIN\tA\t127.0.0.1",
			"miek.nl.\t3600\tCH\tA\t127.0.0.1",
		},
		[...]RR{
			newRR(t, "miek.nl. CH A 127.0.0.1"),
			newRR(t, "miek.nl. IN A 127.0.0.1"),
			newRR(t, "miek.de. IN A 127.0.0.1"),
		}: {"miek.nl.\t3600\tCH\tA\t127.0.0.1",
			"miek.nl.\t3600\tIN\tA\t127.0.0.1",
			"miek.de.\t3600\tIN\tA\t127.0.0.1",
		},
		[...]RR{
			newRR(t, "miek.de. IN A 127.0.0.1"),
			newRR(t, "miek.nl. 200 IN A 127.0.0.1"),
			newRR(t, "miek.nl. 300 IN A 127.0.0.1"),
		}: {"miek.de.\t3600\tIN\tA\t127.0.0.1",
			"miek.nl.\t200\tIN\tA\t127.0.0.1",
		},
	}

	for rr, expected := range testcases {
		out := Dedup([]RR{rr[0], rr[1], rr[2]}, nil)
		for i, o := range out {
			if o.String() != expected[i] {
				t.Fatalf("expected %v, got %v", expected[i], o.String())
			}
		}
	}
}

func BenchmarkDedup(b *testing.B) {
	rrs := []RR{
		newRR(nil, "miEk.nl. 2000 IN A 127.0.0.1"),
		newRR(nil, "mieK.Nl. 1000 IN A 127.0.0.1"),
		newRR(nil, "Miek.nL. 500 IN A 127.0.0.1"),
	}
	m := make(map[string]RR)
	for i := 0; i < b.N; i++ {
		Dedup(rrs, m)
	}
}

func TestNormalizedString(t *testing.T) {
	tests := map[RR]string{
		newRR(t, "mIEk.Nl. 3600 IN A 127.0.0.1"):     "miek.nl.\tIN\tA\t127.0.0.1",
		newRR(t, "m\\ iek.nL. 3600 IN A 127.0.0.1"):  "m\\ iek.nl.\tIN\tA\t127.0.0.1",
		newRR(t, "m\\\tIeK.nl. 3600 in A 127.0.0.1"): "m\\tiek.nl.\tIN\tA\t127.0.0.1",
	}
	for tc, expected := range tests {
		n := normalizedString(tc)
		if n != expected {
			t.Errorf("expected %s, got %s", expected, n)
		}
	}
}

func newRR(t *testing.T, s string) RR {
	r, err := NewRR(s)
	if err != nil {
		t.Logf("newRR: %v", err)
	}
	return r
}

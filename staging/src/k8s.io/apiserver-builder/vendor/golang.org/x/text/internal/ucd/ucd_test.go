package ucd

import (
	"strings"
	"testing"
)

const file = `
# Comments should be skipped
# rune;  bool;  uint; int; float; runes; # Y
0..0005; Y;     0;    2;      -5.25 ;  0 1 2 3 4 5;
6..0007; Yes  ; 6;    1;     -4.25  ;  0006 0007;
8;       T ;    8 ;   0 ;-3.25  ;;# T
9;       True  ;9  ;  -1;-2.25  ;  0009;

# more comments to be ignored
@Part0  

A;       N;   10  ;   -2;  -1.25; ;# N
B;       No;   11 ;   -3;  -0.25; 
C;        False;12;   -4;   0.75;
D;        ;13;-5;1.75;

@Part1   # Another part. 
# We test part comments get removed by not commenting the the next line.
E..10FFFF; F;   14  ; -6;   2.75;
`

var want = []struct {
	start, end rune
}{
	{0x00, 0x05},
	{0x06, 0x07},
	{0x08, 0x08},
	{0x09, 0x09},
	{0x0A, 0x0A},
	{0x0B, 0x0B},
	{0x0C, 0x0C},
	{0x0D, 0x0D},
	{0x0E, 0x10FFFF},
}

func TestGetters(t *testing.T) {
	parts := [][2]string{
		{"Part0", ""},
		{"Part1", "Another part."},
	}
	handler := func(p *Parser) {
		if len(parts) == 0 {
			t.Error("Part handler invoked too many times.")
			return
		}
		want := parts[0]
		parts = parts[1:]
		if got0, got1 := p.String(0), p.Comment(); got0 != want[0] || got1 != want[1] {
			t.Errorf(`part: got %q, %q; want %q"`, got0, got1, want)
		}
	}

	p := New(strings.NewReader(file), KeepRanges, Part(handler))
	for i := 0; p.Next(); i++ {
		start, end := p.Range(0)
		w := want[i]
		if start != w.start || end != w.end {
			t.Fatalf("%d:Range(0); got %#x..%#x; want %#x..%#x", i, start, end, w.start, w.end)
		}
		if w.start == w.end && p.Rune(0) != w.start {
			t.Errorf("%d:Range(0).start: got %U; want %U", i, p.Rune(0), w.start)
		}
		if got, want := p.Bool(1), w.start <= 9; got != want {
			t.Errorf("%d:Bool(1): got %v; want %v", i, got, want)
		}
		if got := p.Rune(4); got != 0 || p.Err() == nil {
			t.Errorf("%d:Rune(%q): got no error; want error", i, p.String(1))
		}
		p.err = nil
		if got := p.Uint(2); rune(got) != start {
			t.Errorf("%d:Uint(2): got %v; want %v", i, got, start)
		}
		if got, want := p.Int(3), 2-i; got != want {
			t.Errorf("%d:Int(3): got %v; want %v", i, got, want)
		}
		if got, want := p.Float(4), -5.25+float64(i); got != want {
			t.Errorf("%d:Int(3): got %v; want %v", i, got, want)
		}
		if got := p.Runes(5); got == nil {
			if p.String(5) != "" {
				t.Errorf("%d:Runes(5): expected non-empty list", i)
			}
		} else {
			if got[0] != start || got[len(got)-1] != end {
				t.Errorf("%d:Runes(5): got %#x; want %#x..%#x", i, got, start, end)
			}
		}
		if got := p.Comment(); got != "" && got != p.String(1) {
			t.Errorf("%d:Comment(): got %v; want %v", i, got, p.String(1))
		}
	}
	if err := p.Err(); err != nil {
		t.Errorf("Parser error: %v", err)
	}
	if len(parts) != 0 {
		t.Errorf("expected %d more invocations of part handler", len(parts))
	}
}

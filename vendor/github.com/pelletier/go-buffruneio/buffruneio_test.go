package buffruneio

import (
	"strings"
	"testing"
)

func assumeRunesArray(t *testing.T, expected []rune, got []rune) {
	if len(expected) != len(got) {
		t.Fatal("expected", len(expected), "runes, but got", len(got))
	}
	for i := 0; i < len(got); i++ {
		if expected[i] != got[i] {
			t.Fatal("expected rune", expected[i], "at index", i, "but got", got[i])
		}
	}
}

func assumeRune(t *testing.T, rd *Reader, r rune) {
	gotRune, err := rd.ReadRune()
	if err != nil {
		t.Fatal("unexpected error", err)
	}
	if gotRune != r {
		t.Fatal("got", string(gotRune),
			"(", []byte(string(gotRune)), ")",
			"expected", string(r),
			"(", []byte(string(r)), ")")
	}
}

func TestReadString(t *testing.T) {
	s := "hello"
	rd := NewReader(strings.NewReader(s))

	assumeRune(t, rd, 'h')
	assumeRune(t, rd, 'e')
	assumeRune(t, rd, 'l')
	assumeRune(t, rd, 'l')
	assumeRune(t, rd, 'o')
	assumeRune(t, rd, EOF)
}

func TestMultipleEOF(t *testing.T) {
	s := ""
	rd := NewReader(strings.NewReader(s))

	assumeRune(t, rd, EOF)
	assumeRune(t, rd, EOF)
}

func TestUnread(t *testing.T) {
	s := "ab"
	rd := NewReader(strings.NewReader(s))

	assumeRune(t, rd, 'a')
	assumeRune(t, rd, 'b')
	rd.UnreadRune()
	assumeRune(t, rd, 'b')
	assumeRune(t, rd, EOF)
}

func TestUnreadEOF(t *testing.T) {
	s := ""
	rd := NewReader(strings.NewReader(s))

	rd.UnreadRune()
	assumeRune(t, rd, EOF)
	assumeRune(t, rd, EOF)
	rd.UnreadRune()
	assumeRune(t, rd, EOF)
}

func TestForget(t *testing.T) {
	s := "hello"
	rd := NewReader(strings.NewReader(s))

	assumeRune(t, rd, 'h')
	assumeRune(t, rd, 'e')
	assumeRune(t, rd, 'l')
	assumeRune(t, rd, 'l')
	rd.Forget()
	if rd.UnreadRune() != ErrNoRuneToUnread {
		t.Fatal("no rune should be available")
	}
}

func TestForgetEmpty(t *testing.T) {
	s := ""
	rd := NewReader(strings.NewReader(s))

	rd.Forget()
	assumeRune(t, rd, EOF)
	rd.Forget()
}

func TestPeekEmpty(t *testing.T) {
	s := ""
	rd := NewReader(strings.NewReader(s))

	runes := rd.Peek(1)
	if len(runes) != 1 {
		t.Fatal("incorrect number of runes", len(runes))
	}
	if runes[0] != EOF {
		t.Fatal("incorrect rune", runes[0])
	}
}

func TestPeek(t *testing.T) {
	s := "a"
	rd := NewReader(strings.NewReader(s))

	runes := rd.Peek(1)
	assumeRunesArray(t, []rune{'a'}, runes)

	runes = rd.Peek(1)
	assumeRunesArray(t, []rune{'a'}, runes)

	assumeRune(t, rd, 'a')
	runes = rd.Peek(1)
	assumeRunesArray(t, []rune{EOF}, runes)

	assumeRune(t, rd, EOF)
}

func TestPeekLarge(t *testing.T) {
	s := "abcdefg"
	rd := NewReader(strings.NewReader(s))

	runes := rd.Peek(100)
	if len(runes) != len(s)+1 {
		t.Fatal("incorrect number of runes", len(runes))
	}
	assumeRunesArray(t, []rune{'a', 'b', 'c', 'd', 'e', 'f', 'g', EOF}, runes)
}

package text

import (
	"bytes"
	"testing"
)

var text = "The quick brown fox jumps over the lazy dog."

func TestWrap(t *testing.T) {
	exp := [][]string{
		{"The", "quick", "brown", "fox"},
		{"jumps", "over", "the", "lazy", "dog."},
	}
	words := bytes.Split([]byte(text), sp)
	got := WrapWords(words, 1, 24, defaultPenalty)
	if len(exp) != len(got) {
		t.Fail()
	}
	for i := range exp {
		if len(exp[i]) != len(got[i]) {
			t.Fail()
		}
		for j := range exp[i] {
			if exp[i][j] != string(got[i][j]) {
				t.Fatal(i, exp[i][j], got[i][j])
			}
		}
	}
}

func TestWrapNarrow(t *testing.T) {
	exp := "The\nquick\nbrown\nfox\njumps\nover\nthe\nlazy\ndog."
	if Wrap(text, 5) != exp {
		t.Fail()
	}
}

func TestWrapOneLine(t *testing.T) {
	exp := "The quick brown fox jumps over the lazy dog."
	if Wrap(text, 500) != exp {
		t.Fail()
	}
}

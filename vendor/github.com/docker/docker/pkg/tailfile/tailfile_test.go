package tailfile

import (
	"io/ioutil"
	"os"
	"testing"
)

func TestTailFile(t *testing.T) {
	f, err := ioutil.TempFile("", "tail-test")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	defer os.RemoveAll(f.Name())
	testFile := []byte(`first line
second line
third line
fourth line
fifth line
next first line
next second line
next third line
next fourth line
next fifth line
last first line
next first line
next second line
next third line
next fourth line
next fifth line
next first line
next second line
next third line
next fourth line
next fifth line
last second line
last third line
last fourth line
last fifth line
truncated line`)
	if _, err := f.Write(testFile); err != nil {
		t.Fatal(err)
	}
	if _, err := f.Seek(0, os.SEEK_SET); err != nil {
		t.Fatal(err)
	}
	expected := []string{"last fourth line", "last fifth line"}
	res, err := TailFile(f, 2)
	if err != nil {
		t.Fatal(err)
	}
	for i, l := range res {
		t.Logf("%s", l)
		if expected[i] != string(l) {
			t.Fatalf("Expected line %s, got %s", expected[i], l)
		}
	}
}

func TestTailFileManyLines(t *testing.T) {
	f, err := ioutil.TempFile("", "tail-test")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	defer os.RemoveAll(f.Name())
	testFile := []byte(`first line
second line
truncated line`)
	if _, err := f.Write(testFile); err != nil {
		t.Fatal(err)
	}
	if _, err := f.Seek(0, os.SEEK_SET); err != nil {
		t.Fatal(err)
	}
	expected := []string{"first line", "second line"}
	res, err := TailFile(f, 10000)
	if err != nil {
		t.Fatal(err)
	}
	for i, l := range res {
		t.Logf("%s", l)
		if expected[i] != string(l) {
			t.Fatalf("Expected line %s, got %s", expected[i], l)
		}
	}
}

func TestTailEmptyFile(t *testing.T) {
	f, err := ioutil.TempFile("", "tail-test")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	defer os.RemoveAll(f.Name())
	res, err := TailFile(f, 10000)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 0 {
		t.Fatal("Must be empty slice from empty file")
	}
}

func TestTailNegativeN(t *testing.T) {
	f, err := ioutil.TempFile("", "tail-test")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	defer os.RemoveAll(f.Name())
	testFile := []byte(`first line
second line
truncated line`)
	if _, err := f.Write(testFile); err != nil {
		t.Fatal(err)
	}
	if _, err := f.Seek(0, os.SEEK_SET); err != nil {
		t.Fatal(err)
	}
	if _, err := TailFile(f, -1); err != ErrNonPositiveLinesNumber {
		t.Fatalf("Expected ErrNonPositiveLinesNumber, got %s", err)
	}
	if _, err := TailFile(f, 0); err != ErrNonPositiveLinesNumber {
		t.Fatalf("Expected ErrNonPositiveLinesNumber, got %s", err)
	}
}

func BenchmarkTail(b *testing.B) {
	f, err := ioutil.TempFile("", "tail-test")
	if err != nil {
		b.Fatal(err)
	}
	defer f.Close()
	defer os.RemoveAll(f.Name())
	for i := 0; i < 10000; i++ {
		if _, err := f.Write([]byte("tailfile pretty interesting line\n")); err != nil {
			b.Fatal(err)
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := TailFile(f, 1000); err != nil {
			b.Fatal(err)
		}
	}
}

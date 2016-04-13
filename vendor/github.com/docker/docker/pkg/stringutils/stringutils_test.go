package stringutils

import "testing"

func testLengthHelper(generator func(int) string, t *testing.T) {
	expectedLength := 20
	s := generator(expectedLength)
	if len(s) != expectedLength {
		t.Fatalf("Length of %s was %d but expected length %d", s, len(s), expectedLength)
	}
}

func testUniquenessHelper(generator func(int) string, t *testing.T) {
	repeats := 25
	set := make(map[string]struct{}, repeats)
	for i := 0; i < repeats; i = i + 1 {
		str := generator(64)
		if len(str) != 64 {
			t.Fatalf("Id returned is incorrect: %s", str)
		}
		if _, ok := set[str]; ok {
			t.Fatalf("Random number is repeated")
		}
		set[str] = struct{}{}
	}
}

func isASCII(s string) bool {
	for _, c := range s {
		if c > 127 {
			return false
		}
	}
	return true
}

func TestGenerateRandomAlphaOnlyStringLength(t *testing.T) {
	testLengthHelper(GenerateRandomAlphaOnlyString, t)
}

func TestGenerateRandomAlphaOnlyStringUniqueness(t *testing.T) {
	testUniquenessHelper(GenerateRandomAlphaOnlyString, t)
}

func TestGenerateRandomAsciiStringLength(t *testing.T) {
	testLengthHelper(GenerateRandomAsciiString, t)
}

func TestGenerateRandomAsciiStringUniqueness(t *testing.T) {
	testUniquenessHelper(GenerateRandomAsciiString, t)
}

func TestGenerateRandomAsciiStringIsAscii(t *testing.T) {
	str := GenerateRandomAsciiString(64)
	if !isASCII(str) {
		t.Fatalf("%s contained non-ascii characters", str)
	}
}

func TestTruncate(t *testing.T) {
	str := "teststring"
	newstr := Truncate(str, 4)
	if newstr != "test" {
		t.Fatalf("Expected test, got %s", newstr)
	}
	newstr = Truncate(str, 20)
	if newstr != "teststring" {
		t.Fatalf("Expected teststring, got %s", newstr)
	}
}

func TestInSlice(t *testing.T) {
	slice := []string{"test", "in", "slice"}

	test := InSlice(slice, "test")
	if !test {
		t.Fatalf("Expected string test to be in slice")
	}
	test = InSlice(slice, "SLICE")
	if !test {
		t.Fatalf("Expected string SLICE to be in slice")
	}
	test = InSlice(slice, "notinslice")
	if test {
		t.Fatalf("Expected string notinslice not to be in slice")
	}
}

func TestShellQuoteArgumentsEmpty(t *testing.T) {
	actual := ShellQuoteArguments([]string{})
	expected := ""
	if actual != expected {
		t.Fatalf("Expected an empty string")
	}
}

func TestShellQuoteArguments(t *testing.T) {
	simpleString := "simpleString"
	complexString := "This is a 'more' complex $tring with some special char *"
	actual := ShellQuoteArguments([]string{simpleString, complexString})
	expected := "simpleString 'This is a '\\''more'\\'' complex $tring with some special char *'"
	if actual != expected {
		t.Fatalf("Expected \"%v\", got \"%v\"", expected, actual)
	}
}

package term

import "testing"

func TestToBytes(t *testing.T) {
	codes, err := ToBytes("ctrl-a,a")
	if err != nil {
		t.Fatal(err)
	}
	if len(codes) != 2 {
		t.Fatalf("Expected 2 codes, got %d", len(codes))
	}
	if codes[0] != 1 || codes[1] != 97 {
		t.Fatalf("Expected '1' '97', got '%d' '%d'", codes[0], codes[1])
	}

	codes, err = ToBytes("shift-z")
	if err == nil {
		t.Fatalf("Expected error, got none")
	}

	codes, err = ToBytes("ctrl-@,ctrl-[,~,ctrl-o")
	if err != nil {
		t.Fatal(err)
	}
	if len(codes) != 4 {
		t.Fatalf("Expected 4 codes, got %d", len(codes))
	}
	if codes[0] != 0 || codes[1] != 27 || codes[2] != 126 || codes[3] != 15 {
		t.Fatalf("Expected '0' '27' '126', '15', got '%d' '%d' '%d' '%d'", codes[0], codes[1], codes[2], codes[3])
	}

	codes, err = ToBytes("DEL,+")
	if err != nil {
		t.Fatal(err)
	}
	if len(codes) != 2 {
		t.Fatalf("Expected 2 codes, got %d", len(codes))
	}
	if codes[0] != 127 || codes[1] != 43 {
		t.Fatalf("Expected '127 '43'', got '%d' '%d'", codes[0], codes[1])
	}
}

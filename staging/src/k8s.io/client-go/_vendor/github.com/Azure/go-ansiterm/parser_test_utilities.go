package ansiterm

import (
	"testing"
)

func createTestParser(s string) (*AnsiParser, *TestAnsiEventHandler) {
	evtHandler := CreateTestAnsiEventHandler()
	parser := CreateParser(s, evtHandler)

	return parser, evtHandler
}

func validateState(t *testing.T, actualState State, expectedStateName string) {
	actualName := "Nil"

	if actualState != nil {
		actualName = actualState.Name()
	}

	if actualName != expectedStateName {
		t.Errorf("Invalid State: '%s' != '%s'", actualName, expectedStateName)
	}
}

func validateFuncCalls(t *testing.T, actualCalls []string, expectedCalls []string) {
	actualCount := len(actualCalls)
	expectedCount := len(expectedCalls)

	if actualCount != expectedCount {
		t.Errorf("Actual   calls: %v", actualCalls)
		t.Errorf("Expected calls: %v", expectedCalls)
		t.Errorf("Call count error: %d != %d", actualCount, expectedCount)
		return
	}

	for i, v := range actualCalls {
		if v != expectedCalls[i] {
			t.Errorf("Actual   calls: %v", actualCalls)
			t.Errorf("Expected calls: %v", expectedCalls)
			t.Errorf("Mismatched calls: %s != %s with lengths %d and %d", v, expectedCalls[i], len(v), len(expectedCalls[i]))
		}
	}
}

func fillContext(context *AnsiContext) {
	context.currentChar = 'A'
	context.paramBuffer = []byte{'C', 'D', 'E'}
	context.interBuffer = []byte{'F', 'G', 'H'}
}

func validateEmptyContext(t *testing.T, context *AnsiContext) {
	var expectedCurrChar byte = 0x0
	if context.currentChar != expectedCurrChar {
		t.Errorf("Currentchar mismatch '%#x' != '%#x'", context.currentChar, expectedCurrChar)
	}

	if len(context.paramBuffer) != 0 {
		t.Errorf("Non-empty parameter buffer: %v", context.paramBuffer)
	}

	if len(context.paramBuffer) != 0 {
		t.Errorf("Non-empty intermediate buffer: %v", context.interBuffer)
	}

}

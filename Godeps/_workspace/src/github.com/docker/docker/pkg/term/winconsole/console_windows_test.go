// +build windows

package winconsole

import (
	"fmt"
	"testing"
)

func helpsTestParseInt16OrDefault(t *testing.T, expectedValue int16, shouldFail bool, input string, defaultValue int16, format string, args ...string) {
	value, err := parseInt16OrDefault(input, defaultValue)
	if nil != err && !shouldFail {
		t.Errorf("Unexpected error returned %v", err)
		t.Errorf(format, args)
	}
	if nil == err && shouldFail {
		t.Errorf("Should have failed as expected\n\tReturned value = %d", value)
		t.Errorf(format, args)
	}
	if expectedValue != value {
		t.Errorf("The value returned does not match expected\n\tExpected:%v\n\t:Actual%v", expectedValue, value)
		t.Errorf(format, args)
	}
}

func TestParseInt16OrDefault(t *testing.T) {
	// empty string
	helpsTestParseInt16OrDefault(t, 0, false, "", 0, "Empty string returns default")
	helpsTestParseInt16OrDefault(t, 2, false, "", 2, "Empty string returns default")

	// normal case
	helpsTestParseInt16OrDefault(t, 0, false, "0", 0, "0 handled correctly")
	helpsTestParseInt16OrDefault(t, 111, false, "111", 2, "Normal")
	helpsTestParseInt16OrDefault(t, 111, false, "+111", 2, "+N")
	helpsTestParseInt16OrDefault(t, -111, false, "-111", 2, "-N")
	helpsTestParseInt16OrDefault(t, 0, false, "+0", 11, "+0")
	helpsTestParseInt16OrDefault(t, 0, false, "-0", 12, "-0")

	// ill formed strings
	helpsTestParseInt16OrDefault(t, 0, true, "abc", 0, "Invalid string")
	helpsTestParseInt16OrDefault(t, 42, true, "+= 23", 42, "Invalid string")
	helpsTestParseInt16OrDefault(t, 42, true, "123.45", 42, "float like")

}

func helpsTestGetNumberOfChars(t *testing.T, expected uint32, fromCoord COORD, toCoord COORD, screenSize COORD, format string, args ...interface{}) {
	actual := getNumberOfChars(fromCoord, toCoord, screenSize)
	mesg := fmt.Sprintf(format, args)
	assertTrue(t, expected == actual, fmt.Sprintf("%s Expected=%d, Actual=%d, Parameters = { fromCoord=%+v, toCoord=%+v, screenSize=%+v", mesg, expected, actual, fromCoord, toCoord, screenSize))
}

func TestGetNumberOfChars(t *testing.T) {
	// Note: The columns and lines are 0 based
	// Also that interval is "inclusive" means will have both start and end chars
	// This test only tests the number opf characters being written

	// all four corners
	maxWindow := COORD{X: 80, Y: 50}
	leftTop := COORD{X: 0, Y: 0}
	rightTop := COORD{X: 79, Y: 0}
	leftBottom := COORD{X: 0, Y: 49}
	rightBottom := COORD{X: 79, Y: 49}

	// same position
	helpsTestGetNumberOfChars(t, 1, COORD{X: 1, Y: 14}, COORD{X: 1, Y: 14}, COORD{X: 80, Y: 50}, "Same position random line")

	// four corners
	helpsTestGetNumberOfChars(t, 1, leftTop, leftTop, maxWindow, "Same position- leftTop")
	helpsTestGetNumberOfChars(t, 1, rightTop, rightTop, maxWindow, "Same position- rightTop")
	helpsTestGetNumberOfChars(t, 1, leftBottom, leftBottom, maxWindow, "Same position- leftBottom")
	helpsTestGetNumberOfChars(t, 1, rightBottom, rightBottom, maxWindow, "Same position- rightBottom")

	// from this char to next char on same line
	helpsTestGetNumberOfChars(t, 2, COORD{X: 0, Y: 0}, COORD{X: 1, Y: 0}, maxWindow, "Next position on same line")
	helpsTestGetNumberOfChars(t, 2, COORD{X: 1, Y: 14}, COORD{X: 2, Y: 14}, maxWindow, "Next position on same line")

	// from this char to next 10 chars on same line
	helpsTestGetNumberOfChars(t, 11, COORD{X: 0, Y: 0}, COORD{X: 10, Y: 0}, maxWindow, "Next position on same line")
	helpsTestGetNumberOfChars(t, 11, COORD{X: 1, Y: 14}, COORD{X: 11, Y: 14}, maxWindow, "Next position on same line")

	helpsTestGetNumberOfChars(t, 5, COORD{X: 3, Y: 11}, COORD{X: 7, Y: 11}, maxWindow, "To and from on same line")

	helpsTestGetNumberOfChars(t, 8, COORD{X: 0, Y: 34}, COORD{X: 7, Y: 34}, maxWindow, "Start of line to middle")
	helpsTestGetNumberOfChars(t, 4, COORD{X: 76, Y: 34}, COORD{X: 79, Y: 34}, maxWindow, "Middle to end of line")

	// multiple lines - 1
	helpsTestGetNumberOfChars(t, 81, COORD{X: 0, Y: 0}, COORD{X: 0, Y: 1}, maxWindow, "one line below same X")
	helpsTestGetNumberOfChars(t, 81, COORD{X: 10, Y: 10}, COORD{X: 10, Y: 11}, maxWindow, "one line below same X")

	// multiple lines - 2
	helpsTestGetNumberOfChars(t, 161, COORD{X: 0, Y: 0}, COORD{X: 0, Y: 2}, maxWindow, "one line below same X")
	helpsTestGetNumberOfChars(t, 161, COORD{X: 10, Y: 10}, COORD{X: 10, Y: 12}, maxWindow, "one line below same X")

	// multiple lines - 3
	helpsTestGetNumberOfChars(t, 241, COORD{X: 0, Y: 0}, COORD{X: 0, Y: 3}, maxWindow, "one line below same X")
	helpsTestGetNumberOfChars(t, 241, COORD{X: 10, Y: 10}, COORD{X: 10, Y: 13}, maxWindow, "one line below same X")

	// full line
	helpsTestGetNumberOfChars(t, 80, COORD{X: 0, Y: 0}, COORD{X: 79, Y: 0}, maxWindow, "Full line - first")
	helpsTestGetNumberOfChars(t, 80, COORD{X: 0, Y: 23}, COORD{X: 79, Y: 23}, maxWindow, "Full line - random")
	helpsTestGetNumberOfChars(t, 80, COORD{X: 0, Y: 49}, COORD{X: 79, Y: 49}, maxWindow, "Full line - last")

	// full screen
	helpsTestGetNumberOfChars(t, 80*50, leftTop, rightBottom, maxWindow, "full screen")

	helpsTestGetNumberOfChars(t, 80*50-1, COORD{X: 1, Y: 0}, rightBottom, maxWindow, "dropping first char to, end of screen")
	helpsTestGetNumberOfChars(t, 80*50-2, COORD{X: 2, Y: 0}, rightBottom, maxWindow, "dropping first two char to, end of screen")

	helpsTestGetNumberOfChars(t, 80*50-1, leftTop, COORD{X: 78, Y: 49}, maxWindow, "from start of screen, till last char-1")
	helpsTestGetNumberOfChars(t, 80*50-2, leftTop, COORD{X: 77, Y: 49}, maxWindow, "from start of screen, till last char-2")

	helpsTestGetNumberOfChars(t, 80*50-5, COORD{X: 4, Y: 0}, COORD{X: 78, Y: 49}, COORD{X: 80, Y: 50}, "from start of screen+4, till last char-1")
	helpsTestGetNumberOfChars(t, 80*50-6, COORD{X: 4, Y: 0}, COORD{X: 77, Y: 49}, COORD{X: 80, Y: 50}, "from start of screen+4, till last char-2")
}

var allForeground = []int16{
	ANSI_FOREGROUND_BLACK,
	ANSI_FOREGROUND_RED,
	ANSI_FOREGROUND_GREEN,
	ANSI_FOREGROUND_YELLOW,
	ANSI_FOREGROUND_BLUE,
	ANSI_FOREGROUND_MAGENTA,
	ANSI_FOREGROUND_CYAN,
	ANSI_FOREGROUND_WHITE,
	ANSI_FOREGROUND_DEFAULT,
}
var allBackground = []int16{
	ANSI_BACKGROUND_BLACK,
	ANSI_BACKGROUND_RED,
	ANSI_BACKGROUND_GREEN,
	ANSI_BACKGROUND_YELLOW,
	ANSI_BACKGROUND_BLUE,
	ANSI_BACKGROUND_MAGENTA,
	ANSI_BACKGROUND_CYAN,
	ANSI_BACKGROUND_WHITE,
	ANSI_BACKGROUND_DEFAULT,
}

func maskForeground(flag WORD) WORD {
	return flag & FOREGROUND_MASK_UNSET
}

func onlyForeground(flag WORD) WORD {
	return flag & FOREGROUND_MASK_SET
}

func maskBackground(flag WORD) WORD {
	return flag & BACKGROUND_MASK_UNSET
}

func onlyBackground(flag WORD) WORD {
	return flag & BACKGROUND_MASK_SET
}

func helpsTestGetWindowsTextAttributeForAnsiValue(t *testing.T, oldValue WORD /*, expected WORD*/, ansi int16, onlyMask WORD, restMask WORD) WORD {
	actual, err := getWindowsTextAttributeForAnsiValue(oldValue, FOREGROUND_MASK_SET, ansi)
	assertTrue(t, nil == err, "Should be no error")
	// assert that other bits are not affected
	if 0 != oldValue {
		assertTrue(t, (actual&restMask) == (oldValue&restMask), "The operation should not have affected other bits actual=%X oldValue=%X ansi=%d", actual, oldValue, ansi)
	}
	return actual
}

func TestBackgroundForAnsiValue(t *testing.T) {
	// Check that nothing else changes
	// background changes
	for _, state1 := range allBackground {
		for _, state2 := range allBackground {
			flag := WORD(0)
			flag = helpsTestGetWindowsTextAttributeForAnsiValue(t, flag, state1, BACKGROUND_MASK_SET, BACKGROUND_MASK_UNSET)
			flag = helpsTestGetWindowsTextAttributeForAnsiValue(t, flag, state2, BACKGROUND_MASK_SET, BACKGROUND_MASK_UNSET)
		}
	}
	// cummulative bcakground changes
	for _, state1 := range allBackground {
		flag := WORD(0)
		for _, state2 := range allBackground {
			flag = helpsTestGetWindowsTextAttributeForAnsiValue(t, flag, state1, BACKGROUND_MASK_SET, BACKGROUND_MASK_UNSET)
			flag = helpsTestGetWindowsTextAttributeForAnsiValue(t, flag, state2, BACKGROUND_MASK_SET, BACKGROUND_MASK_UNSET)
		}
	}
	// change background after foreground
	for _, state1 := range allForeground {
		for _, state2 := range allBackground {
			flag := WORD(0)
			flag = helpsTestGetWindowsTextAttributeForAnsiValue(t, flag, state1, FOREGROUND_MASK_SET, FOREGROUND_MASK_UNSET)
			flag = helpsTestGetWindowsTextAttributeForAnsiValue(t, flag, state2, BACKGROUND_MASK_SET, BACKGROUND_MASK_UNSET)
		}
	}
	// change background after change cumulative
	for _, state1 := range allForeground {
		flag := WORD(0)
		for _, state2 := range allBackground {
			flag = helpsTestGetWindowsTextAttributeForAnsiValue(t, flag, state1, FOREGROUND_MASK_SET, FOREGROUND_MASK_UNSET)
			flag = helpsTestGetWindowsTextAttributeForAnsiValue(t, flag, state2, BACKGROUND_MASK_SET, BACKGROUND_MASK_UNSET)
		}
	}
}

func TestForegroundForAnsiValue(t *testing.T) {
	// Check that nothing else changes
	for _, state1 := range allForeground {
		for _, state2 := range allForeground {
			flag := WORD(0)
			flag = helpsTestGetWindowsTextAttributeForAnsiValue(t, flag, state1, FOREGROUND_MASK_SET, FOREGROUND_MASK_UNSET)
			flag = helpsTestGetWindowsTextAttributeForAnsiValue(t, flag, state2, FOREGROUND_MASK_SET, FOREGROUND_MASK_UNSET)
		}
	}

	for _, state1 := range allForeground {
		flag := WORD(0)
		for _, state2 := range allForeground {
			flag = helpsTestGetWindowsTextAttributeForAnsiValue(t, flag, state1, FOREGROUND_MASK_SET, FOREGROUND_MASK_UNSET)
			flag = helpsTestGetWindowsTextAttributeForAnsiValue(t, flag, state2, FOREGROUND_MASK_SET, FOREGROUND_MASK_UNSET)
		}
	}
	for _, state1 := range allBackground {
		for _, state2 := range allForeground {
			flag := WORD(0)
			flag = helpsTestGetWindowsTextAttributeForAnsiValue(t, flag, state1, BACKGROUND_MASK_SET, BACKGROUND_MASK_UNSET)
			flag = helpsTestGetWindowsTextAttributeForAnsiValue(t, flag, state2, FOREGROUND_MASK_SET, FOREGROUND_MASK_UNSET)
		}
	}
	for _, state1 := range allBackground {
		flag := WORD(0)
		for _, state2 := range allForeground {
			flag = helpsTestGetWindowsTextAttributeForAnsiValue(t, flag, state1, BACKGROUND_MASK_SET, BACKGROUND_MASK_UNSET)
			flag = helpsTestGetWindowsTextAttributeForAnsiValue(t, flag, state2, FOREGROUND_MASK_SET, FOREGROUND_MASK_UNSET)
		}
	}
}

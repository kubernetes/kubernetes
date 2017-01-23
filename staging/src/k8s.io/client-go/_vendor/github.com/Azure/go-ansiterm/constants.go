package ansiterm

const LogEnv = "DEBUG_TERMINAL"

// ANSI constants
// References:
// -- http://www.ecma-international.org/publications/standards/Ecma-048.htm
// -- http://man7.org/linux/man-pages/man4/console_codes.4.html
// -- http://manpages.ubuntu.com/manpages/intrepid/man4/console_codes.4.html
// -- http://en.wikipedia.org/wiki/ANSI_escape_code
// -- http://vt100.net/emu/dec_ansi_parser
// -- http://vt100.net/emu/vt500_parser.svg
// -- http://invisible-island.net/xterm/ctlseqs/ctlseqs.html
// -- http://www.inwap.com/pdp10/ansicode.txt
const (
	// ECMA-48 Set Graphics Rendition
	// Note:
	// -- Constants leading with an underscore (e.g., _ANSI_xxx) are unsupported or reserved
	// -- Fonts could possibly be supported via SetCurrentConsoleFontEx
	// -- Windows does not expose the per-window cursor (i.e., caret) blink times
	ANSI_SGR_RESET              = 0
	ANSI_SGR_BOLD               = 1
	ANSI_SGR_DIM                = 2
	_ANSI_SGR_ITALIC            = 3
	ANSI_SGR_UNDERLINE          = 4
	_ANSI_SGR_BLINKSLOW         = 5
	_ANSI_SGR_BLINKFAST         = 6
	ANSI_SGR_REVERSE            = 7
	_ANSI_SGR_INVISIBLE         = 8
	_ANSI_SGR_LINETHROUGH       = 9
	_ANSI_SGR_FONT_00           = 10
	_ANSI_SGR_FONT_01           = 11
	_ANSI_SGR_FONT_02           = 12
	_ANSI_SGR_FONT_03           = 13
	_ANSI_SGR_FONT_04           = 14
	_ANSI_SGR_FONT_05           = 15
	_ANSI_SGR_FONT_06           = 16
	_ANSI_SGR_FONT_07           = 17
	_ANSI_SGR_FONT_08           = 18
	_ANSI_SGR_FONT_09           = 19
	_ANSI_SGR_FONT_10           = 20
	_ANSI_SGR_DOUBLEUNDERLINE   = 21
	ANSI_SGR_BOLD_DIM_OFF       = 22
	_ANSI_SGR_ITALIC_OFF        = 23
	ANSI_SGR_UNDERLINE_OFF      = 24
	_ANSI_SGR_BLINK_OFF         = 25
	_ANSI_SGR_RESERVED_00       = 26
	ANSI_SGR_REVERSE_OFF        = 27
	_ANSI_SGR_INVISIBLE_OFF     = 28
	_ANSI_SGR_LINETHROUGH_OFF   = 29
	ANSI_SGR_FOREGROUND_BLACK   = 30
	ANSI_SGR_FOREGROUND_RED     = 31
	ANSI_SGR_FOREGROUND_GREEN   = 32
	ANSI_SGR_FOREGROUND_YELLOW  = 33
	ANSI_SGR_FOREGROUND_BLUE    = 34
	ANSI_SGR_FOREGROUND_MAGENTA = 35
	ANSI_SGR_FOREGROUND_CYAN    = 36
	ANSI_SGR_FOREGROUND_WHITE   = 37
	_ANSI_SGR_RESERVED_01       = 38
	ANSI_SGR_FOREGROUND_DEFAULT = 39
	ANSI_SGR_BACKGROUND_BLACK   = 40
	ANSI_SGR_BACKGROUND_RED     = 41
	ANSI_SGR_BACKGROUND_GREEN   = 42
	ANSI_SGR_BACKGROUND_YELLOW  = 43
	ANSI_SGR_BACKGROUND_BLUE    = 44
	ANSI_SGR_BACKGROUND_MAGENTA = 45
	ANSI_SGR_BACKGROUND_CYAN    = 46
	ANSI_SGR_BACKGROUND_WHITE   = 47
	_ANSI_SGR_RESERVED_02       = 48
	ANSI_SGR_BACKGROUND_DEFAULT = 49
	// 50 - 65: Unsupported

	ANSI_MAX_CMD_LENGTH = 4096

	MAX_INPUT_EVENTS = 128
	DEFAULT_WIDTH    = 80
	DEFAULT_HEIGHT   = 24

	ANSI_BEL              = 0x07
	ANSI_BACKSPACE        = 0x08
	ANSI_TAB              = 0x09
	ANSI_LINE_FEED        = 0x0A
	ANSI_VERTICAL_TAB     = 0x0B
	ANSI_FORM_FEED        = 0x0C
	ANSI_CARRIAGE_RETURN  = 0x0D
	ANSI_ESCAPE_PRIMARY   = 0x1B
	ANSI_ESCAPE_SECONDARY = 0x5B
	ANSI_OSC_STRING_ENTRY = 0x5D
	ANSI_COMMAND_FIRST    = 0x40
	ANSI_COMMAND_LAST     = 0x7E
	DCS_ENTRY             = 0x90
	CSI_ENTRY             = 0x9B
	OSC_STRING            = 0x9D
	ANSI_PARAMETER_SEP    = ";"
	ANSI_CMD_G0           = '('
	ANSI_CMD_G1           = ')'
	ANSI_CMD_G2           = '*'
	ANSI_CMD_G3           = '+'
	ANSI_CMD_DECPNM       = '>'
	ANSI_CMD_DECPAM       = '='
	ANSI_CMD_OSC          = ']'
	ANSI_CMD_STR_TERM     = '\\'

	KEY_CONTROL_PARAM_2 = ";2"
	KEY_CONTROL_PARAM_3 = ";3"
	KEY_CONTROL_PARAM_4 = ";4"
	KEY_CONTROL_PARAM_5 = ";5"
	KEY_CONTROL_PARAM_6 = ";6"
	KEY_CONTROL_PARAM_7 = ";7"
	KEY_CONTROL_PARAM_8 = ";8"
	KEY_ESC_CSI         = "\x1B["
	KEY_ESC_N           = "\x1BN"
	KEY_ESC_O           = "\x1BO"

	FILL_CHARACTER = ' '
)

func getByteRange(start byte, end byte) []byte {
	bytes := make([]byte, 0, 32)
	for i := start; i <= end; i++ {
		bytes = append(bytes, byte(i))
	}

	return bytes
}

var ToGroundBytes = getToGroundBytes()
var Executors = getExecuteBytes()

// SPACE		  20+A0 hex  Always and everywhere a blank space
// Intermediate	  20-2F hex   !"#$%&'()*+,-./
var Intermeds = getByteRange(0x20, 0x2F)

// Parameters	  30-3F hex  0123456789:;<=>?
// CSI Parameters 30-39, 3B hex 0123456789;
var CsiParams = getByteRange(0x30, 0x3F)

var CsiCollectables = append(getByteRange(0x30, 0x39), getByteRange(0x3B, 0x3F)...)

// Uppercase	  40-5F hex  @ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_
var UpperCase = getByteRange(0x40, 0x5F)

// Lowercase	  60-7E hex  `abcdefghijlkmnopqrstuvwxyz{|}~
var LowerCase = getByteRange(0x60, 0x7E)

// Alphabetics	  40-7E hex  (all of upper and lower case)
var Alphabetics = append(UpperCase, LowerCase...)

var Printables = getByteRange(0x20, 0x7F)

var EscapeIntermediateToGroundBytes = getByteRange(0x30, 0x7E)
var EscapeToGroundBytes = getEscapeToGroundBytes()

// See http://www.vt100.net/emu/vt500_parser.png for description of the complex
// byte ranges below

func getEscapeToGroundBytes() []byte {
	escapeToGroundBytes := getByteRange(0x30, 0x4F)
	escapeToGroundBytes = append(escapeToGroundBytes, getByteRange(0x51, 0x57)...)
	escapeToGroundBytes = append(escapeToGroundBytes, 0x59)
	escapeToGroundBytes = append(escapeToGroundBytes, 0x5A)
	escapeToGroundBytes = append(escapeToGroundBytes, 0x5C)
	escapeToGroundBytes = append(escapeToGroundBytes, getByteRange(0x60, 0x7E)...)
	return escapeToGroundBytes
}

func getExecuteBytes() []byte {
	executeBytes := getByteRange(0x00, 0x17)
	executeBytes = append(executeBytes, 0x19)
	executeBytes = append(executeBytes, getByteRange(0x1C, 0x1F)...)
	return executeBytes
}

func getToGroundBytes() []byte {
	groundBytes := []byte{0x18}
	groundBytes = append(groundBytes, 0x1A)
	groundBytes = append(groundBytes, getByteRange(0x80, 0x8F)...)
	groundBytes = append(groundBytes, getByteRange(0x91, 0x97)...)
	groundBytes = append(groundBytes, 0x99)
	groundBytes = append(groundBytes, 0x9A)
	groundBytes = append(groundBytes, 0x9C)
	return groundBytes
}

// Delete		     7F hex  Always and everywhere ignored
// C1 Control	  80-9F hex  32 additional control characters
// G1 Displayable A1-FE hex  94 additional displayable characters
// Special		  A0+FF hex  Same as SPACE and DELETE

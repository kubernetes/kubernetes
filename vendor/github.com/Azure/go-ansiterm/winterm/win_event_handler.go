// +build windows

package winterm

import (
	"bytes"
	"log"
	"os"
	"strconv"

	"github.com/Azure/go-ansiterm"
)

type windowsAnsiEventHandler struct {
	fd             uintptr
	file           *os.File
	infoReset      *CONSOLE_SCREEN_BUFFER_INFO
	sr             scrollRegion
	buffer         bytes.Buffer
	attributes     uint16
	inverted       bool
	wrapNext       bool
	drewMarginByte bool
	originMode     bool
	marginByte     byte
	curInfo        *CONSOLE_SCREEN_BUFFER_INFO
	curPos         COORD
	logf           func(string, ...interface{})
}

type Option func(*windowsAnsiEventHandler)

func WithLogf(f func(string, ...interface{})) Option {
	return func(w *windowsAnsiEventHandler) {
		w.logf = f
	}
}

func CreateWinEventHandler(fd uintptr, file *os.File, opts ...Option) ansiterm.AnsiEventHandler {
	infoReset, err := GetConsoleScreenBufferInfo(fd)
	if err != nil {
		return nil
	}

	h := &windowsAnsiEventHandler{
		fd:         fd,
		file:       file,
		infoReset:  infoReset,
		attributes: infoReset.Attributes,
	}
	for _, o := range opts {
		o(h)
	}

	if isDebugEnv := os.Getenv(ansiterm.LogEnv); isDebugEnv == "1" {
		logFile, _ := os.Create("winEventHandler.log")
		logger := log.New(logFile, "", log.LstdFlags)
		if h.logf != nil {
			l := h.logf
			h.logf = func(s string, v ...interface{}) {
				l(s, v...)
				logger.Printf(s, v...)
			}
		} else {
			h.logf = logger.Printf
		}
	}

	if h.logf == nil {
		h.logf = func(string, ...interface{}) {}
	}

	return h
}

type scrollRegion struct {
	top    int16
	bottom int16
}

// simulateLF simulates a LF or CR+LF by scrolling if necessary to handle the
// current cursor position and scroll region settings, in which case it returns
// true. If no special handling is necessary, then it does nothing and returns
// false.
//
// In the false case, the caller should ensure that a carriage return
// and line feed are inserted or that the text is otherwise wrapped.
func (h *windowsAnsiEventHandler) simulateLF(includeCR bool) (bool, error) {
	if h.wrapNext {
		if err := h.Flush(); err != nil {
			return false, err
		}
		h.clearWrap()
	}
	pos, info, err := h.getCurrentInfo()
	if err != nil {
		return false, err
	}
	sr := h.effectiveSr(info.Window)
	if pos.Y == sr.bottom {
		// Scrolling is necessary. Let Windows automatically scroll if the scrolling region
		// is the full window.
		if sr.top == info.Window.Top && sr.bottom == info.Window.Bottom {
			if includeCR {
				pos.X = 0
				h.updatePos(pos)
			}
			return false, nil
		}

		// A custom scroll region is active. Scroll the window manually to simulate
		// the LF.
		if err := h.Flush(); err != nil {
			return false, err
		}
		h.logf("Simulating LF inside scroll region")
		if err := h.scrollUp(1); err != nil {
			return false, err
		}
		if includeCR {
			pos.X = 0
			if err := SetConsoleCursorPosition(h.fd, pos); err != nil {
				return false, err
			}
		}
		return true, nil

	} else if pos.Y < info.Window.Bottom {
		// Let Windows handle the LF.
		pos.Y++
		if includeCR {
			pos.X = 0
		}
		h.updatePos(pos)
		return false, nil
	} else {
		// The cursor is at the bottom of the screen but outside the scroll
		// region. Skip the LF.
		h.logf("Simulating LF outside scroll region")
		if includeCR {
			if err := h.Flush(); err != nil {
				return false, err
			}
			pos.X = 0
			if err := SetConsoleCursorPosition(h.fd, pos); err != nil {
				return false, err
			}
		}
		return true, nil
	}
}

// executeLF executes a LF without a CR.
func (h *windowsAnsiEventHandler) executeLF() error {
	handled, err := h.simulateLF(false)
	if err != nil {
		return err
	}
	if !handled {
		// Windows LF will reset the cursor column position. Write the LF
		// and restore the cursor position.
		pos, _, err := h.getCurrentInfo()
		if err != nil {
			return err
		}
		h.buffer.WriteByte(ansiterm.ANSI_LINE_FEED)
		if pos.X != 0 {
			if err := h.Flush(); err != nil {
				return err
			}
			h.logf("Resetting cursor position for LF without CR")
			if err := SetConsoleCursorPosition(h.fd, pos); err != nil {
				return err
			}
		}
	}
	return nil
}

func (h *windowsAnsiEventHandler) Print(b byte) error {
	if h.wrapNext {
		h.buffer.WriteByte(h.marginByte)
		h.clearWrap()
		if _, err := h.simulateLF(true); err != nil {
			return err
		}
	}
	pos, info, err := h.getCurrentInfo()
	if err != nil {
		return err
	}
	if pos.X == info.Size.X-1 {
		h.wrapNext = true
		h.marginByte = b
	} else {
		pos.X++
		h.updatePos(pos)
		h.buffer.WriteByte(b)
	}
	return nil
}

func (h *windowsAnsiEventHandler) Execute(b byte) error {
	switch b {
	case ansiterm.ANSI_TAB:
		h.logf("Execute(TAB)")
		// Move to the next tab stop, but preserve auto-wrap if already set.
		if !h.wrapNext {
			pos, info, err := h.getCurrentInfo()
			if err != nil {
				return err
			}
			pos.X = (pos.X + 8) - pos.X%8
			if pos.X >= info.Size.X {
				pos.X = info.Size.X - 1
			}
			if err := h.Flush(); err != nil {
				return err
			}
			if err := SetConsoleCursorPosition(h.fd, pos); err != nil {
				return err
			}
		}
		return nil

	case ansiterm.ANSI_BEL:
		h.buffer.WriteByte(ansiterm.ANSI_BEL)
		return nil

	case ansiterm.ANSI_BACKSPACE:
		if h.wrapNext {
			if err := h.Flush(); err != nil {
				return err
			}
			h.clearWrap()
		}
		pos, _, err := h.getCurrentInfo()
		if err != nil {
			return err
		}
		if pos.X > 0 {
			pos.X--
			h.updatePos(pos)
			h.buffer.WriteByte(ansiterm.ANSI_BACKSPACE)
		}
		return nil

	case ansiterm.ANSI_VERTICAL_TAB, ansiterm.ANSI_FORM_FEED:
		// Treat as true LF.
		return h.executeLF()

	case ansiterm.ANSI_LINE_FEED:
		// Simulate a CR and LF for now since there is no way in go-ansiterm
		// to tell if the LF should include CR (and more things break when it's
		// missing than when it's incorrectly added).
		handled, err := h.simulateLF(true)
		if handled || err != nil {
			return err
		}
		return h.buffer.WriteByte(ansiterm.ANSI_LINE_FEED)

	case ansiterm.ANSI_CARRIAGE_RETURN:
		if h.wrapNext {
			if err := h.Flush(); err != nil {
				return err
			}
			h.clearWrap()
		}
		pos, _, err := h.getCurrentInfo()
		if err != nil {
			return err
		}
		if pos.X != 0 {
			pos.X = 0
			h.updatePos(pos)
			h.buffer.WriteByte(ansiterm.ANSI_CARRIAGE_RETURN)
		}
		return nil

	default:
		return nil
	}
}

func (h *windowsAnsiEventHandler) CUU(param int) error {
	if err := h.Flush(); err != nil {
		return err
	}
	h.logf("CUU: [%v]", []string{strconv.Itoa(param)})
	h.clearWrap()
	return h.moveCursorVertical(-param)
}

func (h *windowsAnsiEventHandler) CUD(param int) error {
	if err := h.Flush(); err != nil {
		return err
	}
	h.logf("CUD: [%v]", []string{strconv.Itoa(param)})
	h.clearWrap()
	return h.moveCursorVertical(param)
}

func (h *windowsAnsiEventHandler) CUF(param int) error {
	if err := h.Flush(); err != nil {
		return err
	}
	h.logf("CUF: [%v]", []string{strconv.Itoa(param)})
	h.clearWrap()
	return h.moveCursorHorizontal(param)
}

func (h *windowsAnsiEventHandler) CUB(param int) error {
	if err := h.Flush(); err != nil {
		return err
	}
	h.logf("CUB: [%v]", []string{strconv.Itoa(param)})
	h.clearWrap()
	return h.moveCursorHorizontal(-param)
}

func (h *windowsAnsiEventHandler) CNL(param int) error {
	if err := h.Flush(); err != nil {
		return err
	}
	h.logf("CNL: [%v]", []string{strconv.Itoa(param)})
	h.clearWrap()
	return h.moveCursorLine(param)
}

func (h *windowsAnsiEventHandler) CPL(param int) error {
	if err := h.Flush(); err != nil {
		return err
	}
	h.logf("CPL: [%v]", []string{strconv.Itoa(param)})
	h.clearWrap()
	return h.moveCursorLine(-param)
}

func (h *windowsAnsiEventHandler) CHA(param int) error {
	if err := h.Flush(); err != nil {
		return err
	}
	h.logf("CHA: [%v]", []string{strconv.Itoa(param)})
	h.clearWrap()
	return h.moveCursorColumn(param)
}

func (h *windowsAnsiEventHandler) VPA(param int) error {
	if err := h.Flush(); err != nil {
		return err
	}
	h.logf("VPA: [[%d]]", param)
	h.clearWrap()
	info, err := GetConsoleScreenBufferInfo(h.fd)
	if err != nil {
		return err
	}
	window := h.getCursorWindow(info)
	position := info.CursorPosition
	position.Y = window.Top + int16(param) - 1
	return h.setCursorPosition(position, window)
}

func (h *windowsAnsiEventHandler) CUP(row int, col int) error {
	if err := h.Flush(); err != nil {
		return err
	}
	h.logf("CUP: [[%d %d]]", row, col)
	h.clearWrap()
	info, err := GetConsoleScreenBufferInfo(h.fd)
	if err != nil {
		return err
	}

	window := h.getCursorWindow(info)
	position := COORD{window.Left + int16(col) - 1, window.Top + int16(row) - 1}
	return h.setCursorPosition(position, window)
}

func (h *windowsAnsiEventHandler) HVP(row int, col int) error {
	if err := h.Flush(); err != nil {
		return err
	}
	h.logf("HVP: [[%d %d]]", row, col)
	h.clearWrap()
	return h.CUP(row, col)
}

func (h *windowsAnsiEventHandler) DECTCEM(visible bool) error {
	if err := h.Flush(); err != nil {
		return err
	}
	h.logf("DECTCEM: [%v]", []string{strconv.FormatBool(visible)})
	h.clearWrap()
	return nil
}

func (h *windowsAnsiEventHandler) DECOM(enable bool) error {
	if err := h.Flush(); err != nil {
		return err
	}
	h.logf("DECOM: [%v]", []string{strconv.FormatBool(enable)})
	h.clearWrap()
	h.originMode = enable
	return h.CUP(1, 1)
}

func (h *windowsAnsiEventHandler) DECCOLM(use132 bool) error {
	if err := h.Flush(); err != nil {
		return err
	}
	h.logf("DECCOLM: [%v]", []string{strconv.FormatBool(use132)})
	h.clearWrap()
	if err := h.ED(2); err != nil {
		return err
	}
	info, err := GetConsoleScreenBufferInfo(h.fd)
	if err != nil {
		return err
	}
	targetWidth := int16(80)
	if use132 {
		targetWidth = 132
	}
	if info.Size.X < targetWidth {
		if err := SetConsoleScreenBufferSize(h.fd, COORD{targetWidth, info.Size.Y}); err != nil {
			h.logf("set buffer failed: %v", err)
			return err
		}
	}
	window := info.Window
	window.Left = 0
	window.Right = targetWidth - 1
	if err := SetConsoleWindowInfo(h.fd, true, window); err != nil {
		h.logf("set window failed: %v", err)
		return err
	}
	if info.Size.X > targetWidth {
		if err := SetConsoleScreenBufferSize(h.fd, COORD{targetWidth, info.Size.Y}); err != nil {
			h.logf("set buffer failed: %v", err)
			return err
		}
	}
	return SetConsoleCursorPosition(h.fd, COORD{0, 0})
}

func (h *windowsAnsiEventHandler) ED(param int) error {
	if err := h.Flush(); err != nil {
		return err
	}
	h.logf("ED: [%v]", []string{strconv.Itoa(param)})
	h.clearWrap()

	// [J  -- Erases from the cursor to the end of the screen, including the cursor position.
	// [1J -- Erases from the beginning of the screen to the cursor, including the cursor position.
	// [2J -- Erases the complete display. The cursor does not move.
	// Notes:
	// -- Clearing the entire buffer, versus just the Window, works best for Windows Consoles

	info, err := GetConsoleScreenBufferInfo(h.fd)
	if err != nil {
		return err
	}

	var start COORD
	var end COORD

	switch param {
	case 0:
		start = info.CursorPosition
		end = COORD{info.Size.X - 1, info.Size.Y - 1}

	case 1:
		start = COORD{0, 0}
		end = info.CursorPosition

	case 2:
		start = COORD{0, 0}
		end = COORD{info.Size.X - 1, info.Size.Y - 1}
	}

	err = h.clearRange(h.attributes, start, end)
	if err != nil {
		return err
	}

	// If the whole buffer was cleared, move the window to the top while preserving
	// the window-relative cursor position.
	if param == 2 {
		pos := info.CursorPosition
		window := info.Window
		pos.Y -= window.Top
		window.Bottom -= window.Top
		window.Top = 0
		if err := SetConsoleCursorPosition(h.fd, pos); err != nil {
			return err
		}
		if err := SetConsoleWindowInfo(h.fd, true, window); err != nil {
			return err
		}
	}

	return nil
}

func (h *windowsAnsiEventHandler) EL(param int) error {
	if err := h.Flush(); err != nil {
		return err
	}
	h.logf("EL: [%v]", strconv.Itoa(param))
	h.clearWrap()

	// [K  -- Erases from the cursor to the end of the line, including the cursor position.
	// [1K -- Erases from the beginning of the line to the cursor, including the cursor position.
	// [2K -- Erases the complete line.

	info, err := GetConsoleScreenBufferInfo(h.fd)
	if err != nil {
		return err
	}

	var start COORD
	var end COORD

	switch param {
	case 0:
		start = info.CursorPosition
		end = COORD{info.Size.X, info.CursorPosition.Y}

	case 1:
		start = COORD{0, info.CursorPosition.Y}
		end = info.CursorPosition

	case 2:
		start = COORD{0, info.CursorPosition.Y}
		end = COORD{info.Size.X, info.CursorPosition.Y}
	}

	err = h.clearRange(h.attributes, start, end)
	if err != nil {
		return err
	}

	return nil
}

func (h *windowsAnsiEventHandler) IL(param int) error {
	if err := h.Flush(); err != nil {
		return err
	}
	h.logf("IL: [%v]", strconv.Itoa(param))
	h.clearWrap()
	return h.insertLines(param)
}

func (h *windowsAnsiEventHandler) DL(param int) error {
	if err := h.Flush(); err != nil {
		return err
	}
	h.logf("DL: [%v]", strconv.Itoa(param))
	h.clearWrap()
	return h.deleteLines(param)
}

func (h *windowsAnsiEventHandler) ICH(param int) error {
	if err := h.Flush(); err != nil {
		return err
	}
	h.logf("ICH: [%v]", strconv.Itoa(param))
	h.clearWrap()
	return h.insertCharacters(param)
}

func (h *windowsAnsiEventHandler) DCH(param int) error {
	if err := h.Flush(); err != nil {
		return err
	}
	h.logf("DCH: [%v]", strconv.Itoa(param))
	h.clearWrap()
	return h.deleteCharacters(param)
}

func (h *windowsAnsiEventHandler) SGR(params []int) error {
	if err := h.Flush(); err != nil {
		return err
	}
	strings := []string{}
	for _, v := range params {
		strings = append(strings, strconv.Itoa(v))
	}

	h.logf("SGR: [%v]", strings)

	if len(params) <= 0 {
		h.attributes = h.infoReset.Attributes
		h.inverted = false
	} else {
		for _, attr := range params {

			if attr == ansiterm.ANSI_SGR_RESET {
				h.attributes = h.infoReset.Attributes
				h.inverted = false
				continue
			}

			h.attributes, h.inverted = collectAnsiIntoWindowsAttributes(h.attributes, h.inverted, h.infoReset.Attributes, int16(attr))
		}
	}

	attributes := h.attributes
	if h.inverted {
		attributes = invertAttributes(attributes)
	}
	err := SetConsoleTextAttribute(h.fd, attributes)
	if err != nil {
		return err
	}

	return nil
}

func (h *windowsAnsiEventHandler) SU(param int) error {
	if err := h.Flush(); err != nil {
		return err
	}
	h.logf("SU: [%v]", []string{strconv.Itoa(param)})
	h.clearWrap()
	return h.scrollUp(param)
}

func (h *windowsAnsiEventHandler) SD(param int) error {
	if err := h.Flush(); err != nil {
		return err
	}
	h.logf("SD: [%v]", []string{strconv.Itoa(param)})
	h.clearWrap()
	return h.scrollDown(param)
}

func (h *windowsAnsiEventHandler) DA(params []string) error {
	h.logf("DA: [%v]", params)
	// DA cannot be implemented because it must send data on the VT100 input stream,
	// which is not available to go-ansiterm.
	return nil
}

func (h *windowsAnsiEventHandler) DECSTBM(top int, bottom int) error {
	if err := h.Flush(); err != nil {
		return err
	}
	h.logf("DECSTBM: [%d, %d]", top, bottom)

	// Windows is 0 indexed, Linux is 1 indexed
	h.sr.top = int16(top - 1)
	h.sr.bottom = int16(bottom - 1)

	// This command also moves the cursor to the origin.
	h.clearWrap()
	return h.CUP(1, 1)
}

func (h *windowsAnsiEventHandler) RI() error {
	if err := h.Flush(); err != nil {
		return err
	}
	h.logf("RI: []")
	h.clearWrap()

	info, err := GetConsoleScreenBufferInfo(h.fd)
	if err != nil {
		return err
	}

	sr := h.effectiveSr(info.Window)
	if info.CursorPosition.Y == sr.top {
		return h.scrollDown(1)
	}

	return h.moveCursorVertical(-1)
}

func (h *windowsAnsiEventHandler) IND() error {
	h.logf("IND: []")
	return h.executeLF()
}

func (h *windowsAnsiEventHandler) Flush() error {
	h.curInfo = nil
	if h.buffer.Len() > 0 {
		h.logf("Flush: [%s]", h.buffer.Bytes())
		if _, err := h.buffer.WriteTo(h.file); err != nil {
			return err
		}
	}

	if h.wrapNext && !h.drewMarginByte {
		h.logf("Flush: drawing margin byte '%c'", h.marginByte)

		info, err := GetConsoleScreenBufferInfo(h.fd)
		if err != nil {
			return err
		}

		charInfo := []CHAR_INFO{{UnicodeChar: uint16(h.marginByte), Attributes: info.Attributes}}
		size := COORD{1, 1}
		position := COORD{0, 0}
		region := SMALL_RECT{Left: info.CursorPosition.X, Top: info.CursorPosition.Y, Right: info.CursorPosition.X, Bottom: info.CursorPosition.Y}
		if err := WriteConsoleOutput(h.fd, charInfo, size, position, &region); err != nil {
			return err
		}
		h.drewMarginByte = true
	}
	return nil
}

// cacheConsoleInfo ensures that the current console screen information has been queried
// since the last call to Flush(). It must be called before accessing h.curInfo or h.curPos.
func (h *windowsAnsiEventHandler) getCurrentInfo() (COORD, *CONSOLE_SCREEN_BUFFER_INFO, error) {
	if h.curInfo == nil {
		info, err := GetConsoleScreenBufferInfo(h.fd)
		if err != nil {
			return COORD{}, nil, err
		}
		h.curInfo = info
		h.curPos = info.CursorPosition
	}
	return h.curPos, h.curInfo, nil
}

func (h *windowsAnsiEventHandler) updatePos(pos COORD) {
	if h.curInfo == nil {
		panic("failed to call getCurrentInfo before calling updatePos")
	}
	h.curPos = pos
}

// clearWrap clears the state where the cursor is in the margin
// waiting for the next character before wrapping the line. This must
// be done before most operations that act on the cursor.
func (h *windowsAnsiEventHandler) clearWrap() {
	h.wrapNext = false
	h.drewMarginByte = false
}

// +build windows

package winterm

// effectiveSr gets the current effective scroll region in buffer coordinates
func (h *WindowsAnsiEventHandler) effectiveSr(window SMALL_RECT) scrollRegion {
	top := AddInRange(window.Top, h.sr.top, window.Top, window.Bottom)
	bottom := AddInRange(window.Top, h.sr.bottom, window.Top, window.Bottom)
	if top >= bottom {
		top = window.Top
		bottom = window.Bottom
	}
	return scrollRegion{top: top, bottom: bottom}
}

func (h *WindowsAnsiEventHandler) scrollUp(param int) error {
	info, err := GetConsoleScreenBufferInfo(h.fd)
	if err != nil {
		return err
	}

	sr := h.effectiveSr(info.Window)
	return h.scroll(param, sr, info)
}

func (h *WindowsAnsiEventHandler) scrollDown(param int) error {
	return h.scrollUp(-param)
}

func (h *WindowsAnsiEventHandler) deleteLines(param int) error {
	info, err := GetConsoleScreenBufferInfo(h.fd)
	if err != nil {
		return err
	}

	start := info.CursorPosition.Y
	sr := h.effectiveSr(info.Window)
	// Lines cannot be inserted or deleted outside the scrolling region.
	if start >= sr.top && start <= sr.bottom {
		sr.top = start
		return h.scroll(param, sr, info)
	} else {
		return nil
	}
}

func (h *WindowsAnsiEventHandler) insertLines(param int) error {
	return h.deleteLines(-param)
}

// scroll scrolls the provided scroll region by param lines. The scroll region is in buffer coordinates.
func (h *WindowsAnsiEventHandler) scroll(param int, sr scrollRegion, info *CONSOLE_SCREEN_BUFFER_INFO) error {
	logger.Infof("scroll: scrollTop: %d, scrollBottom: %d", sr.top, sr.bottom)
	logger.Infof("scroll: windowTop: %d, windowBottom: %d", info.Window.Top, info.Window.Bottom)

	// Copy from and clip to the scroll region (full buffer width)
	scrollRect := SMALL_RECT{
		Top:    sr.top,
		Bottom: sr.bottom,
		Left:   0,
		Right:  info.Size.X - 1,
	}

	// Origin to which area should be copied
	destOrigin := COORD{
		X: 0,
		Y: sr.top - SHORT(param),
	}

	char := CHAR_INFO{
		UnicodeChar: ' ',
		Attributes:  h.attributes,
	}

	if err := ScrollConsoleScreenBuffer(h.fd, scrollRect, scrollRect, destOrigin, char); err != nil {
		return err
	}
	return nil
}

func (h *WindowsAnsiEventHandler) deleteCharacters(param int) error {
	info, err := GetConsoleScreenBufferInfo(h.fd)
	if err != nil {
		return err
	}
	return h.scrollLine(param, info.CursorPosition, info)
}

func (h *WindowsAnsiEventHandler) insertCharacters(param int) error {
	return h.deleteCharacters(-param)
}

// scrollLine scrolls a line horizontally starting at the provided position by a number of columns.
func (h *WindowsAnsiEventHandler) scrollLine(columns int, position COORD, info *CONSOLE_SCREEN_BUFFER_INFO) error {
	// Copy from and clip to the scroll region (full buffer width)
	scrollRect := SMALL_RECT{
		Top:    position.Y,
		Bottom: position.Y,
		Left:   position.X,
		Right:  info.Size.X - 1,
	}

	// Origin to which area should be copied
	destOrigin := COORD{
		X: position.X - SHORT(columns),
		Y: position.Y,
	}

	char := CHAR_INFO{
		UnicodeChar: ' ',
		Attributes:  h.attributes,
	}

	if err := ScrollConsoleScreenBuffer(h.fd, scrollRect, scrollRect, destOrigin, char); err != nil {
		return err
	}
	return nil
}

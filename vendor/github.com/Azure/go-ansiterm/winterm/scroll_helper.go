// +build windows

package winterm

// effectiveSr gets the current effective scroll region in buffer coordinates
func (h *windowsAnsiEventHandler) effectiveSr(window SMALL_RECT) scrollRegion {
	top := addInRange(window.Top, h.sr.top, window.Top, window.Bottom)
	bottom := addInRange(window.Top, h.sr.bottom, window.Top, window.Bottom)
	if top >= bottom {
		top = window.Top
		bottom = window.Bottom
	}
	return scrollRegion{top: top, bottom: bottom}
}

func (h *windowsAnsiEventHandler) scrollUp(param int) error {
	info, err := GetConsoleScreenBufferInfo(h.fd)
	if err != nil {
		return err
	}

	sr := h.effectiveSr(info.Window)
	return h.scroll(param, sr, info)
}

func (h *windowsAnsiEventHandler) scrollDown(param int) error {
	return h.scrollUp(-param)
}

func (h *windowsAnsiEventHandler) deleteLines(param int) error {
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

func (h *windowsAnsiEventHandler) insertLines(param int) error {
	return h.deleteLines(-param)
}

// scroll scrolls the provided scroll region by param lines. The scroll region is in buffer coordinates.
func (h *windowsAnsiEventHandler) scroll(param int, sr scrollRegion, info *CONSOLE_SCREEN_BUFFER_INFO) error {
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
		Y: sr.top - int16(param),
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

func (h *windowsAnsiEventHandler) deleteCharacters(param int) error {
	info, err := GetConsoleScreenBufferInfo(h.fd)
	if err != nil {
		return err
	}
	return h.scrollLine(param, info.CursorPosition, info)
}

func (h *windowsAnsiEventHandler) insertCharacters(param int) error {
	return h.deleteCharacters(-param)
}

// scrollLine scrolls a line horizontally starting at the provided position by a number of columns.
func (h *windowsAnsiEventHandler) scrollLine(columns int, position COORD, info *CONSOLE_SCREEN_BUFFER_INFO) error {
	// Copy from and clip to the scroll region (full buffer width)
	scrollRect := SMALL_RECT{
		Top:    position.Y,
		Bottom: position.Y,
		Left:   position.X,
		Right:  info.Size.X - 1,
	}

	// Origin to which area should be copied
	destOrigin := COORD{
		X: position.X - int16(columns),
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

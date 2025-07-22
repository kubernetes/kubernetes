// +build windows

package winterm

import "github.com/Azure/go-ansiterm"

func (h *windowsAnsiEventHandler) clearRange(attributes uint16, fromCoord COORD, toCoord COORD) error {
	// Ignore an invalid (negative area) request
	if toCoord.Y < fromCoord.Y {
		return nil
	}

	var err error

	var coordStart = COORD{}
	var coordEnd = COORD{}

	xCurrent, yCurrent := fromCoord.X, fromCoord.Y
	xEnd, yEnd := toCoord.X, toCoord.Y

	// Clear any partial initial line
	if xCurrent > 0 {
		coordStart.X, coordStart.Y = xCurrent, yCurrent
		coordEnd.X, coordEnd.Y = xEnd, yCurrent

		err = h.clearRect(attributes, coordStart, coordEnd)
		if err != nil {
			return err
		}

		xCurrent = 0
		yCurrent += 1
	}

	// Clear intervening rectangular section
	if yCurrent < yEnd {
		coordStart.X, coordStart.Y = xCurrent, yCurrent
		coordEnd.X, coordEnd.Y = xEnd, yEnd-1

		err = h.clearRect(attributes, coordStart, coordEnd)
		if err != nil {
			return err
		}

		xCurrent = 0
		yCurrent = yEnd
	}

	// Clear remaining partial ending line
	coordStart.X, coordStart.Y = xCurrent, yCurrent
	coordEnd.X, coordEnd.Y = xEnd, yEnd

	err = h.clearRect(attributes, coordStart, coordEnd)
	if err != nil {
		return err
	}

	return nil
}

func (h *windowsAnsiEventHandler) clearRect(attributes uint16, fromCoord COORD, toCoord COORD) error {
	region := SMALL_RECT{Top: fromCoord.Y, Left: fromCoord.X, Bottom: toCoord.Y, Right: toCoord.X}
	width := toCoord.X - fromCoord.X + 1
	height := toCoord.Y - fromCoord.Y + 1
	size := uint32(width) * uint32(height)

	if size <= 0 {
		return nil
	}

	buffer := make([]CHAR_INFO, size)

	char := CHAR_INFO{ansiterm.FILL_CHARACTER, attributes}
	for i := 0; i < int(size); i++ {
		buffer[i] = char
	}

	err := WriteConsoleOutput(h.fd, buffer, COORD{X: width, Y: height}, COORD{X: 0, Y: 0}, &region)
	if err != nil {
		return err
	}

	return nil
}

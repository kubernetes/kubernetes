package ansiterm

type AnsiEventHandler interface {
	// Print
	Print(b byte) error

	// Execute C0 commands
	Execute(b byte) error

	// CUrsor Up
	CUU(int) error

	// CUrsor Down
	CUD(int) error

	// CUrsor Forward
	CUF(int) error

	// CUrsor Backward
	CUB(int) error

	// Cursor to Next Line
	CNL(int) error

	// Cursor to Previous Line
	CPL(int) error

	// Cursor Horizontal position Absolute
	CHA(int) error

	// Vertical line Position Absolute
	VPA(int) error

	// CUrsor Position
	CUP(int, int) error

	// Horizontal and Vertical Position (depends on PUM)
	HVP(int, int) error

	// Text Cursor Enable Mode
	DECTCEM(bool) error

	// Origin Mode
	DECOM(bool) error

	// 132 Column Mode
	DECCOLM(bool) error

	// Erase in Display
	ED(int) error

	// Erase in Line
	EL(int) error

	// Insert Line
	IL(int) error

	// Delete Line
	DL(int) error

	// Insert Character
	ICH(int) error

	// Delete Character
	DCH(int) error

	// Set Graphics Rendition
	SGR([]int) error

	// Pan Down
	SU(int) error

	// Pan Up
	SD(int) error

	// Device Attributes
	DA([]string) error

	// Set Top and Bottom Margins
	DECSTBM(int, int) error

	// Index
	IND() error

	// Reverse Index
	RI() error

	// Flush updates from previous commands
	Flush() error
}

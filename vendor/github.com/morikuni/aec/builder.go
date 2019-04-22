package aec

// Builder is a lightweight syntax to construct customized ANSI.
type Builder struct {
	ANSI ANSI
}

// EmptyBuilder is an initialized Builder.
var EmptyBuilder *Builder

// NewBuilder creates a Builder from existing ANSI.
func NewBuilder(a ...ANSI) *Builder {
	return &Builder{concat(a)}
}

// With is a syntax for With.
func (builder *Builder) With(a ...ANSI) *Builder {
	return NewBuilder(builder.ANSI.With(a...))
}

// Up is a syntax for Up.
func (builder *Builder) Up(n uint) *Builder {
	return builder.With(Up(n))
}

// Down is a syntax for Down.
func (builder *Builder) Down(n uint) *Builder {
	return builder.With(Down(n))
}

// Right is a syntax for Right.
func (builder *Builder) Right(n uint) *Builder {
	return builder.With(Right(n))
}

// Left is a syntax for Left.
func (builder *Builder) Left(n uint) *Builder {
	return builder.With(Left(n))
}

// NextLine is a syntax for NextLine.
func (builder *Builder) NextLine(n uint) *Builder {
	return builder.With(NextLine(n))
}

// PreviousLine is a syntax for PreviousLine.
func (builder *Builder) PreviousLine(n uint) *Builder {
	return builder.With(PreviousLine(n))
}

// Column is a syntax for Column.
func (builder *Builder) Column(col uint) *Builder {
	return builder.With(Column(col))
}

// Position is a syntax for Position.
func (builder *Builder) Position(row, col uint) *Builder {
	return builder.With(Position(row, col))
}

// EraseDisplay is a syntax for EraseDisplay.
func (builder *Builder) EraseDisplay(m EraseMode) *Builder {
	return builder.With(EraseDisplay(m))
}

// EraseLine is a syntax for EraseLine.
func (builder *Builder) EraseLine(m EraseMode) *Builder {
	return builder.With(EraseLine(m))
}

// ScrollUp is a syntax for ScrollUp.
func (builder *Builder) ScrollUp(n int) *Builder {
	return builder.With(ScrollUp(n))
}

// ScrollDown is a syntax for ScrollDown.
func (builder *Builder) ScrollDown(n int) *Builder {
	return builder.With(ScrollDown(n))
}

// Save is a syntax for Save.
func (builder *Builder) Save() *Builder {
	return builder.With(Save)
}

// Restore is a syntax for Restore.
func (builder *Builder) Restore() *Builder {
	return builder.With(Restore)
}

// Hide is a syntax for Hide.
func (builder *Builder) Hide() *Builder {
	return builder.With(Hide)
}

// Show is a syntax for Show.
func (builder *Builder) Show() *Builder {
	return builder.With(Show)
}

// Report is a syntax for Report.
func (builder *Builder) Report() *Builder {
	return builder.With(Report)
}

// Bold is a syntax for Bold.
func (builder *Builder) Bold() *Builder {
	return builder.With(Bold)
}

// Faint is a syntax for Faint.
func (builder *Builder) Faint() *Builder {
	return builder.With(Faint)
}

// Italic is a syntax for Italic.
func (builder *Builder) Italic() *Builder {
	return builder.With(Italic)
}

// Underline is a syntax for Underline.
func (builder *Builder) Underline() *Builder {
	return builder.With(Underline)
}

// BlinkSlow is a syntax for BlinkSlow.
func (builder *Builder) BlinkSlow() *Builder {
	return builder.With(BlinkSlow)
}

// BlinkRapid is a syntax for BlinkRapid.
func (builder *Builder) BlinkRapid() *Builder {
	return builder.With(BlinkRapid)
}

// Inverse is a syntax for Inverse.
func (builder *Builder) Inverse() *Builder {
	return builder.With(Inverse)
}

// Conceal is a syntax for Conceal.
func (builder *Builder) Conceal() *Builder {
	return builder.With(Conceal)
}

// CrossOut is a syntax for CrossOut.
func (builder *Builder) CrossOut() *Builder {
	return builder.With(CrossOut)
}

// BlackF is a syntax for BlackF.
func (builder *Builder) BlackF() *Builder {
	return builder.With(BlackF)
}

// RedF is a syntax for RedF.
func (builder *Builder) RedF() *Builder {
	return builder.With(RedF)
}

// GreenF is a syntax for GreenF.
func (builder *Builder) GreenF() *Builder {
	return builder.With(GreenF)
}

// YellowF is a syntax for YellowF.
func (builder *Builder) YellowF() *Builder {
	return builder.With(YellowF)
}

// BlueF is a syntax for BlueF.
func (builder *Builder) BlueF() *Builder {
	return builder.With(BlueF)
}

// MagentaF is a syntax for MagentaF.
func (builder *Builder) MagentaF() *Builder {
	return builder.With(MagentaF)
}

// CyanF is a syntax for CyanF.
func (builder *Builder) CyanF() *Builder {
	return builder.With(CyanF)
}

// WhiteF is a syntax for WhiteF.
func (builder *Builder) WhiteF() *Builder {
	return builder.With(WhiteF)
}

// DefaultF is a syntax for DefaultF.
func (builder *Builder) DefaultF() *Builder {
	return builder.With(DefaultF)
}

// BlackB is a syntax for BlackB.
func (builder *Builder) BlackB() *Builder {
	return builder.With(BlackB)
}

// RedB is a syntax for RedB.
func (builder *Builder) RedB() *Builder {
	return builder.With(RedB)
}

// GreenB is a syntax for GreenB.
func (builder *Builder) GreenB() *Builder {
	return builder.With(GreenB)
}

// YellowB is a syntax for YellowB.
func (builder *Builder) YellowB() *Builder {
	return builder.With(YellowB)
}

// BlueB is a syntax for BlueB.
func (builder *Builder) BlueB() *Builder {
	return builder.With(BlueB)
}

// MagentaB is a syntax for MagentaB.
func (builder *Builder) MagentaB() *Builder {
	return builder.With(MagentaB)
}

// CyanB is a syntax for CyanB.
func (builder *Builder) CyanB() *Builder {
	return builder.With(CyanB)
}

// WhiteB is a syntax for WhiteB.
func (builder *Builder) WhiteB() *Builder {
	return builder.With(WhiteB)
}

// DefaultB is a syntax for DefaultB.
func (builder *Builder) DefaultB() *Builder {
	return builder.With(DefaultB)
}

// Frame is a syntax for Frame.
func (builder *Builder) Frame() *Builder {
	return builder.With(Frame)
}

// Encircle is a syntax for Encircle.
func (builder *Builder) Encircle() *Builder {
	return builder.With(Encircle)
}

// Overline is a syntax for Overline.
func (builder *Builder) Overline() *Builder {
	return builder.With(Overline)
}

// LightBlackF is a syntax for LightBlueF.
func (builder *Builder) LightBlackF() *Builder {
	return builder.With(LightBlackF)
}

// LightRedF is a syntax for LightRedF.
func (builder *Builder) LightRedF() *Builder {
	return builder.With(LightRedF)
}

// LightGreenF is a syntax for LightGreenF.
func (builder *Builder) LightGreenF() *Builder {
	return builder.With(LightGreenF)
}

// LightYellowF is a syntax for LightYellowF.
func (builder *Builder) LightYellowF() *Builder {
	return builder.With(LightYellowF)
}

// LightBlueF is a syntax for LightBlueF.
func (builder *Builder) LightBlueF() *Builder {
	return builder.With(LightBlueF)
}

// LightMagentaF is a syntax for LightMagentaF.
func (builder *Builder) LightMagentaF() *Builder {
	return builder.With(LightMagentaF)
}

// LightCyanF is a syntax for LightCyanF.
func (builder *Builder) LightCyanF() *Builder {
	return builder.With(LightCyanF)
}

// LightWhiteF is a syntax for LightWhiteF.
func (builder *Builder) LightWhiteF() *Builder {
	return builder.With(LightWhiteF)
}

// LightBlackB is a syntax for LightBlackB.
func (builder *Builder) LightBlackB() *Builder {
	return builder.With(LightBlackB)
}

// LightRedB is a syntax for LightRedB.
func (builder *Builder) LightRedB() *Builder {
	return builder.With(LightRedB)
}

// LightGreenB is a syntax for LightGreenB.
func (builder *Builder) LightGreenB() *Builder {
	return builder.With(LightGreenB)
}

// LightYellowB is a syntax for LightYellowB.
func (builder *Builder) LightYellowB() *Builder {
	return builder.With(LightYellowB)
}

// LightBlueB is a syntax for LightBlueB.
func (builder *Builder) LightBlueB() *Builder {
	return builder.With(LightBlueB)
}

// LightMagentaB is a syntax for LightMagentaB.
func (builder *Builder) LightMagentaB() *Builder {
	return builder.With(LightMagentaB)
}

// LightCyanB is a syntax for LightCyanB.
func (builder *Builder) LightCyanB() *Builder {
	return builder.With(LightCyanB)
}

// LightWhiteB is a syntax for LightWhiteB.
func (builder *Builder) LightWhiteB() *Builder {
	return builder.With(LightWhiteB)
}

// Color3BitF is a syntax for Color3BitF.
func (builder *Builder) Color3BitF(c RGB3Bit) *Builder {
	return builder.With(Color3BitF(c))
}

// Color3BitB is a syntax for Color3BitB.
func (builder *Builder) Color3BitB(c RGB3Bit) *Builder {
	return builder.With(Color3BitB(c))
}

// Color8BitF is a syntax for Color8BitF.
func (builder *Builder) Color8BitF(c RGB8Bit) *Builder {
	return builder.With(Color8BitF(c))
}

// Color8BitB is a syntax for Color8BitB.
func (builder *Builder) Color8BitB(c RGB8Bit) *Builder {
	return builder.With(Color8BitB(c))
}

// FullColorF is a syntax for FullColorF.
func (builder *Builder) FullColorF(r, g, b uint8) *Builder {
	return builder.With(FullColorF(r, g, b))
}

// FullColorB is a syntax for FullColorB.
func (builder *Builder) FullColorB(r, g, b uint8) *Builder {
	return builder.With(FullColorB(r, g, b))
}

// RGB3BitF is a syntax for Color3BitF with NewRGB3Bit.
func (builder *Builder) RGB3BitF(r, g, b uint8) *Builder {
	return builder.Color3BitF(NewRGB3Bit(r, g, b))
}

// RGB3BitB is a syntax for Color3BitB with NewRGB3Bit.
func (builder *Builder) RGB3BitB(r, g, b uint8) *Builder {
	return builder.Color3BitB(NewRGB3Bit(r, g, b))
}

// RGB8BitF is a syntax for Color8BitF with NewRGB8Bit.
func (builder *Builder) RGB8BitF(r, g, b uint8) *Builder {
	return builder.Color8BitF(NewRGB8Bit(r, g, b))
}

// RGB8BitB is a syntax for Color8BitB with NewRGB8Bit.
func (builder *Builder) RGB8BitB(r, g, b uint8) *Builder {
	return builder.Color8BitB(NewRGB8Bit(r, g, b))
}

func init() {
	EmptyBuilder = &Builder{empty}
}

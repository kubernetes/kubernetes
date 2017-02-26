package loreley

import (
	"bytes"
	"fmt"
	"os"
	"regexp"
	"strings"
	"text/template"

	"golang.org/x/crypto/ssh/terminal"
)

const (
	// CodeStart is an escape sequence for starting color coding
	CodeStart = "\x1b"

	// CodeEnd is an escape sequence fo ending color coding
	CodeEnd = `m`

	// AttrForeground is an escape sequence part for setting foreground
	AttrForeground = `3`

	// AttrBackground is an escape sequence part for setting background
	AttrBackground = `4`

	// AttrDefault is an escape sequence part for resetting foreground
	// or background
	AttrDefault = `9`

	// AttrReset is an esscape sequence part for resetting all attributes
	AttrReset = `0`

	// AttrReverse is an escape sequence part for setting reverse display
	AttrReverse = `7`

	// AttrNoReverse is an escape sequence part for setting reverse display off
	AttrNoReverse = `27`

	// AttrUnderline is an escape sequence part for setting underline display
	AttrUnderline = `4`

	// AttrNoUnderline is an escape sequence part for setting underline display off
	AttrNoUnderline = `24`

	// AttrBold is an escape sequence part for setting bold mode on
	AttrBold = `1`

	// AttrNoBold is an escape sequence part for setting bold mode off
	AttrNoBold = `22`

	// AttrForeground256 is an escape sequence part for setting foreground
	// color in 256 color mode
	AttrForeground256 = `38;5`

	// AttrBackground256 is an escape sequence part for setting background
	// color in 256 color mode
	AttrBackground256 = `48;5`

	// StyleReset is a placeholder for resetting all attributes.
	StyleReset = `{reset}`

	// StyleForeground is a Sprintf template for setting foreground color.
	StyleForeground = `{fg %d}`

	// StyleBackground is a Sprintf template for setting background color.
	StyleBackground = `{bg %d}`

	// StyleNoFg is a placeholder for resetting foreground color to the default
	// state.
	StyleNoFg = `{nofg}`

	// StyleNoBg is a placeholder for resetting background color to the default
	// state.
	StyleNoBg = `{nobg}`

	// StyleBold is a placeholder for setting bold display mode on.
	StyleBold = `{bold}`

	// StyleNoBold is a placeholder for setting bold display mode off.
	StyleNoBold = `{nobold}`

	// StyleReverse is a placeholder for setting reverse display mode on.
	StyleReverse = `{reverse}`

	// StyleNoReverse is a placeholder for setting reverse display mode off.
	StyleNoReverse = `{noreverse}`

	// StyleUnderline is a placeholder for setting reverse display mode on.
	StyleUnderline = `{underline}`

	// StyleNoUnderline is a placeholder for setting underline display mode off.
	StyleNoUnderline = `{nounderline}`
)

type (
	// Color represents type for foreground or background color, 256-color
	// based.
	Color int
)

const (
	// DefaultColor represents default foreground and background color.
	DefaultColor Color = 0
)

type (
	// ColorizeMode represents default behaviour for colorizing or not
	// colorizing output/
	ColorizeMode int
)

const (
	// ColorizeAlways will tell loreley to colorize output no matter of what.
	ColorizeAlways ColorizeMode = iota

	// ColorizeOnTTY will tell loreley to colorize output only on TTY.
	ColorizeOnTTY

	// ColorizeNever turns of output colorizing.
	ColorizeNever
)

var (
	// CodeRegexp is an regular expression for matching escape codes.
	CodeRegexp = regexp.MustCompile(CodeStart + `[^` + CodeEnd + `]+` + CodeEnd)

	// DelimLeft is used for match template syntax (see Go-lang templates).
	DelimLeft = `{`

	// DelimRight is used for match template syntax (see Go-lang templates).
	DelimRight = `}`

	// Colorize controls whether loreley will output color codes. By default,
	// loreley will try to detect TTY and print colorized output only if
	// TTY is present.
	Colorize = ColorizeOnTTY

	// TTYStream is a stream FD (os.Stderr or os.Stdout), which
	// will be checked by loreley when Colorize = ColorizeOnTTY.
	TTYStream = int(os.Stderr.Fd())
)

// State represents foreground and background color, bold and reversed modes
// in between of Style template execution, which may be usefull for various
// extension functions.
type State struct {
	foreground Color
	background Color

	bold       bool
	reversed   bool
	underlined bool
}

// String returns current state representation as loreley template.
func (state State) String() string {
	styles := []string{}

	if state.foreground == DefaultColor {
		styles = append(styles, StyleNoFg)
	} else {
		styles = append(
			styles,
			fmt.Sprintf(StyleForeground, state.foreground-1),
		)
	}

	if state.background == DefaultColor {
		styles = append(styles, StyleNoBg)
	} else {
		styles = append(
			styles,
			fmt.Sprintf(StyleBackground, state.background-1),
		)
	}

	if state.bold {
		styles = append(styles, StyleBold)
	} else {
		styles = append(styles, StyleNoBold)
	}

	if state.reversed {
		styles = append(styles, StyleReverse)
	} else {
		styles = append(styles, StyleNoReverse)
	}

	if state.underlined {
		styles = append(styles, StyleUnderline)
	} else {
		styles = append(styles, StyleNoUnderline)
	}

	return strings.Join(styles, "")
}

// Style is a compiled style. Can be used as text/template.
type Style struct {
	*template.Template

	state *State

	// NoColors can be set to true to disable escape sequence output,
	// but still render underlying template.
	NoColors bool
}

// GetState returns current style state in the middle of template execution.
func (style *Style) GetState() State {
	return *style.state
}

// SetState sets current style state in the middle of template execution.
func (style *Style) SetState(state State) {
	*style.state = state
}

// ExecuteToString is same, as text/template Execute, but return string
// as a result.
func (style *Style) ExecuteToString(
	data map[string]interface{},
) (string, error) {
	buffer := bytes.Buffer{}

	err := style.Template.Execute(&buffer, data)
	if err != nil {
		return "", err
	}

	return buffer.String(), nil
}

func (style *Style) putReset() string {
	style.state.background = DefaultColor
	style.state.foreground = DefaultColor
	style.state.bold = false
	style.state.reversed = false

	return style.getStyleCodes(AttrReset)
}

func (style *Style) putBackground(color Color) string {
	if color == -1 {
		return style.putDefaultBackground()
	}

	style.state.background = color + 1

	return style.getStyleCodes(AttrBackground256, fmt.Sprint(color))
}

func (style *Style) putDefaultBackground() string {
	style.state.background = DefaultColor

	return style.getStyleCodes(
		AttrBackground + AttrDefault,
	)
}

func (style *Style) putDefaultForeground() string {
	style.state.foreground = DefaultColor

	return style.getStyleCodes(
		AttrForeground + AttrDefault,
	)
}

func (style *Style) putForeground(color Color) string {
	if color == -1 {
		return style.putDefaultForeground()
	}

	style.state.foreground = color + 1

	return style.getStyleCodes(AttrForeground256, fmt.Sprint(color))
}

func (style *Style) putBold() string {
	style.state.bold = true

	return style.getStyleCodes(AttrBold)
}

func (style *Style) putNoBold() string {
	style.state.bold = false

	return style.getStyleCodes(AttrNoBold)
}

func (style *Style) putReverse() string {
	style.state.reversed = true

	return style.getStyleCodes(AttrReverse)
}

func (style *Style) putNoReverse() string {
	style.state.reversed = false

	return style.getStyleCodes(AttrNoReverse)
}

func (style *Style) putUnderline() string {
	style.state.underlined = true

	return style.getStyleCodes(AttrUnderline)
}

func (style *Style) putNoUnderline() string {
	style.state.underlined = false

	return style.getStyleCodes(AttrNoUnderline)
}

func (style *Style) putTransitionFrom(
	text string,
	nextBackground Color,
) string {
	previousBackground := style.state.background - 1
	previousForeground := style.state.foreground - 1

	return style.putForeground(previousBackground) +
		style.putBackground(nextBackground) +
		text +
		style.putForeground(previousForeground)
}

func (style *Style) putTransitionTo(nextBackground Color, text string) string {
	previousForeground := style.state.foreground - 1

	return style.putForeground(nextBackground) +
		text +
		style.putBackground(nextBackground) +
		style.putForeground(previousForeground)
}

func (style *Style) getStyleCodes(attr ...string) string {
	if style.NoColors {
		return ``
	}

	return fmt.Sprintf(`%s[%sm`, CodeStart, strings.Join(attr, `;`))
}

// Compile compiles style which is specified by string into Style,
// optionally adding extended formatting functions.
//
// Avaiable functions to extend Go-lang template syntax:
// * {bg <color>} - changes background color to the specified <color>.
// * {fg <color>} - changes foreground color to the specified <color>.
// * {nobg} - resets background color to the default.
// * {nofg} - resets background color to the default.
// * {bold} - set display mode to bold.
// * {nobold} - disable bold display mode.
// * {reverse} - set display mode to reverse.
// * {noreverse} - disable reverse display mode.
// * {reset} - resets all previously set styles.
// * {from <text> <bg>} - renders <text> which foreground color equals
//   current background color, and background color with specified <bg>
//   color. Applies current foreground color and specified <bg> color
//   to the following statements.
// * {to <fg> <text>} - renders <text> with specified <fg> color as
//   foreground color, and then use it as background color for the
//   following statements.
//
// `extensions` is a additional functions, which will be available in
// the template.
func Compile(
	text string,
	extensions map[string]interface{},
) (*Style, error) {
	style := &Style{
		state: &State{},
	}

	functions := map[string]interface{}{
		`bg`:          style.putBackground,
		`fg`:          style.putForeground,
		`nobg`:        style.putDefaultBackground,
		`nofg`:        style.putDefaultForeground,
		`bold`:        style.putBold,
		`nobold`:      style.putNoBold,
		`reverse`:     style.putReverse,
		`noreverse`:   style.putNoReverse,
		`nounderline`: style.putNoUnderline,
		`underline`:   style.putUnderline,
		`reset`:       style.putReset,
		`from`:        style.putTransitionFrom,
		`to`:          style.putTransitionTo,
	}

	for name, function := range extensions {
		functions[name] = function
	}

	template, err := template.New(`style`).Delims(
		DelimLeft,
		DelimRight,
	).Funcs(functions).Parse(
		text,
	)

	if err != nil {
		return nil, err
	}

	style.Template = template

	switch Colorize {
	case ColorizeNever:
		style.NoColors = true

	case ColorizeOnTTY:
		style.NoColors = !HasTTY(TTYStream)

	}

	return style, nil
}

// CompileAndExecuteToString compiles Compile specified template and
// immediately execute it with given `data`.
func CompileAndExecuteToString(
	text string,
	extensions map[string]interface{},
	data map[string]interface{},
) (string, error) {
	style, err := Compile(text, extensions)
	if err != nil {
		return "", err
	}

	return style.ExecuteToString(data)
}

// CompileWithReset is same as Compile, but appends {reset} to the end
// of given style string.
func CompileWithReset(
	text string,
	extensions map[string]interface{},
) (*Style, error) {
	return Compile(text+StyleReset, extensions)
}

// TrimStyles removes all escape codes from the given string.
func TrimStyles(input string) string {
	return CodeRegexp.ReplaceAllLiteralString(input, ``)
}

// HasTTY will return true if specified file descriptor is bound to a TTY.
func HasTTY(fd int) bool {
	return terminal.IsTerminal(fd)
}

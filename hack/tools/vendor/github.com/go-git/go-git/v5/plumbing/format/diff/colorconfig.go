package diff

import "github.com/go-git/go-git/v5/plumbing/color"

// A ColorKey is a key into a ColorConfig map and also equal to the key in the
// diff.color subsection of the config. See
// https://github.com/git/git/blob/v2.26.2/diff.c#L83-L106.
type ColorKey string

// ColorKeys.
const (
	Context                   ColorKey = "context"
	Meta                      ColorKey = "meta"
	Frag                      ColorKey = "frag"
	Old                       ColorKey = "old"
	New                       ColorKey = "new"
	Commit                    ColorKey = "commit"
	Whitespace                ColorKey = "whitespace"
	Func                      ColorKey = "func"
	OldMoved                  ColorKey = "oldMoved"
	OldMovedAlternative       ColorKey = "oldMovedAlternative"
	OldMovedDimmed            ColorKey = "oldMovedDimmed"
	OldMovedAlternativeDimmed ColorKey = "oldMovedAlternativeDimmed"
	NewMoved                  ColorKey = "newMoved"
	NewMovedAlternative       ColorKey = "newMovedAlternative"
	NewMovedDimmed            ColorKey = "newMovedDimmed"
	NewMovedAlternativeDimmed ColorKey = "newMovedAlternativeDimmed"
	ContextDimmed             ColorKey = "contextDimmed"
	OldDimmed                 ColorKey = "oldDimmed"
	NewDimmed                 ColorKey = "newDimmed"
	ContextBold               ColorKey = "contextBold"
	OldBold                   ColorKey = "oldBold"
	NewBold                   ColorKey = "newBold"
)

// A ColorConfig is a color configuration. A nil or empty ColorConfig
// corresponds to no color.
type ColorConfig map[ColorKey]string

// A ColorConfigOption sets an option on a ColorConfig.
type ColorConfigOption func(ColorConfig)

// WithColor sets the color for key.
func WithColor(key ColorKey, color string) ColorConfigOption {
	return func(cc ColorConfig) {
		cc[key] = color
	}
}

// defaultColorConfig is the default color configuration. See
// https://github.com/git/git/blob/v2.26.2/diff.c#L57-L81.
var defaultColorConfig = ColorConfig{
	Context:                   color.Normal,
	Meta:                      color.Bold,
	Frag:                      color.Cyan,
	Old:                       color.Red,
	New:                       color.Green,
	Commit:                    color.Yellow,
	Whitespace:                color.BgRed,
	Func:                      color.Normal,
	OldMoved:                  color.BoldMagenta,
	OldMovedAlternative:       color.BoldBlue,
	OldMovedDimmed:            color.Faint,
	OldMovedAlternativeDimmed: color.FaintItalic,
	NewMoved:                  color.BoldCyan,
	NewMovedAlternative:       color.BoldYellow,
	NewMovedDimmed:            color.Faint,
	NewMovedAlternativeDimmed: color.FaintItalic,
	ContextDimmed:             color.Faint,
	OldDimmed:                 color.FaintRed,
	NewDimmed:                 color.FaintGreen,
	ContextBold:               color.Bold,
	OldBold:                   color.BoldRed,
	NewBold:                   color.BoldGreen,
}

// NewColorConfig returns a new ColorConfig.
func NewColorConfig(options ...ColorConfigOption) ColorConfig {
	cc := make(ColorConfig)
	for key, value := range defaultColorConfig {
		cc[key] = value
	}
	for _, option := range options {
		option(cc)
	}
	return cc
}

// Reset returns the ANSI escape sequence to reset the color with key set from
// cc. If no color was set then no reset is needed so it returns the empty
// string.
func (cc ColorConfig) Reset(key ColorKey) string {
	if cc[key] == "" {
		return ""
	}
	return color.Reset
}

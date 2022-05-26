package color

// TODO read colors from a github.com/go-git/go-git/plumbing/format/config.Config struct
// TODO implement color parsing, see https://github.com/git/git/blob/v2.26.2/color.c

// Colors. See https://github.com/git/git/blob/v2.26.2/color.h#L24-L53.
const (
	Normal       = ""
	Reset        = "\033[m"
	Bold         = "\033[1m"
	Red          = "\033[31m"
	Green        = "\033[32m"
	Yellow       = "\033[33m"
	Blue         = "\033[34m"
	Magenta      = "\033[35m"
	Cyan         = "\033[36m"
	BoldRed      = "\033[1;31m"
	BoldGreen    = "\033[1;32m"
	BoldYellow   = "\033[1;33m"
	BoldBlue     = "\033[1;34m"
	BoldMagenta  = "\033[1;35m"
	BoldCyan     = "\033[1;36m"
	FaintRed     = "\033[2;31m"
	FaintGreen   = "\033[2;32m"
	FaintYellow  = "\033[2;33m"
	FaintBlue    = "\033[2;34m"
	FaintMagenta = "\033[2;35m"
	FaintCyan    = "\033[2;36m"
	BgRed        = "\033[41m"
	BgGreen      = "\033[42m"
	BgYellow     = "\033[43m"
	BgBlue       = "\033[44m"
	BgMagenta    = "\033[45m"
	BgCyan       = "\033[46m"
	Faint        = "\033[2m"
	FaintItalic  = "\033[2;3m"
	Reverse      = "\033[7m"
)

package configs

const (
	// EXT_COPYUP is a directive to copy up the contents of a directory when
	// a tmpfs is mounted over it.
	EXT_COPYUP = 1 << iota //nolint:golint,revive // ignore "don't use ALL_CAPS" warning
)

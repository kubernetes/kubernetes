package getter

// ClientMode is the mode that the client operates in.
type ClientMode uint

const (
	ClientModeInvalid ClientMode = iota

	// ClientModeAny downloads anything it can. In this mode, dst must
	// be a directory. If src is a file, it is saved into the directory
	// with the basename of the URL. If src is a directory or archive,
	// it is unpacked directly into dst.
	ClientModeAny

	// ClientModeFile downloads a single file. In this mode, dst must
	// be a file path (doesn't have to exist). src must point to a single
	// file. It is saved as dst.
	ClientModeFile

	// ClientModeDir downloads a directory. In this mode, dst must be
	// a directory path (doesn't have to exist). src must point to an
	// archive or directory (such as in s3).
	ClientModeDir
)

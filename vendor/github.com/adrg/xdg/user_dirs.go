package xdg

// XDG user directories environment variables.
const (
	envDesktopDir     = "XDG_DESKTOP_DIR"
	envDownloadDir    = "XDG_DOWNLOAD_DIR"
	envDocumentsDir   = "XDG_DOCUMENTS_DIR"
	envMusicDir       = "XDG_MUSIC_DIR"
	envPicturesDir    = "XDG_PICTURES_DIR"
	envVideosDir      = "XDG_VIDEOS_DIR"
	envTemplatesDir   = "XDG_TEMPLATES_DIR"
	envPublicShareDir = "XDG_PUBLICSHARE_DIR"
)

// UserDirectories defines the locations of well known user directories.
type UserDirectories struct {
	// Desktop defines the location of the user's desktop directory.
	Desktop string

	// Download defines a suitable location for user downloaded files.
	Download string

	// Documents defines a suitable location for user document files.
	Documents string

	// Music defines a suitable location for user audio files.
	Music string

	// Pictures defines a suitable location for user image files.
	Pictures string

	// VideosDir defines a suitable location for user video files.
	Videos string

	// Templates defines a suitable location for user template files.
	Templates string

	// PublicShare defines a suitable location for user shared files.
	PublicShare string
}

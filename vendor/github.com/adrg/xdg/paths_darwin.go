package xdg

import (
	"os"
	"path/filepath"
)

func homeDir() string {
	if home := os.Getenv("HOME"); home != "" {
		return home
	}

	return "/"
}

func initDirs(home string) {
	initBaseDirs(home)
	initUserDirs(home)
}

func initBaseDirs(home string) {
	homeAppSupport := filepath.Join(home, "Library", "Application Support")
	rootAppSupport := "/Library/Application Support"

	// Initialize standard directories.
	baseDirs.dataHome = xdgPath(envDataHome, homeAppSupport)
	baseDirs.data = xdgPaths(envDataDirs, rootAppSupport)
	baseDirs.configHome = xdgPath(envConfigHome, homeAppSupport)
	baseDirs.config = xdgPaths(envConfigDirs,
		filepath.Join(home, "Library", "Preferences"),
		rootAppSupport,
		"/Library/Preferences",
	)
	baseDirs.stateHome = xdgPath(envStateHome, homeAppSupport)
	baseDirs.cacheHome = xdgPath(envCacheHome, filepath.Join(home, "Library", "Caches"))
	baseDirs.runtime = xdgPath(envRuntimeDir, homeAppSupport)

	// Initialize non-standard directories.
	baseDirs.applications = []string{
		"/Applications",
	}

	baseDirs.fonts = []string{
		filepath.Join(home, "Library/Fonts"),
		"/Library/Fonts",
		"/System/Library/Fonts",
		"/Network/Library/Fonts",
	}
}

func initUserDirs(home string) {
	UserDirs.Desktop = xdgPath(envDesktopDir, filepath.Join(home, "Desktop"))
	UserDirs.Download = xdgPath(envDownloadDir, filepath.Join(home, "Downloads"))
	UserDirs.Documents = xdgPath(envDocumentsDir, filepath.Join(home, "Documents"))
	UserDirs.Music = xdgPath(envMusicDir, filepath.Join(home, "Music"))
	UserDirs.Pictures = xdgPath(envPicturesDir, filepath.Join(home, "Pictures"))
	UserDirs.Videos = xdgPath(envVideosDir, filepath.Join(home, "Movies"))
	UserDirs.Templates = xdgPath(envTemplatesDir, filepath.Join(home, "Templates"))
	UserDirs.PublicShare = xdgPath(envPublicShareDir, filepath.Join(home, "Public"))
}

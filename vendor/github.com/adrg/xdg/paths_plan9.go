package xdg

import (
	"os"
	"path/filepath"
)

func homeDir() string {
	if home := os.Getenv("home"); home != "" {
		return home
	}

	return "/"
}

func initDirs(home string) {
	initBaseDirs(home)
	initUserDirs(home)
}

func initBaseDirs(home string) {
	homeLibDir := filepath.Join(home, "lib")
	rootLibDir := "/lib"

	// Initialize standard directories.
	baseDirs.dataHome = xdgPath(envDataHome, homeLibDir)
	baseDirs.data = xdgPaths(envDataDirs, rootLibDir)
	baseDirs.configHome = xdgPath(envConfigHome, homeLibDir)
	baseDirs.config = xdgPaths(envConfigDirs, rootLibDir)
	baseDirs.stateHome = xdgPath(envStateHome, filepath.Join(homeLibDir, "state"))
	baseDirs.cacheHome = xdgPath(envCacheHome, filepath.Join(homeLibDir, "cache"))
	baseDirs.runtime = xdgPath(envRuntimeDir, "/tmp")

	// Initialize non-standard directories.
	baseDirs.applications = []string{
		filepath.Join(home, "bin"),
		"/bin",
	}

	baseDirs.fonts = []string{
		filepath.Join(homeLibDir, "font"),
		"/lib/font",
	}
}

func initUserDirs(home string) {
	UserDirs.Desktop = xdgPath(envDesktopDir, filepath.Join(home, "desktop"))
	UserDirs.Download = xdgPath(envDownloadDir, filepath.Join(home, "downloads"))
	UserDirs.Documents = xdgPath(envDocumentsDir, filepath.Join(home, "documents"))
	UserDirs.Music = xdgPath(envMusicDir, filepath.Join(home, "music"))
	UserDirs.Pictures = xdgPath(envPicturesDir, filepath.Join(home, "pictures"))
	UserDirs.Videos = xdgPath(envVideosDir, filepath.Join(home, "videos"))
	UserDirs.Templates = xdgPath(envTemplatesDir, filepath.Join(home, "templates"))
	UserDirs.PublicShare = xdgPath(envPublicShareDir, filepath.Join(home, "public"))
}

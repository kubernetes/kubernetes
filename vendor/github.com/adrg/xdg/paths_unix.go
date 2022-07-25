//go:build aix || dragonfly || freebsd || (js && wasm) || nacl || linux || netbsd || openbsd || solaris
// +build aix dragonfly freebsd js,wasm nacl linux netbsd openbsd solaris

package xdg

import (
	"os"
	"path/filepath"
	"strconv"

	"github.com/adrg/xdg/internal/pathutil"
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
	// Initialize standard directories.
	baseDirs.dataHome = xdgPath(envDataHome, filepath.Join(home, ".local", "share"))
	baseDirs.data = xdgPaths(envDataDirs, "/usr/local/share", "/usr/share")
	baseDirs.configHome = xdgPath(envConfigHome, filepath.Join(home, ".config"))
	baseDirs.config = xdgPaths(envConfigDirs, "/etc/xdg")
	baseDirs.stateHome = xdgPath(envStateHome, filepath.Join(home, ".local", "state"))
	baseDirs.cacheHome = xdgPath(envCacheHome, filepath.Join(home, ".cache"))
	baseDirs.runtime = xdgPath(envRuntimeDir, filepath.Join("/run/user", strconv.Itoa(os.Getuid())))

	// Initialize non-standard directories.
	appDirs := []string{
		filepath.Join(baseDirs.dataHome, "applications"),
		filepath.Join(home, ".local/share/applications"),
		"/usr/local/share/applications",
		"/usr/share/applications",
	}

	fontDirs := []string{
		filepath.Join(baseDirs.dataHome, "fonts"),
		filepath.Join(home, ".fonts"),
		filepath.Join(home, ".local/share/fonts"),
		"/usr/local/share/fonts",
		"/usr/share/fonts",
	}

	for _, dir := range baseDirs.data {
		appDirs = append(appDirs, filepath.Join(dir, "applications"))
		fontDirs = append(fontDirs, filepath.Join(dir, "fonts"))
	}

	baseDirs.applications = pathutil.Unique(appDirs, Home)
	baseDirs.fonts = pathutil.Unique(fontDirs, Home)
}

func initUserDirs(home string) {
	UserDirs.Desktop = xdgPath(envDesktopDir, filepath.Join(home, "Desktop"))
	UserDirs.Download = xdgPath(envDownloadDir, filepath.Join(home, "Downloads"))
	UserDirs.Documents = xdgPath(envDocumentsDir, filepath.Join(home, "Documents"))
	UserDirs.Music = xdgPath(envMusicDir, filepath.Join(home, "Music"))
	UserDirs.Pictures = xdgPath(envPicturesDir, filepath.Join(home, "Pictures"))
	UserDirs.Videos = xdgPath(envVideosDir, filepath.Join(home, "Videos"))
	UserDirs.Templates = xdgPath(envTemplatesDir, filepath.Join(home, "Templates"))
	UserDirs.PublicShare = xdgPath(envPublicShareDir, filepath.Join(home, "Public"))
}

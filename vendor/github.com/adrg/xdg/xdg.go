package xdg

import (
	"os"
	"path/filepath"

	"github.com/adrg/xdg/internal/pathutil"
)

var (
	// Home contains the path of the user's home directory.
	Home string

	// DataHome defines the base directory relative to which user-specific
	// data files should be stored. This directory is defined by the
	// $XDG_DATA_HOME environment variable. If the variable is not set,
	// a default equal to $HOME/.local/share should be used.
	DataHome string

	// DataDirs defines the preference-ordered set of base directories to
	// search for data files in addition to the DataHome base directory.
	// This set of directories is defined by the $XDG_DATA_DIRS environment
	// variable. If the variable is not set, the default directories
	// to be used are /usr/local/share and /usr/share, in that order. The
	// DataHome directory is considered more important than any of the
	// directories defined by DataDirs. Therefore, user data files should be
	// written relative to the DataHome directory, if possible.
	DataDirs []string

	// ConfigHome defines the base directory relative to which user-specific
	// configuration files should be written. This directory is defined by
	// the $XDG_CONFIG_HOME environment variable. If the variable is not
	// not set, a default equal to $HOME/.config should be used.
	ConfigHome string

	// ConfigDirs defines the preference-ordered set of base directories to
	// search for configuration files in addition to the ConfigHome base
	// directory. This set of directories is defined by the $XDG_CONFIG_DIRS
	// environment variable. If the variable is not set, a default equal
	// to /etc/xdg should be used. The ConfigHome directory is considered
	// more important than any of the directories defined by ConfigDirs.
	// Therefore, user config files should be written relative to the
	// ConfigHome directory, if possible.
	ConfigDirs []string

	// StateHome defines the base directory relative to which user-specific
	// state files should be stored. This directory is defined by the
	// $XDG_STATE_HOME environment variable. If the variable is not set,
	// a default equal to ~/.local/state should be used.
	StateHome string

	// CacheHome defines the base directory relative to which user-specific
	// non-essential (cached) data should be written. This directory is
	// defined by the $XDG_CACHE_HOME environment variable. If the variable
	// is not set, a default equal to $HOME/.cache should be used.
	CacheHome string

	// RuntimeDir defines the base directory relative to which user-specific
	// non-essential runtime files and other file objects (such as sockets,
	// named pipes, etc.) should be stored. This directory is defined by the
	// $XDG_RUNTIME_DIR environment variable. If the variable is not set,
	// applications should fall back to a replacement directory with similar
	// capabilities. Applications should use this directory for communication
	// and synchronization purposes and should not place larger files in it,
	// since it might reside in runtime memory and cannot necessarily be
	// swapped out to disk.
	RuntimeDir string

	// UserDirs defines the locations of well known user directories.
	UserDirs UserDirectories

	// FontDirs defines the common locations where font files are stored.
	FontDirs []string

	// ApplicationDirs defines the common locations of applications.
	ApplicationDirs []string

	// baseDirs defines the locations of base directories.
	baseDirs baseDirectories
)

func init() {
	Reload()
}

// Reload refreshes base and user directories by reading the environment.
// Defaults are applied for XDG variables which are empty or not present
// in the environment.
func Reload() {
	// Initialize home directory.
	Home = homeDir()

	// Initialize base and user directories.
	initDirs(Home)

	// Set standard directories.
	DataHome = baseDirs.dataHome
	DataDirs = baseDirs.data
	ConfigHome = baseDirs.configHome
	ConfigDirs = baseDirs.config
	StateHome = baseDirs.stateHome
	CacheHome = baseDirs.cacheHome
	RuntimeDir = baseDirs.runtime

	// Set non-standard directories.
	FontDirs = baseDirs.fonts
	ApplicationDirs = baseDirs.applications
}

// DataFile returns a suitable location for the specified data file.
// The relPath parameter must contain the name of the data file, and
// optionally, a set of parent directories (e.g. appname/app.data).
// If the specified directories do not exist, they will be created relative
// to the base data directory. On failure, an error containing the
// attempted paths is returned.
func DataFile(relPath string) (string, error) {
	return baseDirs.dataFile(relPath)
}

// ConfigFile returns a suitable location for the specified config file.
// The relPath parameter must contain the name of the config file, and
// optionally, a set of parent directories (e.g. appname/app.yaml).
// If the specified directories do not exist, they will be created relative
// to the base config directory. On failure, an error containing the
// attempted paths is returned.
func ConfigFile(relPath string) (string, error) {
	return baseDirs.configFile(relPath)
}

// StateFile returns a suitable location for the specified state file. State
// files are usually volatile data files, not suitable to be stored relative
// to the $XDG_DATA_HOME directory.
// The relPath parameter must contain the name of the state file, and
// optionally, a set of parent directories (e.g. appname/app.state).
// If the specified directories do not exist, they will be created relative
// to the base state directory. On failure, an error containing the
// attempted paths is returned.
func StateFile(relPath string) (string, error) {
	return baseDirs.stateFile(relPath)
}

// CacheFile returns a suitable location for the specified cache file.
// The relPath parameter must contain the name of the cache file, and
// optionally, a set of parent directories (e.g. appname/app.cache).
// If the specified directories do not exist, they will be created relative
// to the base cache directory. On failure, an error containing the
// attempted paths is returned.
func CacheFile(relPath string) (string, error) {
	return baseDirs.cacheFile(relPath)
}

// RuntimeFile returns a suitable location for the specified runtime file.
// The relPath parameter must contain the name of the runtime file, and
// optionally, a set of parent directories (e.g. appname/app.pid).
// If the specified directories do not exist, they will be created relative
// to the base runtime directory. On failure, an error containing the
// attempted paths is returned.
func RuntimeFile(relPath string) (string, error) {
	return baseDirs.runtimeFile(relPath)
}

// SearchDataFile searches for specified file in the data search paths.
// The relPath parameter must contain the name of the data file, and
// optionally, a set of parent directories (e.g. appname/app.data). If the
// file cannot be found, an error specifying the searched paths is returned.
func SearchDataFile(relPath string) (string, error) {
	return baseDirs.searchDataFile(relPath)
}

// SearchConfigFile searches for the specified file in config search paths.
// The relPath parameter must contain the name of the config file, and
// optionally, a set of parent directories (e.g. appname/app.yaml). If the
// file cannot be found, an error specifying the searched paths is returned.
func SearchConfigFile(relPath string) (string, error) {
	return baseDirs.searchConfigFile(relPath)
}

// SearchStateFile searches for the specified file in the state search path.
// The relPath parameter must contain the name of the state file, and
// optionally, a set of parent directories (e.g. appname/app.state). If the
// file cannot be found, an error specifying the searched path is returned.
func SearchStateFile(relPath string) (string, error) {
	return baseDirs.searchStateFile(relPath)
}

// SearchCacheFile searches for the specified file in the cache search path.
// The relPath parameter must contain the name of the cache file, and
// optionally, a set of parent directories (e.g. appname/app.cache). If the
// file cannot be found, an error specifying the searched path is returned.
func SearchCacheFile(relPath string) (string, error) {
	return baseDirs.searchCacheFile(relPath)
}

// SearchRuntimeFile searches for the specified file in the runtime search path.
// The relPath parameter must contain the name of the runtime file, and
// optionally, a set of parent directories (e.g. appname/app.pid). If the
// file cannot be found, an error specifying the searched path is returned.
func SearchRuntimeFile(relPath string) (string, error) {
	return baseDirs.searchRuntimeFile(relPath)
}

func xdgPath(name, defaultPath string) string {
	dir := pathutil.ExpandHome(os.Getenv(name), Home)
	if dir != "" && filepath.IsAbs(dir) {
		return dir
	}

	return defaultPath
}

func xdgPaths(name string, defaultPaths ...string) []string {
	dirs := pathutil.Unique(filepath.SplitList(os.Getenv(name)), Home)
	if len(dirs) != 0 {
		return dirs
	}

	return pathutil.Unique(defaultPaths, Home)
}

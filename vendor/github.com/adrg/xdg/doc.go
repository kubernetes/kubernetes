/*
Package xdg provides an implementation of the XDG Base Directory Specification.
The specification defines a set of standard paths for storing application files
including data and configuration files. For portability and flexibility reasons,
applications should use the XDG defined locations instead of hardcoding paths.
The package also includes the locations of well known user directories.

The current implementation supports most flavors of Unix, Windows, Mac OS and Plan 9.

	For more information regarding the XDG Base Directory Specification see:
	https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html

	For more information regarding the XDG user directories see:
	https://wiki.archlinux.org/index.php/XDG_user_directories

	For more information regarding the Windows Known Folders see:
	https://docs.microsoft.com/en-us/windows/win32/shell/known-folders

Usage

XDG Base Directory
	package main

	import (
		"log"

		"github.com/adrg/xdg"
	)

	func main() {
		// XDG Base Directory paths.
		log.Println("Home data directory:", xdg.DataHome)
		log.Println("Data directories:", xdg.DataDirs)
		log.Println("Home config directory:", xdg.ConfigHome)
		log.Println("Config directories:", xdg.ConfigDirs)
		log.Println("Home state directory:", xdg.StateHome)
		log.Println("Cache directory:", xdg.CacheHome)
		log.Println("Runtime directory:", xdg.RuntimeDir)

		// Other common directories.
		log.Println("Home directory:", xdg.Home)
		log.Println("Application directories:", xdg.ApplicationDirs)
		log.Println("Font directories:", xdg.FontDirs)

		// Obtain a suitable location for application config files.
		// ConfigFile takes one parameter which must contain the name of the file,
		// but it can also contain a set of parent directories. If the directories
		// don't exist, they will be created relative to the base config directory.
		configFilePath, err := xdg.ConfigFile("appname/config.yaml")
		if err != nil {
			log.Fatal(err)
		}
		log.Println("Save the config file at:", configFilePath)

		// For other types of application files use:
		// xdg.DataFile()
		// xdg.StateFile()
		// xdg.CacheFile()
		// xdg.RuntimeFile()

		// Finding application config files.
		// SearchConfigFile takes one parameter which must contain the name of
		// the file, but it can also contain a set of parent directories relative
		// to the config search paths (xdg.ConfigHome and xdg.ConfigDirs).
		configFilePath, err = xdg.SearchConfigFile("appname/config.yaml")
		if err != nil {
			log.Fatal(err)
		}
		log.Println("Config file was found at:", configFilePath)

		// For other types of application files use:
		// xdg.SearchDataFile()
		// xdg.SearchStateFile()
		// xdg.SearchCacheFile()
		// xdg.SearchRuntimeFile()
	}

XDG user directories
	package main

	import (
		"log"

		"github.com/adrg/xdg"
	)

	func main() {
		// XDG user directories.
		log.Println("Desktop directory:", xdg.UserDirs.Desktop)
		log.Println("Download directory:", xdg.UserDirs.Download)
		log.Println("Documents directory:", xdg.UserDirs.Documents)
		log.Println("Music directory:", xdg.UserDirs.Music)
		log.Println("Pictures directory:", xdg.UserDirs.Pictures)
		log.Println("Videos directory:", xdg.UserDirs.Videos)
		log.Println("Templates directory:", xdg.UserDirs.Templates)
		log.Println("Public directory:", xdg.UserDirs.PublicShare)
	}
*/
package xdg

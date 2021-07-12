package ssocreds

import "os"

func getHomeDirectory() string {
	return os.Getenv("USERPROFILE")
}

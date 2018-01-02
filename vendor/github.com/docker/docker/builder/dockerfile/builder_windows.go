package dockerfile

func defaultShellForPlatform(platform string) []string {
	if platform == "linux" {
		return []string{"/bin/sh", "-c"}
	}
	return []string{"cmd", "/S", "/C"}
}

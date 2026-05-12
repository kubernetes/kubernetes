package dbus

func getDefaultAuthMethods(user string) []Auth {
	return []Auth{AuthCookieSha1(user, getHomeDir())}
}

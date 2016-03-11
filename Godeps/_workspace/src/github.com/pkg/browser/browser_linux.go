package browser

func openBrowser(url string) error {
	return runCmd("xdg-open", url)
}

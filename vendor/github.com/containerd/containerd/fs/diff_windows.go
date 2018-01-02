package fs

import "os"

func detectDirDiff(upper, lower string) *diffDirOptions {
	return nil
}

func compareSysStat(s1, s2 interface{}) (bool, error) {
	// TODO: Use windows specific sys type
	return false, nil
}

func compareCapabilities(p1, p2 string) (bool, error) {
	// TODO: Use windows equivalent
	return true, nil
}

func isLinked(os.FileInfo) bool {
	return false
}

// +build !linux

package archive

func getWhiteoutConverter(format WhiteoutFormat) tarWhiteoutConverter {
	return nil
}

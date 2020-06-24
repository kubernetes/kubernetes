package mountinfo

import "io"

func parseMountTable(_ FilterFunc) ([]*Info, error) {
	// Do NOT return an error!
	return nil, nil
}

func parseInfoFile(_ io.Reader, f FilterFunc) ([]*Info, error) {
	return parseMountTable(f)
}

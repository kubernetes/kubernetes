package mountinfo

func parseMountTable(_ FilterFunc) ([]*Info, error) {
	// Do NOT return an error!
	return nil, nil
}

func mounted(_ string) (bool, error) {
	return false, nil
}

// validatePayload returns an error if any path in the payload is invalid,
// otherwise it returns a copy of the payload with normalized/cleaned keys.
func validatePayload(payload map[string]FileProjection) (map[string]FileProjection, error) {
	cleanPayload := make(map[string]FileProjection, len(payload))

	for k, content := range payload {
		if k == "" {
			return nil, fmt.Errorf("invalid path: must not be empty: %q", k)
		}

		// Normalize first (important on Windows: handles mixed slashes),
		// then clean so we store a canonical key.
		ck := filepath.Clean(filepath.FromSlash(k))
		if ck == "." {
			return nil, fmt.Errorf("invalid path: must not be '.'")
		}

		if err := validatePath(ck); err != nil {
			return nil, err
		}

		cleanPayload[ck] = content
	}
	return cleanPayload, nil
}

// validatePath validates a single path, returning an error if the path is invalid.
func validatePath(targetPath string) error {
	if targetPath == "" {
		return fmt.Errorf("invalid path: must not be empty: %q", targetPath)
	}

	// Windows-specific hardening:
	// - reject any volume-qualified path (C:, \\server\share, etc)
	// - reject rooted paths (\foo) and absolute paths
	// - reject ':' anywhere (defense-in-depth against "\C::" segments)
	if runtime.GOOS == "windows" {
		if filepath.VolumeName(targetPath) != "" ||
			filepath.IsAbs(targetPath) ||
			strings.ContainsRune(targetPath, ':') {
			return fmt.Errorf("invalid path: must be relative path: %s", targetPath)
		}
	}

	// Keep POSIX abs check too (also blocks /foo on Windows inputs)
	if path.IsAbs(targetPath) {
		return fmt.Errorf("invalid path: must be relative path: %s", targetPath)
	}

	if len(targetPath) > maxPathLength {
		return fmt.Errorf("invalid path: must be less than or equal to %d characters", maxPathLength)
	}

	items := strings.Split(targetPath, string(os.PathSeparator))
	for _, item := range items {
		if item == ".." {
			return fmt.Errorf("invalid path: must not contain '..': %s", targetPath)
		}
		if len(item) > maxFileNameLength {
			return fmt.Errorf("invalid path: filenames must be less than or equal to %d characters", maxFileNameLength)
		}
	}

	if strings.HasPrefix(items[0], "..") && len(items[0]) > 2 {
		return fmt.Errorf("invalid path: must not start with '..': %s", targetPath)
	}

	return nil
}


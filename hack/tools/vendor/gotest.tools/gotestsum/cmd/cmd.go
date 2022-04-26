package cmd

// Next splits args into the next positional argument and any remaining args.
func Next(args []string) (string, []string) {
	switch len(args) {
	case 0:
		return "", nil
	case 1:
		return args[0], nil
	default:
		return args[0], args[1:]
	}
}

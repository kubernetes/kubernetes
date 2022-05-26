package toml

// ValueStringRepresentation transforms an interface{} value into its toml string representation.
func ValueStringRepresentation(v interface{}, commented string, indent string, ord MarshalOrder, arraysOneElementPerLine bool) (string, error) {
	return tomlValueStringRepresentation(v, commented, indent, ord, arraysOneElementPerLine)
}

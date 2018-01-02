key "" {
	policy = "read"
}

key "foo/" {
	policy = "write"
}

key "foo/bar/" {
	policy = "read"
}

key "foo/bar/baz" {
	policy = "deny"
}

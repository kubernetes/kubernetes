package bbolt

// mlock locks memory of db file
func mlock(_ *DB, _ int) error {
	panic("mlock is supported only on UNIX systems")
}

// munlock unlocks memory of db file
func munlock(_ *DB, _ int) error {
	panic("munlock is supported only on UNIX systems")
}

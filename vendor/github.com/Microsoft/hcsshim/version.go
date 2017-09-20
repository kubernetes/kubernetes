package hcsshim

// IsTP4 returns whether the currently running Windows build is at least TP4.
func IsTP4() bool {
	// HNSCall was not present in TP4
	return procHNSCall.Find() != nil
}

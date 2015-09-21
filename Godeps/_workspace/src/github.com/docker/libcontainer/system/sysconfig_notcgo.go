// +build !cgo

package system

func GetClockTicks() int {
	// TODO figure out a better alternative for platforms where we're missing cgo
	return 100
}

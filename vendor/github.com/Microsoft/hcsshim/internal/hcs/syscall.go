package hcs

//go:generate go run ../../mksyscall_windows.go -output zsyscall_windows.go syscall.go

//sys hcsFormatWritableLayerVhd(handle uintptr) (hr error) = computestorage.HcsFormatWritableLayerVhd

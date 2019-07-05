package winio

//go:generate go run $GOROOT/src/syscall/mksyscall_windows.go -output zsyscall_windows.go file.go pipe.go sd.go fileinfo.go privilege.go backup.go

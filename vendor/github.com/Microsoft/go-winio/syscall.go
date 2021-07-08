package winio

//go:generate go run golang.org/x/sys/windows/mkwinsyscall -output zsyscall_windows.go file.go pipe.go sd.go fileinfo.go privilege.go backup.go hvsock.go

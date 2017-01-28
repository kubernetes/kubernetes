set -e
BASE=$(echo $1 | sed s/.go$//)
go tool cgo -godefs ${BASE}_ignore.go | gofmt > ${BASE}_$2_$3.go
find ${BASE}_$2_$3.go -size 0 -delete

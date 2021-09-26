package vendor_pkg

//go:generate mockgen -destination mock.go -package vendor_pkg golang.org/x/tools/present Elem

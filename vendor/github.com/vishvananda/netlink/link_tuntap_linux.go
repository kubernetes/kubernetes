package netlink

// ideally golang.org/x/sys/unix would define IfReq but it only has
// IFNAMSIZ, hence this minimalistic implementation
const (
	SizeOfIfReq = 40
	IFNAMSIZ    = 16
)

type ifReq struct {
	Name  [IFNAMSIZ]byte
	Flags uint16
	pad   [SizeOfIfReq - IFNAMSIZ - 2]byte
}

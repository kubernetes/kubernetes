package misc

import "fmt"
import "net"
import "strconv"
import "strings"
import "encoding/asn1"
import "encoding/binary"

const (
	TOKEN_NOOP    byte = (1 << 0)
	TOKEN_CONTEXT byte = (1 << 1)
	TOKEN_DATA    byte = (1 << 2)
	TOKEN_MIC     byte = (1 << 3)

	TOKEN_CONTEXT_NEXT byte = (1 << 4)
	TOKEN_WRAPPED      byte = (1 << 5)
	TOKEN_ENCRYPTED    byte = (1 << 6)
	TOKEN_SEND_MIC     byte = (1 << 7)
)

/* ParseOid returns an asn1.ObjectIdentifier based on the dotted form input string. */
func ParseOid(oids string) (oid asn1.ObjectIdentifier) {
	components := strings.Split(oids, ".")
	if len(components) > 0 {
		oid = make([]int, len(components))
		for i, component := range components {
			val, err := strconv.Atoi(component)
			if err != nil {
				fmt.Printf("Error parsing OID \"%s\".\n", oids)
				oid = nil
				return
			}
			oid[i] = val
		}
	}
	return
}

/* SendToken sends a token from the sample GSS client to the sample GSS server, or vice-versa. */
func SendToken(conn net.Conn, tag byte, token []byte) {
	tlen := uint32(len(token))

	if tag != 0 {
		binary.Write(conn, binary.BigEndian, tag)
	}
	binary.Write(conn, binary.BigEndian, tlen)
	if tlen > 0 {
		conn.Write(token)
	}
}

/* RecvToken reads a token sent by SendToken over a newtork connection. */
func RecvToken(conn net.Conn) (tag byte, token []byte) {
	var tlen uint32
	tmp := make([]byte, 1)

	n, err := conn.Read(tmp)
	if n == 0 {
		fmt.Printf("reading token flags: 0 bytes read\n")
		return
	}
	if err != nil {
		fmt.Printf("Error reading flag: %s.\n", err)
		return
	}
	tag = tmp[0]
	if tag != 0 {
		err = binary.Read(conn, binary.BigEndian, &tlen)
		if err != nil {
			fmt.Printf("Error reading tag: %s.\n", err)
			return
		}
	} else {
		tags := make([]byte, 3)
		_, err := conn.Read(tags)
		if err != nil {
			fmt.Printf("Error reading length: %s.\n", err)
			return
		}
		tlen = uint32((tag << 24) | (tags[0] << 16) | (tags[1] << 8) | tags[2])
	}
	if tlen > 0 {
		token = make([]byte, tlen)
		_, err = conn.Read(token)
		if err != nil {
			fmt.Printf("Error reading: %s.\n", err)
			return
		}
	}
	return
}

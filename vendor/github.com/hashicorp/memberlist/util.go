package memberlist

import (
	"bytes"
	"compress/lzw"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net"
	"time"

	"github.com/hashicorp/go-msgpack/codec"
)

// pushPullScale is the minimum number of nodes
// before we start scaling the push/pull timing. The scale
// effect is the log2(Nodes) - log2(pushPullScale). This means
// that the 33rd node will cause us to double the interval,
// while the 65th will triple it.
const pushPullScaleThreshold = 32

/*
 * Contains an entry for each private block:
 * 10.0.0.0/8
 * 100.64.0.0/10
 * 127.0.0.0/8
 * 169.254.0.0/16
 * 172.16.0.0/12
 * 192.168.0.0/16
 */
var privateBlocks []*net.IPNet

var loopbackBlock *net.IPNet

const (
	// Constant litWidth 2-8
	lzwLitWidth = 8
)

func init() {
	// Seed the random number generator
	rand.Seed(time.Now().UnixNano())

	// Add each private block
	privateBlocks = make([]*net.IPNet, 6)

	_, block, err := net.ParseCIDR("10.0.0.0/8")
	if err != nil {
		panic(fmt.Sprintf("Bad cidr. Got %v", err))
	}
	privateBlocks[0] = block

	_, block, err = net.ParseCIDR("100.64.0.0/10")
	if err != nil {
		panic(fmt.Sprintf("Bad cidr. Got %v", err))
	}
	privateBlocks[1] = block

	_, block, err = net.ParseCIDR("127.0.0.0/8")
	if err != nil {
		panic(fmt.Sprintf("Bad cidr. Got %v", err))
	}
	privateBlocks[2] = block

	_, block, err = net.ParseCIDR("169.254.0.0/16")
	if err != nil {
		panic(fmt.Sprintf("Bad cidr. Got %v", err))
	}
	privateBlocks[3] = block

	_, block, err = net.ParseCIDR("172.16.0.0/12")
	if err != nil {
		panic(fmt.Sprintf("Bad cidr. Got %v", err))
	}
	privateBlocks[4] = block

	_, block, err = net.ParseCIDR("192.168.0.0/16")
	if err != nil {
		panic(fmt.Sprintf("Bad cidr. Got %v", err))
	}
	privateBlocks[5] = block

	_, block, err = net.ParseCIDR("127.0.0.0/8")
	if err != nil {
		panic(fmt.Sprintf("Bad cidr. Got %v", err))
	}
	loopbackBlock = block
}

// Decode reverses the encode operation on a byte slice input
func decode(buf []byte, out interface{}) error {
	r := bytes.NewReader(buf)
	hd := codec.MsgpackHandle{}
	dec := codec.NewDecoder(r, &hd)
	return dec.Decode(out)
}

// Encode writes an encoded object to a new bytes buffer
func encode(msgType messageType, in interface{}) (*bytes.Buffer, error) {
	buf := bytes.NewBuffer(nil)
	buf.WriteByte(uint8(msgType))
	hd := codec.MsgpackHandle{}
	enc := codec.NewEncoder(buf, &hd)
	err := enc.Encode(in)
	return buf, err
}

// GetPrivateIP returns the first private IP address found in a list of
// addresses.
func GetPrivateIP(addresses []net.Addr) (net.IP, error) {
	var candidates []net.IP

	// Find private IPv4 address
	for _, rawAddr := range addresses {
		var ip net.IP
		switch addr := rawAddr.(type) {
		case *net.IPAddr:
			ip = addr.IP
		case *net.IPNet:
			ip = addr.IP
		default:
			continue
		}

		if ip.To4() == nil {
			continue
		}
		if !IsPrivateIP(ip.String()) {
			continue
		}
		candidates = append(candidates, ip)
	}
	numIps := len(candidates)
	switch numIps {
	case 0:
		return nil, fmt.Errorf("No private IP address found")
	case 1:
		return candidates[0], nil
	default:
		return nil, fmt.Errorf("Multiple private IPs found. Please configure one.")
	}
}

// Returns a random offset between 0 and n
func randomOffset(n int) int {
	if n == 0 {
		return 0
	}
	return int(rand.Uint32() % uint32(n))
}

// suspicionTimeout computes the timeout that should be used when
// a node is suspected
func suspicionTimeout(suspicionMult, n int, interval time.Duration) time.Duration {
	nodeScale := math.Ceil(math.Log10(float64(n + 1)))
	timeout := time.Duration(suspicionMult) * time.Duration(nodeScale) * interval
	return timeout
}

// retransmitLimit computes the limit of retransmissions
func retransmitLimit(retransmitMult, n int) int {
	nodeScale := math.Ceil(math.Log10(float64(n + 1)))
	limit := retransmitMult * int(nodeScale)
	return limit
}

// shuffleNodes randomly shuffles the input nodes using the Fisher-Yates shuffle
func shuffleNodes(nodes []*nodeState) {
	n := len(nodes)
	for i := n - 1; i > 0; i-- {
		j := rand.Intn(i + 1)
		nodes[i], nodes[j] = nodes[j], nodes[i]
	}
}

// pushPushScale is used to scale the time interval at which push/pull
// syncs take place. It is used to prevent network saturation as the
// cluster size grows
func pushPullScale(interval time.Duration, n int) time.Duration {
	// Don't scale until we cross the threshold
	if n <= pushPullScaleThreshold {
		return interval
	}

	multiplier := math.Ceil(math.Log2(float64(n))-math.Log2(pushPullScaleThreshold)) + 1.0
	return time.Duration(multiplier) * interval
}

// moveDeadNodes moves all the nodes in the dead state
// to the end of the slice and returns the index of the first dead node.
func moveDeadNodes(nodes []*nodeState) int {
	numDead := 0
	n := len(nodes)
	for i := 0; i < n-numDead; i++ {
		if nodes[i].State != stateDead {
			continue
		}

		// Move this node to the end
		nodes[i], nodes[n-numDead-1] = nodes[n-numDead-1], nodes[i]
		numDead++
		i--
	}
	return n - numDead
}

// kRandomNodes is used to select up to k random nodes, excluding a given
// node and any non-alive nodes. It is possible that less than k nodes are returned.
func kRandomNodes(k int, excludes []string, nodes []*nodeState) []*nodeState {
	n := len(nodes)
	kNodes := make([]*nodeState, 0, k)
OUTER:
	// Probe up to 3*n times, with large n this is not necessary
	// since k << n, but with small n we want search to be
	// exhaustive
	for i := 0; i < 3*n && len(kNodes) < k; i++ {
		// Get random node
		idx := randomOffset(n)
		node := nodes[idx]

		// Exclude node if match
		for _, exclude := range excludes {
			if node.Name == exclude {
				continue OUTER
			}
		}

		// Exclude if not alive
		if node.State != stateAlive {
			continue
		}

		// Check if we have this node already
		for j := 0; j < len(kNodes); j++ {
			if node == kNodes[j] {
				continue OUTER
			}
		}

		// Append the node
		kNodes = append(kNodes, node)
	}
	return kNodes
}

// makeCompoundMessage takes a list of messages and generates
// a single compound message containing all of them
func makeCompoundMessage(msgs [][]byte) *bytes.Buffer {
	// Create a local buffer
	buf := bytes.NewBuffer(nil)

	// Write out the type
	buf.WriteByte(uint8(compoundMsg))

	// Write out the number of message
	buf.WriteByte(uint8(len(msgs)))

	// Add the message lengths
	for _, m := range msgs {
		binary.Write(buf, binary.BigEndian, uint16(len(m)))
	}

	// Append the messages
	for _, m := range msgs {
		buf.Write(m)
	}

	return buf
}

// decodeCompoundMessage splits a compound message and returns
// the slices of individual messages. Also returns the number
// of truncated messages and any potential error
func decodeCompoundMessage(buf []byte) (trunc int, parts [][]byte, err error) {
	if len(buf) < 1 {
		err = fmt.Errorf("missing compound length byte")
		return
	}
	numParts := uint8(buf[0])
	buf = buf[1:]

	// Check we have enough bytes
	if len(buf) < int(numParts*2) {
		err = fmt.Errorf("truncated len slice")
		return
	}

	// Decode the lengths
	lengths := make([]uint16, numParts)
	for i := 0; i < int(numParts); i++ {
		lengths[i] = binary.BigEndian.Uint16(buf[i*2 : i*2+2])
	}
	buf = buf[numParts*2:]

	// Split each message
	for idx, msgLen := range lengths {
		if len(buf) < int(msgLen) {
			trunc = int(numParts) - idx
			return
		}

		// Extract the slice, seek past on the buffer
		slice := buf[:msgLen]
		buf = buf[msgLen:]
		parts = append(parts, slice)
	}
	return
}

// Returns if the given IP is in a private block
func IsPrivateIP(ip_str string) bool {
	ip := net.ParseIP(ip_str)
	for _, priv := range privateBlocks {
		if priv.Contains(ip) {
			return true
		}
	}
	return false
}

// Returns if the given IP is in a loopback block
func isLoopbackIP(ip_str string) bool {
	ip := net.ParseIP(ip_str)
	return loopbackBlock.Contains(ip)
}

// compressPayload takes an opaque input buffer, compresses it
// and wraps it in a compress{} message that is encoded.
func compressPayload(inp []byte) (*bytes.Buffer, error) {
	var buf bytes.Buffer
	compressor := lzw.NewWriter(&buf, lzw.LSB, lzwLitWidth)

	_, err := compressor.Write(inp)
	if err != nil {
		return nil, err
	}

	// Ensure we flush everything out
	if err := compressor.Close(); err != nil {
		return nil, err
	}

	// Create a compressed message
	c := compress{
		Algo: lzwAlgo,
		Buf:  buf.Bytes(),
	}
	return encode(compressMsg, &c)
}

// decompressPayload is used to unpack an encoded compress{}
// message and return its payload uncompressed
func decompressPayload(msg []byte) ([]byte, error) {
	// Decode the message
	var c compress
	if err := decode(msg, &c); err != nil {
		return nil, err
	}
	return decompressBuffer(&c)
}

// decompressBuffer is used to decompress the buffer of
// a single compress message, handling multiple algorithms
func decompressBuffer(c *compress) ([]byte, error) {
	// Verify the algorithm
	if c.Algo != lzwAlgo {
		return nil, fmt.Errorf("Cannot decompress unknown algorithm %d", c.Algo)
	}

	// Create a uncompressor
	uncomp := lzw.NewReader(bytes.NewReader(c.Buf), lzw.LSB, lzwLitWidth)
	defer uncomp.Close()

	// Read all the data
	var b bytes.Buffer
	_, err := io.Copy(&b, uncomp)
	if err != nil {
		return nil, err
	}

	// Return the uncompressed bytes
	return b.Bytes(), nil
}

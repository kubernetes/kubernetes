package dns

// StringToType is the reverse of TypeToString, needed for string parsing.
var StringToType = reverseInt16(TypeToString)

// StringToClass is the reverse of ClassToString, needed for string parsing.
var StringToClass = reverseInt16(ClassToString)

// StringToOpcode is a map of opcodes to strings.
var StringToOpcode = reverseInt(OpcodeToString)

// StringToRcode is a map of rcodes to strings.
var StringToRcode = reverseInt(RcodeToString)

func init() {
	// Preserve previous NOTIMP typo, see github.com/miekg/dns/issues/733.
	StringToRcode["NOTIMPL"] = RcodeNotImplemented
}

// StringToAlgorithm is the reverse of AlgorithmToString.
var StringToAlgorithm = reverseInt8(AlgorithmToString)

// StringToHash is a map of names to hash IDs.
var StringToHash = reverseInt8(HashToString)

// StringToCertType is the reverseof CertTypeToString.
var StringToCertType = reverseInt16(CertTypeToString)

// Reverse a map
func reverseInt8(m map[uint8]string) map[string]uint8 {
	n := make(map[string]uint8, len(m))
	for u, s := range m {
		n[s] = u
	}
	return n
}

func reverseInt16(m map[uint16]string) map[string]uint16 {
	n := make(map[string]uint16, len(m))
	for u, s := range m {
		n[s] = u
	}
	return n
}

func reverseInt(m map[int]string) map[string]int {
	n := make(map[string]int, len(m))
	for u, s := range m {
		n[s] = u
	}
	return n
}

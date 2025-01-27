package ntlmssp

// Version is a struct representing https://msdn.microsoft.com/en-us/library/cc236654.aspx
type Version struct {
	ProductMajorVersion uint8
	ProductMinorVersion uint8
	ProductBuild        uint16
	_                   [3]byte
	NTLMRevisionCurrent uint8
}

// DefaultVersion returns a Version with "sensible" defaults (Windows 7)
func DefaultVersion() Version {
	return Version{
		ProductMajorVersion: 6,
		ProductMinorVersion: 1,
		ProductBuild:        7601,
		NTLMRevisionCurrent: 15,
	}
}

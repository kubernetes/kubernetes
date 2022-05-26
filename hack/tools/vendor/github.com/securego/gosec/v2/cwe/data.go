package cwe

import "fmt"

const (
	// Acronym is the acronym of CWE
	Acronym = "CWE"
	// Version the CWE version
	Version = "4.4"
	// ReleaseDateUtc the release Date of CWE Version
	ReleaseDateUtc = "2021-03-15"
	// Organization MITRE
	Organization = "MITRE"
	// Description the description of CWE
	Description = "The MITRE Common Weakness Enumeration"
)

var (
	// InformationURI link to the published CWE PDF
	InformationURI = fmt.Sprintf("https://cwe.mitre.org/data/published/cwe_v%s.pdf/", Version)
	// DownloadURI link to the zipped XML of the CWE list
	DownloadURI = fmt.Sprintf("https://cwe.mitre.org/data/xml/cwec_v%s.xml.zip", Version)

	data = map[string]*Weakness{}

	weaknesses = []*Weakness{
		{
			ID:          "118",
			Description: "The software does not restrict or incorrectly restricts operations within the boundaries of a resource that is accessed using an index or pointer, such as memory or files.",
			Name:        "Incorrect Access of Indexable Resource ('Range Error')",
		},
		{
			ID:          "190",
			Description: "The software performs a calculation that can produce an integer overflow or wraparound, when the logic assumes that the resulting value will always be larger than the original value. This can introduce other weaknesses when the calculation is used for resource management or execution control.",
			Name:        "Integer Overflow or Wraparound",
		},
		{
			ID:          "200",
			Description: "The product exposes sensitive information to an actor that is not explicitly authorized to have access to that information.",
			Name:        "Exposure of Sensitive Information to an Unauthorized Actor",
		},
		{
			ID:          "22",
			Description: "The software uses external input to construct a pathname that is intended to identify a file or directory that is located underneath a restricted parent directory, but the software does not properly neutralize special elements within the pathname that can cause the pathname to resolve to a location that is outside of the restricted directory.",
			Name:        "Improper Limitation of a Pathname to a Restricted Directory ('Path Traversal')",
		},
		{
			ID:          "242",
			Description: "The program calls a function that can never be guaranteed to work safely.",
			Name:        "Use of Inherently Dangerous Function",
		},
		{
			ID:          "276",
			Description: "During installation, installed file permissions are set to allow anyone to modify those files.",
			Name:        "Incorrect Default Permissions",
		},
		{
			ID:          "295",
			Description: "The software does not validate, or incorrectly validates, a certificate.",
			Name:        "Improper Certificate Validation",
		},
		{
			ID:          "310",
			Description: "Weaknesses in this category are related to the design and implementation of data confidentiality and integrity. Frequently these deal with the use of encoding techniques, encryption libraries, and hashing algorithms. The weaknesses in this category could lead to a degradation of the quality data if they are not addressed.",
			Name:        "Cryptographic Issues",
		},
		{
			ID:          "322",
			Description: "The software performs a key exchange with an actor without verifying the identity of that actor.",
			Name:        "Key Exchange without Entity Authentication",
		},
		{
			ID:          "326",
			Description: "The software stores or transmits sensitive data using an encryption scheme that is theoretically sound, but is not strong enough for the level of protection required.",
			Name:        "Inadequate Encryption Strength",
		},
		{
			ID:          "327",
			Description: "The use of a broken or risky cryptographic algorithm is an unnecessary risk that may result in the exposure of sensitive information.",
			Name:        "Use of a Broken or Risky Cryptographic Algorithm",
		},
		{
			ID:          "338",
			Description: "The product uses a Pseudo-Random Number Generator (PRNG) in a security context, but the PRNG's algorithm is not cryptographically strong.",
			Name:        "Use of Cryptographically Weak Pseudo-Random Number Generator (PRNG)",
		},
		{
			ID:          "377",
			Description: "Creating and using insecure temporary files can leave application and system data vulnerable to attack.",
			Name:        "Insecure Temporary File",
		},
		{
			ID:          "409",
			Description: "The software does not handle or incorrectly handles a compressed input with a very high compression ratio that produces a large output.",
			Name:        "Improper Handling of Highly Compressed Data (Data Amplification)",
		},
		{
			ID:          "703",
			Description: "The software does not properly anticipate or handle exceptional conditions that rarely occur during normal operation of the software.",
			Name:        "Improper Check or Handling of Exceptional Conditions",
		},
		{
			ID:          "78",
			Description: "The software constructs all or part of an OS command using externally-influenced input from an upstream component, but it does not neutralize or incorrectly neutralizes special elements that could modify the intended OS command when it is sent to a downstream component.",
			Name:        "Improper Neutralization of Special Elements used in an OS Command ('OS Command Injection')",
		},
		{
			ID:          "79",
			Description: "The software does not neutralize or incorrectly neutralizes user-controllable input before it is placed in output that is used as a web page that is served to other users.",
			Name:        "Improper Neutralization of Input During Web Page Generation ('Cross-site Scripting')",
		},
		{
			ID:          "798",
			Description: "The software contains hard-coded credentials, such as a password or cryptographic key, which it uses for its own inbound authentication, outbound communication to external components, or encryption of internal data.",
			Name:        "Use of Hard-coded Credentials",
		},
		{
			ID:          "88",
			Description: "The software constructs a string for a command to executed by a separate component\nin another control sphere, but it does not properly delimit the\nintended arguments, options, or switches within that command string.",
			Name:        "Improper Neutralization of Argument Delimiters in a Command ('Argument Injection')",
		},
		{
			ID:          "89",
			Description: "The software constructs all or part of an SQL command using externally-influenced input from an upstream component, but it does not neutralize or incorrectly neutralizes special elements that could modify the intended SQL command when it is sent to a downstream component.",
			Name:        "Improper Neutralization of Special Elements used in an SQL Command ('SQL Injection')",
		},
	}
)

func init() {
	for _, weakness := range weaknesses {
		data[weakness.ID] = weakness
	}
}

// Get Retrieves a CWE weakness by it's id
func Get(id string) *Weakness {
	weakness, ok := data[id]
	if ok && weakness != nil {
		return weakness
	}
	return nil
}

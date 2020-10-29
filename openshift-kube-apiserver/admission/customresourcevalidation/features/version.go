package features

var (
	// scos is a setting to enable CentOS Stream CoreOS-only modifications
	scos = false
)

// IsSCOS returns true if CentOS Stream CoreOS-only modifications are enabled
func IsSCOS() bool {
	return scos
}

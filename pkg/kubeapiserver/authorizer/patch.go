package authorizer

var skipSystemMastersAuthorizer = false

// SkipSystemMastersAuthorizer disable implicitly added system/master authz, and turn it into another authz mode "SystemMasters", to be added via authorization-mode
func SkipSystemMastersAuthorizer() {
	skipSystemMastersAuthorizer = true
}

//go:build !linux
// +build !linux

package selinux

func attrPath(string) string {
	return ""
}

func readCon(string) (string, error) {
	return "", nil
}

func writeCon(string, string) error {
	return nil
}

func setDisabled() {}

func getEnabled() bool {
	return false
}

func classIndex(string) (int, error) {
	return -1, nil
}

func setFileLabel(string, string) error {
	return nil
}

func lSetFileLabel(string, string) error {
	return nil
}

func fileLabel(string) (string, error) {
	return "", nil
}

func lFileLabel(string) (string, error) {
	return "", nil
}

func setFSCreateLabel(string) error {
	return nil
}

func fsCreateLabel() (string, error) {
	return "", nil
}

func currentLabel() (string, error) {
	return "", nil
}

func pidLabel(int) (string, error) {
	return "", nil
}

func execLabel() (string, error) {
	return "", nil
}

func canonicalizeContext(string) (string, error) {
	return "", nil
}

func computeCreateContext(string, string, string) (string, error) {
	return "", nil
}

func calculateGlbLub(string, string) (string, error) {
	return "", nil
}

func peerLabel(uintptr) (string, error) {
	return "", nil
}

func setKeyLabel(string) error {
	return nil
}

func (c Context) get() string {
	return ""
}

func newContext(string) (Context, error) {
	return Context{}, nil
}

func clearLabels() {
}

func reserveLabel(string) {
}

func isMLSEnabled() bool {
	return false
}

func enforceMode() int {
	return Disabled
}

func setEnforceMode(int) error {
	return nil
}

func defaultEnforceMode() int {
	return Disabled
}

func releaseLabel(string) {
}

func roFileLabel() string {
	return ""
}

func kvmContainerLabels() (string, string) {
	return "", ""
}

func initContainerLabels() (string, string) {
	return "", ""
}

func containerLabels() (string, string) {
	return "", ""
}

func securityCheckContext(string) error {
	return nil
}

func copyLevel(string, string) (string, error) {
	return "", nil
}

func chcon(string, string, bool) error {
	return nil
}

func dupSecOpt(string) ([]string, error) {
	return nil, nil
}

func getDefaultContextWithLevel(string, string, string) (string, error) {
	return "", nil
}

func label(_ string) string {
	return ""
}

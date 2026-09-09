//go:build !linux

package selinux

func readConThreadSelf(string) (string, error) {
	return "", nil
}

func writeConThreadSelf(string, string) error {
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

func pidLabel(int) (string, error) {
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

func peerLabel(int) (string, error) {
	return "", nil
}

func setKeyLabel(string) error {
	return nil
}

func keyLabel() (string, error) {
	return "", nil
}

func (c Context) get() string {
	return ""
}

func newContext(string) (Context, error) {
	return Context{}, nil
}

func clearLabels() {
}

func reserveLabel(string) error {
	return nil
}

func checkLabel(string) error {
	return nil
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

func kvmContainerLabel() (string, error) {
	return "", nil
}

func initContainerLabels() (string, string) {
	return "", ""
}

func initContainerLabel() (string, error) {
	return "", nil
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

func getSeUserByName(string) (string, string, error) {
	return "", "", nil
}

func getDefaultContextWithLevel(string, string, string) (string, error) {
	return "", nil
}

func label(_ string) string {
	return ""
}

func setProcessKind(string, ProcessKind) (string, error) {
	return "", nil
}

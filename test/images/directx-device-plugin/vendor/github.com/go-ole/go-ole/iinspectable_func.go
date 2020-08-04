// +build !windows

package ole

func (v *IInspectable) GetIids() ([]*GUID, error) {
	return []*GUID{}, NewError(E_NOTIMPL)
}

func (v *IInspectable) GetRuntimeClassName() (string, error) {
	return "", NewError(E_NOTIMPL)
}

func (v *IInspectable) GetTrustLevel() (uint32, error) {
	return uint32(0), NewError(E_NOTIMPL)
}

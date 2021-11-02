// +build !windows

package ole

func getClassInfo(disp *IProvideClassInfo) (tinfo *ITypeInfo, err error) {
	return nil, NewError(E_NOTIMPL)
}

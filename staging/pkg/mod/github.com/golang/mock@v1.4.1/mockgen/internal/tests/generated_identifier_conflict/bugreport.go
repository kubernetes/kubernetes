//go:generate mockgen -destination bugreport_mock.go -package bugreport -source=bugreport.go

package bugreport

type Example interface {
	// _m and _mr were used by the buggy code: the '_' prefix was there hoping
	// that no one will use method argument names starting with '_' reducing
	// the chance of collision with generated identifiers.
	// m and mr are used by the bugfixed new code, the '_' prefix has been
	// removed because the new code generator changes the names of the
	// generated identifiers in case they would collide with identifiers
	// coming from argument names.
	Method(_m, _mr, m, mr int)

	VarargMethod(_s, _x, a, ret int, varargs ...int)
}

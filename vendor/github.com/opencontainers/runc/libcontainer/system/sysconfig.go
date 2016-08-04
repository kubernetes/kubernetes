// +build cgo,linux cgo,freebsd

package system

/*
#include <unistd.h>
#include <limits.h>

int GetLongBit() {
#ifdef _SC_LONG_BIT
    int longbits;

    longbits = sysconf(_SC_LONG_BIT);
    if (longbits <  0) {
        longbits = (CHAR_BIT * sizeof(long));
    }
    return longbits;
#else
    return (CHAR_BIT * sizeof(long));
#endif
}
*/
import "C"

func GetClockTicks() int {
	return int(C.sysconf(C._SC_CLK_TCK))
}

func GetLongBit() int {
	return int(C.GetLongBit())
}

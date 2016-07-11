package stacktrace

import "runtime"

// Capture captures a stacktrace for the current calling go program
//
// skip is the number of frames to skip
func Capture(userSkip int) Stacktrace {
	var (
		skip   = userSkip + 1 // add one for our own function
		frames []Frame
		prevPc uintptr
	)
	for i := skip; ; i++ {
		pc, file, line, ok := runtime.Caller(i)
		//detect if caller is repeated to avoid loop, gccgo
		//currently runs  into a loop without this check
		if !ok || pc == prevPc {
			break
		}
		frames = append(frames, NewFrame(pc, file, line))
		prevPc = pc
	}
	return Stacktrace{
		Frames: frames,
	}
}

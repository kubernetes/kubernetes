package stacktrace

import "runtime"

// Caputure captures a stacktrace for the current calling go program
//
// skip is the number of frames to skip
func Capture(userSkip int) Stacktrace {
	var (
		skip   = userSkip + 1 // add one for our own function
		frames []Frame
	)
	for i := skip; ; i++ {
		pc, file, line, ok := runtime.Caller(i)
		if !ok {
			break
		}
		frames = append(frames, NewFrame(pc, file, line))
	}
	return Stacktrace{
		Frames: frames,
	}
}

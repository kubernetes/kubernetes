package tracefs

import "fmt"

// UprobeToken creates the PATH:OFFSET(REF_CTR_OFFSET) token for the tracefs api.
func UprobeToken(args ProbeArgs) string {
	po := fmt.Sprintf("%s:%#x", args.Path, args.Offset)

	if args.RefCtrOffset != 0 {
		// This is not documented in Documentation/trace/uprobetracer.txt.
		// elixir.bootlin.com/linux/v5.15-rc7/source/kernel/trace/trace.c#L5564
		po += fmt.Sprintf("(%#x)", args.RefCtrOffset)
	}

	return po
}

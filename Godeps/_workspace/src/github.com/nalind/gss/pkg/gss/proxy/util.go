package proxy

import "fmt"
import "io"

/* DisplayProxyStatus prints status error messages associated with the passed-in Status object. */
func DisplayProxyStatus(when string, status Status) {
	fmt.Printf("Error \"%s\" ", status.MajorStatusString)
	if len(when) > 0 {
		fmt.Printf("while %s", when)
	}
	if len(status.MinorStatusString) > 0 {
		fmt.Printf(" (%s)", status.MinorStatusString)
	} else {
		fmt.Printf(" (minor code = 0x%x)", status.MinorStatus)
	}
	fmt.Printf(".\n")
}

/* DisplayProxyFlags logs the contents of the passed-in flags. */
func DisplayProxyFlags(flags Flags, complete bool, file io.Writer) {
	if flags.Deleg {
		fmt.Fprintf(file, "context flag: GSS_C_DELEG_FLAG\n")
	}
	if flags.DelegPolicy {
		fmt.Fprintf(file, "context flag: GSS_C_DELEG_POLICY_FLAG\n")
	}
	if flags.Mutual {
		fmt.Fprintf(file, "context flag: GSS_C_MUTUAL_FLAG\n")
	}
	if flags.Replay {
		fmt.Fprintf(file, "context flag: GSS_C_REPLAY_FLAG\n")
	}
	if flags.Sequence {
		fmt.Fprintf(file, "context flag: GSS_C_SEQUENCE_FLAG\n")
	}
	if flags.Anon {
		fmt.Fprintf(file, "context flag: GSS_C_ANON_FLAG\n")
	}
	if flags.Conf {
		fmt.Fprintf(file, "context flag: GSS_C_CONF_FLAG \n")
	}
	if flags.Integ {
		fmt.Fprintf(file, "context flag: GSS_C_INTEG_FLAG \n")
	}
	if complete {
		if flags.Trans {
			fmt.Fprintf(file, "context flag: GSS_C_TRANS_FLAG \n")
		}
		if flags.ProtReady {
			fmt.Fprintf(file, "context flag: GSS_C_PROT_READY_FLAG \n")
		}
	}
}

package gss

import (
	"bytes"
	"encoding/asn1"
	"errors"
	"fmt"
	"io"
)

/* DisplayError prints error messages associated with the passed-in major and minor error codes. */
func DisplayGSSError(when string, major, minor uint32, mech *asn1.ObjectIdentifier) {
	fmt.Println(NewGSSError(when, major, minor, mech))
}

func NewGSSError(when string, major, minor uint32, mech *asn1.ObjectIdentifier) error {
	var b bytes.Buffer
	fmt.Fprint(&b, DisplayStatus(major, C_GSS_CODE, nil)[3])
	if len(when) > 0 {
		fmt.Fprintf(&b, " while %s", when)
	}
	if mech != nil {
		fmt.Fprintf(&b, " (")
		fmt.Fprint(&b, DisplayStatus(minor, C_MECH_CODE, *mech)[3])
		fmt.Fprintf(&b, ")")
	}

	return errors.New(b.String())
}

/* DisplayGSSFlags logs the contents of the passed-in flags. */
func DisplayGSSFlags(flags Flags, complete bool, file io.Writer) {
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

package config

import (
	"fmt"
	"io"
	"strings"
)

// An Encoder writes config files to an output stream.
type Encoder struct {
	w io.Writer
}

// NewEncoder returns a new encoder that writes to w.
func NewEncoder(w io.Writer) *Encoder {
	return &Encoder{w}
}

// Encode writes the config in git config format to the stream of the encoder.
func (e *Encoder) Encode(cfg *Config) error {
	for _, s := range cfg.Sections {
		if err := e.encodeSection(s); err != nil {
			return err
		}
	}

	return nil
}

func (e *Encoder) encodeSection(s *Section) error {
	if len(s.Options) > 0 {
		if err := e.printf("[%s]\n", s.Name); err != nil {
			return err
		}

		if err := e.encodeOptions(s.Options); err != nil {
			return err
		}
	}

	for _, ss := range s.Subsections {
		if err := e.encodeSubsection(s.Name, ss); err != nil {
			return err
		}
	}

	return nil
}

func (e *Encoder) encodeSubsection(sectionName string, s *Subsection) error {
	//TODO: escape
	if err := e.printf("[%s \"%s\"]\n", sectionName, s.Name); err != nil {
		return err
	}

	return e.encodeOptions(s.Options)
}

func (e *Encoder) encodeOptions(opts Options) error {
	for _, o := range opts {
		pattern := "\t%s = %s\n"
		if strings.Contains(o.Value, "\\") {
			pattern = "\t%s = %q\n"
		}

		if err := e.printf(pattern, o.Key, o.Value); err != nil {
			return err
		}
	}

	return nil
}

func (e *Encoder) printf(msg string, args ...interface{}) error {
	_, err := fmt.Fprintf(e.w, msg, args...)
	return err
}

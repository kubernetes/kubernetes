package auth

import (
	"bytes"
	"strings"
	"testing"

	"k8s.io/cli-runtime/pkg/genericiooptions"
)

func TestNewCmdAuth_Help(t *testing.T) {
	var outBuf, errBuf bytes.Buffer
	streams := genericiooptions.IOStreams{
		In:     bytes.NewReader(nil),
		Out:    &outBuf,
		ErrOut: &errBuf,
	}

	cmd := NewCmdAuth(nil, streams)

	// 🔧 Tell Cobra to print help to these buffers
	cmd.SetOut(&outBuf)
	cmd.SetErr(&errBuf)

	cmd.SetArgs([]string{"--help"})

	err := cmd.Execute()
	if err != nil {
		t.Fatalf("Expected no error, got: %v", err)
	}

	// Cobra may print to ErrOut or Out depending on context
	helpText := outBuf.String() + errBuf.String()

	if !strings.Contains(helpText, "Inspect authorization") {
		t.Errorf("Expected help output to contain 'Inspect authorization', got:\n%s", helpText)
	}
}


func TestNewCmdAuth_SubcommandsRegistered(t *testing.T) {
	var outBuf, errBuf bytes.Buffer
	streams := genericiooptions.IOStreams{
		In:     bytes.NewReader(nil),
		Out:    &outBuf,
		ErrOut: &errBuf,
	}
	cmd := NewCmdAuth(nil, streams)

	expected := []string{"can-i", "reconcile", "whoami"}
	found := map[string]bool{}
	for _, sub := range cmd.Commands() {
		found[sub.Name()] = true
	}

	for _, name := range expected {
		if !found[name] {
			t.Errorf("Expected subcommand '%s' to be registered", name)
		}
	}
}

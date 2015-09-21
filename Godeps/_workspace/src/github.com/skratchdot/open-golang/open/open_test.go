package open

import "testing"

func TestRun(t *testing.T) {
	// shouldn't error
	input := "https://google.com/"
	err := Run(input)
	if err != nil {
		t.Errorf("open.Run(\"%s\") threw an error: %s", input, err)
	}

	// should error
	input = "xxxxxxxxxxxxxxx"
	err = Run(input)
	if err == nil {
		t.Errorf("Run(\"%s\") did not throw an error as expected", input)
	}
}

func TestStart(t *testing.T) {
	// shouldn't error
	input := "https://google.com/"
	err := Start(input)
	if err != nil {
		t.Errorf("open.Start(\"%s\") threw an error: %s", input, err)
	}

	// shouldn't error
	input = "xxxxxxxxxxxxxxx"
	err = Start(input)
	if err != nil {
		t.Errorf("open.Start(\"%s\") shouldn't even fail on invalid input: %s", input, err)
	}
}

func TestRunWith(t *testing.T) {
	// shouldn't error
	input := "https://google.com/"
	app := "firefox"
	err := RunWith(input, app)
	if err != nil {
		t.Errorf("open.RunWith(\"%s\", \"%s\") threw an error: %s", input, app, err)
	}

	// should error
	app = "xxxxxxxxxxxxxxx"
	err = RunWith(input, app)
	if err == nil {
		t.Errorf("RunWith(\"%s\", \"%s\") did not throw an error as expected", input, app)
	}
}

func TestStartWith(t *testing.T) {
	// shouldn't error
	input := "https://google.com/"
	app := "firefox"
	err := StartWith(input, app)
	if err != nil {
		t.Errorf("open.StartWith(\"%s\", \"%s\") threw an error: %s", input, app, err)
	}

	// shouldn't error
	input = "[<Invalid URL>]"
	err = StartWith(input, app)
	if err != nil {
		t.Errorf("StartWith(\"%s\", \"%s\") shouldn't even fail on invalid input: %s", input, app, err)
	}

}

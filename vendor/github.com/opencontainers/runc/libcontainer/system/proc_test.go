package system

import "testing"

func TestParseStartTime(t *testing.T) {
	data := map[string]Stat_t{
		"4902 (gunicorn: maste) S 4885 4902 4902 0 -1 4194560 29683 29929 61 83 78 16 96 17 20 0 1 0 9126532 52965376 1903 18446744073709551615 4194304 7461796 140733928751520 140733928698072 139816984959091 0 0 16781312 137447943 1 0 0 17 3 0 0 9 0 0 9559488 10071156 33050624 140733928758775 140733928758945 140733928758945 140733928759264 0": {
			PID:       4902,
			Name:      "gunicorn: maste",
			State:     'S',
			StartTime: 9126532,
		},
		"9534 (cat) R 9323 9534 9323 34828 9534 4194304 95 0 0 0 0 0 0 0 20 0 1 0 9214966 7626752 168 18446744073709551615 4194304 4240332 140732237651568 140732237650920 140570710391216 0 0 0 0 0 0 0 17 1 0 0 0 0 0 6340112 6341364 21553152 140732237653865 140732237653885 140732237653885 140732237656047 0": {
			PID:       9534,
			Name:      "cat",
			State:     'R',
			StartTime: 9214966,
		},

		"24767 (irq/44-mei_me) S 2 0 0 0 -1 2129984 0 0 0 0 0 0 0 0 -51 0 1 0 8722075 0 0 18446744073709551615 0 0 0 0 0 0 0 2147483647 0 0 0 0 17 1 50 1 0 0 0 0 0 0 0 0 0 0 0": {
			PID:       24767,
			Name:      "irq/44-mei_me",
			State:     'S',
			StartTime: 8722075,
		},
	}
	for line, expected := range data {
		st, err := parseStat(line)
		if err != nil {
			t.Fatal(err)
		}
		if st.PID != expected.PID {
			t.Fatalf("expected PID %q but received %q", expected.PID, st.PID)
		}
		if st.State != expected.State {
			t.Fatalf("expected state %q but received %q", expected.State, st.State)
		}
		if st.Name != expected.Name {
			t.Fatalf("expected name %q but received %q", expected.Name, st.Name)
		}
		if st.StartTime != expected.StartTime {
			t.Fatalf("expected start time %q but received %q", expected.StartTime, st.StartTime)
		}
	}
}

package winio

import (
	"io"
	"io/ioutil"
	"os"
	"syscall"
	"testing"
)

var testFileName string

func TestMain(m *testing.M) {
	f, err := ioutil.TempFile("", "tmp")
	if err != nil {
		panic(err)
	}
	testFileName = f.Name()
	f.Close()
	defer os.Remove(testFileName)
	os.Exit(m.Run())
}

func makeTestFile(makeADS bool) error {
	os.Remove(testFileName)
	f, err := os.Create(testFileName)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.Write([]byte("testing 1 2 3\n"))
	if err != nil {
		return err
	}
	if makeADS {
		a, err := os.Create(testFileName + ":ads.txt")
		if err != nil {
			return err
		}
		defer a.Close()
		_, err = a.Write([]byte("alternate data stream\n"))
		if err != nil {
			return err
		}
	}
	return nil
}

func TestBackupRead(t *testing.T) {
	err := makeTestFile(true)
	if err != nil {
		t.Fatal(err)
	}

	f, err := os.Open(testFileName)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	r := NewBackupFileReader(f, false)
	defer r.Close()
	b, err := ioutil.ReadAll(r)
	if err != nil {
		t.Fatal(err)
	}
	if len(b) == 0 {
		t.Fatal("no data")
	}
}

func TestBackupStreamRead(t *testing.T) {
	err := makeTestFile(true)
	if err != nil {
		t.Fatal(err)
	}

	f, err := os.Open(testFileName)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	r := NewBackupFileReader(f, false)
	defer r.Close()

	br := NewBackupStreamReader(r)
	gotData := false
	gotAltData := false
	for {
		hdr, err := br.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}

		switch hdr.Id {
		case BackupData:
			if gotData {
				t.Fatal("duplicate data")
			}
			if hdr.Name != "" {
				t.Fatalf("unexpected name %s", hdr.Name)
			}
			b, err := ioutil.ReadAll(br)
			if err != nil {
				t.Fatal(err)
			}
			if string(b) != "testing 1 2 3\n" {
				t.Fatalf("incorrect data %v", b)
			}
			gotData = true
		case BackupAlternateData:
			if gotAltData {
				t.Fatal("duplicate alt data")
			}
			if hdr.Name != ":ads.txt:$DATA" {
				t.Fatalf("incorrect name %s", hdr.Name)
			}
			b, err := ioutil.ReadAll(br)
			if err != nil {
				t.Fatal(err)
			}
			if string(b) != "alternate data stream\n" {
				t.Fatalf("incorrect data %v", b)
			}
			gotAltData = true
		default:
			t.Fatalf("unknown stream ID %d", hdr.Id)
		}
	}
	if !gotData || !gotAltData {
		t.Fatal("missing stream")
	}
}

func TestBackupStreamWrite(t *testing.T) {
	f, err := os.Create(testFileName)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	w := NewBackupFileWriter(f, false)
	defer w.Close()

	data := "testing 1 2 3\n"
	altData := "alternate stream\n"

	br := NewBackupStreamWriter(w)
	err = br.WriteHeader(&BackupHeader{Id: BackupData, Size: int64(len(data))})
	if err != nil {
		t.Fatal(err)
	}
	n, err := br.Write([]byte(data))
	if err != nil {
		t.Fatal(err)
	}
	if n != len(data) {
		t.Fatal("short write")
	}

	err = br.WriteHeader(&BackupHeader{Id: BackupAlternateData, Size: int64(len(altData)), Name: ":ads.txt:$DATA"})
	if err != nil {
		t.Fatal(err)
	}
	n, err = br.Write([]byte(altData))
	if err != nil {
		t.Fatal(err)
	}
	if n != len(altData) {
		t.Fatal("short write")
	}

	f.Close()

	b, err := ioutil.ReadFile(testFileName)
	if err != nil {
		t.Fatal(err)
	}
	if string(b) != data {
		t.Fatalf("wrong data %v", b)
	}

	b, err = ioutil.ReadFile(testFileName + ":ads.txt")
	if err != nil {
		t.Fatal(err)
	}
	if string(b) != altData {
		t.Fatalf("wrong data %v", b)
	}
}

func makeSparseFile() error {
	os.Remove(testFileName)
	f, err := os.Create(testFileName)
	if err != nil {
		return err
	}
	defer f.Close()

	const (
		FSCTL_SET_SPARSE    = 0x000900c4
		FSCTL_SET_ZERO_DATA = 0x000980c8
	)

	err = syscall.DeviceIoControl(syscall.Handle(f.Fd()), FSCTL_SET_SPARSE, nil, 0, nil, 0, nil, nil)
	if err != nil {
		return err
	}

	_, err = f.Write([]byte("testing 1 2 3\n"))
	if err != nil {
		return err
	}

	_, err = f.Seek(1000000, 0)
	if err != nil {
		return err
	}

	_, err = f.Write([]byte("more data later\n"))
	if err != nil {
		return err
	}

	return nil
}

func TestBackupSparseFile(t *testing.T) {
	err := makeSparseFile()
	if err != nil {
		t.Fatal(err)
	}

	f, err := os.Open(testFileName)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	r := NewBackupFileReader(f, false)
	defer r.Close()

	br := NewBackupStreamReader(r)
	for {
		hdr, err := br.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}

		t.Log(hdr)
	}
}
